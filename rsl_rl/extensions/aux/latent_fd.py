# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Latent forward dynamics: SSL, K-step prediction in latent space against an EMA target."""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from torch.nn import functional
from typing import Any

from rsl_rl.modules import MLP
from rsl_rl.storage import RolloutStorage

from .base import AuxObjective


class LatentFdAux(AuxObjective):
    """Action-conditioned K-step latent prediction against a slow (EMA) target encoder.

    multimodal_rl's ``ForwardDynamics`` mechanics: residual transition ``z' = z + mlp([z, a])``,
    projector on the prediction side only, plain MSE against the raw EMA-encoder latent of the
    true next feature (no target projector, no normalization), EMA updated per gradient step.
    The recursion state IS z — the controllability filter: the representation must encode what
    is predictable under the agent's own actions. The EMA target is anti-collapse plumbing,
    not grounding.

    The transition is also conditioned on ``condition_group`` (robot state r, default on — the
    reference encoder ingests proprio directly, ours reaches it only as an attention QUERY whose
    value never enters z). Two things this buys: input parity with :class:`StateFdAux`, whose
    predictor reads the same r, and no reward for re-encoding proprio into z — the easy,
    already-available modality — leaving object content as z's only contribution. As in the SL
    variant, step k reads the TRUE ``r_{t+k}``, so the chain is open-loop in z alone.

    Loss composition (``autoregressive=True``): ``L = (Σ_k L_TF^k + Σ_{k>1} L_AR^k) / (2K-1)``
    — the teacher-forced sum is the reference loss verbatim (fresh-encoded true ``z_k`` each
    step: K encoder gradient paths per window), and the open-loop chain (prediction fed back
    from ``z_0``) adds the multi-step consistency the reference lacks. A pure open-loop unroll
    gives the encoder only ONE gradient path (through ``z_0``) and mostly trains the transition
    — the wave-2 failure mode. ``autoregressive=False`` is reference-exact (TF only). Actions
    are the executed rollout actions ``a_{t+k}`` in both branches.
    """

    def __init__(
        self,
        storage: RolloutStorage,
        actor: nn.Module,
        ema_tau: float = 0.99,
        autoregressive: bool = True,
        transition_hidden_dims: tuple[int, ...] | list[int] = (512, 256),
        **kwargs: Any,
    ) -> None:
        """Initialize the latent objective; extra kwargs go to :class:`AuxObjective`.

        The base predictor is the projector (reference: Linear-ELU-Linear, set
        ``predictor_hidden_dims`` accordingly at the call site). Conditioning enters the
        transition only — the projector stays z -> z (BYOL-side plumbing).
        """
        kwargs.setdefault("condition_group", "prediction_conditioning")
        extractor = actor.extractors[kwargs.get("extractor_group", "extractor_input")]
        action_dim = storage.actions.shape[-1]
        kwargs.setdefault("predictor_hidden_dims", (extractor.latent_dim,))
        super().__init__(
            storage=storage,
            actor=actor,
            predictor_input_dim=extractor.latent_dim,
            predictor_output_dim=extractor.latent_dim,
            **kwargs,
        )
        assert self.unroll_steps >= 1, "LatentFdAux needs unroll_steps >= 1."
        self.ema_tau = ema_tau
        self.autoregressive = autoregressive
        activation = kwargs.get("activation", "elu")
        self.transition = MLP(
            extractor.latent_dim + action_dim + self.cond_dim,
            extractor.latent_dim,
            transition_hidden_dims,
            activation,
        )
        self.ema_extractor = copy.deepcopy(extractor)
        self.ema_extractor.requires_grad_(False)
        self._finalize()

    def _obs_keys(self) -> list[str]:
        return [self.condition_group] if self.condition_group is not None else []

    def _step(self, z: torch.Tensor, action: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        """Residual transition (reference DynamicsMLP predicts the state difference)."""
        parts = [z, action] if cond is None else [z, action, cond]
        return z + self.transition(torch.cat(parts, dim=-1))

    def _loss(
        self, flat: dict[str, torch.Tensor], actions: torch.Tensor, idx: torch.Tensor, num_envs: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        losses_tf, losses_ar, metrics = [], [], {}
        # Both encoder streams read STORED rows at fixed offsets — nothing here depends on the
        # AR recursion, so each runs ONCE over its whole (steps x mb) block instead of once per
        # step (see StateFdAux._encode_span for the same argument).
        z_live = self._encode_span(flat, idx, num_envs, self.unroll_steps)
        with torch.no_grad():
            z_tgt = self._encode_span(flat, idx, num_envs, self.unroll_steps + 1, self.ema_extractor)
            # Do-nothing floor: distance between consecutive targets. A learned loss that only
            # matches this is exploiting temporal smoothness, not dynamics.
            metrics["ZPrediction/floor"] = functional.mse_loss(z_tgt[0], z_tgt[1]).item()
        z_ar, cond = None, None
        for k in range(self.unroll_steps):
            action = actions[idx + k * num_envs]
            if self.condition_group is not None:  # TRUE r_{t+k}, both branches (SL-variant parity)
                cond = self._cond(flat[self.condition_group][idx + k * num_envs], update_stats=(k == 0))
            target = z_tgt[k + 1]
            # TF branch (reference-exact): fresh-encoded true z_k, 1-step prediction.
            z_tf = z_live[k]
            losses_tf.append(functional.mse_loss(self.predictor(self._step(z_tf, action, cond)), target))
            # AR branch (deviation): open-loop chain from z_0, multi-step consistency.
            if self.autoregressive:
                z_ar = self._step(z_tf if k == 0 else z_ar, action, cond)
                if k > 0:  # k = 0 is identical to the TF term — count it once
                    losses_ar.append(functional.mse_loss(self.predictor(z_ar), target))
        loss = torch.stack(losses_tf + losses_ar).mean()
        metrics["loss/z_mse"] = loss.item()
        metrics["ZPrediction/total"] = loss.item()
        metrics["ZPrediction/tf"] = torch.stack(losses_tf).mean().item()
        if losses_ar:
            metrics["ZPrediction/ar"] = torch.stack(losses_ar).mean().item()
        return loss, metrics

    def _encode_span(
        self,
        flat: dict[str, torch.Tensor],
        idx: torch.Tensor,
        num_envs: int,
        steps: int,
        extractor: nn.Module | None = None,
    ) -> torch.Tensor:
        """Encode ``steps`` consecutive offsets of ``idx`` in one pass -> ``(steps, mb, latent)``.

        ``flat`` is the (t, env)-flattened storage, so step k of window start ``i`` is the row
        ``i + k * num_envs``; concatenating the offsets keeps the block order (step, sample).
        """
        rows = torch.cat([idx + k * num_envs for k in range(steps)])
        return self._encode(flat, rows, extractor=extractor).view(steps, len(idx), -1)

    def _post_step(self) -> None:
        with torch.no_grad():
            for p_ema, p in zip(self.ema_extractor.parameters(), self.extractor.parameters()):
                p_ema.lerp_(p, 1.0 - self.ema_tau)
            # Normalizer statistics track the live encoder directly (they are slow-moving already).
            for b_ema, b in zip(self.ema_extractor.buffers(), self.extractor.buffers()):
                b_ema.copy_(b)
