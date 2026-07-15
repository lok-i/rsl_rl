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
        feat_group: str = "img_feat",
        ema_tau: float = 0.99,
        autoregressive: bool = True,
        transition_hidden_dims: tuple[int, ...] | list[int] = (512, 256),
        **kwargs: Any,
    ) -> None:
        """Initialize the latent objective; extra kwargs go to :class:`AuxObjective`.

        The base predictor is the projector (reference: Linear-ELU-Linear, set
        ``predictor_hidden_dims`` accordingly at the call site).
        """
        encoder = actor.encoders[feat_group]
        action_dim = storage.actions.shape[-1]
        kwargs.setdefault("predictor_hidden_dims", (encoder.latent_dim,))
        super().__init__(
            storage=storage,
            actor=actor,
            predictor_input_dim=encoder.latent_dim,
            predictor_output_dim=encoder.latent_dim,
            feat_group=feat_group,
            **kwargs,
        )
        assert self.unroll_steps >= 1, "LatentFdAux needs unroll_steps >= 1."
        self.ema_tau = ema_tau
        self.autoregressive = autoregressive
        activation = kwargs.get("activation", "elu")
        self.transition = MLP(encoder.latent_dim + action_dim, encoder.latent_dim, transition_hidden_dims, activation)
        self.ema_encoder = copy.deepcopy(encoder)
        self.ema_encoder.requires_grad_(False)
        self._finalize()

    def _obs_keys(self) -> list[str]:
        return [self.feat_group]

    def _step(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Residual transition (reference DynamicsMLP predicts the state difference)."""
        return z + self.transition(torch.cat([z, action], dim=-1))

    def _loss(
        self, flat: dict[str, torch.Tensor], actions: torch.Tensor, idx: torch.Tensor, num_envs: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        feats = flat[self.feat_group]
        losses_tf, losses_ar, metrics = [], [], {}
        with torch.no_grad():
            # Do-nothing floor: distance between consecutive targets. A learned loss that only
            # matches this is exploiting temporal smoothness, not dynamics.
            tgt_prev = self.ema_encoder(feats[idx])
            tgt_next = self.ema_encoder(feats[idx + num_envs])
            metrics["latent_floor"] = functional.mse_loss(tgt_prev, tgt_next).item()
        z_ar = None
        for k in range(self.unroll_steps):
            action = actions[idx + k * num_envs]
            with torch.no_grad():
                target = self.ema_encoder(feats[idx + (k + 1) * num_envs]) if k > 0 else tgt_next
            # TF branch (reference-exact): fresh-encoded true z_k, 1-step prediction.
            z_tf = self.encoder(feats[idx + k * num_envs])
            losses_tf.append(functional.mse_loss(self.predictor(self._step(z_tf, action)), target))
            # AR branch (deviation): open-loop chain from z_0, multi-step consistency.
            if self.autoregressive:
                z_ar = self._step(z_tf if k == 0 else z_ar, action)
                if k > 0:  # k = 0 is identical to the TF term — count it once
                    losses_ar.append(functional.mse_loss(self.predictor(z_ar), target))
        loss = torch.stack(losses_tf + losses_ar).mean()
        metrics["loss/latent_mse"] = loss.item()
        metrics["latent_tf_mse"] = torch.stack(losses_tf).mean().item()
        if losses_ar:
            metrics["latent_ar_mse"] = torch.stack(losses_ar).mean().item()
        return loss, metrics

    def _post_step(self) -> None:
        with torch.no_grad():
            for p_ema, p in zip(self.ema_encoder.parameters(), self.encoder.parameters()):
                p_ema.lerp_(p, 1.0 - self.ema_tau)
            # Normalizer statistics track the live encoder directly (they are slow-moving already).
            for b_ema, b in zip(self.ema_encoder.buffers(), self.encoder.buffers()):
                b_ema.copy_(b)
