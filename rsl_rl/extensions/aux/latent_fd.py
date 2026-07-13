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

    multimodal_rl's ``ForwardDynamics`` mechanics: residual transition
    ``z' = z + mlp([z, a])``, projector on the prediction side only, plain MSE against the raw
    EMA-encoder latent of the true next feature (no target projector, no normalization), EMA
    updated per optimizer step. The recursion state IS z — the controllability filter: the
    representation must encode what is predictable under the agent's own actions. The EMA
    target is anti-collapse plumbing, not grounding.

    One deliberate deviation, flagged: the reference teacher-forces its sequence loss (every
    step re-encodes the true obs; 1-step FD summed over the window — the temporal-smoothness
    shortcut applies, but the encoder gets a gradient path at every step).
    ``autoregressive=True`` (default) instead feeds the prediction back open-loop for
    ``unroll_steps`` — multi-step in the latent space proper. Actions are the executed rollout
    actions ``a_{t+k}`` either way.
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

    def _loss(
        self, flat: dict[str, torch.Tensor], actions: torch.Tensor, idx: torch.Tensor, num_envs: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        feats = flat[self.feat_group]
        z = self.encoder(feats[idx])
        losses, metrics = [], {}
        with torch.no_grad():
            target = self.ema_encoder(feats[idx + num_envs])
            # Do-nothing floor: distance between consecutive targets. A learned loss that only
            # matches this is exploiting temporal smoothness, not dynamics.
            floor = functional.mse_loss(self.ema_encoder(feats[idx]), target)
            metrics["aux/latent_floor"] = floor.item()
        for k in range(self.unroll_steps):
            z_in = z if (self.autoregressive or k == 0) else self.encoder(feats[idx + k * num_envs])
            # Residual transition (reference DynamicsMLP predicts the state difference).
            z = z_in + self.transition(torch.cat([z_in, actions[idx + k * num_envs]], dim=-1))
            if k > 0:
                with torch.no_grad():
                    target = self.ema_encoder(feats[idx + (k + 1) * num_envs])
            step_loss = functional.mse_loss(self.predictor(z), target)
            losses.append(step_loss)
            metrics[f"aux/latent_k{k + 1}_mse"] = step_loss.item()
        loss = torch.stack(losses).mean()
        metrics["aux/latent_mse"] = loss.item()
        return loss, metrics

    def _post_step(self) -> None:
        with torch.no_grad():
            for p_ema, p in zip(self.ema_encoder.parameters(), self.encoder.parameters()):
                p_ema.lerp_(p, 1.0 - self.ema_tau)
            # Normalizer statistics track the live encoder directly (they are slow-moving already).
            for b_ema, b in zip(self.ema_encoder.buffers(), self.encoder.buffers()):
                b_ema.copy_(b)
