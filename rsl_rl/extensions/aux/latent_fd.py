# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Latent forward dynamics: SSL, ``(z_t, a_t) -> sg[ema_encoder(feat_t+1)]`` (SPR-style)."""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from torch.nn import functional
from typing import Any

from rsl_rl.storage import RolloutStorage

from .base import AuxObjective


class LatentFdAux(AuxObjective):
    """Action-conditioned latent prediction against a slow (EMA) target encoder.

    The controllability filter: the representation must encode what is predictable under the
    agent's own actions. The EMA target is anti-collapse plumbing, not grounding — its
    parameters follow the live encoder by polyak averaging and receive no gradients.
    """

    requires_next = True

    def __init__(
        self,
        storage: RolloutStorage,
        actor: nn.Module,
        feat_group: str = "img_feat",
        ema_tau: float = 0.995,
        **kwargs: Any,
    ) -> None:
        """Initialize the latent objective; extra kwargs go to :class:`AuxObjective`."""
        encoder = actor.encoders[feat_group]
        action_dim = storage.actions.shape[-1]
        super().__init__(
            storage=storage,
            actor=actor,
            predictor_input_dim=encoder.latent_dim + action_dim,
            predictor_output_dim=encoder.latent_dim,
            feat_group=feat_group,
            **kwargs,
        )
        self.ema_tau = ema_tau
        self.ema_encoder = copy.deepcopy(encoder)
        self.ema_encoder.requires_grad_(False)
        self._finalize()

    def _obs_keys(self) -> list[str]:
        return [self.feat_group]

    def _loss(
        self, flat: dict[str, torch.Tensor], actions: torch.Tensor, idx: torch.Tensor, num_envs: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        z = self.encoder(flat[self.feat_group][idx])
        pred = self.predictor(torch.cat([z, actions[idx]], dim=-1))
        with torch.no_grad():
            target = self.ema_encoder(flat[self.feat_group][idx + num_envs])
        # BYOL/SPR cosine loss on l2-normalized latents
        pred, target = functional.normalize(pred, dim=-1), functional.normalize(target, dim=-1)
        loss = (2.0 - 2.0 * (pred * target).sum(dim=-1)).mean()
        return loss, {"aux/latent_cos": loss.item()}

    def _post_step(self) -> None:
        with torch.no_grad():
            for p_ema, p in zip(self.ema_encoder.parameters(), self.encoder.parameters()):
                p_ema.lerp_(p, 1.0 - self.ema_tau)
            # Normalizer statistics track the live encoder directly (they are slow-moving already).
            for b_ema, b in zip(self.ema_encoder.buffers(), self.encoder.buffers()):
                b_ema.copy_(b)
