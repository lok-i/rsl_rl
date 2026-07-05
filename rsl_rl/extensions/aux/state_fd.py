# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grounded state-space forward dynamics: SL, ``(z_t, cond_t, a_t) -> state @ t+1``."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.storage import RolloutStorage

from .base import AuxObjective


class StateFdAux(AuxObjective):
    """Predict the next physical state from the feature latent, conditioning obs, and action.

    Targets are pinned by the simulator (no collapse, no EMA plumbing). The conditioning
    groups (e.g. proprioception) make the robot part predictable without vision — so watch
    the per-slice metrics: the image-dependent residual is the object state.
    """

    requires_next = True

    def __init__(
        self,
        storage: RolloutStorage,
        actor: nn.Module,
        feat_group: str = "img_feat",
        target_group: str = "aux_state",
        condition_groups: tuple[str, ...] | list[str] = (),
        target_slices: dict[str, tuple[int, int]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the state-FD objective; extra kwargs go to :class:`AuxObjective`."""
        obs: TensorDict = storage.observations
        encoder = actor.encoders[feat_group]
        self.target_slices = self._resolve_slices(obs[target_group].shape[-1], target_slices)
        target_dim = sum(b - a for a, b in self.target_slices.values())
        cond_dim = sum(obs[g].shape[-1] for g in condition_groups)
        action_dim = storage.actions.shape[-1]
        super().__init__(
            storage=storage,
            actor=actor,
            predictor_input_dim=encoder.latent_dim + cond_dim + action_dim,
            predictor_output_dim=target_dim,
            feat_group=feat_group,
            **kwargs,
        )
        self.target_group = target_group
        self.condition_groups = list(condition_groups)
        self.cond_normalizer = EmpiricalNormalization(cond_dim) if cond_dim > 0 else nn.Identity()
        self.target_normalizer = EmpiricalNormalization(target_dim)
        self._finalize()

    def _obs_keys(self) -> list[str]:
        return [self.feat_group, self.target_group, *self.condition_groups]

    def _loss(
        self, flat: dict[str, torch.Tensor], actions: torch.Tensor, idx: torch.Tensor, num_envs: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        parts = [self.encoder(flat[self.feat_group][idx])]
        if self.condition_groups:
            cond = torch.cat([flat[g][idx] for g in self.condition_groups], dim=-1)
            if isinstance(self.cond_normalizer, EmpiricalNormalization):
                self.cond_normalizer.update(cond)
            parts.append(self.cond_normalizer(cond))
        parts.append(actions[idx])
        target = self._select_target(flat[self.target_group][idx + num_envs])
        self.target_normalizer.update(target)
        target = self.target_normalizer(target)
        sq_err = (self.predictor(torch.cat(parts, dim=-1)) - target).square()
        loss = sq_err.mean()
        metrics = {"aux/fd_mse": loss.item(), **self._slice_metrics(sq_err, "aux/fd_")}
        return loss, metrics
