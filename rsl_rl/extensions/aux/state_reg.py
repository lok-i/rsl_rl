# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Privileged state regression: SL, ``z_t -> state slice @ t`` (e.g. object pose)."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.storage import RolloutStorage

from .base import AuxObjective


class StateRegAux(AuxObjective):
    """Regress a privileged (noise-free) state slice from the feature latent.

    The engineered-relevance filter — and the cheapest probe of whether the features carry
    the designated state at all.
    """

    requires_next = False

    def __init__(
        self,
        storage: RolloutStorage,
        actor: nn.Module,
        feat_group: str = "img_feat",
        target_group: str = "aux_state",
        target_slices: dict[str, tuple[int, int]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the regression objective; extra kwargs go to :class:`AuxObjective`."""
        obs: TensorDict = storage.observations
        encoder = actor.encoders[feat_group]
        self.target_slices = self._resolve_slices(obs[target_group].shape[-1], target_slices)
        target_dim = sum(b - a for a, b in self.target_slices.values())
        super().__init__(
            storage=storage,
            actor=actor,
            predictor_input_dim=encoder.latent_dim,
            predictor_output_dim=target_dim,
            feat_group=feat_group,
            **kwargs,
        )
        self.target_group = target_group
        self.target_normalizer = EmpiricalNormalization(target_dim)
        self._finalize()

    def _obs_keys(self) -> list[str]:
        return [self.feat_group, self.target_group]

    def _loss(
        self, flat: dict[str, torch.Tensor], actions: torch.Tensor, idx: torch.Tensor, num_envs: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        z = self.encoder(flat[self.feat_group][idx])
        target = self._select_target(flat[self.target_group][idx])
        self.target_normalizer.update(target)
        target = self.target_normalizer(target)
        sq_err = (self.predictor(z) - target).square()
        loss = sq_err.mean()
        metrics = {"aux/reg_mse": loss.item(), **self._slice_metrics(sq_err, "aux/reg_")}
        return loss, metrics
