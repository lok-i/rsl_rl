# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Privileged state regression: SL, ``(z_t, cond_t) -> state slice @ t`` (e.g. object pose)."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.storage import RolloutStorage

from .base import AuxObjective


class StateRegAux(AuxObjective):
    """Regress a privileged (noise-free) state slice from the feature latent + robot conditioning.

    The engineered-relevance filter — and the cheapest probe of whether the features carry
    the designated state at all. Deliberately dynamics-free (unroll_steps = 0): its value as
    the control variant is measuring decodability with no temporal machinery.

    Conditioning exists because the target frame and the camera frame differ by robot
    kinematics (waist between head camera and base): a base-frame target is not identifiable
    from the image alone. Condition on the robot slice only — object state in the input would
    bypass the image and void the probe.
    """

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
        kwargs["unroll_steps"] = 0
        obs: TensorDict = storage.observations
        encoder = actor.encoders[feat_group]
        self.target_slices = self._resolve_slices(obs[target_group].shape[-1], target_slices)
        target_dim = sum(b - a for a, b in self.target_slices.values())
        cond_dim = self.cond_dim_of(obs, kwargs.get("condition_group"), kwargs.get("condition_slices"))
        super().__init__(
            storage=storage,
            actor=actor,
            predictor_input_dim=encoder.latent_dim + cond_dim,
            predictor_output_dim=target_dim,
            feat_group=feat_group,
            **kwargs,
        )
        self.target_group = target_group
        self.target_normalizer = EmpiricalNormalization(target_dim)
        self._finalize()

    def _obs_keys(self) -> list[str]:
        keys = [self.feat_group, self.target_group]
        if self.condition_group is not None and self.condition_group not in keys:
            keys.append(self.condition_group)
        return keys

    def _loss(
        self, flat: dict[str, torch.Tensor], actions: torch.Tensor, idx: torch.Tensor, num_envs: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        parts = [self.encoder(flat[self.feat_group][idx])]
        if self.condition_group is not None:
            parts.append(self._cond(flat[self.condition_group][idx], update_stats=True))
        target = self._select_target(flat[self.target_group][idx])
        self.target_normalizer.update(target)
        target = self.target_normalizer(target)
        sq_err = (self.predictor(torch.cat(parts, dim=-1)) - target).square()
        loss = sq_err.mean()
        metrics = {"loss/reg_mse": loss.item(), **self._slice_metrics(sq_err, "reg_mse_")}
        return loss, metrics
