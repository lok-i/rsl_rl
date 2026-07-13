# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grounded state-space forward dynamics: SL, full-rollout autoregression in physical state space."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.storage import RolloutStorage

from .base import AuxObjective


class StateFdAux(AuxObjective):
    """Autoregress the physical state across the whole rollout: ``(ŝ_t, z_t, cond_t, a_t) -> ŝ_{t+1}``.

    AnyAdapter's autoregressive world model (OpenTrack ``compute_world_model_loss``,
    ``world_model_autoregressive=True``), state-for-state: the recursion state is the *physical*
    state (z is a per-step measurement input), seeded with the TRUE state at the window start,
    predictions fed back over all T rollout steps in one scan, and episode resets handled by
    masking the loss at done steps and resetting the carry to the true post-reset state (no
    window masking). Minibatches are env columns — the time axis stays whole, one backward
    through the full scan. Deviations from the reference: the encoder reads the real image
    feature at each step (future images cannot be imagined, unlike their proprio history), and
    the loss is normalized MSE instead of per-slice-weighted L1 (the target normalizer plays
    the weights' role and keeps the same lens as StateRegAux).

    The true-s seed makes step 1 nearly image-free (dead reckoning), but autoregression decay
    over T steps means the image is the only error-correcting input for the rest of the window.
    Keep the target slices to the image-dependent state (object) and let the conditioning
    (proprio/command) supply the robot side.
    """

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
        kwargs["unroll_steps"] = 0  # horizon = whole rollout; base window indexing unused
        obs: TensorDict = storage.observations
        encoder = actor.encoders[feat_group]
        self.target_slices = self._resolve_slices(obs[target_group].shape[-1], target_slices)
        target_dim = sum(b - a for a, b in self.target_slices.values())
        cond_dim = sum(obs[g].shape[-1] for g in condition_groups)
        action_dim = storage.actions.shape[-1]
        super().__init__(
            storage=storage,
            actor=actor,
            predictor_input_dim=target_dim + encoder.latent_dim + cond_dim + action_dim,
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

    def update(self, storage: RolloutStorage, actor: nn.Module) -> dict[str, float]:
        """Trajectory minibatching: sample env columns, autoregress over the full time axis."""
        num_t, num_envs = storage.num_transitions_per_env, storage.num_envs
        obs = storage.observations
        mini_batch_size = num_envs // self.num_mini_batches
        if mini_batch_size == 0:
            return {}

        totals: dict[str, float] = {}
        num_updates = 0
        for _ in range(self.num_epochs):
            perm = torch.randperm(num_envs, device=storage.dones.device)
            for i in range(self.num_mini_batches):
                cols = perm[i * mini_batch_size : (i + 1) * mini_batch_size]
                loss, metrics = self._scan_loss(obs, storage.actions, storage.dones, cols, num_t)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for group in self.optimizer.param_groups for p in group["params"]], self.max_grad_norm
                )
                self.optimizer.step()
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + value
                num_updates += 1
        out = {key: value / num_updates for key, value in totals.items()}
        out.update(self._latent_metrics(obs[self.feat_group].flatten(0, 1)))
        return out

    def _scan_loss(
        self, obs: TensorDict, actions: torch.Tensor, dones: torch.Tensor, cols: torch.Tensor, num_t: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        target = self._select_target(obs[self.target_group][:, cols])  # (T, mb, D), true states
        self.target_normalizer.update(target.flatten(0, 1))
        target = self.target_normalizer(target)
        cond = None
        if self.condition_groups:
            cond = torch.cat([obs[g][:, cols] for g in self.condition_groups], dim=-1)
            if isinstance(self.cond_normalizer, EmpiricalNormalization):
                self.cond_normalizer.update(cond.flatten(0, 1))
            cond = self.cond_normalizer(cond)
        not_done = 1.0 - dones.squeeze(-1)[:, cols].float()  # (T, mb)

        s_hat = target[0]  # seed: TRUE state at the window start (reference behavior)
        step_losses, sq_errs = [], []
        for t in range(num_t - 1):
            parts = [s_hat, self.encoder(obs[self.feat_group][t, cols])]
            if cond is not None:
                parts.append(cond[t])
            parts.append(actions[t, cols])
            s_hat = self.predictor(torch.cat(parts, dim=-1))
            # Loss masked at resets; carry reset to the true post-reset state (target[t+1] IS
            # the next observation of step t).
            sq_err = (s_hat - target[t + 1]).square()
            mask = not_done[t]
            step_losses.append((sq_err.mean(-1) * mask).sum() / mask.sum().clamp(min=1.0))
            sq_errs.append(sq_err[mask.bool()])
            s_hat = torch.where(not_done[t].unsqueeze(-1).bool(), s_hat, target[t + 1])
        loss = torch.stack(step_losses).mean()
        metrics = {"aux/fd_mse": loss.item()}
        with torch.no_grad():
            half = len(step_losses) // 2
            metrics["aux/fd_mse_early"] = torch.stack(step_losses[:half]).mean().item()
            metrics["aux/fd_mse_late"] = torch.stack(step_losses[half:]).mean().item()
            metrics.update(self._slice_metrics(torch.cat(sq_errs, dim=0), "aux/fd_"))
        return loss, metrics
