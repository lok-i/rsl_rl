# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Supervised forward dynamics: SL, K-step prediction of a simulator-supervised target."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.storage import RolloutStorage

from .base import AuxObjective


class StateFdAux(AuxObjective):
    """Supervised forward dynamics over K-step windows: ``(ŝ_k, z_k, cond_k, a_k) -> ŝ_{k+1}``.

    The target ``s`` is any simulator-supervised vector (physical object state, current
    up-face color, task-reward rates — whatever ``prediction_target`` carries), so "state" here
    means the supervised recursion state, not physical state only. Mechanics follow AnyAdapter's
    autoregressive world model (OpenTrack ``compute_world_model_loss``): the rollout is chunked
    into windows of ``unroll_steps`` predictions (reference ``unroll_length=10``), episode
    resets handled by masking the loss at done steps and resetting the carry to the true
    post-reset state (no window masking). Loss is L1 on normalized targets (reference:
    per-slice-weighted L1 — the normalizer plays the weights' role). Minibatches are env
    columns — the time axis of each window stays whole, one backward through all windows.
    Deviation from the reference: the encoder reads the real image feature at each step
    (future images cannot be imagined, unlike their proprio history).

    Two switches span the SL family (one predictor, one input layout):

    - ``start_with_current_step``: shift the K window targets from ``{s_{t+1}..s_{t+K}}`` to
      ``{s_t..s_{t+K-1}}``. The first prediction is then a same-step regression
      ``(0, z_t, cond_t, 0) -> s_t`` (state/action slots zeroed — no transition is involved),
      and with ``autoregress`` the chain seeds from that estimate instead of the true state
      (fully image-grounded window, no dead-reckoning crutch).
    - ``autoregress``: feed predictions back along the window (default). ``False`` = teacher
      forcing — every FD step reads the TRUE previous state.

    ``start_with_current_step=True, autoregress=False, unroll_steps=1`` is the dynamics-free
    decodability probe (the retired StateRegAux, modulo two constant-zero input blocks).

    With the default true-s window seed, the first window steps are nearly image-free (dead
    reckoning); the later steps are only correctable through z. Keep the target slices to the
    image-dependent state (object) and condition on the robot slice only (object state in the
    input would bypass the image).
    """

    def __init__(
        self,
        storage: RolloutStorage,
        actor: nn.Module,
        target_group: str = "prediction_target",
        target_slices: dict[str, tuple[int, int]] | None = None,
        autoregress: bool = True,
        start_with_current_step: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the supervised-FD objective; extra kwargs go to :class:`AuxObjective`."""
        kwargs.setdefault("condition_group", "prediction_conditioning")
        obs: TensorDict = storage.observations
        extractor = actor.extractors[kwargs.get("extractor_group", "extractor_input")]
        self.target_slices = self._resolve_slices(obs[target_group].shape[-1], target_slices)
        target_dim = sum(b - a for a, b in self.target_slices.values())
        cond_dim = self.cond_dim_of(obs, kwargs.get("condition_group"), kwargs.get("condition_slices"))
        action_dim = storage.actions.shape[-1]
        super().__init__(
            storage=storage,
            actor=actor,
            predictor_input_dim=target_dim + extractor.latent_dim + cond_dim + action_dim,
            predictor_output_dim=target_dim,
            **kwargs,
        )
        assert 1 <= self.unroll_steps <= storage.num_transitions_per_env - 1, (
            f"StateFdAux window ({self.unroll_steps}) must fit the rollout "
            f"({storage.num_transitions_per_env} steps)."
        )
        self.autoregress = autoregress
        self.start_with_current_step = start_with_current_step
        # every window contributes K predictions -> K encoder rows per sampled env column
        self._rows_per_col = len(self._window_starts(storage.num_transitions_per_env)) * self.unroll_steps
        self.target_group = target_group
        self.target_normalizer = EmpiricalNormalization(target_dim)
        self._finalize()

    def _obs_keys(self) -> list[str]:
        keys = [self.target_group]
        if self.condition_group is not None and self.condition_group not in keys:
            keys.append(self.condition_group)
        return keys

    def _window_starts(self, num_t: int) -> range:
        """Disjoint window starts covering the rollout (last partial window dropped)."""
        return range(0, num_t - self.unroll_steps, self.unroll_steps)

    def sample_loss(self, storage: RolloutStorage) -> tuple[torch.Tensor, dict[str, float]] | None:
        """One env-column minibatch over all windows (joint mode)."""
        mini_batch_size = self._sample_count(self._rows_per_col, storage.num_envs, storage.num_envs)
        if mini_batch_size == 0:
            return None
        cols = torch.randint(storage.num_envs, (mini_batch_size,), device=storage.dones.device)
        return self._scan_loss(storage, cols)

    def update(self, storage: RolloutStorage, actor: nn.Module) -> dict[str, float]:
        """Sequential mode: env-column minibatches, autoregress over the window'd time axis."""
        mini_batch_size = storage.num_envs // self.num_mini_batches
        if mini_batch_size == 0:
            return {}
        totals: dict[str, float] = {}
        num_updates = 0
        for _ in range(self.num_epochs):
            perm = torch.randperm(storage.num_envs, device=storage.dones.device)
            for i in range(self.num_mini_batches):
                cols = perm[i * mini_batch_size : (i + 1) * mini_batch_size]
                loss, metrics = self._scan_loss(storage, cols)
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
        out.update(self.latent_metrics(storage))
        return out

    def _scan_loss(self, storage: RolloutStorage, cols: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        obs, num_t = storage.observations, storage.num_transitions_per_env
        target = self._select_target(obs[self.target_group][:, cols])  # (T, mb, D), true states
        self.target_normalizer.update(target.flatten(0, 1))
        target = self.target_normalizer(target)
        cond = self._cond(obs[self.condition_group][:, cols], update_stats=True) if self.condition_group else None
        not_done = 1.0 - storage.dones.squeeze(-1)[:, cols].float()  # (T, mb)

        # Every encode in a window reads STORED observations — only s_hat is sequential.
        # So the extractor runs once over the whole (windows x K x mb) block instead of
        # once per step: same math (it is row-independent), ~K-fold fewer kernel launches,
        # and GEMM shapes K times larger. Activations were already all live in one
        # backward graph, so peak memory is unchanged.
        starts = self._window_starts(num_t)
        z = self._encode_span(obs, starts, cols)

        step_losses, err_sums, err_counts = [], [], []
        for w, start in enumerate(starts):
            if self.start_with_current_step:
                # k=0 same-step regression: (0, z_t, cond_t, 0) -> s_t. No transition is
                # involved (no mask); the chain seed is the estimate iff autoregressive.
                parts = [torch.zeros_like(target[start]), z[w, 0]]
                if cond is not None:
                    parts.append(cond[start])
                parts.append(torch.zeros_like(storage.actions[start, cols]))
                s_hat = self.predictor(torch.cat(parts, dim=-1))
                err = (s_hat - target[start]).abs()
                step_losses.append(err.mean())
                err_sums.append(err.sum(0))
                err_counts.append(not_done.new_tensor(float(err.shape[0])))
                if not self.autoregress:
                    s_hat = target[start]
                fd_steps = self.unroll_steps - 1
            else:
                s_hat = target[start]  # seed: TRUE state at the window start (reference behavior)
                fd_steps = self.unroll_steps
            for t in range(start, start + fd_steps):
                parts = [s_hat, z[w, t - start]]
                if cond is not None:
                    parts.append(cond[t])
                parts.append(storage.actions[t, cols])
                s_hat = self.predictor(torch.cat(parts, dim=-1))
                # Loss masked at resets; carry reset to the true post-reset state (target[t+1]
                # IS the next observation of step t). L1 on normalized targets (reference).
                err = (s_hat - target[t + 1]).abs()
                mask = not_done[t]
                step_losses.append((err.mean(-1) * mask).sum() / mask.sum().clamp(min=1.0))
                # Masked SUM + count, never `err[mask.bool()]`: a boolean index has a
                # data-dependent output shape, so it forces a device->host sync on every
                # step of every window of every minibatch — hundreds per update, purely to
                # feed a diagnostic. Sum/count stays on device and divides out identically.
                err_sums.append((err * mask.unsqueeze(-1)).sum(0))
                err_counts.append(mask.sum())
                if self.autoregress:
                    s_hat = torch.where(not_done[t].unsqueeze(-1).bool(), s_hat, target[t + 1])
                else:  # teacher forcing: every step reads the TRUE previous state
                    s_hat = target[t + 1]
        loss = torch.stack(step_losses).mean()
        metrics = {"loss/fd_l1": loss.item()}
        with torch.no_grad():
            # Early/late halves WITHIN each window: early rides the window seed (dead
            # reckoning), late is image-corrected only — the image-dependence diagnostic.
            if self.unroll_steps >= 2:
                per_window = torch.stack(step_losses).view(-1, self.unroll_steps)
                half = self.unroll_steps // 2
                metrics["fd_l1_early"] = per_window[:, :half].mean().item()
                metrics["fd_l1_late"] = per_window[:, half:].mean().item()
            # Row-weighted per-dim mean == the old concat-then-mean (every step contributes
            # its own row count), so the logged slices are unchanged.
            per_dim = torch.stack(err_sums).sum(0) / torch.stack(err_counts).sum().clamp(min=1.0)
            metrics.update(self._slice_metrics(per_dim, "fd_l1_"))
        return loss, metrics

    def _encode_span(self, obs: dict, starts: range, cols: torch.Tensor) -> torch.Tensor:
        """Encode every window's K steps in ONE extractor pass -> ``(windows, K, mb, latent)``.

        Advanced indexing pairs the two index tensors elementwise, so the flat row order is
        (window, step, column) — exactly the view the caller unflattens to.
        """
        k, mb = self.unroll_steps, len(cols)
        steps = torch.cat([torch.arange(s, s + k, device=cols.device) for s in starts])
        z = self._encode(obs, (steps.repeat_interleave(mb), cols.repeat(len(steps))))
        return z.view(len(starts), k, mb, -1)
