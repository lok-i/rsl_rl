# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for auxiliary representation objectives.

An auxiliary objective trains the actor's extractor (plus its own
predictor head) with a supervised or self-supervised loss, complementary to PPO returns. It is
invoked by :class:`~rsl_rl.algorithms.PPOAux` — either after the PPO epochs (``sequential``) or
as an accumulated gradient inside each PPO minibatch (``joint``) — and reads temporally-ordered
``(obs_t, a_t, ..., obs_t+K)`` views straight from the rollout storage; no extra buffers.

Metric key convention: ``loss/<name>`` marks the objective's actual loss (routed to the
``Loss/`` logger section); every other key is a diagnostic and carries its own fully
qualified section, which the algorithm passes through verbatim — the producer owns the
name. Objective-fit diagnostics go to ``ZPrediction/*``; the sections describing z itself
(``ZAttention/*``, ``ZCapacity/*``) belong to the extractor, so that the extractor-only
row reports them too and they stay comparable across objectives.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from itertools import chain
from tensordict import TensorDict

from rsl_rl.modules import MLP, EmpiricalNormalization
from rsl_rl.storage import RolloutStorage


class AuxObjective(nn.Module):
    """Owns a predictor MLP and an optimizer over (encoder + own modules) parameters.

    Subclasses implement :meth:`_obs_keys` (observation groups they read) and :meth:`_loss`
    (per-minibatch loss + metrics), and call :meth:`_finalize` at the end of their ``__init__``.
    The algorithm stays agnostic to the objective's nature (SL vs SSL): the contract is
    ``update(storage, actor) -> metrics`` (sequential) or ``sample_loss(storage) -> (loss,
    metrics)`` (joint — one freshly-sampled minibatch, backward done by the caller).
    """

    def __init__(
        self,
        storage: RolloutStorage,
        actor: nn.Module,
        predictor_input_dim: int,
        predictor_output_dim: int,
        extractor_group: str = "extractor_input",
        predictor_hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        activation: str = "elu",
        learning_rate: float = 1e-3,
        num_epochs: int = 1,
        num_mini_batches: int = 4,
        max_grad_norm: float = 1.0,
        unroll_steps: int = 1,
        mini_batch_rows_per_env: float | None = None,
        condition_group: str | None = None,
        condition_slices: dict[str, tuple[int, int]] | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the predictor and common hyperparameters; subclasses call ``_finalize`` last.

        ``unroll_steps`` is the prediction horizon K: minibatch samples are window starts t whose
        K transitions are reset-free, and ``_loss`` may index ``idx + k * num_envs`` for any
        k <= K. K = 0 means no temporal structure (every stored step is a sample).

        ``mini_batch_rows_per_env`` is the aux batch budget in ENCODER-GRADIENT ROWS PER ENV
        (see :meth:`_sample_count`) — the unit that makes objectives with different sampling
        units draw the same encoder batch, at any ``num_envs``. ``None`` = the plain
        ``available // num_mini_batches`` split.

        ``condition_group`` names the conditioning observation group (default:
        ``prediction_conditioning`` in the FD variants — robot state only, never object
        state, which would bypass the image); ``condition_slices`` optionally narrows it.
        """
        super().__init__()
        self.extractor_group = extractor_group
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_mini_batches = num_mini_batches
        self.max_grad_norm = max_grad_norm
        self.unroll_steps = unroll_steps
        self.mini_batch_rows_per_env = mini_batch_rows_per_env
        self.device = device
        self.condition_group = condition_group
        if condition_group is not None:
            obs: TensorDict = storage.observations
            self.cond_slices = self._resolve_slices(obs[condition_group].shape[-1], condition_slices)
            self.cond_dim = sum(b - a for a, b in self.cond_slices.values())
            self.cond_normalizer = EmpiricalNormalization(self.cond_dim)
        else:
            self.cond_slices, self.cond_dim, self.cond_normalizer = {}, 0, nn.Identity()
        self.predictor = MLP(predictor_input_dim, predictor_output_dim, predictor_hidden_dims, activation)
        # Tuple hides the reference from Module registration: the actor owns the encoder,
        # so it must not appear in this module's state_dict.
        self._extractor_ref = (actor.extractors[extractor_group],)

    @property
    def extractor(self) -> nn.Module:
        """The actor's extractor this objective trains."""
        return self._extractor_ref[0]

    def _extractor_groups(self) -> tuple[str, ...]:
        """Observation groups the extractor consumes."""
        return getattr(self.extractor, "input_groups", None) or (self.extractor_group,)

    def _encode(
        self, tensors: dict | TensorDict, index: torch.Tensor | tuple, extractor: nn.Module | None = None
    ) -> torch.Tensor:
        """Encode rows ``index`` of the extractor's input groups (``tensors``: flat dict or TensorDict)."""
        enc = self.extractor if extractor is None else extractor
        groups = getattr(enc, "input_groups", None) or (self.extractor_group,)
        return enc(*(tensors[g][index] for g in groups))

    @staticmethod
    def cond_dim_of(obs: TensorDict, group: str | None, slices: dict[str, tuple[int, int]] | None) -> int:
        """Conditioning width for predictor sizing, resolvable before ``super().__init__``."""
        if group is None:
            return 0
        resolved = AuxObjective._resolve_slices(obs[group].shape[-1], slices)
        return sum(b - a for a, b in resolved.values())

    def _cond(self, x: torch.Tensor, update_stats: bool) -> torch.Tensor:
        """Slice-select and normalize conditioning rows ``x`` of the condition group."""
        cond = torch.cat([x[..., a:b] for a, b in self.cond_slices.values()], dim=-1)
        if update_stats:
            self.cond_normalizer.update(cond.reshape(-1, self.cond_dim))
        return self.cond_normalizer(cond)

    def _finalize(self) -> None:
        """Build the optimizer over extractor + own trainable params; print the architecture. Call last in init."""
        params = [p for p in chain(self.parameters(), self.extractor.parameters()) if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        n_own = sum(p.numel() for p in self.parameters() if p.requires_grad)
        cond = f", cond='{self.condition_group}'({self.cond_dim})" if self.condition_group else ""
        rows = (
            f", rows/env={self.mini_batch_rows_per_env:g}"
            if self.mini_batch_rows_per_env is not None
            else f", mini_batches={self.num_mini_batches}"
        )
        print(
            f"Aux Objective: {type(self).__name__} (K={self.unroll_steps}, "
            f"extractor_group='{self.extractor_group}'{cond}{rows}, lr={self.learning_rate:g}, "
            f"{n_own:,} trainable params — train-only, dropped at deployment)"
        )
        for name, mod in self.named_children():
            if name.startswith("ema_") or isinstance(mod, nn.Identity):
                continue  # EMA copy = frozen extractor dup (already printed); Identity = noise
            print(f"  ({name}): " + repr(mod).replace("\n", "\n  "))

    # --- sampling ---

    def _flat_views(self, storage: RolloutStorage) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        groups = {*self._obs_keys(), *self._extractor_groups()}
        flat = {g: storage.observations[g].flatten(0, 1) for g in groups}
        return flat, storage.actions.flatten(0, 1)

    def _sample_count(self, rows_per_sample: int, available: int, num_envs: int) -> int:
        """Resolve the samples per aux call, from the ``mini_batch_rows_per_env`` budget if set.

        Two objectives can sample in different units (env COLUMNS whose whole time axis is
        scanned, vs flat ``(t, env)`` window starts) yet still owe the shared encoder the same
        batch. The comparable unit is encoder rows that receive a gradient per aux call:
        ``rows_per_sample`` (how many the caller's unit costs) x samples. Budgeting it
        per-env keeps the ask ``num_envs``-agnostic, so an ``-e`` change rescales both
        objectives identically. ``None`` keeps the plain ``num_mini_batches`` split.
        """
        if self.mini_batch_rows_per_env is None:
            count = available // self.num_mini_batches
        else:
            count = int(self.mini_batch_rows_per_env * num_envs) // max(1, rows_per_sample)
        return min(max(count, 1), available)

    def _valid_indices(self, storage: RolloutStorage) -> torch.Tensor:
        """Flat (t, n) start indices whose K-step window is reset-free (all steps if K = 0)."""
        num_t, num_envs = storage.num_transitions_per_env, storage.num_envs
        if self.unroll_steps > 0:
            dones = storage.dones.squeeze(-1)[: num_t - 1]
            valid = dones.unfold(0, self.unroll_steps, 1).sum(-1) == 0  # (num_t - K, num_envs)
            return torch.arange(valid.numel(), device=valid.device)[valid.flatten(0, 1)]
        return torch.arange(num_t * num_envs, device=storage.dones.device)

    def sample_loss(self, storage: RolloutStorage) -> tuple[torch.Tensor, dict[str, float]] | None:
        """One freshly-sampled minibatch loss (joint mode); caller does backward + step."""
        flat, actions = self._flat_views(storage)
        indices = self._valid_indices(storage)
        # one encoder pass per unrolled step -> K rows per sampled window start
        mini_batch_size = self._sample_count(max(1, self.unroll_steps), len(indices), storage.num_envs)
        if mini_batch_size == 0:
            return None
        idx = indices[torch.randint(len(indices), (mini_batch_size,), device=indices.device)]
        return self._loss(flat, actions, idx, storage.num_envs)

    # --- sequential mode ---

    def update(self, storage: RolloutStorage, actor: nn.Module) -> dict[str, float]:
        """Run optimization epochs over the stored rollout and return mean metrics."""
        flat, actions = self._flat_views(storage)
        indices = self._valid_indices(storage)
        mini_batch_size = len(indices) // self.num_mini_batches
        if mini_batch_size == 0:
            return {}

        totals: dict[str, float] = {}
        num_updates = 0
        for _ in range(self.num_epochs):
            perm = indices[torch.randperm(len(indices), device=indices.device)]
            for i in range(self.num_mini_batches):
                idx = perm[i * mini_batch_size : (i + 1) * mini_batch_size]
                loss, metrics = self._loss(flat, actions, idx, storage.num_envs)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for group in self.optimizer.param_groups for p in group["params"]], self.max_grad_norm
                )
                self.optimizer.step()
                self._post_step()
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + value
                num_updates += 1
        return {key: value / num_updates for key, value in totals.items()}

    # NOTE: the z capacity/collapse diagnostics (RankMe, std) live on the EXTRACTOR
    # (``CrossAttentionExtractor.metrics`` -> ``ZCapacity/*``), not here. They used to be
    # duplicated: a storage re-encode of 4096 rows on this side and the last minibatch on
    # the extractor's. The two tracked each other to ~1% while the storage version cost an
    # extra encoder forward every iteration and could not be reported by the extractor-only
    # (``-Ext``) row at all — which is precisely the baseline the aux rows are read against.
    # One definition, owned by the module the metric describes.

    # --- subclass interface ---

    def _obs_keys(self) -> list[str]:
        """Observation groups this objective reads from the storage."""
        raise NotImplementedError

    def _loss(
        self, flat: dict[str, torch.Tensor], actions: torch.Tensor, idx: torch.Tensor, num_envs: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the minibatch loss and scalar metrics. ``idx + k * num_envs`` indexes t+k."""
        raise NotImplementedError

    def _post_step(self) -> None:
        """Run after each gradient application (e.g. EMA update). No-op by default."""
        pass

    # --- helpers for state-target variants ---

    @staticmethod
    def _resolve_slices(dim: int, target_slices: dict[str, tuple[int, int]] | None) -> dict[str, tuple[int, int]]:
        """Named column ranges of the target group; defaults to the whole group."""
        return dict(target_slices) if target_slices else {"all": (0, dim)}

    def _select_target(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate the configured target slices."""
        return torch.cat([x[..., a:b] for a, b in self.target_slices.values()], dim=-1)

    def _slice_metrics(self, err: torch.Tensor, prefix: str) -> dict[str, float]:
        """Per-slice mean error in normalized target space.

        Catches e.g. object-state error hiding behind an easy proprio-dominated total.
        """
        if len(self.target_slices) <= 1:
            return {}
        out, offset = {}, 0
        for name, (a, b) in self.target_slices.items():
            width = b - a
            out[f"{prefix}{name}"] = err[..., offset : offset + width].mean().item()
            offset += width
        return out
