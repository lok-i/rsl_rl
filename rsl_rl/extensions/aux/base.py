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
``Loss/`` logger section); every other key is a diagnostic (routed to ``Auxiliaries/``).
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
        condition_group: str | None = None,
        condition_slices: dict[str, tuple[int, int]] | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the predictor and common hyperparameters; subclasses call ``_finalize`` last.

        ``unroll_steps`` is the prediction horizon K: minibatch samples are window starts t whose
        K transitions are reset-free, and ``_loss`` may index ``idx + k * num_envs`` for any
        k <= K. K = 0 means no temporal structure (every stored step is a sample).

        ``condition_group`` names the conditioning observation group (default:
        ``prediction_conditioning`` in the SL variants — robot state only, never object
        state, which would bypass the image); ``condition_slices`` optionally narrows it.
        """
        super().__init__()
        self.extractor_group = extractor_group
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_mini_batches = num_mini_batches
        self.max_grad_norm = max_grad_norm
        self.unroll_steps = unroll_steps
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
        print(
            f"Aux Objective: {type(self).__name__} (K={self.unroll_steps}, "
            f"extractor_group='{self.extractor_group}'{cond}, lr={self.learning_rate:g}, "
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
        mini_batch_size = len(indices) // self.num_mini_batches
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
        out = {key: value / num_updates for key, value in totals.items()}
        out.update(self.latent_metrics(storage))
        return out

    @torch.no_grad()
    def latent_metrics(self, storage: RolloutStorage, max_samples: int = 4096) -> dict[str, float]:
        """Capacity/collapse diagnostics of the latent z over a rollout sample.

        RankMe (Garrido et al. 2023): exp-entropy of the normalized singular-value spectrum —
        the effective number of independent dimensions z actually uses (elbow-plot metric for
        latent_dim ablations). latent_std: mean per-dim std, the cheap collapse alarm.
        """
        flat = {g: storage.observations[g].flatten(0, 1) for g in self._extractor_groups()}
        num_rows = next(iter(flat.values())).shape[0]
        idx = torch.randperm(num_rows, device=storage.dones.device)[:max_samples]
        z = self._encode(flat, idx)
        sv = torch.linalg.svdvals(z.float())
        p = sv / sv.sum() + 1e-12
        rankme = torch.exp(-(p * p.log()).sum())
        return {"latent_rankme": rankme.item(), "latent_std": z.std(dim=0).mean().item()}

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
