# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for auxiliary representation objectives.

An auxiliary objective trains the actor's :class:`~rsl_rl.modules.FeatureEncoder` (plus its own
predictor head) with a supervised or self-supervised loss, complementary to PPO returns. It is
invoked by :class:`~rsl_rl.algorithms.PPOAux` after each PPO update and reads temporally-ordered
``(obs_t, a_t, ..., obs_t+K)`` views straight from the rollout storage — no extra buffers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from itertools import chain

from rsl_rl.modules import MLP
from rsl_rl.storage import RolloutStorage


class AuxObjective(nn.Module):
    """Owns a predictor MLP and an optimizer over (encoder + own modules) parameters.

    Subclasses implement :meth:`_obs_keys` (observation groups they read) and :meth:`_loss`
    (per-minibatch loss + metrics), and call :meth:`_finalize` at the end of their ``__init__``.
    The algorithm stays agnostic to the objective's nature (SL vs SSL): the whole contract is
    ``update(storage, actor) -> metrics``.
    """

    def __init__(
        self,
        storage: RolloutStorage,
        actor: nn.Module,
        predictor_input_dim: int,
        predictor_output_dim: int,
        feat_group: str = "img_feat",
        predictor_hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        activation: str = "elu",
        learning_rate: float = 1e-3,
        num_epochs: int = 1,
        num_mini_batches: int = 4,
        max_grad_norm: float = 1.0,
        unroll_steps: int = 1,
        device: str = "cpu",
    ) -> None:
        """Initialize the predictor and common hyperparameters; subclasses call ``_finalize`` last.

        ``unroll_steps`` is the prediction horizon K: minibatch samples are window starts t whose
        K transitions are reset-free, and ``_loss`` may index ``idx + k * num_envs`` for any
        k <= K. K = 0 means no temporal structure (every stored step is a sample).
        """
        super().__init__()
        self.feat_group = feat_group
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_mini_batches = num_mini_batches
        self.max_grad_norm = max_grad_norm
        self.unroll_steps = unroll_steps
        self.device = device
        self.predictor = MLP(predictor_input_dim, predictor_output_dim, predictor_hidden_dims, activation)
        # Tuple hides the reference from Module registration: the actor owns the encoder,
        # so it must not appear in this module's state_dict.
        self._encoder_ref = (actor.encoders[feat_group],)

    @property
    def encoder(self) -> nn.Module:
        """The actor's feature encoder this objective trains."""
        return self._encoder_ref[0]

    def _finalize(self) -> None:
        """Build the optimizer over encoder + own trainable parameters. Call last in subclass init."""
        params = [p for p in chain(self.parameters(), self.encoder.parameters()) if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)

    def update(self, storage: RolloutStorage, actor: nn.Module) -> dict[str, float]:
        """Run optimization epochs over the stored rollout and return mean metrics."""
        num_t, num_envs = storage.num_transitions_per_env, storage.num_envs
        flat = {g: storage.observations[g].flatten(0, 1) for g in self._obs_keys()}
        actions = storage.actions.flatten(0, 1)

        # Flat index of (t, n) is t * num_envs + n, so idx + k * num_envs addresses t+k for the
        # same env. A K-step window starting at t is valid iff no done occurs in [t, t+K).
        if self.unroll_steps > 0:
            dones = storage.dones.squeeze(-1)[: num_t - 1]
            valid = dones.unfold(0, self.unroll_steps, 1).sum(-1) == 0  # (num_t - K, num_envs)
            indices = torch.arange(valid.numel(), device=valid.device)[valid.flatten(0, 1)]
        else:
            indices = torch.arange(num_t * num_envs, device=storage.dones.device)

        mini_batch_size = len(indices) // self.num_mini_batches
        if mini_batch_size == 0:
            return {}

        totals: dict[str, float] = {}
        num_updates = 0
        for _ in range(self.num_epochs):
            perm = indices[torch.randperm(len(indices), device=indices.device)]
            for i in range(self.num_mini_batches):
                idx = perm[i * mini_batch_size : (i + 1) * mini_batch_size]
                loss, metrics = self._loss(flat, actions, idx, num_envs)
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
        out.update(self._latent_metrics(flat[self.feat_group]))
        return out

    @torch.no_grad()
    def _latent_metrics(self, feats: torch.Tensor, max_samples: int = 4096) -> dict[str, float]:
        """Capacity/collapse diagnostics of the latent z over a rollout sample.

        RankMe (Garrido et al. 2023): exp-entropy of the normalized singular-value spectrum —
        the effective number of independent dimensions z actually uses (elbow-plot metric for
        latent_dim ablations). z_std: mean per-dim std, the cheap collapse alarm.
        """
        sample = feats[torch.randperm(feats.shape[0], device=feats.device)[:max_samples]]
        z = self.encoder(sample)
        sv = torch.linalg.svdvals(z.float())
        p = sv / sv.sum() + 1e-12
        rankme = torch.exp(-(p * p.log()).sum())
        return {"aux/z_rankme": rankme.item(), "aux/z_std": z.std(dim=0).mean().item()}

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
        """Run after each optimizer step (e.g. EMA update). No-op by default."""
        pass

    # --- helpers for state-target variants ---

    @staticmethod
    def _resolve_slices(dim: int, target_slices: dict[str, tuple[int, int]] | None) -> dict[str, tuple[int, int]]:
        """Named column ranges of the target group; defaults to the whole group."""
        return dict(target_slices) if target_slices else {"all": (0, dim)}

    def _select_target(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate the configured target slices."""
        return torch.cat([x[..., a:b] for a, b in self.target_slices.values()], dim=-1)

    def _slice_metrics(self, sq_err: torch.Tensor, prefix: str) -> dict[str, float]:
        """Compute per-slice MSE in normalized target space.

        Catches e.g. object-state error hiding behind an easy proprio-dominated total.
        """
        if len(self.target_slices) <= 1:
            return {}
        out, offset = {}, 0
        for name, (a, b) in self.target_slices.items():
            width = b - a
            out[f"{prefix}{name}_mse"] = sq_err[..., offset : offset + width].mean().item()
            offset += width
        return out
