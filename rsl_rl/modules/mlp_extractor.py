# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MLP extractor: trainable projection over frozen-backbone feature observations."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.modules.mlp import MLP
from rsl_rl.modules.normalization import EmpiricalNormalization


class MlpExtractor(nn.Module):
    """Normalizer + MLP projecting a feature observation group to a compact latent.

    The extractor is the shared trainable representation between the policy and any auxiliary
    objective (see ``rsl_rl.extensions.aux``): PPO gradients reach it through the policy head,
    auxiliary gradients through the aux predictor.

    Consumes a flat (B, D) group, or a dict group (``concatenate_terms=False``) whose terms
    are each flattened and concatenated in sorted-key order (deterministic weight identity).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256,),
        activation: str = "elu",
        obs_normalization: bool = True,
        layer_norm: bool = False,
    ) -> None:
        """Initialize the extractor from the raw feature dim and the desired latent dim.

        ``layer_norm`` normalizes the latent's scale so z cannot be drowned by (or drown) the
        empirically-normalized plain groups it is concatenated with downstream.
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.normalizer = EmpiricalNormalization(input_dim) if obs_normalization else nn.Identity()
        self.mlp = MLP(input_dim, latent_dim, hidden_dims, activation)
        self.out_norm = nn.LayerNorm(latent_dim) if layer_norm else nn.Identity()
        self.input_groups: tuple[str, ...] = ()
        self.term_keys: list[str] | None = None  # dict-group term order; None = flat group

    @classmethod
    def from_obs(cls, obs: TensorDict, group: str, **cfg: Any) -> MlpExtractor:
        """Build from an observation template: ``group`` -> (B, D) flat, or a dict of terms."""
        x = obs[group]
        if hasattr(x, "keys"):  # dict group: every term flattened + concatenated
            term_keys = sorted(x.keys())
            input_dim = sum(x[k].reshape(x[k].shape[0], -1).shape[-1] for k in term_keys)
            enc = cls(input_dim, **cfg)
            enc.term_keys = term_keys
        else:
            if len(x.shape) != 2:
                raise ValueError(f"MlpExtractor expects a flat observation group, got {x.shape} for '{group}'.")
            enc = cls(x.shape[-1], **cfg)
        enc.input_groups = (group,)
        return enc

    def _flat(self, x: torch.Tensor | TensorDict) -> torch.Tensor:
        if self.term_keys is None:
            return x
        return torch.cat([x[k].reshape(x[k].shape[0], -1) for k in self.term_keys], dim=-1)

    def forward(self, x: torch.Tensor | TensorDict) -> torch.Tensor:
        """Project the normalized feature group (flat tensor or dict of terms) to the latent."""
        return self.out_norm(self.mlp(self.normalizer(self._flat(x))))

    def update_normalization(self, x: torch.Tensor | TensorDict) -> None:
        """Update the input-normalization statistics."""
        if isinstance(self.normalizer, EmpiricalNormalization):
            self.normalizer.update(self._flat(x))
