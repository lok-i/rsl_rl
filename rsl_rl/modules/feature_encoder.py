# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Feature encoder: trainable projection over frozen-backbone feature observations."""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.mlp import MLP
from rsl_rl.modules.normalization import EmpiricalNormalization


class FeatureEncoder(nn.Module):
    """Normalizer + MLP projecting a flat feature observation to a compact latent.

    The encoder is the shared trainable representation between the policy and any auxiliary
    objective (see ``rsl_rl.extensions.aux``): PPO gradients reach it through the policy head,
    auxiliary gradients through the aux predictor.
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
        """Initialize the encoder from the raw feature dim and the desired latent dim.

        ``layer_norm`` normalizes the latent's scale so z cannot be drowned by (or drown) the
        empirically-normalized plain groups it is concatenated with downstream.
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.normalizer = EmpiricalNormalization(input_dim) if obs_normalization else nn.Identity()
        self.mlp = MLP(input_dim, latent_dim, hidden_dims, activation)
        self.out_norm = nn.LayerNorm(latent_dim) if layer_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project normalized features to the latent."""
        return self.out_norm(self.mlp(self.normalizer(x)))

    def update_normalization(self, x: torch.Tensor) -> None:
        """Update the input-normalization statistics."""
        if isinstance(self.normalizer, EmpiricalNormalization):
            self.normalizer.update(x)
