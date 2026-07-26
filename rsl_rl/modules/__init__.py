# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Building blocks for neural models."""

from .amp import AmpMixin, resolve_amp_dtype, set_model_amp
from .cnn import CNN
from .cross_attention import CrossAttentionExtractor
from .distribution import BetaDistribution, Distribution, GaussianDistribution, HeteroscedasticGaussianDistribution
from .mlp import MLP
from .mlp_extractor import MlpExtractor
from .mlp_with_adapter import Adapter, MLPWithAdapter, ModularNormMLPWithAdapter
from .mlp_with_sidecar import MLPWithSidecar, ModularNormMLPWithSidecar, Sidecar
from .modular_norm_mlp import ModularNormMLP
from .normalization import EmpiricalDiscountedVariationNormalization, EmpiricalNormalization
from .rnn import RNN, HiddenState

__all__ = [
    "CNN",
    "MLP",
    "RNN",
    "Adapter",
    "AmpMixin",
    "BetaDistribution",
    "CrossAttentionExtractor",
    "Distribution",
    "EmpiricalDiscountedVariationNormalization",
    "EmpiricalNormalization",
    "GaussianDistribution",
    "HeteroscedasticGaussianDistribution",
    "HiddenState",
    "MLPWithAdapter",
    "MLPWithSidecar",
    "MlpExtractor",
    "ModularNormMLP",
    "ModularNormMLPWithAdapter",
    "ModularNormMLPWithSidecar",
    "Sidecar",
    "resolve_amp_dtype",
    "set_model_amp",
]
