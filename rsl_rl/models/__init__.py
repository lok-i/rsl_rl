# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural models for the learning algorithm."""

from .cnn_model import CNNModel
from .feature_encoder_model import FeatureEncoderAdapterModel, FeatureEncoderMLPModel
from .mlp_adapter_model import MLPWithAdapterModel, ModularNormMLPWithAdapterModel
from .mlp_sidecar_model import MLPWithSidecarModel, ModularNormMLPWithSidecarModel
from .mlp_model import MLPModel, ModularNormMLPModel
from .rnn_model import RNNModel

__all__ = [
    "CNNModel",
    "FeatureEncoderAdapterModel",
    "FeatureEncoderMLPModel",
    "MLPModel",
    "ModularNormMLPModel",
    "MLPWithAdapterModel",
    "MLPWithSidecarModel",
    "ModularNormMLPWithAdapterModel",
    "ModularNormMLPWithSidecarModel",
    "RNNModel",
]
