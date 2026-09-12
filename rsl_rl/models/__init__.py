# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural models for the learning algorithm."""

from .cnn_model import CNNModel
from .extractor_model import ExtractorSonicAdapterModel
from .mlp_adapter_model import MLPWithAdapterModel, ModularNormMLPWithAdapterModel
from .mlp_model import MLPModel, ModularNormMLPModel
from .mlp_sidecar_model import MLPWithSidecarModel, ModularNormMLPWithSidecarModel
from .rnn_model import RNNModel
from .sonic_adapter_model import SonicWithAdapterModel
from .sonic_base_model import SonicBaseModel

__all__ = [
    "CNNModel",
    "ExtractorSonicAdapterModel",
    "MLPModel",
    "MLPWithAdapterModel",
    "MLPWithSidecarModel",
    "ModularNormMLPModel",
    "ModularNormMLPWithAdapterModel",
    "ModularNormMLPWithSidecarModel",
    "RNNModel",
    "SonicBaseModel",
    "SonicWithAdapterModel",
]
