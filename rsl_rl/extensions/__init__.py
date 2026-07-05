# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extensions for the learning algorithms."""

from .aux import AuxObjective, LatentFdAux, StateFdAux, StateRegAux
from .rnd import RandomNetworkDistillation, resolve_rnd_config
from .symmetry import Symmetry, resolve_symmetry_config

__all__ = [
    "AuxObjective",
    "LatentFdAux",
    "RandomNetworkDistillation",
    "StateFdAux",
    "StateRegAux",
    "Symmetry",
    "resolve_rnd_config",
    "resolve_symmetry_config",
]
