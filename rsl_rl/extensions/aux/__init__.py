# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Auxiliary representation objectives for PPOAux (one file per variant)."""

from .base import AuxObjective
from .latent_fd import LatentFdAux
from .state_fd import StateFdAux
from .state_reg import StateRegAux

__all__ = ["AuxObjective", "LatentFdAux", "StateFdAux", "StateRegAux"]
