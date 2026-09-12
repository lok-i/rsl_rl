# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility tests for the frozen TextOp model path."""

from __future__ import annotations

import torch
from pathlib import Path
from tensordict import TensorDict

import pytest

from rsl_rl.models import ModularNormMLPModel, ModularNormMLPWithAdapterModel

NUM_ENVS = 4
OBS_DIM = 8
NUM_ACTIONS = 4
HIDDEN_DIMS = [16, 12]
OBS_GROUPS = {"actor": ["policy"]}


def _obs() -> TensorDict:
    """Return a flat policy observation matching the TextOp model contract."""
    return TensorDict(
        {"policy": torch.randn(NUM_ENVS, OBS_DIM)},
        batch_size=[NUM_ENVS],
    )


def _distribution_cfg() -> dict[str, object]:
    """Return a fresh distribution config because model construction consumes it."""
    return {
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
    }


def _base(obs: TensorDict) -> ModularNormMLPModel:
    """Build the frozen-base architecture used by the TextOp compatibility path."""
    return ModularNormMLPModel(
        obs=obs,
        obs_groups=OBS_GROUPS,
        obs_set="actor",
        output_dim=NUM_ACTIONS,
        hidden_dims=HIDDEN_DIMS,
        activation="elu",
        obs_normalization=True,
        distribution_cfg=_distribution_cfg(),
    )


def _adapted(obs: TensorDict, checkpoint: Path) -> ModularNormMLPWithAdapterModel:
    """Load the base through the all-adapters-skipped path used by Mocke."""
    return ModularNormMLPWithAdapterModel(
        obs=obs,
        obs_groups=OBS_GROUPS,
        obs_set="actor",
        output_dim=NUM_ACTIONS,
        hidden_dims=HIDDEN_DIMS,
        activation="elu",
        obs_normalization=True,
        distribution_cfg=_distribution_cfg(),
        adapter_obs_group="policy",
        rank=[None] * (len(HIDDEN_DIMS) + 1),
        base_checkpoint=str(checkpoint),
    )


def test_textop_compatibility_path_reproduces_frozen_base(tmp_path: Path) -> None:
    """Skipping every adapter must reproduce the checkpointed base exactly."""
    torch.manual_seed(0)
    obs = _obs()
    base = _base(obs).eval()
    checkpoint = tmp_path / "textop_ported.pt"
    torch.save({"model_state_dict": base.state_dict()}, checkpoint)

    adapted = _adapted(obs, checkpoint).eval()

    torch.testing.assert_close(adapted(obs), base(obs), rtol=0.0, atol=0.0)


def test_textop_compatibility_path_rejects_incomplete_checkpoint(tmp_path: Path) -> None:
    """A truncated frozen base must not leave random weights behind silently."""
    obs = _obs()
    state_dict = _base(obs).state_dict()
    state_dict.pop("mlp.2.weight")
    checkpoint = tmp_path / "truncated.pt"
    torch.save({"model_state_dict": state_dict}, checkpoint)

    with pytest.raises(RuntimeError, match=r"missing=.*mlp\.2\.weight"):
        _adapted(obs, checkpoint)
