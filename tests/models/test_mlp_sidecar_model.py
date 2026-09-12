# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the MLPWithSidecarModel."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models.mlp_sidecar_model import MLPWithSidecarModel

NUM_ENVS = 4
OBS_DIM = 8
AUG_DIM = 6
NUM_ACTIONS = 4
OBS_GROUPS = {"actor": ["policy"], "critic": ["policy"]}
HIDDEN_DIMS = [16, 16]
SIDECAR_HIDDEN_DIMS = [16]


def _make_obs() -> TensorDict:
    """Create a TensorDict with policy and augmentation groups."""
    return TensorDict(
        {
            "policy": torch.randn(NUM_ENVS, OBS_DIM),
            "augmentation": torch.randn(NUM_ENVS, AUG_DIM),
        },
        batch_size=[NUM_ENVS],
    )


def _make_model(**kwargs: object) -> tuple[MLPWithSidecarModel, TensorDict]:
    """Create a sidecar model with default stochastic config."""
    obs = _make_obs()
    defaults: dict[str, object] = {
        "hidden_dims": HIDDEN_DIMS,
        "sidecar_hidden_dims": SIDECAR_HIDDEN_DIMS,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    }
    defaults.update(kwargs)
    model = MLPWithSidecarModel(
        obs,
        OBS_GROUPS,
        "actor",
        NUM_ACTIONS,
        sidecar_obs_group="augmentation",
        **defaults,
    )
    return model, obs


class TestMLPWithSidecarModelConstruction:
    """Tests for model construction and the zero-init invariant."""

    def test_deterministic_output_shape(self) -> None:
        """Deterministic forward should return (num_envs, num_actions)."""
        model, obs = _make_model()
        model.eval()
        out = model(obs)
        assert out.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_stochastic_output_shape(self) -> None:
        """Stochastic forward should return (num_envs, num_actions)."""
        model, obs = _make_model()
        out = model(obs, stochastic_output=True)
        assert out.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_base_frozen_by_default(self) -> None:
        """Base MLP parameters should be frozen by default."""
        model, _ = _make_model()
        for p in model.mlp.base.parameters():
            assert not p.requires_grad

    def test_sidecar_trainable(self) -> None:
        """Sidecar parameters should be trainable."""
        model, _ = _make_model()
        trainable = [p for p in model.mlp.sidecar.parameters() if p.requires_grad]
        assert len(trainable) > 0


class TestMLPWithSidecarModelForward:
    """Tests for forward pass correctness."""

    def test_no_nan_in_forward(self) -> None:
        """Forward pass should not produce NaN."""
        model, obs = _make_model()
        out = model(obs, stochastic_output=True)
        assert not torch.isnan(out).any()

    def test_backward_produces_gradients(self) -> None:
        """Backward through the deterministic action should reach only the sidecar."""
        model, obs = _make_model()
        out = model(obs)
        out.sum().backward()
        # Sidecar should have gradients
        sidecar_grads = [p.grad for p in model.mlp.sidecar.parameters() if p.grad is not None]
        assert len(sidecar_grads) > 0
        # Base should not
        for p in model.mlp.base.parameters():
            assert p.grad is None

    def test_no_nan_in_backward(self) -> None:
        """Gradients should not be NaN."""
        model, obs = _make_model()
        out = model(obs)
        out.sum().backward()
        for p in model.parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any()


class TestMLPWithSidecarModelNormalization:
    """Tests for normalization behavior."""

    def test_update_normalization_only_updates_sidecar(self) -> None:
        """update_normalization should only change the sidecar normalizer stats."""
        model, _ = _make_model(obs_normalization=True)
        model.train()

        base_mean_before = model.obs_normalizer._mean.clone()
        sidecar_mean_before = model.sidecar_normalizer._mean.clone()

        for _ in range(10):
            model.update_normalization(_make_obs())

        assert torch.allclose(model.obs_normalizer._mean, base_mean_before), "Base normalizer should not change"
        assert not torch.allclose(model.sidecar_normalizer._mean, sidecar_mean_before), (
            "Sidecar normalizer should update"
        )


class TestMLPWithSidecarModelOptions:
    """Tests for head_init_gain and condition_on_base_output options."""

    def test_custom_head_init_gain(self) -> None:
        """Model with custom sidecar_head_init_gain should produce valid output."""
        model, obs = _make_model(sidecar_head_init_gain=0.05)
        out = model(obs, stochastic_output=True)
        assert out.shape == (NUM_ENVS, NUM_ACTIONS)
        assert not torch.isnan(out).any()

    def test_condition_on_base_output_does_not_break_forward(self) -> None:
        """Model with condition_on_base_output should produce valid output."""
        model, obs = _make_model(condition_on_base_output=True)
        out = model(obs, stochastic_output=True)
        assert out.shape == (NUM_ENVS, NUM_ACTIONS)
        assert not torch.isnan(out).any()

    def test_condition_on_base_output_with_custom_head_gain(self) -> None:
        """Both options combined should produce valid output."""
        model, obs = _make_model(condition_on_base_output=True, sidecar_head_init_gain=0.05)
        out = model(obs, stochastic_output=True)
        assert out.shape == (NUM_ENVS, NUM_ACTIONS)
        assert not torch.isnan(out).any()
