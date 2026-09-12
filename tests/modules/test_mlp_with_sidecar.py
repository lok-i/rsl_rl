# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the MLPWithSidecar module."""

from __future__ import annotations

import torch

import pytest

from rsl_rl.modules.mlp import MLP
from rsl_rl.modules.mlp_with_sidecar import MLPWithSidecar, Sidecar

INPUT_DIM = 8
OUTPUT_DIM = 4
HIDDEN_DIMS = [16, 16]
SIDECAR_INPUT_DIM = 12
SIDECAR_HIDDEN_DIMS = [16]
BATCH = 4


class TestSidecar:
    """Tests for the near-zero-initialized Sidecar module."""

    def test_near_zero_output_at_init(self) -> None:
        """Sidecar output should be near-zero at construction (head is Xavier with small gain)."""
        sidecar = Sidecar(SIDECAR_INPUT_DIM, OUTPUT_DIM, SIDECAR_HIDDEN_DIMS)
        x = torch.randn(BATCH, SIDECAR_INPUT_DIM)
        assert sidecar(x).abs().max() < 0.1, "Sidecar output should be near-zero at init"

    def test_small_head_gain_reduces_output(self) -> None:
        """Smaller head_init_gain should produce smaller initial output."""
        torch.manual_seed(0)
        default = Sidecar(SIDECAR_INPUT_DIM, OUTPUT_DIM, SIDECAR_HIDDEN_DIMS, head_init_gain=0.01)
        torch.manual_seed(0)
        large = Sidecar(SIDECAR_INPUT_DIM, OUTPUT_DIM, SIDECAR_HIDDEN_DIMS, head_init_gain=1.0)

        x = torch.randn(BATCH, SIDECAR_INPUT_DIM)
        assert default(x).abs().max() < large(x).abs().max()

    def test_gradients_flow(self) -> None:
        """Backward pass should produce non-zero gradients for trunk and head."""
        sidecar = Sidecar(SIDECAR_INPUT_DIM, OUTPUT_DIM, SIDECAR_HIDDEN_DIMS)
        x = torch.randn(BATCH, SIDECAR_INPUT_DIM)
        loss = sidecar(x).sum()
        loss.backward()
        trunk_grads = [p.grad for p in sidecar.trunk.parameters() if p.grad is not None]
        head_grads = [p.grad for p in [sidecar.head.weight] if p.grad is not None]
        assert len(trunk_grads) > 0, "Trunk should receive gradients"
        assert len(head_grads) > 0, "Head should receive gradients"

    def test_empty_hidden_dims(self) -> None:
        """Empty hidden_dims should create a linear-only sidecar (Identity trunk)."""
        sidecar = Sidecar(SIDECAR_INPUT_DIM, OUTPUT_DIM, hidden_dims=[])
        x = torch.randn(BATCH, SIDECAR_INPUT_DIM)
        assert sidecar(x).shape == (BATCH, OUTPUT_DIM)


class TestMLPWithSidecar:
    """Tests for the composed base + sidecar module."""

    def test_near_reproduces_base_at_init(self) -> None:
        """At construction the composed network should approximately reproduce the base."""
        torch.manual_seed(0)
        base = MLP(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIMS)
        composed = MLPWithSidecar.from_base_mlp(base, sidecar_input_dim=SIDECAR_INPUT_DIM)

        torch.manual_seed(42)
        x = torch.randn(BATCH, INPUT_DIM)
        sidecar_x = torch.randn(BATCH, SIDECAR_INPUT_DIM)
        expected = base(x)
        actual = composed(x, sidecar_x)
        assert torch.allclose(expected, actual, atol=0.1), "Composed should be near base at init"

    def test_freeze_base(self) -> None:
        """Frozen base parameters should have requires_grad=False."""
        base = MLP(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIMS)
        composed = MLPWithSidecar.from_base_mlp(base, sidecar_input_dim=SIDECAR_INPUT_DIM, freeze_base=True)
        for p in composed.base.parameters():
            assert not p.requires_grad

    def test_unfreeze_base(self) -> None:
        """Unfrozen base parameters should have requires_grad=True."""
        base = MLP(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIMS)
        composed = MLPWithSidecar.from_base_mlp(base, sidecar_input_dim=SIDECAR_INPUT_DIM, freeze_base=False)
        for p in composed.base.parameters():
            assert p.requires_grad

    def test_condition_on_base_output_widens_sidecar_input(self) -> None:
        """With condition_on_base_output, sidecar input_dim should include base output_dim."""
        base = MLP(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIMS)
        composed = MLPWithSidecar.from_base_mlp(
            base,
            sidecar_input_dim=SIDECAR_INPUT_DIM,
            condition_on_base_output=True,
        )
        first_linear = next(m for m in composed.sidecar.trunk.modules() if isinstance(m, torch.nn.Linear))
        assert first_linear.in_features == SIDECAR_INPUT_DIM + OUTPUT_DIM

    def test_condition_on_base_output_still_near_zero_at_init(self) -> None:
        """Base-output conditioning should not change the near-zero-init property."""
        torch.manual_seed(0)
        base = MLP(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIMS)
        composed = MLPWithSidecar.from_base_mlp(
            base,
            sidecar_input_dim=SIDECAR_INPUT_DIM,
            condition_on_base_output=True,
        )
        x = torch.randn(BATCH, INPUT_DIM)
        sidecar_x = torch.randn(BATCH, SIDECAR_INPUT_DIM)
        expected = composed.base(x)
        actual = composed(x, sidecar_x)
        assert torch.allclose(expected, actual, atol=0.1), "Should still be near base at init"

    def test_forward_output_shape(self) -> None:
        """Output shape should match base output_dim."""
        composed = MLPWithSidecar(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIMS, sidecar_input_dim=SIDECAR_INPUT_DIM)
        x = torch.randn(BATCH, INPUT_DIM)
        sidecar_x = torch.randn(BATCH, SIDECAR_INPUT_DIM)
        assert composed(x, sidecar_x).shape == (BATCH, OUTPUT_DIM)

    def test_sidecar_only_gradients_when_frozen(self) -> None:
        """With frozen base, only sidecar params should accumulate gradients."""
        base = MLP(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIMS)
        composed = MLPWithSidecar.from_base_mlp(base, sidecar_input_dim=SIDECAR_INPUT_DIM, freeze_base=True)
        x = torch.randn(BATCH, INPUT_DIM)
        sidecar_x = torch.randn(BATCH, SIDECAR_INPUT_DIM)
        loss = composed(x, sidecar_x).sum()
        loss.backward()
        for p in composed.base.parameters():
            assert p.grad is None
        sidecar_grads = [p.grad for p in composed.sidecar.parameters() if p.grad is not None]
        assert len(sidecar_grads) > 0

    def test_rejects_base_with_last_activation(self) -> None:
        """Should raise if the base ends with a non-Linear layer."""
        base = MLP(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIMS, last_activation="elu")
        with pytest.raises(ValueError, match="Linear layer"):
            MLPWithSidecar.from_base_mlp(base, sidecar_input_dim=SIDECAR_INPUT_DIM)
