# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the modular-norm MLP module."""

import torch
import torch.nn as nn

import pytest

from rsl_rl.modules.modular_norm_mlp import ModularNormMLP, orthogonalize


def _spectral_scales(mlp: ModularNormMLP) -> list[tuple[nn.Linear, float]]:
    """Return each linear layer with its expected spectral scale."""
    return [
        (module, (module.out_features / module.in_features) ** 0.5) for module in mlp if isinstance(module, nn.Linear)
    ]


class TestOrthogonalize:
    """Tests for the Newton-Schulz orthogonalization."""

    def test_singular_values_are_one(self) -> None:
        """All singular values of the orthogonalized matrix should be close to one."""
        torch.manual_seed(0)
        matrix = orthogonalize(torch.randn(64, 32))
        singular_values = torch.linalg.svdvals(matrix)
        assert torch.allclose(singular_values, torch.ones_like(singular_values), atol=0.05)

    def test_scale_invariance(self) -> None:
        """The result should be invariant to the scale of the input."""
        torch.manual_seed(0)
        matrix = torch.randn(32, 64)
        assert torch.allclose(orthogonalize(matrix), orthogonalize(100.0 * matrix), atol=1e-4)


class TestModularNormMLP:
    """Tests for ``ModularNormMLP``."""

    def test_bias_free_with_target_norm_buffers(self) -> None:
        """All linear layers should be bias-free and carry a ``target_norm`` budget of 1 / num_layers."""
        mlp = ModularNormMLP(8, 4, [16, 16])
        linear_layers = [module for module in mlp if isinstance(module, nn.Linear)]
        for linear in linear_layers:
            assert linear.bias is None
            assert torch.allclose(linear.target_norm, torch.tensor(1.0 / len(linear_layers)))

    def test_initialization_on_manifold(self) -> None:
        """Initial weights should have singular values equal to the spectral scale sqrt(out / in)."""
        torch.manual_seed(0)
        mlp = ModularNormMLP(8, 4, [16, 16])
        for linear, scale in _spectral_scales(mlp):
            singular_values = torch.linalg.svdvals(linear.weight)
            assert torch.allclose(singular_values, torch.full_like(singular_values, scale), atol=0.05)

    def test_project_weights_returns_to_manifold(self) -> None:
        """Perturbed weights should be back on the constraint manifold after projection."""
        torch.manual_seed(0)
        mlp = ModularNormMLP(8, 4, [16, 16])
        with torch.no_grad():
            for linear, _ in _spectral_scales(mlp):
                linear.weight += 0.5 * torch.randn_like(linear.weight)
        mlp.project_weights()
        for linear, scale in _spectral_scales(mlp):
            singular_values = torch.linalg.svdvals(linear.weight)
            assert torch.allclose(singular_values, torch.full_like(singular_values, scale), atol=0.05)

    def test_dualize_gradients(self) -> None:
        """Dualized gradients should have spectral norm equal to the spectral scale times the norm budget.

        Small singular values of an ill-conditioned gradient converge slowly under Newton-Schulz, so only the
        spectral norm (largest singular value) is checked tightly; the rest must not exceed it.
        """
        torch.manual_seed(0)
        mlp = ModularNormMLP(8, 4, [16, 16])
        mlp(torch.randn(32, 8)).square().sum().backward()
        mlp.dualize_gradients()
        for linear, scale in _spectral_scales(mlp):
            singular_values = torch.linalg.svdvals(linear.weight.grad)
            expected = scale * linear.target_norm
            assert abs(singular_values.max().item() - expected) < 0.05 * expected
            assert singular_values.max().item() < expected * 1.1

    def test_forward_matches_plain_sequential(self) -> None:
        """The forward pass should be a plain MLP forward (manifold only constrains the weights)."""
        torch.manual_seed(0)
        mlp = ModularNormMLP(8, 4, [16, 16], activation="elu")
        x = torch.randn(5, 8)
        expected = x
        for layer in mlp:
            expected = layer(expected)
        assert torch.allclose(mlp(x), expected)

    def test_non_lipschitz_activation_raises(self) -> None:
        """Activations that are not 1-Lipschitz should be rejected."""
        with pytest.raises(ValueError):
            ModularNormMLP(8, 4, [16, 16], activation="sigmoid")
