# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn

from .mlp import MLP


def orthogonalize(matrix: torch.Tensor) -> torch.Tensor:
    """Orthogonalize a matrix with a six-step Newton-Schulz iteration.

    Computes a (semi-)orthogonal matrix close to the input, i.e. the nearest matrix with all singular values equal
    to one. The result is invariant to the scale of the input.

    Coefficients by @YouJiacheng, found by optimization with a stability loss (idea by @leloykun):
    https://twitter.com/YouJiacheng/status/1893704552689303901

    Args:
        matrix: The 2D matrix to orthogonalize.

    Returns:
        The orthogonalized matrix.
    """
    abc_list = [
        (3955 / 1024, -8306 / 1024, 5008 / 1024),
        (3735 / 1024, -6681 / 1024, 3463 / 1024),
        (3799 / 1024, -6499 / 1024, 3211 / 1024),
        (4019 / 1024, -6385 / 1024, 2906 / 1024),
        (2677 / 1024, -3029 / 1024, 1162 / 1024),
        (2172 / 1024, -1833 / 1024, 682 / 1024),
    ]
    transpose = matrix.shape[1] > matrix.shape[0]
    if transpose:
        matrix = matrix.T
    matrix = matrix / torch.linalg.norm(matrix)
    for a, b, c in abc_list:
        gram = matrix.T @ matrix
        identity = torch.eye(gram.shape[0], device=matrix.device, dtype=matrix.dtype)
        matrix = matrix @ (a * identity + b * gram + c * gram @ gram)
    if transpose:
        matrix = matrix.T
    return matrix


class UnitLipschitzGELU(nn.GELU):
    """GELU activation rescaled to be 1-Lipschitz (its maximum derivative is ~1.1289)."""

    def __init__(self) -> None:
        """Initialize the activation with the tanh approximation."""
        super().__init__(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the rescaled GELU."""
        return super().forward(x) / 1.1289


class ModularNormMLP(MLP):
    """Multi-Layer Perceptron normed in the modular norm.

    A bias-free MLP with 1-Lipschitz activations whose linear layers live on the spectral-norm manifold
    ``norm(W) = sqrt(out_features / in_features)``. Weights are initialized (semi-)orthogonal on the manifold.

    Training in the modular norm replaces gradient clipping with two operations around the optimizer step:
    - :meth:`dualize_gradients`: orthogonalize each layer's gradient and rescale it by the layer's spectral
      scale and its ``target_norm`` budget (``1 / num_layers``), so the update has unit modular norm.
    - :meth:`project_weights`: re-project the weights back onto the constraint manifold after the step.

    Reference:
        Large et al., "Scalable Optimization in the Modular Norm." https://arxiv.org/abs/2405.14813
    """

    _supported_activations = ("relu", "elu", "gelu")
    """Activations that are 1-Lipschitz (gelu is rescaled to be)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        hidden_dims: tuple[int, ...] | list[int],
        activation: str = "elu",
        last_activation: str | None = None,
    ) -> None:
        """Initialize the modular-norm MLP.

        Args:
            input_dim: Dimension of the input.
            output_dim: Dimension of the output.
            hidden_dims: Dimensions of the hidden layers. A value of ``-1`` indicates that the dimension should be
                inferred from the input dimension.
            activation: Activation function. Must be 1-Lipschitz.
            last_activation: Activation function of the last layer. None results in a linear last layer.
        """
        for name in (activation, last_activation):
            if name is not None and name not in self._supported_activations:
                raise ValueError(
                    f"Activation '{name}' is not supported for the modular norm. The activation must be 1-Lipschitz."
                    f" Supported activations: {self._supported_activations}."
                )
        super().__init__(input_dim, output_dim, hidden_dims, activation, last_activation, bias=False)

        # Replace GELU activations with the 1-Lipschitz rescaled variant
        for idx, module in enumerate(self):
            if isinstance(module, nn.GELU):
                self[idx] = UnitLipschitzGELU()

        # Register the per-layer norm budget and initialize the weights on the manifold
        linear_layers = [module for module in self if isinstance(module, nn.Linear)]
        for linear in linear_layers:
            linear.register_buffer("target_norm", torch.tensor(1.0 / len(linear_layers)))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize all linear layers with (semi-)orthogonal weights on the constraint manifold."""
        with torch.no_grad():
            for module in self:
                if isinstance(module, nn.Linear):
                    weight = torch.randn_like(module.weight)
                    weight = orthogonalize(weight) * self._spectral_scale(module)
                    module.weight.copy_(weight)

    def project_weights(self) -> None:
        """Project the weights of all linear layers back onto the constraint manifold."""
        with torch.no_grad():
            for module in self:
                if isinstance(module, nn.Linear):
                    weight = orthogonalize(module.weight) * self._spectral_scale(module)
                    module.weight.copy_(weight)

    def dualize_gradients(self) -> None:
        """Dualize the gradients of all linear layers, replacing gradient clipping."""
        with torch.no_grad():
            for module in self:
                if isinstance(module, nn.Linear) and module.weight.grad is not None:
                    grad = orthogonalize(module.weight.grad) * self._spectral_scale(module) * module.target_norm
                    module.weight.grad.copy_(grad)

    @staticmethod
    def _spectral_scale(linear: nn.Linear) -> torch.Tensor:
        """Return the spectral scale ``sqrt(out_features / in_features)`` of a linear layer."""
        return torch.sqrt(torch.tensor(linear.out_features / linear.in_features))
