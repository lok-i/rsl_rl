# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sidecar: per-network action-residual for base-policy adaptation (RPL; Tom Silver et al. 2018)."""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation

from .mlp import MLP
from .modular_norm_mlp import ModularNormMLP


class Sidecar(nn.Module):
    """A zero-initialized side network whose output is added to a base network's output.

    The trunk (hidden layers) uses standard initialization; the output head is zero-initialized,
    so ``forward(x) == 0`` at construction and the composed ``base + sidecar`` reproduces the
    base exactly.

    Args:
        input_dim: Sidecar input width (may differ from the base input).
        output_dim: Must match the base output width.
        hidden_dims: Sidecar-private hidden layer widths. Empty → linear zero-init projection.
        activation: Activation between hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256,),
        activation: str = "elu",
        init_gain: float | None = None,
    ) -> None:
        super().__init__()
        act = resolve_nn_activation(activation)

        if hidden_dims:
            layers: list[nn.Module] = []
            prev = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev, h))
                layers.append(act)
                prev = h
            self.trunk = nn.Sequential(*layers)
            head_in = prev
        else:
            self.trunk = nn.Identity()
            head_in = input_dim

        self.head = nn.Linear(head_in, output_dim, bias=False)
        nn.init.zeros_(self.head.weight)

        if init_gain is not None:
            self._init_trunk_xavier(init_gain)

    def _init_trunk_xavier(self, gain: float) -> None:
        """Re-initialize trunk linear layers with Xavier uniform scaled by ``gain``."""
        for module in self.trunk.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the (zero-at-init) residual action."""
        return self.head(self.trunk(x))


class MLPWithSidecar(nn.Module):
    """A frozen base MLP augmented with a trainable :class:`Sidecar` at the output.

    ``a = base(obs) + sidecar(sidecar_obs [, base(obs)])``

    The sidecar is a separate MLP with its own architecture and a zero-initialized output head,
    so the composed network starts from the base behaviour exactly. The first (sidecar) input may
    differ in width from the base input — e.g. an augmentation observation group that the base
    policy was never trained on.

    When ``condition_on_base_output`` is enabled, the base action is concatenated to the sidecar
    input so the residual can condition on what the base is already doing.

    Contrast with :class:`~rsl_rl.modules.MLPWithAdapter` which applies per-layer weight
    residuals (LoRA; Hu et al. 2021).

    Args:
        input_dim: Dimension of the base input.
        output_dim: Dimension of the output.
        hidden_dims: Dimensions of the base hidden layers.
        activation: Activation function of the base MLP.
        last_activation: Activation of the base MLP's last layer (``None`` → linear).
        sidecar_input_dim: Input dimension of the sidecar. ``None`` reuses ``input_dim``.
        sidecar_hidden_dims: Hidden layer widths of the sidecar network.
        sidecar_activation: Activation function of the sidecar trunk.
        sidecar_init_gain: If set, re-initialize the sidecar trunk with Xavier uniform scaled by this gain.
        condition_on_base_output: If ``True``, the base output is concatenated to the sidecar input.
        freeze_base: If ``True`` (default), the base weights are frozen.
    """

    base_mlp_class: type[MLP] = MLP
    """The class used to build the frozen base. Subclasses override this."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        hidden_dims: tuple[int, ...] | list[int],
        activation: str = "elu",
        last_activation: str | None = None,
        sidecar_input_dim: int | None = None,
        sidecar_hidden_dims: tuple[int, ...] | list[int] = (256,),
        sidecar_activation: str = "elu",
        sidecar_init_gain: float | None = None,
        condition_on_base_output: bool = False,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        base = self.base_mlp_class(input_dim, output_dim, hidden_dims, activation, last_activation)
        self._set_base(base, freeze_base)
        self.condition_on_base_output = condition_on_base_output
        self._build_sidecar(
            sidecar_input_dim if sidecar_input_dim is not None else input_dim,
            sidecar_hidden_dims,
            sidecar_activation,
            sidecar_init_gain,
        )

    @classmethod
    def from_base_mlp(
        cls,
        base: MLP,
        sidecar_input_dim: int | None = None,
        sidecar_hidden_dims: tuple[int, ...] | list[int] = (256,),
        sidecar_activation: str = "elu",
        sidecar_init_gain: float | None = None,
        condition_on_base_output: bool = False,
        freeze_base: bool = True,
    ) -> MLPWithSidecar:
        """Wrap an already-built (e.g. checkpoint-loaded) base MLP with a sidecar.

        Unlike :meth:`__init__` this keeps the supplied base verbatim (its loaded weights),
        which is the path used to adapt a pretrained policy.
        """
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj._set_base(base, freeze_base)
        obj.condition_on_base_output = condition_on_base_output
        base_input_dim = next(m for m in base if isinstance(m, nn.Linear)).in_features
        obj._build_sidecar(
            sidecar_input_dim if sidecar_input_dim is not None else base_input_dim,
            sidecar_hidden_dims,
            sidecar_activation,
            sidecar_init_gain,
        )
        return obj

    def _set_base(self, base: MLP, freeze_base: bool = True) -> None:
        """Register the base and optionally freeze it."""
        if not isinstance(base[-1], nn.Linear):
            raise ValueError(
                "MLPWithSidecar requires a base ending in a Linear layer (no last_activation and an "
                f"integer output_dim); got a trailing {type(base[-1]).__name__}."
            )
        if freeze_base:
            base.requires_grad_(False)
        self.base = base

    def _build_sidecar(
        self,
        sidecar_input_dim: int,
        sidecar_hidden_dims: tuple[int, ...] | list[int],
        sidecar_activation: str,
        init_gain: float | None = None,
    ) -> None:
        """Build the zero-initialized sidecar network matching the base output width."""
        base_last: nn.Linear = self.base[-1]  # type: ignore[assignment]
        output_dim: int = base_last.out_features
        if self.condition_on_base_output:
            sidecar_input_dim += output_dim
        self.sidecar = Sidecar(sidecar_input_dim, output_dim, sidecar_hidden_dims, sidecar_activation, init_gain)

    def forward(self, obs: torch.Tensor, sidecar_obs: torch.Tensor | None = None) -> torch.Tensor:
        """Forward: ``base(obs) + sidecar(sidecar_obs [, base(obs)])``.

        Args:
            obs: Input to the base.
            sidecar_obs: Input to the sidecar. ``None`` reuses ``obs``.
        """
        base_output = self.base(obs)
        if sidecar_obs is None:
            sidecar_obs = obs
        if self.condition_on_base_output:
            sidecar_obs = torch.cat([sidecar_obs, base_output], dim=-1)
        return base_output + self.sidecar(sidecar_obs)


class ModularNormMLPWithSidecar(MLPWithSidecar):
    """An :class:`MLPWithSidecar` whose frozen base is a :class:`~rsl_rl.modules.ModularNormMLP`."""

    base_mlp_class: type[MLP] = ModularNormMLP
