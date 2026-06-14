# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn

from .mlp import MLP
from .modular_norm_mlp import ModularNormMLP


class Adapter(nn.Module):
    """A single zero-initialized LoRA-style adapter for one linear layer.

    The adapter realizes a trainable weight delta ``Delta W`` that is added to a frozen base layer's
    output (both consume the same input, so the pair computes ``(W + Delta W) x``). It interpolates
    between two regimes via ``rank``:

    - ``rank <= 0`` or ``rank >= min(in, out)``: **full-rank** delta, a single dense ``Linear``.
    - ``0 < rank < min(in, out)``: **low-rank** delta ``Delta W = B A`` with inner dimension ``rank``.

    The output-side factor is zero-initialized (the dense weight for full-rank, the up-projection ``B``
    for low-rank), so ``Delta W = 0`` at construction and the adapted network is identical to the base.

    Args:
        input_dim: Input dimension of the layer the adapter augments.
        output_dim: Output dimension of the layer the adapter augments.
        rank: Rank of the delta. ``<= 0`` (or ``>= min(input_dim, output_dim)``) selects a full-rank delta.
        alpha: Scale applied to the delta. For low-rank the effective scale is ``alpha / rank`` (LoRA
            convention), decoupling magnitude from rank; for full-rank it is ``alpha``.
    """

    def __init__(self, input_dim: int, output_dim: int, rank: int = -1, alpha: float = 1.0) -> None:
        """Initialize the adapter, zero-initializing the output-side factor."""
        super().__init__()

        self.low_rank = 0 < rank < min(input_dim, output_dim)
        if self.low_rank:
            self.down = nn.Linear(input_dim, rank, bias=False)
            self.up = nn.Linear(rank, output_dim, bias=False)
            nn.init.normal_(self.down.weight, std=1.0 / rank**0.5)
            nn.init.zeros_(self.up.weight)
            self.scale = alpha / rank
        else:
            self.delta = nn.Linear(input_dim, output_dim, bias=False)
            nn.init.zeros_(self.delta.weight)
            self.scale = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the (scaled) weight delta applied to ``x``."""
        if self.low_rank:
            return self.scale * self.up(self.down(x))
        return self.scale * self.delta(x)

    def delta_weight(self) -> torch.Tensor:
        """Return the effective weight delta ``Delta W`` (shape ``[out, in]``) this adapter adds.

        Useful for logging the adapter magnitude or merging it into the base weight at deploy time.
        """
        if self.low_rank:
            return self.scale * (self.up.weight @ self.down.weight)
        return self.scale * self.delta.weight


class MLPWithAdapter(nn.Module):
    """An MLP with a frozen base and a trainable, zero-initialized adapter on every linear layer.

    The base is a standard :class:`~rsl_rl.modules.MLP` whose weights are frozen; each of its linear
    layers is augmented with an :class:`Adapter`. The forward pass interleaves the two branches in the
    LoRA sense -- at every layer the frozen weight ``W_i`` and the trainable delta ``Delta W_i`` consume
    the same running activation, so the layer computes ``(W_i + Delta W_i) h``. Because the adapters are
    zero-initialized, the network reproduces the base exactly at construction.

    The first adapter may take a distinct input (``adapter_obs``) -- e.g. a dedicated observation group
    of new conditioning features whose width differs from the base input. If none is supplied at call
    time, the base input is reused.
    """

    base_mlp_class: type[MLP] = MLP
    """The class used to build the frozen base. Subclasses override this to change the base layer type."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        hidden_dims: tuple[int, ...] | list[int],
        activation: str = "elu",
        last_activation: str | None = None,
        adapter_input_dim: int | None = None,
        rank: int | list[int] = -1,
        alpha: float = 1.0,
    ) -> None:
        """Initialize the MLP, build the frozen base, and strap a zero-initialized adapter on each layer.

        Args:
            input_dim: Dimension of the base input.
            output_dim: Dimension of the output.
            hidden_dims: Dimensions of the hidden layers.
            activation: Activation function of the base MLP.
            last_activation: Activation of the base MLP's last layer. None results in a linear last layer.
            adapter_input_dim: Input dimension of the first (input) adapter. ``None`` reuses ``input_dim``.
            rank: Adapter rank, a scalar broadcast to all layers or a per-layer list (entry 0 is the input
                adapter, the last entry the output adapter). See :class:`Adapter` for the rank semantics.
            alpha: Adapter scale (see :class:`Adapter`).
        """
        super().__init__()
        base = self.base_mlp_class(input_dim, output_dim, hidden_dims, activation, last_activation)
        self._set_base(base)
        self._build_adapters(adapter_input_dim if adapter_input_dim is not None else input_dim, rank, alpha)

    @classmethod
    def from_base_mlp(
        cls,
        base: MLP,
        adapter_input_dim: int | None = None,
        rank: int | list[int] = -1,
        alpha: float = 1.0,
    ) -> MLPWithAdapter:
        """Wrap an already-built (e.g. checkpoint-loaded) base MLP, strapping adapters on its layers.

        Unlike :meth:`__init__` this keeps the supplied base verbatim (its loaded weights), which is the
        path used to adapt a pretrained policy.

        Args:
            base: The base MLP to freeze and adapt. Its weights are kept as-is.
            adapter_input_dim: Input dimension of the input adapter. ``None`` reuses the base input width.
            rank: Adapter rank (scalar or per-layer list). See :class:`Adapter`.
            alpha: Adapter scale.
        """
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj._set_base(base)
        base_input_dim = obj.base_linears[0].in_features
        obj._build_adapters(adapter_input_dim if adapter_input_dim is not None else base_input_dim, rank, alpha)
        return obj

    def _set_base(self, base: MLP) -> None:
        """Register the base as a frozen submodule and cache its linear layers and activation."""
        # The interleaved forward iterates only the linear layers, so a trailing op (a ``last_activation``
        # or the ``Unflatten`` of a tuple ``output_dim``) would be silently dropped. Require a linear tail.
        if not isinstance(base[-1], nn.Linear):
            raise ValueError(
                "MLPWithAdapter requires a base ending in a Linear layer (no last_activation and an integer "
                f"output_dim); got a trailing {type(base[-1]).__name__}."
            )
        base.requires_grad_(False)
        self.base = base
        # Cache references to the base's linear layers (not re-registered; they live under ``self.base``).
        self.base_linears = [module for module in base if isinstance(module, nn.Linear)]
        # Activation between layers (shared, stateless); ``None`` only if the base is a single linear layer.
        self.activation = next((module for module in base if not isinstance(module, nn.Linear)), None)

    def _build_adapters(self, adapter_input_dim: int, rank: int | list[int], alpha: float) -> None:
        """Build one zero-initialized adapter per base linear layer."""
        num_layers = len(self.base_linears)
        ranks = list(rank) if isinstance(rank, (list, tuple)) else [rank] * num_layers
        if len(ranks) != num_layers:
            raise ValueError(f"rank list has {len(ranks)} entries but the base has {num_layers} linear layers.")
        self.adapters = nn.ModuleList(
            Adapter(
                adapter_input_dim if i == 0 else linear.in_features,
                linear.out_features,
                ranks[i],
                alpha,
            )
            for i, linear in enumerate(self.base_linears)
        )

    def forward(self, obs: torch.Tensor, adapter_obs: torch.Tensor | None = None) -> torch.Tensor:
        """Interleaved forward pass over the frozen base and the trainable adapters.

        Args:
            obs: Input to the base.
            adapter_obs: Input to the first adapter. ``None`` reuses ``obs``.
        """
        if adapter_obs is None:
            adapter_obs = obs
        base_hidden = self.base_linears[0](obs)
        adapter_hidden = self.adapters[0](adapter_obs)
        for i in range(1, len(self.base_linears)):
            out = self.activation(base_hidden + adapter_hidden)
            base_hidden = self.base_linears[i](out)
            adapter_hidden = self.adapters[i](out)
        return base_hidden + adapter_hidden


class ModularNormMLPWithAdapter(MLPWithAdapter):
    """An :class:`MLPWithAdapter` whose frozen base is a :class:`~rsl_rl.modules.ModularNormMLP`.

    This is the variant used to adapt the (bias-free, modular-norm) TextOp WBC: load the pretrained
    weights into the base, freeze it, and train only the strapped free-Adam adapters. The adapters
    themselves are plain :class:`Adapter` modules (no modular-norm constraint).
    """

    base_mlp_class: type[MLP] = ModularNormMLP
