# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared spine for multi-input ONNX exports (frozen-base deploy path).

One ONNX input per observation GROUP (dict groups expand to one input per term), so a
C++ node fills named buffers instead of hand-counting offsets into a single flat vector.
Ports are DERIVED from the model (``obs_groups`` / ``tokenizer_obs_group`` /
``adapter_obs_groups`` / ``extractor.input_groups``), never written twice.

Two folds keep the exported graph minimal (see :func:`merge_adapters`, :class:`FoldedFSQ`):
adapters collapse into the base weights, FSQ constants collapse into buffers.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from dataclasses import dataclass, replace
from tensordict import TensorDict


@dataclass(frozen=True)
class Port:
    """One ONNX input: a named tensor and the observation group(s) that fill it.

    ``groups`` holds more than one entry only after :meth:`_OnnxExportBase.alias_ports`
    merged content-identical groups (e.g. ``augmentation`` and ``q_motion_cmd``, which are
    the same term bundle by construction).
    """

    name: str
    shape: tuple[int, ...]
    groups: tuple[str, ...]
    term: str | None = None


def capture_obs_shapes(obs: TensorDict) -> dict:
    """Snapshot per-group (and per-term, for dict groups) shapes minus the batch axis.

    Stored on the model at construction so the export wrapper can size its ports without
    a live environment.
    """
    shapes: dict = {}
    for key in obs.keys():  # noqa: SIM118 — TensorDict keys() is not a dict view
        value = obs[key]
        if hasattr(value, "keys"):  # dict group (concatenate_terms=False)
            shapes[key] = {t: tuple(value[t].shape[1:]) for t in list(value.keys())}
        else:
            shapes[key] = tuple(value.shape[1:])
    return shapes


def merge_adapters(adapted: nn.Module) -> tuple[nn.Sequential, int]:
    """Fold a :class:`~rsl_rl.modules.MLPWithAdapter` into a plain MLP.

    ``MLPWithAdapter.forward`` feeds every adapter ``i >= 1`` the same activation its base
    linear gets, so ``W_i h + dW_i h == (W_i + dW_i) h`` — a pure weight merge. Adapter 0
    reads the adapter stream instead, which merges as a column block::

        W_0 x + dW_0 a  ==  [W_0 | dW_0] [x ; a]

    so the whole adapted network collapses to a plain ``Sequential`` whose input is the base
    input concatenated with the adapter latent. Exact in real arithmetic; fp32 reassociation
    puts the residual at ~1e-7 relative (the export parity check reports the measured value).

    The merge runs on a CPU copy on purpose: ``Adapter.delta_weight()`` is a matmul
    (``up.weight @ down.weight``), so on CUDA with TF32 enabled — which is what
    ``configure_torch_backends()`` sets for train and play — the low-rank delta would be
    folded at a 10-bit mantissa and cost ~1e-4 on the exported actions.

    Returns:
        ``(merged_mlp, adapter_dim)`` — ``adapter_dim`` is the width appended to the input.
    """
    adapted = copy.deepcopy(adapted).to("cpu")
    linears, adapters = adapted.base_linears, adapted.adapters
    adapter_dim = 0
    merged: list[nn.Module] = []
    for i, linear in enumerate(linears):
        adapter = adapters[i]
        delta = None if adapter is None else adapter.delta_weight().detach()
        weight = linear.weight.detach().clone()
        if i == 0 and delta is not None:
            adapter_dim = delta.shape[1]
            weight = torch.cat([weight, delta], dim=1)
        elif delta is not None:
            weight = weight + delta
        new = nn.Linear(weight.shape[1], weight.shape[0], bias=linear.bias is not None)
        with torch.no_grad():
            new.weight.copy_(weight)
            if linear.bias is not None:
                new.bias.copy_(linear.bias.detach())
        merged.append(new)

    activation = adapted.activation
    layers: list[nn.Module] = []
    for i, linear in enumerate(merged):
        if i:
            layers.append(copy.deepcopy(activation))
        layers.append(linear)
    return nn.Sequential(*layers), adapter_dim


def plain_mlp_copy(mlp: nn.Sequential) -> nn.Sequential:
    """Deep-copy an un-adapted MLP (the no-adapter counterpart of :func:`merge_adapters`)."""
    return copy.deepcopy(mlp)


class FoldedFSQ(nn.Module):
    """Finite Scalar Quantization with its level-derived constants precomputed.

    ``half_l``, ``shift``, ``offset`` and ``half_width`` are pure functions of ``levels``, so
    they become buffers instead of graph nodes. The straight-through term is dropped (it is
    the identity at inference), leaving Add/Tanh/Mul/Sub/Round/Div — all opset-18 native.
    ``torch.round`` and ONNX ``Round`` both break ties to even.
    """

    def __init__(self, levels: torch.Tensor, eps: float = 1e-3) -> None:
        """Precompute the quantizer constants from the per-dimension codebook sizes."""
        super().__init__()
        levels = levels.to(torch.float32)
        half_l = (levels - 1) * (1 + eps) / 2
        offset = torch.where(levels % 2 == 0, 0.5, 0.0)
        self.register_buffer("half_l", half_l)
        self.register_buffer("offset", offset)
        self.register_buffer("shift", (offset / half_l).atanh())
        self.register_buffer("half_width", torch.div(levels, 2, rounding_mode="floor"))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize the latent to the FSQ grid, rescaled to [-1, 1]."""
        bounded = (z + self.shift).tanh() * self.half_l - self.offset
        return bounded.round() / self.half_width


class _OnnxExportBase(nn.Module):
    """Base for multi-input ONNX wrappers: port bookkeeping + tracing metadata.

    Subclasses call :meth:`_set_slots` with one :class:`Port` per positional ``forward``
    argument. Ports are 1:1 with ONNX inputs until :meth:`alias_ports` merges duplicates.
    """

    is_recurrent: bool = False

    def __init__(self, verbose: bool = False) -> None:
        """Initialize with an empty port table."""
        super().__init__()
        self.verbose = verbose
        self._slots: list[Port] = []
        self._ports: list[Port] = []
        self._slot_to_port: list[int] = []

    # --- port table ---

    def _set_slots(self, slots: list[Port]) -> None:
        """Declare the positional inputs of ``forward`` (one port each, no aliasing yet)."""
        self._slots = list(slots)
        self._ports = list(slots)
        self._slot_to_port = list(range(len(slots)))

    def alias_ports(self, aliases: dict[str, str]) -> None:
        """Merge content-identical groups onto one ONNX input.

        Args:
            aliases: ``{group_or_port_name: canonical_port_name}``. The canonical port must
                exist and have the same shape; the alias disappears from ``input_names`` and
                its ``forward`` slot is fed from the canonical input.

        The caller owns the equivalence proof (vibe compares ordered observation-term lists);
        this only rewires. Shapes are checked here so a wrong alias fails at export.
        """
        by_name = {p.name: p for p in self._ports}
        redirect: dict[str, str] = {}
        for alias, canonical in aliases.items():
            if alias == canonical or alias not in by_name:
                continue  # unknown alias: group not exported (e.g. a disabled query row)
            if canonical not in by_name:
                raise ValueError(f"alias target '{canonical}' is not an input port.")
            if by_name[alias].shape != by_name[canonical].shape:
                raise ValueError(
                    f"cannot alias '{alias}' {by_name[alias].shape} onto "
                    f"'{canonical}' {by_name[canonical].shape}: shape mismatch."
                )
            redirect[alias] = canonical

        slot_names = [self._ports[i].name for i in self._slot_to_port]
        kept: list[Port] = []
        for port in self._ports:
            if port.name in redirect:
                continue
            merged = tuple(
                g for a, c in redirect.items() if c == port.name for g in by_name[a].groups
            )
            kept.append(replace(port, groups=port.groups + merged) if merged else port)
        index = {p.name: i for i, p in enumerate(kept)}
        self._slot_to_port = [index[redirect.get(n, n)] for n in slot_names]
        self._ports = kept

    def _gather(self, inputs: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        """Expand the ONNX inputs to one tensor per ``forward`` slot (undoing aliasing)."""
        return [inputs[i] for i in self._slot_to_port]

    # --- export interface (consumed by the runner's torch.onnx.export call) ---

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Zero tensors matching every ONNX input, batch size 1."""
        return tuple(torch.zeros(1, *p.shape) for p in self._ports)

    @property
    def input_names(self) -> list[str]:
        """ONNX input tensor names, in ``forward`` order."""
        return [p.name for p in self._ports]

    @property
    def output_names(self) -> list[str]:
        """ONNX output tensor names."""
        return ["actions"]

    @property
    def layout(self) -> list[dict]:
        """Port table for the deploy manifest (names, shapes, source groups)."""
        return [
            {"name": p.name, "shape": list(p.shape), "groups": list(p.groups), "term": p.term}
            for p in self._ports
        ]
