# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapter model over a frozen SONIC base: LoRA on the action decoder only.

The g1 encoder and FSQ quantizer stay untouched — a weight delta upstream of the
quantizer would be snapped away by the rounding until it crosses half a grid step
(gradient only via STE), and token space already has its own adaptation interface
(``latent_residual``). The decoder is where new conditioning belongs.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.modules import MLPWithAdapter

from ._onnx_export import Port, merge_adapters
from .mlp_adapter_model import AdapterStreamMixin
from .sonic_base_model import SonicBaseModel, _OnnxSonicBaseModel


class SonicWithAdapterModel(AdapterStreamMixin, SonicBaseModel):
    """A :class:`SonicBaseModel` with trainable, zero-initialized LoRA adapters on the decoder.

    The base stream (tokenizer + proprio) runs raw, exactly as the frozen base;
    the adapter stream (``adapter_obs_group``) has its own trainable normalizer
    and conditions the first adapter layer. Zero-init adapters reproduce the
    base bit-exactly at construction.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        adapter_obs_group: str | list[str] = "augmentation",
        rank: int | list[int | None] = -1,
        alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Load the frozen base (see :class:`SonicBaseModel`), then strap decoder adapters."""
        # Parent loads base_checkpoint into the plain decoder (keys match pre-strap).
        super().__init__(obs=obs, obs_groups=obs_groups, obs_set=obs_set, output_dim=output_dim, **kwargs)

        adapter_dim = self._init_adapter_stream(obs, adapter_obs_group, obs_normalization=True)
        # The adapters ride the DECODER, so its freeze flag — not the coarse
        # freeze_base — decides whether the base weights below them are trainable.
        self.decoder = MLPWithAdapter.from_base_mlp(
            self.decoder, adapter_input_dim=adapter_dim, rank=rank, alpha=alpha,
            freeze_base=self.freeze_decoder,
        )
        self._print_param_summary(self.freeze_decoder)

    @property
    def _adapted_mlp(self) -> MLPWithAdapter:
        """The decoder carries the adapters."""
        return self.decoder

    def _decode(self, tokens: torch.Tensor, obs: TensorDict) -> torch.Tensor:
        """Adapted decoder pass, conditioned on the adapter stream."""
        base_input = torch.cat([tokens, self._get_proprio(obs)], dim=-1)
        return self.decoder(base_input, self._get_adapter_latent(obs))

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a multi-input ONNX wrapper (base ports + the adapter stream)."""
        return _OnnxSonicAdapterModel(self, verbose)


class _OnnxSonicAdapterModel(_OnnxSonicBaseModel):
    """Exportable SONIC + LoRA: the base graph with the adapters folded into the decoder.

    The adapter stream contributes one input per group and its normalizer; the LoRA weights
    live in the merged decoder (see :func:`~rsl_rl.models._onnx_export.merge_adapters`), so
    the exported graph has no adapter branch at all.
    """

    def _merge_decoder(self, model: SonicWithAdapterModel) -> nn.Sequential:  # type: ignore[override]
        merged, _ = merge_adapters(model.decoder)
        return merged

    def _adapter_ports(self, model: SonicWithAdapterModel) -> list[Port]:  # type: ignore[override]
        self.adapter_normalizer = copy.deepcopy(model.adapter_normalizer)
        return [Port(g, model.export_shapes[g], (g,)) for g in model.adapter_obs_groups]

    def _adapter_latent(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        return self.adapter_normalizer(torch.cat(inputs, dim=-1))
