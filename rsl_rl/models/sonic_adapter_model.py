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
    """A :class:`SonicBaseModel` with trainable, zero-initialized LoRA adapters.

    The base stream (tokenizer + proprio) runs raw, exactly as the frozen base;
    the adapter stream (``adapter_obs_group``) has its own trainable normalizer
    and conditions the first adapter layer. Zero-init adapters reproduce the
    base bit-exactly at construction.

    ``adapt_encoder`` / ``adapt_decoder`` choose WHERE the adapters attach — the axis that
    actually varies here, since ``freeze_base`` keeps the released weights frozen either
    way. It is also what decides whether the token cache is legal:

    ==================================  ==========================================  =========
    configuration                       adapted                                     tokens
    ==================================  ==========================================  =========
    ``adapt_decoder`` (**vibe**)        decoder only; new conditioning enters at    cacheable
                                        decoder layer 0
    ``adapt_encoder``                   encoder only; conditioning enters upstream  recomputed
                                        of FSQ
    both                                the full stack                              recomputed
    ==================================  ==========================================  =========

    **Adapting the encoder is a real design decision, not just a flag.** A weight delta
    upstream of the quantizer is snapped away by the rounding until it crosses half a grid
    step (gradient only via the straight-through estimator), and token space already has a
    native adaptation interface in ``latent_residual``. Decoder-only is the default for
    that reason; the encoder path exists for downstream users who want to fine-tune the
    tokenizer, and it costs the token cache.
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
        adapt_encoder: bool = False,
        adapt_decoder: bool = True,
        **kwargs: Any,
    ) -> None:
        """Load the frozen base (see :class:`SonicBaseModel`), then strap the adapters."""
        # Parent loads base_checkpoint into the plain halves (keys match pre-strap).
        super().__init__(obs=obs, obs_groups=obs_groups, obs_set=obs_set, output_dim=output_dim, **kwargs)
        if not (adapt_encoder or adapt_decoder):
            raise ValueError(
                "SonicWithAdapterModel needs adapt_encoder or adapt_decoder; with neither "
                "and a frozen base the model has no trainable parameters at all."
            )
        self.adapt_encoder, self.adapt_decoder = adapt_encoder, adapt_decoder

        adapter_dim = self._init_adapter_stream(obs, adapter_obs_group, obs_normalization=True)
        strap = dict(adapter_input_dim=adapter_dim, rank=rank, alpha=alpha, freeze_base=self.freeze_base)
        if adapt_decoder:
            self.decoder = MLPWithAdapter.from_base_mlp(self.decoder, **strap)
        if adapt_encoder:
            self.encoder = MLPWithAdapter.from_base_mlp(self.encoder, **strap)
            # The encoder now carries trainable weights, so its output is no longer a pure
            # function of a fixed input and the cached tokens would go stale mid-update.
            self.cache_tokens = False
        self._print_param_summary(self.freeze_base)

    @property
    def _adapted_mlp(self) -> MLPWithAdapter:
        """The half whose per-layer adapter norms are logged (decoder when both are on)."""
        return self.decoder if self.adapt_decoder else self.encoder

    def _adapter_cond(self, obs: TensorDict) -> torch.Tensor:
        """Adapter stream for this forward — computed once, shared by both halves."""
        return self._get_adapter_latent(obs)

    def _encode_mlp(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        """Encoder pass, conditioned on the adapter stream when the encoder is adapted."""
        return self.encoder(x, cond) if self.adapt_encoder else self.encoder(x)

    def _decode(
        self, tokens: torch.Tensor, obs: TensorDict, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Decoder pass, conditioned on the adapter stream when the decoder is adapted."""
        base_input = torch.cat([tokens, self._get_proprio(obs)], dim=-1)
        if not self.adapt_decoder:
            return self.decoder(base_input)
        return self.decoder(base_input, self._adapter_cond(obs) if cond is None else cond)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a multi-input ONNX wrapper (base ports + the adapter stream)."""
        if self.adapt_encoder:
            raise NotImplementedError(
                "ONNX export folds adapters into the decoder only; an adapted encoder "
                "would need the same merge upstream of the folded FSQ."
            )
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
