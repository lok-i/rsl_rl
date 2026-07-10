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

import torch
from tensordict import TensorDict
from typing import Any

from rsl_rl.modules import MLPWithAdapter

from .mlp_adapter_model import AdapterStreamMixin
from .sonic_base_model import SonicBaseModel


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
        self.decoder = MLPWithAdapter.from_base_mlp(
            self.decoder, adapter_input_dim=adapter_dim, rank=rank, alpha=alpha,
            freeze_base=self.freeze_base,
        )
        self._print_param_summary(self.freeze_base)

    @property
    def _adapted_mlp(self) -> MLPWithAdapter:
        """The decoder carries the adapters."""
        return self.decoder

    def _decode(self, tokens: torch.Tensor, obs: TensorDict) -> torch.Tensor:
        """Adapted decoder pass, conditioned on the adapter stream."""
        base_input = torch.cat([tokens, self._get_proprio(obs)], dim=-1)
        return self.decoder(base_input, self._get_adapter_latent(obs))
