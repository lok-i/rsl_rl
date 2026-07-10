# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Models whose feature observation groups pass through trainable :class:`FeatureEncoder` s.

Aux-agnostic by design: these models only expose the encoder(s); which auxiliary objective
trains them (if any) is decided by the algorithm (see ``rsl_rl.algorithms.PPOAux``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.modules import EmpiricalNormalization, FeatureEncoder, HiddenState

from .mlp_adapter_model import ModularNormMLPWithAdapterModel
from .mlp_model import MLPModel
from .sonic_adapter_model import SonicWithAdapterModel


def _build_encoders(obs: TensorDict, encoder_cfg: dict[str, dict] | None, encoders: dict | None) -> dict:
    """Build (or validate) one FeatureEncoder per configured observation group."""
    if encoders is not None:
        return dict(encoders)
    if not encoder_cfg:
        raise ValueError("Either 'encoder_cfg' or 'encoders' must be provided.")
    built = {}
    for group, cfg in encoder_cfg.items():
        if len(obs[group].shape) != 2:
            raise ValueError(f"FeatureEncoder expects a flat observation group, got {obs[group].shape} for '{group}'.")
        built[group] = FeatureEncoder(obs[group].shape[-1], **cfg)
    return built


class FeatureEncoderMLPModel(MLPModel):
    """An :class:`MLPModel` where configured observation groups are encoded before the MLP.

    Groups named in ``encoder_cfg`` are projected by a :class:`FeatureEncoder` (own normalizer);
    the remaining groups follow the plain MLPModel path (shared normalizer, direct concat).
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        encoder_cfg: dict[str, dict] | None = None,
        encoders: dict | None = None,
    ) -> None:
        """Initialize the MLP model with per-group feature encoders (see ``encoder_cfg``)."""
        # Plain-dict attribute: safe before nn.Module.__init__, registered as ModuleDict after.
        self._encoders = _build_encoders(obs, encoder_cfg, encoders)
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )
        self.encoders = nn.ModuleDict(self._encoders)

    # --- overrides ---

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Split active groups into encoded and plain; report only plain dims to the parent."""
        active_obs_groups = obs_groups[obs_set]
        self.encoded_obs_groups = [g for g in active_obs_groups if g in self._encoders]
        if not self.encoded_obs_groups:
            raise ValueError(
                f"None of the active observation groups {active_obs_groups} has an encoder. "
                "Use MLPModel instead if this is intentional."
            )
        plain_groups = [g for g in active_obs_groups if g not in self._encoders]
        obs_dim = 0
        for g in plain_groups:
            if len(obs[g].shape) != 2:
                raise ValueError(f"The MLP model only supports 1D observations, got {obs[g].shape} for '{g}'.")
            obs_dim += obs[g].shape[-1]
        return plain_groups, obs_dim

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self.obs_dim + sum(self._encoders[g].latent_dim for g in self.encoded_obs_groups)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent from encoder latents and plain normalized groups."""
        latent_enc = self.encode(obs)
        if not self.obs_groups:
            return latent_enc
        return torch.cat([super().get_latent(obs), latent_enc], dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update plain-group and encoder normalization statistics."""
        if self.obs_groups:
            super().update_normalization(obs)
        for g in self.encoded_obs_groups:
            self.encoders[g].update_normalization(obs[g])

    # --- aux interface ---

    def encode(self, obs: TensorDict) -> torch.Tensor:
        """Concatenated encoder latents (the representation shared with auxiliary objectives)."""
        return torch.cat([self.encoders[g](obs[g]) for g in self.encoded_obs_groups], dim=-1)

    # --- export (not supported yet) ---

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        raise NotImplementedError("JIT export for FeatureEncoderMLPModel is not implemented yet.")

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        raise NotImplementedError("ONNX export for FeatureEncoderMLPModel is not implemented yet.")


class EncodedAdapterStreamMixin:
    """Feature-encoded adapter stream for frozen-base + LoRA models.

    The frozen base stream is untouched. The adapter stream may mix encoded groups (through
    their FeatureEncoders) and plain groups (through a trainable stream normalizer);
    ``adapter_obs_group`` accepts a single group name or a list. Exposes ``encoders`` and
    ``encode()`` — the contract PPOAux objectives bind to.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        adapter_obs_group: str | list[str] = "img_feat",
        encoder_cfg: dict[str, dict] | None = None,
        encoders: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the host adapter model with a feature-encoded adapter stream."""
        self._encoders = _build_encoders(obs, encoder_cfg, encoders)
        stream_groups = [adapter_obs_group] if isinstance(adapter_obs_group, str) else list(adapter_obs_group)
        # The host wraps adapter_obs_group in a list; our _concat_dim override flattens it and
        # reports encoder latent dims so the LoRA adapter is strapped with the effective input dim.
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            adapter_obs_group=stream_groups,  # type: ignore[arg-type]
            **kwargs,
        )
        self.encoders = nn.ModuleDict(self._encoders)
        for name, enc in self.encoders.items():
            n_params = sum(p.numel() for p in enc.parameters())
            print(f"  + feature encoder['{name}']: {n_params:,} trainable params (not in summary above)")
        self.adapter_obs_groups = stream_groups
        self.encoded_adapter_groups = [g for g in stream_groups if g in self._encoders]
        if not self.encoded_adapter_groups:
            raise ValueError(f"No adapter-stream group in {stream_groups} has an encoder.")
        self.plain_adapter_groups = [g for g in stream_groups if g not in self._encoders]
        # Replace the host's stream normalizer (sized for the full effective dim) with one
        # covering only the plain groups; encoded groups normalize inside their encoders.
        plain_dim = sum(obs[g].shape[-1] for g in self.plain_adapter_groups)
        if plain_dim > 0:
            self.adapter_normalizer = EmpiricalNormalization(plain_dim)
        else:
            self.adapter_normalizer = nn.Identity()

    # --- overrides ---

    def _concat_dim(self, obs: TensorDict, groups: list) -> int:  # type: ignore[override]
        """Effective stream dim: encoder latents for encoded groups, raw dims otherwise."""
        flat = [g for item in groups for g in ([item] if isinstance(item, str) else item)]
        return sum(self._encoders[g].latent_dim if g in self._encoders else obs[g].shape[-1] for g in flat)

    def _get_adapter_latent(self, obs: TensorDict) -> torch.Tensor:
        parts = []
        if self.plain_adapter_groups:
            plain = torch.cat([obs[g] for g in self.plain_adapter_groups], dim=-1)
            parts.append(self.adapter_normalizer(plain))
        parts += [self.encoders[g](obs[g]) for g in self.encoded_adapter_groups]
        return torch.cat(parts, dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update the plain-stream normalizer and encoder normalizers; base stays frozen."""
        if self.plain_adapter_groups and isinstance(self.adapter_normalizer, EmpiricalNormalization):
            plain = torch.cat([obs[g] for g in self.plain_adapter_groups], dim=-1)
            self.adapter_normalizer.update(plain)
        for g in self.encoded_adapter_groups:
            self.encoders[g].update_normalization(obs[g])

    # --- aux interface ---

    def encode(self, obs: TensorDict) -> torch.Tensor:
        """Concatenated encoder latents (the representation shared with auxiliary objectives)."""
        return torch.cat([self.encoders[g](obs[g]) for g in self.encoded_adapter_groups], dim=-1)


class FeatureEncoderAdapterModel(EncodedAdapterStreamMixin, ModularNormMLPWithAdapterModel):
    """A :class:`ModularNormMLPWithAdapterModel` whose adapter stream is feature-encoded."""


class FeatureEncoderSonicAdapterModel(EncodedAdapterStreamMixin, SonicWithAdapterModel):
    """A :class:`SonicWithAdapterModel` whose adapter stream is feature-encoded."""
