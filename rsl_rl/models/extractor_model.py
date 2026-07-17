# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Models whose feature observation groups pass through trainable extractors.

Aux-agnostic by design: these models only expose the extractor(s); which auxiliary objective
trains them (if any) is decided by the algorithm (see ``rsl_rl.algorithms.PPOAux``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.modules import EmpiricalNormalization, HiddenState
from rsl_rl.utils import resolve_callable

from .mlp_adapter_model import ModularNormMLPWithAdapterModel
from .mlp_model import MLPModel
from .sonic_adapter_model import SonicWithAdapterModel


def _build_extractors(obs: TensorDict, extractor_cfg: dict[str, dict] | None, extractors: dict | None) -> dict:
    """Build (or validate) one extractor per configured observation group.

    Each per-group cfg may name its extractor class via ``class_name`` (default:
    :class:`MlpExtractor`); construction goes through the class's ``from_obs``, which sets
    ``input_groups`` — the group names the extractor consumes. Call sites gather inputs via
    :func:`_extractor_inputs`.
    """
    if extractors is not None:
        return dict(extractors)
    if not extractor_cfg:
        raise ValueError("Either 'extractor_cfg' or 'extractors' must be provided.")
    built = {}
    for group, cfg in extractor_cfg.items():
        cfg = dict(cfg)
        cls = resolve_callable(cfg.pop("class_name", "rsl_rl.modules.MlpExtractor"))
        built[group] = cls.from_obs(obs, group, **cfg)
    return built


def _extractor_inputs(extractor: nn.Module, obs: TensorDict, group: str) -> list:
    """Gather an extractor's input values (its declared groups; fallback: its own group)."""
    return [obs[g] for g in (getattr(extractor, "input_groups", None) or (group,))]


def _print_extractors(extractors: nn.ModuleDict) -> None:
    """Print each extractor's architecture + param count (not in the host's summary table)."""
    for name, ext in extractors.items():
        n_params = sum(p.numel() for p in ext.parameters() if p.requires_grad)
        print(f"  + extractor['{name}']: {n_params:,} trainable params (not in summary above)")
        print("      " + repr(ext).replace("\n", "\n      "))


class ExtractorMLPModel(MLPModel):
    """An :class:`MLPModel` where configured observation groups pass through extractors.

    Groups named in ``extractor_cfg`` are projected by their extractor (own normalizer);
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
        extractor_cfg: dict[str, dict] | None = None,
        extractors: dict | None = None,
    ) -> None:
        """Initialize the MLP model with per-group extractors (see ``extractor_cfg``)."""
        # Plain-dict attribute: safe before nn.Module.__init__, registered as ModuleDict after.
        self._extractors = _build_extractors(obs, extractor_cfg, extractors)
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
        self.extractors = nn.ModuleDict(self._extractors)
        _print_extractors(self.extractors)

    # --- overrides ---

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Split active groups into extracted and plain; report only plain dims to the parent."""
        active_obs_groups = obs_groups[obs_set]
        self.extracted_obs_groups = [g for g in active_obs_groups if g in self._extractors]
        if not self.extracted_obs_groups:
            raise ValueError(
                f"None of the active observation groups {active_obs_groups} has an extractor. "
                "Use MLPModel instead if this is intentional."
            )
        plain_groups = [g for g in active_obs_groups if g not in self._extractors]
        obs_dim = 0
        for g in plain_groups:
            if len(obs[g].shape) != 2:
                raise ValueError(f"The MLP model only supports 1D observations, got {obs[g].shape} for '{g}'.")
            obs_dim += obs[g].shape[-1]
        return plain_groups, obs_dim

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self.obs_dim + sum(self._extractors[g].latent_dim for g in self.extracted_obs_groups)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent from extractor latents and plain normalized groups."""
        latent_ext = self.encode(obs)
        if not self.obs_groups:
            return latent_ext
        return torch.cat([super().get_latent(obs), latent_ext], dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update plain-group and extractor normalization statistics."""
        if self.obs_groups:
            super().update_normalization(obs)
        for g in self.extracted_obs_groups:
            self.extractors[g].update_normalization(*_extractor_inputs(self.extractors[g], obs, g))

    # --- aux interface ---

    def encode(self, obs: TensorDict) -> torch.Tensor:
        """Concatenated extractor latents (the representation shared with auxiliary objectives)."""
        return torch.cat(
            [self.extractors[g](*_extractor_inputs(self.extractors[g], obs, g)) for g in self.extracted_obs_groups],
            dim=-1,
        )

    # --- export (not supported yet) ---

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        raise NotImplementedError("JIT export for ExtractorMLPModel is not implemented yet.")

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        raise NotImplementedError("ONNX export for ExtractorMLPModel is not implemented yet.")


class ExtractedAdapterStreamMixin:
    """Extractor-fed adapter stream for frozen-base + LoRA models.

    The frozen base stream is untouched. The adapter stream may mix extracted groups (through
    their extractors) and plain groups (through a trainable stream normalizer);
    ``adapter_obs_group`` accepts a single group name or a list. Exposes ``extractors`` and
    ``encode()`` — the contract PPOAux objectives bind to.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        adapter_obs_group: str | list[str] = "extractor_input",
        extractor_cfg: dict[str, dict] | None = None,
        extractors: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the host adapter model with an extractor-fed adapter stream."""
        self._extractors = _build_extractors(obs, extractor_cfg, extractors)
        stream_groups = [adapter_obs_group] if isinstance(adapter_obs_group, str) else list(adapter_obs_group)
        # The host wraps adapter_obs_group in a list; our _concat_dim override flattens it and
        # reports extractor latent dims so the LoRA adapter is strapped with the effective input dim.
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            adapter_obs_group=stream_groups,  # type: ignore[arg-type]
            **kwargs,
        )
        self.extractors = nn.ModuleDict(self._extractors)
        _print_extractors(self.extractors)
        self.adapter_obs_groups = stream_groups
        self.extracted_adapter_groups = [g for g in stream_groups if g in self._extractors]
        if not self.extracted_adapter_groups:
            raise ValueError(f"No adapter-stream group in {stream_groups} has an extractor.")
        self.plain_adapter_groups = [g for g in stream_groups if g not in self._extractors]
        # Replace the host's stream normalizer (sized for the full effective dim) with one
        # covering only the plain groups; extracted groups normalize inside their extractors.
        plain_dim = sum(obs[g].shape[-1] for g in self.plain_adapter_groups)
        if plain_dim > 0:
            self.adapter_normalizer = EmpiricalNormalization(plain_dim)
        else:
            self.adapter_normalizer = nn.Identity()

    # --- overrides ---

    def _concat_dim(self, obs: TensorDict, groups: list) -> int:  # type: ignore[override]
        """Effective stream dim: extractor latents for extracted groups, raw dims otherwise."""
        flat = [g for item in groups for g in ([item] if isinstance(item, str) else item)]
        return sum(self._extractors[g].latent_dim if g in self._extractors else obs[g].shape[-1] for g in flat)

    def _get_adapter_latent(self, obs: TensorDict) -> torch.Tensor:
        parts = []
        if self.plain_adapter_groups:
            plain = torch.cat([obs[g] for g in self.plain_adapter_groups], dim=-1)
            parts.append(self.adapter_normalizer(plain))
        parts += [
            self.extractors[g](*_extractor_inputs(self.extractors[g], obs, g)) for g in self.extracted_adapter_groups
        ]
        return torch.cat(parts, dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update the plain-stream normalizer and extractor normalizers; base stays frozen."""
        if self.plain_adapter_groups and isinstance(self.adapter_normalizer, EmpiricalNormalization):
            plain = torch.cat([obs[g] for g in self.plain_adapter_groups], dim=-1)
            self.adapter_normalizer.update(plain)
        for g in self.extracted_adapter_groups:
            self.extractors[g].update_normalization(*_extractor_inputs(self.extractors[g], obs, g))

    # --- aux interface ---

    def encode(self, obs: TensorDict) -> torch.Tensor:
        """Concatenated extractor latents (the representation shared with auxiliary objectives)."""
        return torch.cat(
            [self.extractors[g](*_extractor_inputs(self.extractors[g], obs, g)) for g in self.extracted_adapter_groups],
            dim=-1,
        )


class ExtractorAdapterModel(ExtractedAdapterStreamMixin, ModularNormMLPWithAdapterModel):
    """A :class:`ModularNormMLPWithAdapterModel` whose adapter stream is extractor-fed."""


class ExtractorSonicAdapterModel(ExtractedAdapterStreamMixin, SonicWithAdapterModel):
    """A :class:`SonicWithAdapterModel` whose adapter stream is extractor-fed."""
