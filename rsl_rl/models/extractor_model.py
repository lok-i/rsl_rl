# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Models whose feature observation groups pass through trainable extractors.

Aux-agnostic by design: these models only expose the extractor(s); which auxiliary objective
trains them (if any) is decided by the algorithm (see ``rsl_rl.algorithms.PPOAux``).
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.modules import CrossAttentionExtractor, EmpiricalNormalization
from rsl_rl.utils import resolve_callable

from ._onnx_export import Port
from .sonic_adapter_model import SonicWithAdapterModel, _OnnxSonicAdapterModel


def _build_extractors(obs: TensorDict, extractor_cfg: dict[str, dict] | None, extractors: dict | None) -> dict:
    """Build (or validate) one extractor per configured observation group.

    Each per-group cfg names its extractor class via ``class_name``. Construction goes
    through the class's ``from_obs``, which sets ``input_groups`` — the group names the
    extractor consumes. Call sites gather inputs via :func:`_extractor_inputs`.
    """
    if extractors is not None:
        return dict(extractors)
    if not extractor_cfg:
        raise ValueError("Either 'extractor_cfg' or 'extractors' must be provided.")
    built = {}
    for group, cfg in extractor_cfg.items():
        cfg = dict(cfg)
        if "class_name" not in cfg:
            raise ValueError(f"extractor_cfg['{group}'] requires 'class_name'.")
        cls = resolve_callable(cfg.pop("class_name"))
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


class ExtractorSonicAdapterModel(ExtractedAdapterStreamMixin, SonicWithAdapterModel):
    """A :class:`SonicWithAdapterModel` whose adapter stream is extractor-fed."""

    def as_onnx(self, verbose: bool = False, with_attn: bool = True) -> nn.Module:
        """Return a multi-input ONNX wrapper (base ports + extractor token/query ports).

        ``with_attn`` adds the attention map as a second output — already computed, so it is
        free, and it is what a deploy-side "where is it looking" overlay reads.
        """
        return _OnnxExtractorSonicAdapterModel(self, verbose, with_attn=with_attn)


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


class _OnnxCrossAttention(nn.Module):
    """Export copy of :class:`~rsl_rl.modules.CrossAttentionExtractor`.

    Same math, but token/query groups arrive as positional tensors (ONNX has no dict inputs)
    and the module-level debug tap is dropped — the attention map leaves as a return value
    instead of a side effect.
    """

    def __init__(self, extractor: CrossAttentionExtractor) -> None:
        """Deep-copy the trained weights and normalizers into a trace-friendly layout."""
        super().__init__()
        self.token_normalizers = nn.ModuleList(
            copy.deepcopy(extractor.token_normalizers[t]) for t in extractor.token_terms
        )
        self.query_normalizers = nn.ModuleList(
            copy.deepcopy(extractor.query_normalizers[g]) for g in extractor.query_groups
        )
        self.w_q = nn.ModuleList(copy.deepcopy(extractor.w_q[g]) for g in extractor.query_groups)
        self.w_k = copy.deepcopy(extractor.w_k)
        self.w_v = copy.deepcopy(extractor.w_v)
        self.learned_queries = (
            None if extractor.learned_queries is None else nn.Parameter(extractor.learned_queries.detach().clone())
        )
        self.proj = copy.deepcopy(extractor.proj)
        self.out_norm = copy.deepcopy(extractor.out_norm)
        self.attn_scale = extractor.attn_dim**0.5
        self.num_token_inputs = len(extractor.token_terms)

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool the token inputs under every query row; returns ``(latent, attention)``."""
        token_inputs = inputs[: self.num_token_inputs]
        query_inputs = inputs[self.num_token_inputs :]
        tokens = torch.cat([norm(x) for norm, x in zip(self.token_normalizers, token_inputs)], dim=1)
        keys, values = self.w_k(tokens), self.w_v(tokens)
        rows = [w_q(norm(qv)).unsqueeze(1) for w_q, norm, qv in zip(self.w_q, self.query_normalizers, query_inputs)]
        q = torch.cat(rows, dim=1)
        if self.learned_queries is not None:
            q = torch.cat([q, self.learned_queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)], dim=1)
        attn = torch.softmax(q @ keys.transpose(-2, -1) / self.attn_scale, dim=-1)
        return self.out_norm(self.proj((attn @ values).flatten(1))), attn


class _OnnxFlatExtractor(nn.Module):
    """Export copy of an extractor that consumes one tensor observation group."""

    def __init__(self, extractor: nn.Module) -> None:
        """Deep-copy the extractor; the tuple return keeps the caller uniform."""
        super().__init__()
        self.extractor = copy.deepcopy(extractor)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Project the feature group to its latent without an attention output."""
        return self.extractor(x), None


def _wrap_extractor(extractor: nn.Module, group: str, shapes: dict) -> tuple[nn.Module, list[Port]]:
    """Build the export copy of one extractor plus the ONNX ports feeding it.

    Ports follow ``extractor.input_groups``: the dict token group expands to one input per
    term (``<group>__<term>``, in the extractor's own ``token_terms`` order — never sorted-key
    luck), each query group is one flat input.
    """
    if isinstance(extractor, CrossAttentionExtractor):
        ports = [Port(f"{group}__{term}", shapes[group][term], (group,), term=term) for term in extractor.token_terms]
        ports += [Port(g, shapes[g], (g,)) for g in extractor.query_groups]
        return _OnnxCrossAttention(extractor), ports
    if getattr(extractor, "term_keys", None) is None and not isinstance(shapes[group], dict):
        return _OnnxFlatExtractor(extractor), [Port(group, shapes[group], (group,))]
    raise NotImplementedError(
        f"ONNX export for extractor {type(extractor).__name__} on a dict group is not implemented."
    )


class _OnnxExtractorSonicAdapterModel(_OnnxSonicAdapterModel):
    """Exportable SONIC + LoRA whose adapter latent is ``[normalized plain groups | z]``.

    The extractor lives inside the graph (its normalizers included); only the frozen vision
    backbone stays outside — deployment streams its tokens in on the ``kv_tokens__*`` ports.
    """

    def __init__(self, model: ExtractorSonicAdapterModel, verbose: bool = False, with_attn: bool = True) -> None:
        """Build the export copy; ``with_attn`` exposes the attention map as an extra output."""
        self.with_attn = with_attn
        super().__init__(model, verbose)

    def _adapter_ports(self, model: ExtractorSonicAdapterModel) -> list[Port]:  # type: ignore[override]
        self.adapter_normalizer = copy.deepcopy(model.adapter_normalizer)
        self.num_plain_inputs = len(model.plain_adapter_groups)
        ports = [Port(g, model.export_shapes[g], (g,)) for g in model.plain_adapter_groups]
        self.extractors = nn.ModuleList()
        self.extractor_input_counts: list[int] = []
        for group in model.extracted_adapter_groups:
            wrapper, extractor_ports = _wrap_extractor(model.extractors[group], group, model.export_shapes)
            self.extractors.append(wrapper)
            self.extractor_input_counts.append(len(extractor_ports))
            ports += extractor_ports
        self.num_attn_outputs = sum(isinstance(e, _OnnxCrossAttention) for e in self.extractors)
        self.with_attn = self.with_attn and self.num_attn_outputs > 0
        return ports

    def _adapter_latent(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        # Order matches ExtractedAdapterStreamMixin._get_adapter_latent: plain groups first.
        parts, attentions = [], []
        if self.num_plain_inputs:
            plain = torch.cat(inputs[: self.num_plain_inputs], dim=-1)
            parts.append(self.adapter_normalizer(plain))
        offset = self.num_plain_inputs
        for extractor, count in zip(self.extractors, self.extractor_input_counts):
            latent, attn = extractor(*inputs[offset : offset + count])
            offset += count
            parts.append(latent)
            if attn is not None:
                attentions.append(attn)
        self._attentions = attentions
        return torch.cat(parts, dim=-1)

    def _extra_outputs(self) -> list[torch.Tensor]:
        return self._attentions if self.with_attn else []

    @property
    def output_names(self) -> list[str]:
        """ONNX output names (one ``attn_*`` per attention-producing extractor)."""
        names = ["actions"]
        if self.with_attn:
            names += [f"attn_{i}" if i else "attn" for i in range(self.num_attn_outputs)]
        return names
