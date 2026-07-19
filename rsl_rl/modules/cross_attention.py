# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cross-attention extractor: multi-source-queried pooling over dense feature tokens."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.modules.normalization import EmpiricalNormalization

# Debug tap: the most-recent extractor to run a forward pass. Viewers read
# ``LATEST_ATTENTION.last_attn`` to overlay attention on the camera feed; never
# consulted during training (a detached pointer write, no graph, no copy).
LATEST_ATTENTION: CrossAttentionExtractor | None = None


class CrossAttentionExtractor(nn.Module):
    """Single-head cross-attention pooling of vision tokens into a compact latent.

    Each **query group** is one (B, d_g) vector -> one attention ROW via its own ``W_q``
    (``Q = [W_q^g x_g for g in query_groups ; learned queries]``); tokens give K/V
    (``K = f W_k``, ``V = f W_v``); ``z = LayerNorm(proj(flatten(softmax(Q K^T/sqrt(d)) V)))``.
    Query VALUES are not carried into z (pure attention pooling) — the sys0 hierarchy routes
    command values to control directly (adapter stream), the query only steers *where to look*
    (docs/extractor_ahead.md). Token count P is free at runtime (resolution-agnostic).

    Consumes ONE dict token group (``concatenate_terms=False``, roles EXPLICIT via
    ``token_terms``) + N vector query groups (``concatenate_terms=True``, one row each).
    ``input_groups = (token_group, *query_groups)`` — the host gathers them positionally.

    ``num_heads`` is reserved for the multi-head extension (docs/extractor_ahead.md step 1);
    only ``num_heads=1`` is implemented today.
    """

    def __init__(
        self,
        token_dim: int,
        latent_dim: int,
        token_terms: tuple[str, ...] | list[str],
        query_groups: tuple[str, ...] | list[str],
        query_dims: dict[str, int],
        attn_dim: int = 64,
        num_heads: int = 1,
        num_learned_queries: int = 0,
        obs_normalization: bool = True,
        layer_norm: bool = True,
    ) -> None:
        """Initialize from the token channel dim and the per-query-group source dims."""
        super().__init__()
        assert num_heads == 1, "multi-head is step-1 (docs/extractor_ahead.md); single-head only"
        assert query_groups or num_learned_queries > 0, "need a query group and/or learned queries"
        self.token_dim = token_dim
        self.latent_dim = latent_dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.token_terms = list(token_terms)
        self.query_groups = list(query_groups)
        self.num_learned_queries = num_learned_queries
        # Per-term token normalizers (shared over P within a term — tokens are a set;
        # separate per term — different sources have different statistics).
        self.token_normalizers = nn.ModuleDict(
            {t: EmpiricalNormalization(token_dim) if obs_normalization else nn.Identity() for t in self.token_terms}
        )
        # Per-group query normalizer + projection: one attention row per query group.
        self.query_normalizers = nn.ModuleDict(
            {g: EmpiricalNormalization(query_dims[g]) if obs_normalization else nn.Identity()
             for g in self.query_groups}
        )
        self.w_q = nn.ModuleDict({g: nn.Linear(query_dims[g], attn_dim, bias=False) for g in self.query_groups})
        self.w_k = nn.Linear(token_dim, attn_dim, bias=False)
        self.w_v = nn.Linear(token_dim, attn_dim, bias=False)
        self.learned_queries = (
            nn.Parameter(torch.randn(num_learned_queries, attn_dim) / attn_dim**0.5)
            if num_learned_queries > 0
            else None
        )
        num_queries = len(self.query_groups) + num_learned_queries
        self.proj = nn.Linear(num_queries * attn_dim, latent_dim)
        self.out_norm = nn.LayerNorm(latent_dim) if layer_norm else nn.Identity()
        self.input_groups: tuple[str, ...] = ()
        # Debug tap: last forward's attention (B, Q, P); row order = query_labels.
        self.query_labels = [g.removeprefix("q_") for g in self.query_groups] + [
            f"learned{i}" for i in range(num_learned_queries)
        ]
        self.last_attn: torch.Tensor | None = None

    @classmethod
    def from_obs(
        cls,
        obs: TensorDict,
        group: str,
        token_terms: tuple[str, ...] | list[str],
        query_groups: tuple[str, ...] | list[str] = (),
        **cfg: Any,
    ) -> CrossAttentionExtractor:
        """Build from an obs template: dict token group + one (B, d) vector group per query row."""
        token_td = obs[group]
        if not hasattr(token_td, "keys"):
            raise ValueError(f"token group '{group}' must be a dict (concatenate_terms=False).")
        token_dims = {t: token_td[t].shape[-1] for t in token_terms}
        bad = not token_terms or len(set(token_dims.values())) != 1 or any(token_td[t].ndim != 3 for t in token_terms)
        if bad:
            raise ValueError(f"token_terms {list(token_terms)} must all be (B, P, C), one shared C; got {token_dims}.")
        query_dims = {}
        for g in query_groups:
            xg = obs[g]
            if hasattr(xg, "keys") or len(xg.shape) != 2:
                raise ValueError(f"query group '{g}' must be a flat (B, d) vector group (concatenate_terms=True).")
            query_dims[g] = xg.shape[-1]
        enc = cls(
            token_dim=next(iter(token_dims.values())),
            token_terms=token_terms,
            query_groups=query_groups,
            query_dims=query_dims,
            **cfg,
        )
        enc.input_groups = (group, *query_groups)
        return enc

    def forward(self, token_td: TensorDict, *query_vecs: torch.Tensor) -> torch.Tensor:
        """Pool tokens (queried by each query group) into the (B, latent_dim) latent.

        ``query_vecs`` are positional, in ``query_groups`` order (host gathers them from
        ``input_groups``).
        """
        tokens = torch.cat([self.token_normalizers[t](token_td[t]) for t in self.token_terms], dim=1)  # (B, P, C)
        keys, values = self.w_k(tokens), self.w_v(tokens)  # (B, P, d)
        rows = [
            self.w_q[g](self.query_normalizers[g](qv)).unsqueeze(1)  # (B, 1, d)
            for g, qv in zip(self.query_groups, query_vecs, strict=True)
        ]
        q = torch.cat(rows, dim=1) if rows else keys.new_empty((tokens.shape[0], 0, self.attn_dim))
        if self.learned_queries is not None:
            q = torch.cat([q, self.learned_queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)], dim=1)
        attn = torch.softmax(q @ keys.transpose(-2, -1) / self.attn_dim**0.5, dim=-1)  # (B, Q, P)
        global LATEST_ATTENTION
        self.last_attn = attn.detach()
        LATEST_ATTENTION = self
        return self.out_norm(self.proj((attn @ values).flatten(1)))

    def update_normalization(self, token_td: TensorDict, *query_vecs: torch.Tensor) -> None:
        """Update per-term token and per-group query normalization statistics."""
        for t in self.token_terms:
            norm = self.token_normalizers[t]
            if isinstance(norm, EmpiricalNormalization):
                norm.update(token_td[t].reshape(-1, token_td[t].shape[-1]))
        for g, qv in zip(self.query_groups, query_vecs, strict=True):
            norm = self.query_normalizers[g]
            if isinstance(norm, EmpiricalNormalization):
                norm.update(qv)
