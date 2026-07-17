# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cross-attention extractor: command-queried pooling over dense feature tokens."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.modules.normalization import EmpiricalNormalization


class CrossAttentionExtractor(nn.Module):
    """Single-head cross-attention pooling of a token set into a compact latent.

    ``Q = [g W_q ; m learned queries]`` over token keys/values (``K = f W_k``, ``V = f W_v``,
    ``Z = softmax(Q K^T / sqrt(d)) V``, ``z = LayerNorm(proj(flatten(Z)))``). Both sides project
    into the shared attention dim ``d``; the token count P is free at runtime (resolution-
    agnostic). Same trainable-representation contract as :class:`MlpExtractor` (``latent_dim``,
    ``update_normalization``, PPO + aux gradients).

    Consumes ONE dict observation group (``concatenate_terms=False``); term roles are
    EXPLICIT config, never inferred: ``token_terms`` -> (B, P_i, C) token sets, concatenated
    along P into K/V; ``query_terms`` -> (B, q_i) vectors, concatenated into the ``W_q``
    query source. The two lists must exactly partition the group's terms. An empty
    ``query_terms`` degenerates to pure learned-query pooling.
    """

    def __init__(
        self,
        token_dim: int,
        latent_dim: int,
        token_terms: tuple[str, ...] | list[str],
        query_terms: tuple[str, ...] | list[str] = (),
        query_dim: int = 0,
        attn_dim: int = 64,
        num_learned_queries: int = 3,
        obs_normalization: bool = True,
        layer_norm: bool = True,
    ) -> None:
        """Initialize from the token channel dim and the total query source dim."""
        super().__init__()
        assert query_dim > 0 or num_learned_queries > 0, "need a command query and/or learned queries"
        assert (query_dim > 0) == bool(query_terms), "query_dim and query_terms must agree"
        self.token_dim = token_dim
        self.latent_dim = latent_dim
        self.query_dim = query_dim
        self.attn_dim = attn_dim
        self.token_terms = list(token_terms)
        self.query_terms = list(query_terms)
        # Per-term token normalizers (shared over P within a term — tokens are a set;
        # separate per term — different sources have different statistics).
        self.token_normalizers = nn.ModuleDict(
            {t: EmpiricalNormalization(token_dim) if obs_normalization else nn.Identity() for t in self.token_terms}
        )
        self.query_normalizer = (
            EmpiricalNormalization(query_dim) if (obs_normalization and query_dim > 0) else nn.Identity()
        )
        self.w_q = nn.Linear(query_dim, attn_dim, bias=False) if query_dim > 0 else None
        self.w_k = nn.Linear(token_dim, attn_dim, bias=False)
        self.w_v = nn.Linear(token_dim, attn_dim, bias=False)
        self.learned_queries = (
            nn.Parameter(torch.randn(num_learned_queries, attn_dim) / attn_dim**0.5)
            if num_learned_queries > 0
            else None
        )
        num_queries = num_learned_queries + (1 if query_dim > 0 else 0)
        self.proj = nn.Linear(num_queries * attn_dim, latent_dim)
        self.out_norm = nn.LayerNorm(latent_dim) if layer_norm else nn.Identity()
        self.input_groups: tuple[str, ...] = ()

    @classmethod
    def from_obs(
        cls,
        obs: TensorDict,
        group: str,
        token_terms: tuple[str, ...] | list[str] = (),
        query_terms: tuple[str, ...] | list[str] = (),
        **cfg: Any,
    ) -> CrossAttentionExtractor:
        """Build from an observation template: ``group`` is a dict group; roles are explicit."""
        x = obs[group]
        if not hasattr(x, "keys"):
            raise ValueError(f"CrossAttentionExtractor expects a dict group (concatenate_terms=False) for '{group}'.")
        keys = set(x.keys())
        declared = set(token_terms) | set(query_terms)
        if not token_terms or declared != keys or len(declared) != len(token_terms) + len(query_terms):
            raise ValueError(
                f"token_terms {list(token_terms)} + query_terms {list(query_terms)} must exactly "
                f"partition group '{group}' terms {sorted(keys)}."
            )
        token_dims = {t: x[t].shape[-1] for t in token_terms}
        if len(set(token_dims.values())) != 1 or any(len(x[t].shape) != 3 for t in token_terms):
            raise ValueError(f"token terms must all be (B, P, C) with one shared C, got {token_dims}.")
        if any(len(x[t].shape) != 2 for t in query_terms):
            raise ValueError(f"query terms must be (B, q) vectors, got { {t: x[t].shape for t in query_terms} }.")
        query_dim = sum(x[t].shape[-1] for t in query_terms)
        enc = cls(
            token_dim=next(iter(token_dims.values())),
            token_terms=token_terms,
            query_terms=query_terms,
            query_dim=query_dim,
            **cfg,
        )
        enc.input_groups = (group,)
        return enc

    def forward(self, x: TensorDict) -> torch.Tensor:
        """Pool the dict group's token terms with its query terms into the (B, latent_dim) latent."""
        tokens = torch.cat([self.token_normalizers[t](x[t]) for t in self.token_terms], dim=1)  # (B, P, C)
        keys, values = self.w_k(tokens), self.w_v(tokens)  # (B, P, d)
        queries = []
        if self.w_q is not None:
            query_src = torch.cat([x[t] for t in self.query_terms], dim=-1)
            queries.append(self.w_q(self.query_normalizer(query_src)).unsqueeze(1))
        if self.learned_queries is not None:
            queries.append(self.learned_queries.unsqueeze(0).expand(tokens.shape[0], -1, -1))
        q = torch.cat(queries, dim=1)  # (B, m+1, d)
        attn = torch.softmax(q @ keys.transpose(-2, -1) / self.attn_dim**0.5, dim=-1)
        return self.out_norm(self.proj((attn @ values).flatten(1)))

    def update_normalization(self, x: TensorDict) -> None:
        """Update per-term token and query normalization statistics."""
        for t in self.token_terms:
            norm = self.token_normalizers[t]
            if isinstance(norm, EmpiricalNormalization):
                norm.update(x[t].reshape(-1, x[t].shape[-1]))
        if self.query_terms and isinstance(self.query_normalizer, EmpiricalNormalization):
            self.query_normalizer.update(torch.cat([x[t] for t in self.query_terms], dim=-1))
