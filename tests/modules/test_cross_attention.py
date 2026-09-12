# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused contracts for the supported cross-attention extractor path."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.modules import CrossAttentionExtractor

BATCH = 4
NUM_PATCHES = 6
TOKEN_DIM = 8
QUERY_DIM = 5
ATTN_DIM = 4
LATENT_DIM = 7


def _obs() -> TensorDict:
    """Return one token term and two explicit query groups."""
    return TensorDict(
        {
            "kv_tokens": TensorDict(
                {"img_tokens": torch.randn(BATCH, NUM_PATCHES, TOKEN_DIM)},
                batch_size=[BATCH],
            ),
            "q_task": torch.randn(BATCH, QUERY_DIM),
            "q_proprio": torch.randn(BATCH, QUERY_DIM),
        },
        batch_size=[BATCH],
    )


def _extractor(obs: TensorDict) -> CrossAttentionExtractor:
    """Build the single-token-term, explicit-query configuration used by Vibe."""
    return CrossAttentionExtractor.from_obs(
        obs=obs,
        group="kv_tokens",
        token_terms=["img_tokens"],
        query_groups=["q_task", "q_proprio"],
        latent_dim=LATENT_DIM,
        attn_dim=ATTN_DIM,
        num_heads=1,
        num_learned_queries=0,
        obs_normalization=False,
        layer_norm=True,
    )


def test_supported_shape_and_attention_contract() -> None:
    """The extractor returns one latent and one attention row per explicit query."""
    obs = _obs()
    extractor = _extractor(obs)

    latent = extractor(obs["kv_tokens"], obs["q_task"], obs["q_proprio"])

    assert latent.shape == (BATCH, LATENT_DIM)
    assert extractor.last_attn is not None
    assert extractor.last_attn.shape == (BATCH, 2, NUM_PATCHES)
    torch.testing.assert_close(
        extractor.last_attn.sum(dim=-1),
        torch.ones(BATCH, 2),
    )


def test_policy_gradient_reaches_tokens_queries_and_extractor() -> None:
    """Policy loss must reach both input streams and every trainable projection."""
    obs = _obs()
    obs["kv_tokens"]["img_tokens"].requires_grad_()
    obs["q_task"].requires_grad_()
    obs["q_proprio"].requires_grad_()
    extractor = _extractor(obs)

    extractor(obs["kv_tokens"], obs["q_task"], obs["q_proprio"]).square().mean().backward()

    assert obs["kv_tokens"]["img_tokens"].grad is not None
    assert obs["q_task"].grad is not None
    assert obs["q_proprio"].grad is not None
    assert all(parameter.grad is not None for parameter in extractor.parameters())


def test_normalization_updates_each_supported_input_stream() -> None:
    """Token and query normalizers own independent running statistics."""
    obs = _obs()
    extractor = CrossAttentionExtractor.from_obs(
        obs=obs,
        group="kv_tokens",
        token_terms=["img_tokens"],
        query_groups=["q_task", "q_proprio"],
        latent_dim=LATENT_DIM,
        attn_dim=ATTN_DIM,
        obs_normalization=True,
    )

    extractor.update_normalization(obs["kv_tokens"], obs["q_task"], obs["q_proprio"])

    assert extractor.token_normalizers["img_tokens"].count > 0
    assert extractor.query_normalizers["q_task"].count > 0
    assert extractor.query_normalizers["q_proprio"].count > 0
