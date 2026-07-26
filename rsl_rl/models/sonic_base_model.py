# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SONIC-style frozen base WBC: tokenizer encoder -> FSQ -> action decoder.

Robot-agnostic port of the GEAR-SONIC universal-token policy (Luo et al. 2025,
arXiv:2511.07820) restricted to a single encoder mode. All dimensions are
inferred from the observation TensorDict; joint ordering, action scaling, and
observation construction are the environment's responsibility (any robot-order
permutation is baked into the ported checkpoint, not this module).

Token flow::

    obs[tokenizer_obs_group] -> encoder MLP -> (num_tokens, token_dim) latent
                             -> FSQ quantize [+ latent_residual]
                             -> decoder MLP([tokens | proprio]) -> action mean
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import AmpMixin, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories

TOKEN_CACHE_KEY = "_sonic_tokens"
"""Observation key carrying pre-quantized tokens through the rollout storage (see
:meth:`SonicBaseModel.storage_obs`)."""

from ._onnx_export import (
    FoldedFSQ,
    Port,
    _OnnxExportBase,
    capture_obs_shapes,
    plain_mlp_copy,
)


def fsq_quantize(z: torch.Tensor, levels: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Finite Scalar Quantization (Mentzer et al. 2023), vector_quantize_pytorch semantics.

    Args:
        z: Latent tensor (..., d) with d == len(levels).
        levels: Per-dimension codebook sizes, shape (d,), integer dtype.
        eps: Bound-shrink factor (vector_quantize_pytorch default).

    Returns:
        Quantized tensor in [-1, 1], same shape as ``z`` (straight-through gradient).
    """
    half_l = (levels - 1) * (1 + eps) / 2
    offset = torch.where(levels % 2 == 0, 0.5, 0.0)
    shift = (offset / half_l).atanh()
    bounded = (z + shift).tanh() * half_l - offset
    quantized = bounded + (bounded.round() - bounded).detach()  # round with STE
    half_width = levels // 2
    return quantized / half_width


def _mlp(
    input_dim: int, hidden_dims: tuple[int, ...] | list[int], output_dim: int, activation: str
) -> nn.Sequential:
    """Plain [Linear, Act]*N + Linear stack matching the SONIC BaseModule MLP layout."""
    act_cls = getattr(nn, activation)
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dims[0]), act_cls()]
    for i in range(len(hidden_dims)):
        if i == len(hidden_dims) - 1:
            layers.append(nn.Linear(hidden_dims[i], output_dim))
        else:
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(act_cls())
    return nn.Sequential(*layers)


class SonicBaseModel(AmpMixin, nn.Module):
    """Frozen SONIC base policy (single encoder mode).

    Two observation streams:

    - **tokenizer stream** -- ``tokenizer_obs_group``: future-reference window,
      encoded to FSQ tokens.
    - **proprio stream** -- ``obs_groups[obs_set]``: proprioceptive history,
      concatenated with the flattened tokens and decoded to actions.

    ``latent_residual`` (post-quantization additive correction in token space)
    is the native adaptation hook; trainable heads strap onto it in phase 2.

    **Which half is frozen decides what a PPO update may skip.** An update runs
    ``num_learning_epochs x num_mini_batches`` forward/backward passes over one
    rollout; any sub-graph whose inputs AND weights are fixed for the whole update
    produces the same numbers every pass. Three branches, selected by
    ``freeze_encoder`` / ``freeze_decoder`` (both default to ``freeze_base``):

    ==================================  ====================================================
    branch                              invariant across the update
    ==================================  ====================================================
    ``freeze_encoder`` (fine-tune dec)  encoder + FSQ -> tokens; cached via
                                        :meth:`storage_obs`, ~15% of the actor's update cost
    ``freeze_decoder`` (fine-tune enc)  nothing in the token path (the encoder trains);
                                        the frozen decoder's backward is grad-input only,
                                        which autograd already does for ``requires_grad=False``
    neither                             nothing; the naive full fine-tune
    ==================================  ====================================================

    The token cache is bit-exact by construction: a frozen encoder over a fixed
    ``tokenizer`` row is a pure function, so the 20 recomputations it replaces were
    already producing identical values.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        tokenizer_obs_group: str = "tokenizer",
        num_tokens: int = 2,
        token_dim: int = 32,
        fsq_levels: int | list[int] = 32,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (2048, 1024, 512, 512),
        decoder_hidden_dims: tuple[int, ...] | list[int] = (2048, 2048, 1024, 1024, 512, 512),
        activation: str = "SiLU",
        distribution_cfg: dict | None = None,
        base_checkpoint: str | None = None,
        freeze_base: bool = True,
        freeze_encoder: bool | None = None,
        freeze_decoder: bool | None = None,
        cache_tokens: bool = True,
    ) -> None:
        """Build the encoder/FSQ/decoder stack; dims inferred from ``obs``.

        Args:
            freeze_base: Coarse default for both halves (kept for backward compatibility).
            freeze_encoder: Freeze the tokenizer encoder. ``None`` inherits ``freeze_base``.
            freeze_decoder: Freeze the action decoder. ``None`` inherits ``freeze_base``.
            cache_tokens: Route tokens through the rollout storage when the encoder is
                frozen, so the update never re-runs it. Off = always recompute.
        """
        super().__init__()

        # Resolve stream dims from the observation dict.
        self.obs_groups = obs_groups[obs_set]
        self.tokenizer_obs_group = tokenizer_obs_group
        # Per-group obs shapes, kept for ONNX export (ports sized without a live env).
        self.export_shapes = capture_obs_shapes(obs)
        proprio_dim = sum(obs[g].shape[-1] for g in self.obs_groups)
        tokenizer_dim = obs[tokenizer_obs_group].shape[-1]

        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.token_total_dim = num_tokens * token_dim
        if isinstance(fsq_levels, int):
            fsq_levels = [fsq_levels] * token_dim
        assert len(fsq_levels) == token_dim, "fsq_levels must have token_dim entries"
        self.register_buffer(
            "fsq_levels", torch.tensor(fsq_levels, dtype=torch.long), persistent=False
        )

        # Distribution (optional; deterministic when None).
        if distribution_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            decoder_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            decoder_output_dim = output_dim

        self.encoder = _mlp(tokenizer_dim, encoder_hidden_dims, self.token_total_dim, activation)
        self.decoder = _mlp(
            self.token_total_dim + proprio_dim, decoder_hidden_dims, decoder_output_dim, activation
        )

        if base_checkpoint is not None:
            payload = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
            state_dict = payload.get("model_state_dict", payload)
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            missing = [k for k in missing if not k.startswith("distribution")]
            if missing or unexpected:
                raise RuntimeError(
                    f"base_checkpoint mismatch: missing={missing}, unexpected={unexpected}"
                )
            # Seed the action std from the ported ckpt's meta — the GEAR ckpt
            # keeps action_std outside the module tree, so load_state_dict
            # can't restore it. Starting at the base's converged per-dim std
            # keeps adapter exploration inside the frozen WBC's competent band
            # (same behavior textop gets for free via distribution.std_param).
            action_std = payload.get("meta", {}).get("action_std")
            if action_std is not None and self.distribution is not None:
                std = torch.as_tensor(action_std, dtype=torch.float32)
                with torch.no_grad():
                    if hasattr(self.distribution, "std_param"):
                        self.distribution.std_param.copy_(std)
                    elif hasattr(self.distribution, "log_std_param"):
                        self.distribution.log_std_param.copy_(std.log())

        self.freeze_base = freeze_base
        self.freeze_encoder = freeze_base if freeze_encoder is None else freeze_encoder
        self.freeze_decoder = freeze_base if freeze_decoder is None else freeze_decoder
        for module, frozen in ((self.encoder, self.freeze_encoder), (self.decoder, self.freeze_decoder)):
            if frozen:
                for p in module.parameters():
                    p.requires_grad = False
        # Cache only helps when the encoder is a pure function for the whole update.
        self.cache_tokens = cache_tokens and self.freeze_encoder
        self._last_tokens: torch.Tensor | None = None

    # --- token pipeline ---

    def encode_tokens(
        self, obs: TensorDict, latent_residual: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Tokenizer obs -> quantized tokens, flattened to (B, token_total_dim).

        Returns the cached tokens when the rollout storage carries them (see
        :meth:`storage_obs`); otherwise runs the encoder and stashes the result for
        :meth:`cache_obs` to pick up. ``latent_residual`` is applied post-cache — it is
        a trainable correction, so it must never be baked into the stored tokens.
        """
        if self.cache_tokens and TOKEN_CACHE_KEY in obs.keys():
            tokens = obs[TOKEN_CACHE_KEY]
        else:
            z = self.encoder(obs[self.tokenizer_obs_group])
            z = z.view(*z.shape[:-1], self.num_tokens, self.token_dim)
            tokens = fsq_quantize(z, self.fsq_levels)
            tokens = tokens.reshape(*tokens.shape[:-2], self.token_total_dim)
            self._last_tokens = tokens
        if latent_residual is not None:
            tokens = tokens + latent_residual
        return tokens

    def _get_proprio(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[g] for g in self.obs_groups], dim=-1)

    # --- rollout-storage layout (see AmpMixin's sibling contract in the algorithm) ---

    def storage_obs(self, obs: TensorDict) -> TensorDict:
        """Observation layout the rollout storage should carry.

        Swaps the raw ``tokenizer`` window for the tokens it encodes to: 64 floats in
        place of 360, so the cache SHRINKS storage rather than growing it. Off (identity)
        unless the encoder is frozen — a trainable encoder needs the raw window every pass.
        """
        if not self.cache_tokens:
            return obs
        zeros = torch.zeros(*obs.batch_size, self.token_total_dim, device=obs.device)
        return self._with_tokens(obs, zeros)

    def cache_obs(self, obs: TensorDict) -> TensorDict:
        """Return ``obs`` in storage layout, carrying the tokens just computed by ``forward``."""
        if not self.cache_tokens:
            return obs
        if self._last_tokens is None:
            raise RuntimeError("cache_obs() called before a forward pass produced tokens.")
        return self._with_tokens(obs, self._last_tokens.detach())

    def _with_tokens(self, obs: TensorDict, tokens: torch.Tensor) -> TensorDict:
        """``obs`` with the tokenizer window replaced by ``tokens`` (a shallow re-select)."""
        cached = obs.select(*(k for k in obs.keys() if k != self.tokenizer_obs_group))
        cached[TOKEN_CACHE_KEY] = tokens
        return cached

    # --- Model API ---

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
        latent_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Deterministic action mean (or distribution output) for the given obs."""
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        # The encoder stays OUT of autocast: FSQ rounds to a grid of 32 levels over
        # [-1, 1], and bf16's ~0.06 of resolution at the pre-round scale would flip
        # boundary cases — a DISCRETE 1/16 jump in a token, far coarser than the
        # smooth rounding bf16 costs elsewhere. It is also the half that gets cached
        # out of the update entirely (see storage_obs), so autocasting it buys nothing.
        tokens = self.encode_tokens(obs, latent_residual)
        # Decoder body under autocast (no-op unless set_amp_dtype was called); head fp32.
        with self._amp_body():
            decoded = self._decode(tokens, obs)
        return self._head(self._head_input(decoded), stochastic_output)

    def _decode(self, tokens: torch.Tensor, obs: TensorDict) -> torch.Tensor:
        """Run the decoder over [tokens | proprio]. Adapter variants override this."""
        return self.decoder(torch.cat([tokens, self._get_proprio(obs)], dim=-1))

    def _head(self, mlp_output: torch.Tensor, stochastic_output: bool) -> torch.Tensor:
        """Distribution head (identity when the model is deterministic)."""
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset internal state (no-op; the base is stateless)."""

    def get_hidden_state(self) -> HiddenState:
        """Return the recurrent hidden state (``None``: stateless)."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach hidden state for TBPTT (no-op)."""

    def update_normalization(self, obs: TensorDict) -> None:
        """No normalizers: the SONIC release policy runs on raw observations."""
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        """Mean of the current output distribution."""
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        """Standard deviation of the current output distribution."""
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Entropy of the current output distribution."""
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Raw parameters of the current output distribution."""
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Log-probabilities of outputs under the current distribution."""
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """KL divergence between two distribution parameterizations."""
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        """JIT export (not implemented)."""
        raise NotImplementedError("JIT export for SonicBaseModel is not implemented yet.")

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a multi-input ONNX wrapper (one input per observation group)."""
        return _OnnxSonicBaseModel(self, verbose)


class _OnnxSonicBaseModel(_OnnxExportBase):
    """Exportable SONIC base: tokenizer -> encoder -> FSQ -> decoder -> action.

    Inputs are ``(tokenizer, *obs_groups, *adapter_stream)``; the adapter tail is empty here
    and supplied by subclasses through :meth:`_adapter_ports` / :meth:`_adapter_latent`, so
    the decoder path is written once for the whole SONIC family (the adapted decoder is a
    plain MLP after :func:`~rsl_rl.models._onnx_export.merge_adapters`).
    """

    def __init__(self, model: SonicBaseModel, verbose: bool = False) -> None:
        """Build the export copy: deep-copied encoder, folded FSQ, merged decoder."""
        super().__init__(verbose)
        self.encoder = copy.deepcopy(model.encoder)
        self.fsq = FoldedFSQ(model.fsq_levels)
        self.num_tokens = model.num_tokens
        self.token_dim = model.token_dim
        self.decoder = self._merge_decoder(model)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None
            else nn.Identity()
        )

        shapes = model.export_shapes
        base_slots = [
            Port(model.tokenizer_obs_group, shapes[model.tokenizer_obs_group],
                 (model.tokenizer_obs_group,)),
            *(Port(g, shapes[g], (g,)) for g in model.obs_groups),
        ]
        self.num_base_inputs = len(base_slots)
        self._set_slots(base_slots + self._adapter_ports(model))

    # --- hooks (overridden by the adapter / extractor variants) ---

    def _merge_decoder(self, model: SonicBaseModel) -> nn.Sequential:
        """Return the plain (un-adapted) decoder."""
        return plain_mlp_copy(model.decoder)

    def _adapter_ports(self, model: SonicBaseModel) -> list[Port]:
        """Extra inputs feeding the adapter latent (none for the plain base)."""
        return []

    def _adapter_latent(self, inputs: list[torch.Tensor]) -> torch.Tensor | None:
        """Return the adapter latent appended to the decoder input (``None`` for the base)."""
        return None

    def _extra_outputs(self) -> list[torch.Tensor]:
        """Return diagnostic outputs produced while building the adapter latent (none here)."""
        return []

    # --- forward ---

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Deterministic action for one observation, inputs in ``input_names`` order."""
        slots = self._gather(inputs)
        latent = self.encoder(slots[0])
        latent = latent.reshape(latent.shape[0], self.num_tokens, self.token_dim)
        tokens = self.fsq(latent).reshape(latent.shape[0], self.num_tokens * self.token_dim)
        parts = [tokens, *slots[1:self.num_base_inputs]]
        adapter_latent = self._adapter_latent(slots[self.num_base_inputs:])
        if adapter_latent is not None:
            parts.append(adapter_latent)
        actions = self.deterministic_output(self.decoder(torch.cat(parts, dim=-1)))
        extras = self._extra_outputs()
        return (actions, *extras) if extras else actions
