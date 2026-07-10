# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapter model: per-layer weight-residual for base-policy adaptation (LoRA; Hu et al. 2021)."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, HiddenState, MLPWithAdapter, ModularNormMLPWithAdapter

from .mlp_model import MLPModel


class AdapterStreamMixin:
    """Adapter-stream plumbing shared by all frozen-base + LoRA models.

    Provides the trainable adapter stream (own normalizer), diagnostics, and the
    parameter summary. The host model calls :meth:`_init_adapter_stream` after its
    base is built and exposes the adapted network via :attr:`_adapted_mlp`.
    """

    @property
    def _adapted_mlp(self) -> MLPWithAdapter:
        """The :class:`MLPWithAdapter` carrying the LoRA adapters. Override per host."""
        return self.mlp  # type: ignore[attr-defined]

    def _init_adapter_stream(
        self, obs: TensorDict, adapter_obs_group: str | list[str], obs_normalization: bool
    ) -> int:
        """Set up the adapter stream (groups + normalizer); returns its effective dim."""
        self.adapter_obs_groups = (
            [adapter_obs_group] if isinstance(adapter_obs_group, str) else list(adapter_obs_group)
        )
        adapter_dim = self._concat_dim(obs, self.adapter_obs_groups)
        self.adapter_normalizer = EmpiricalNormalization(adapter_dim) if obs_normalization else torch.nn.Identity()
        return adapter_dim

    def _concat_dim(self, obs: TensorDict, groups: list[str]) -> int:
        """Effective adapter-stream dimension."""
        return sum(obs[group].shape[-1] for group in groups)

    def _get_adapter_latent(self, obs: TensorDict) -> torch.Tensor:
        latent = torch.cat([obs[group] for group in self.adapter_obs_groups], dim=-1)
        return self.adapter_normalizer(latent)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update only the adapter normalizer; the frozen base is never updated."""
        if isinstance(self.adapter_normalizer, EmpiricalNormalization):
            latent = torch.cat([obs[group] for group in self.adapter_obs_groups], dim=-1)
            self.adapter_normalizer.update(latent)

    def adapter_diagnostics(self) -> dict[str, float]:
        """Per-layer adapter weight norms under ``AdapterStats/``."""
        diag: dict[str, float] = {}
        for i, adapter in enumerate(self._adapted_mlp.adapters):
            if adapter is not None:
                diag[f"AdapterStats/layer_{i}_delta_w_norm"] = adapter.delta_weight().norm().item()
        return diag

    def _print_param_summary(self, freeze_base: bool) -> None:
        """Pretty-print per-component trainable / total param counts."""
        def _count(module: torch.nn.Module) -> tuple[int, int]:
            total = sum(p.numel() for p in module.parameters())
            train = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return train, total

        sep = "─" * 62
        print(f"\n{sep}")
        print(f"  {type(self).__name__}  (base {'frozen' if freeze_base else 'unfrozen'})")
        print(sep)
        print(f"  {'component':<28} {'trainable':>12} {'total':>12}")
        print(f"  {'─' * 28} {'─' * 12} {'─' * 12}")

        grand_train, grand_total = 0, 0

        # base layers
        for i, linear in enumerate(self._adapted_mlp.base_linears):
            tr, tot = _count(linear)
            tag = "frozen" if not tr else ""
            print(f"  base linear[{i}] {tag:<12} {tr:>12,} {tot:>12,}")
            grand_train += tr
            grand_total += tot

        # adapters
        for i, adapter in enumerate(self._adapted_mlp.adapters):
            if adapter is not None:
                tr, tot = _count(adapter)
                print(f"  adapter[{i}]{'':16} {tr:>12,} {tot:>12,}")
                grand_train += tr
                grand_total += tot
            else:
                print(f"  adapter[{i}]{'':16} {'—':>12} {'—':>12}")

        # normalizers
        for name, mod in [
            ("base normalizer", getattr(self, "obs_normalizer", None)),
            ("adapter normalizer", self.adapter_normalizer),
        ]:
            if mod is None:
                continue
            tr, tot = _count(mod)
            if tot:
                print(f"  {name:<28} {tr:>12,} {tot:>12,}")
                grand_train += tr
                grand_total += tot

        # distribution (action std)
        if getattr(self, "distribution", None) is not None:
            tr, tot = _count(self.distribution)
            if tot:
                print(f"  {'distribution (std)':<28} {tr:>12,} {tot:>12,}")
                grand_train += tr
                grand_total += tot

        print(f"  {'─' * 28} {'─' * 12} {'─' * 12}")
        print(f"  {'TOTAL':<28} {grand_train:>12,} {grand_total:>12,}")
        pct = 100 * grand_train / grand_total if grand_total else 0
        print(f"  trainable: {pct:.1f}%")
        print(f"{sep}\n")


class MLPWithAdapterModel(AdapterStreamMixin, MLPModel):
    """An :class:`MLPModel` with a pretrained MLP base and trainable adapters.

    Two observation streams feed the network:

    - **base stream** -- ``obs_groups[obs_set]``, normalized by the (frozen) base normalizer.
    - **adapter stream** -- ``adapter_obs_group`` (e.g. object state), with its own trainable normalizer.

    Subclass :class:`ModularNormMLPWithAdapterModel` when the frozen base is a ``ModularNormMLP``.
    """

    _adapter_cls: type[MLPWithAdapter] = MLPWithAdapter
    """Adapter module class. Mirrors the module-layer split."""

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
        adapter_obs_group: str = "augmentation",
        rank: int | list[int | None] = -1,
        alpha: float = 1.0,
        base_checkpoint: str | None = None,
        freeze_base: bool = True,
    ) -> None:
        """Build the base MLP, load the pretrained weights, and strap the adapters."""
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

        # Load pretrained base BEFORE strapping (self.mlp is still a plain MLP whose keys match).
        if base_checkpoint is not None:
            if not obs_normalization:
                raise ValueError("base_checkpoint requires obs_normalization=True to load the base normalizer.")
            payload = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
            is_ported = isinstance(payload, dict) and "model_state_dict" in payload
            state_dict = payload["model_state_dict"] if is_ported else payload
            _, unexpected = self.load_state_dict(state_dict, strict=False)
            if unexpected:
                raise RuntimeError(f"base_checkpoint has unexpected keys: {unexpected}")

        # Adapter stream: separate obs group + its own (trainable) normalizer.
        adapter_dim = self._init_adapter_stream(obs, adapter_obs_group, obs_normalization)

        # Strap adapter on the (now loaded) base.
        self.mlp = self._adapter_cls.from_base_mlp(
            self.mlp, adapter_input_dim=adapter_dim, rank=rank, alpha=alpha, freeze_base=freeze_base
        )

        # Remove modular-norm hooks if present: the adapted base must never be dualized/projected.
        for hook in ("dualize_gradients", "project_weights"):
            if hasattr(self, hook):
                delattr(self, hook)

        self._print_param_summary(freeze_base)

    # --- overrides ---

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Forward the base stream through the adapted MLP, conditioned on the adapter stream."""
        base_latent = self.get_latent(obs, masks, hidden_state)
        adapter_latent = self._get_adapter_latent(obs)
        mlp_output = self.mlp(base_latent, adapter_latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def as_jit(self) -> torch.nn.Module:
        """JIT export (not implemented)."""
        raise NotImplementedError("JIT export for MLPWithAdapterModel is not implemented yet.")

    def as_onnx(self, verbose: bool) -> torch.nn.Module:
        """ONNX export (not implemented)."""
        raise NotImplementedError("ONNX export for MLPWithAdapterModel is not implemented yet.")


class ModularNormMLPWithAdapterModel(MLPWithAdapterModel):
    """An :class:`MLPWithAdapterModel` whose frozen base is a :class:`~rsl_rl.modules.ModularNormMLP`."""

    _adapter_cls = ModularNormMLPWithAdapter

    def _make_mlp(
        self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...] | list[int], activation: str
    ) -> ModularNormMLP:  # noqa: F821
        """Build a ModularNormMLP base (matches the pretrained checkpoint's layer type)."""
        from rsl_rl.modules import ModularNormMLP
        return ModularNormMLP(input_dim, output_dim, hidden_dims, activation)
