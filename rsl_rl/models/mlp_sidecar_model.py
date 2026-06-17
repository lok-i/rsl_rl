# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sidecar model: per-network action-residual for base-policy adaptation (RPL; Tom Silver et al. 2018)."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, HiddenState, MLPWithSidecar, ModularNormMLPWithSidecar

from .mlp_model import MLPModel


class MLPWithSidecarModel(MLPModel):
    """An :class:`MLPModel` with a pretrained MLP base and a trainable sidecar network.

    Two observation streams feed the network:

    - **base stream** — ``obs_groups[obs_set]``, normalized by the (frozen) base normalizer.
    - **sidecar stream** — ``sidecar_obs_group`` (e.g. object state), with its own trainable normalizer.

    Subclass :class:`ModularNormMLPWithSidecarModel` when the frozen base is a ``ModularNormMLP``.
    """

    _sidecar_cls: type[MLPWithSidecar] = MLPWithSidecar
    """Sidecar module class. Mirrors the module-layer split."""

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
        sidecar_obs_group: str = "augmentation",
        sidecar_hidden_dims: tuple[int, ...] | list[int] = (256,),
        sidecar_activation: str = "elu",
        wbc_checkpoint: str | None = None,
        freeze_base: bool = True,
    ) -> None:
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
        if wbc_checkpoint is not None:
            if not obs_normalization:
                raise ValueError("wbc_checkpoint requires obs_normalization=True to load the base normalizer.")
            payload = torch.load(wbc_checkpoint, map_location="cpu", weights_only=False)
            is_ported = isinstance(payload, dict) and "model_state_dict" in payload
            state_dict = payload["model_state_dict"] if is_ported else payload
            _, unexpected = self.load_state_dict(state_dict, strict=False)
            if unexpected:
                raise RuntimeError(f"wbc_checkpoint has unexpected keys: {unexpected}")

        # Sidecar stream: separate obs group + its own (trainable) normalizer.
        self.sidecar_obs_groups = [sidecar_obs_group]
        sidecar_dim = self._concat_dim(obs, self.sidecar_obs_groups)
        self.sidecar_normalizer = (
            EmpiricalNormalization(sidecar_dim) if obs_normalization else torch.nn.Identity()
        )

        # Strap sidecar on the (now loaded) base.
        self.mlp = self._sidecar_cls.from_base_mlp(
            self.mlp,
            sidecar_input_dim=sidecar_dim,
            sidecar_hidden_dims=sidecar_hidden_dims,
            sidecar_activation=sidecar_activation,
            freeze_base=freeze_base,
        )

        # Remove modular-norm hooks if present: the off-manifold base (frozen or fine-tuned with free-Adam)
        # must never be dualized/projected, and the sidecar needs neither hook.
        for hook in ("dualize_gradients", "project_weights"):
            if hasattr(self, hook):
                delattr(self, hook)

        self._print_param_summary(freeze_base)

    # --- helpers ---

    @staticmethod
    def _concat_dim(obs: TensorDict, groups: list[str]) -> int:
        return sum(obs[group].shape[-1] for group in groups)

    def _get_sidecar_latent(self, obs: TensorDict) -> torch.Tensor:
        latent = torch.cat([obs[group] for group in self.sidecar_obs_groups], dim=-1)
        return self.sidecar_normalizer(latent)

    def _print_param_summary(self, freeze_base: bool) -> None:
        """Pretty-print per-component trainable / total param counts."""
        def _count(module):
            total = sum(p.numel() for p in module.parameters())
            train = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return train, total

        sep = "─" * 62
        print(f"\n{sep}")
        print(f"  MLPWithSidecarModel  (base {'frozen' if freeze_base else 'unfrozen'})")
        print(sep)
        print(f"  {'component':<28} {'trainable':>12} {'total':>12}")
        print(f"  {'─'*28} {'─'*12} {'─'*12}")

        grand_train, grand_total = 0, 0

        # base
        tr, tot = _count(self.mlp.base)
        tag = "frozen" if not tr else ""
        print(f"  base {tag:<22} {tr:>12,} {tot:>12,}")
        grand_train += tr; grand_total += tot

        # sidecar
        tr, tot = _count(self.mlp.sidecar)
        print(f"  sidecar{'':19} {tr:>12,} {tot:>12,}")
        grand_train += tr; grand_total += tot

        # normalizers
        for name, mod in [("base normalizer", self.obs_normalizer), ("sidecar normalizer", self.sidecar_normalizer)]:
            tr, tot = _count(mod)
            if tot:
                print(f"  {name:<28} {tr:>12,} {tot:>12,}")
                grand_train += tr; grand_total += tot

        # distribution (action std)
        if self.distribution is not None:
            tr, tot = _count(self.distribution)
            if tot:
                print(f"  {'distribution (std)':<28} {tr:>12,} {tot:>12,}")
                grand_train += tr; grand_total += tot

        print(f"  {'─'*28} {'─'*12} {'─'*12}")
        print(f"  {'TOTAL':<28} {grand_train:>12,} {grand_total:>12,}")
        pct = 100 * grand_train / grand_total if grand_total else 0
        print(f"  trainable: {pct:.1f}%")
        print(f"{sep}\n")

    # --- overrides ---

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        base_latent = self.get_latent(obs, masks, hidden_state)
        sidecar_latent = self._get_sidecar_latent(obs)
        mlp_output = self.mlp(base_latent, sidecar_latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def update_normalization(self, obs: TensorDict) -> None:
        """Update only the sidecar normalizer; base normalizer stays frozen."""
        if self.obs_normalization:
            latent = torch.cat([obs[group] for group in self.sidecar_obs_groups], dim=-1)
            self.sidecar_normalizer.update(latent)  # type: ignore

    def sidecar_diagnostics(self) -> dict[str, float]:
        """Sidecar head weight norm under ``SidecarStats/``."""
        return {"SidecarStats/head_weight_norm": self.mlp.sidecar.head.weight.norm().item()}

    def as_jit(self) -> torch.nn.Module:
        raise NotImplementedError("JIT export for MLPWithSidecarModel is not implemented yet.")

    def as_onnx(self, verbose: bool) -> torch.nn.Module:
        raise NotImplementedError("ONNX export for MLPWithSidecarModel is not implemented yet.")


class ModularNormMLPWithSidecarModel(MLPWithSidecarModel):
    """An :class:`MLPWithSidecarModel` whose frozen base is a :class:`~rsl_rl.modules.ModularNormMLP`.

    Used to adapt the bias-free, modular-norm TextOp WBC with a free-Adam sidecar.
    """

    _sidecar_cls = ModularNormMLPWithSidecar

    def _make_mlp(self, input_dim, output_dim, hidden_dims, activation):
        from rsl_rl.modules import ModularNormMLP
        return ModularNormMLP(input_dim, output_dim, hidden_dims, activation)
