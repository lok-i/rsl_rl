# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, HiddenState, ModularNormMLPWithAdapter

from .mlp_model import MLPModel


class MLPWithAdapterModel(MLPModel):
    """An :class:`MLPModel` whose MLP is a frozen base with a trainable LoRA adapter (:class:`MLPWithAdapter`).

    The base is a pretrained, frozen policy (e.g. the TextOp WBC); only the strapped adapters are trained.
    Two observation streams feed the network:

    - **base stream** -- the standard ``obs_groups[obs_set]`` groups, normalized by the (frozen) base
      normalizer and fed to the frozen base layers. For the WBC this is its exact training observation.
    - **adapter stream** -- a separate ``adapter_obs_group`` (e.g. object state), normalized by its own
      (trainable) normalizer and fed to the input adapter. This is the new conditioning signal.

    The base weights and the base observation normalizer are frozen (the WBC must keep seeing its trained
    input distribution); only the adapters and the adapter normalizer learn. The modular-norm
    ``dualize_gradients`` / ``project_weights`` hooks are intentionally NOT exposed: the base is off its
    spectral manifold (TextOp trained dualize-only), so projecting it every step would destroy it -- the
    free-Adam adapters need neither hook.
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
        adapter_obs_group: str = "adapter",
        rank: int | list[int] = -1,
        alpha: float = 1.0,
        wbc_checkpoint: str | None = None,
    ) -> None:
        """Initialize the model, load + freeze the base, and strap a zero-init adapter on it.

        Args:
            obs: Observation TensorDict (used to size the base and adapter streams).
            obs_groups: Mapping of observation sets to groups. ``obs_groups[obs_set]`` is the base stream.
            obs_set: Observation set for this model (``"actor"``).
            output_dim: Action dimension.
            hidden_dims: Hidden dims of the base MLP (must match the pretrained base, e.g. the WBC).
            activation: Activation of the base MLP (the WBC uses ``elu``).
            obs_normalization: Normalize both streams (required to load a checkpoint's base normalizer).
            distribution_cfg: Output distribution config (Gaussian for PPO).
            adapter_obs_group: Name of the observation group feeding the adapter (e.g. object state).
            rank: Adapter rank, scalar or per-layer list (see :class:`~rsl_rl.modules.Adapter`).
            alpha: Adapter scale.
            wbc_checkpoint: Path to a pretrained base checkpoint (a ``*_ported.pt`` payload or a raw state
                dict). Loaded into the base before strapping, so the adapted model starts as the base.
        """
        # The base (WBC) is a modular-norm MLP; build it as such, then strap the adapter on top.
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
            modular_norm=True,
        )

        # Load the pretrained base (weights + base normalizer + action std) BEFORE strapping: at this point
        # ``self.mlp`` is still a plain ModularNormMLP whose keys match the checkpoint.
        if wbc_checkpoint is not None:
            if not obs_normalization:
                raise ValueError("wbc_checkpoint requires obs_normalization=True to load the base normalizer.")
            payload = torch.load(wbc_checkpoint, map_location="cpu", weights_only=False)
            is_ported = isinstance(payload, dict) and "model_state_dict" in payload
            state_dict = payload["model_state_dict"] if is_ported else payload
            _, unexpected = self.load_state_dict(state_dict, strict=False)
            if unexpected:
                raise RuntimeError(f"wbc_checkpoint has unexpected keys: {unexpected}")

        # Adapter stream: separate group + its own (trainable) normalizer.
        self.adapter_obs_groups = [adapter_obs_group]
        adapter_dim = self._concat_dim(obs, self.adapter_obs_groups)
        self.adapter_normalizer = EmpiricalNormalization(adapter_dim) if obs_normalization else torch.nn.Identity()

        # Strap the adapter on the (now loaded) frozen base. The input adapter takes the adapter stream.
        self.mlp = ModularNormMLPWithAdapter.from_base_mlp(
            self.mlp, adapter_input_dim=adapter_dim, rank=rank, alpha=alpha
        )

        # Drop the modular-norm hooks: the frozen base must never be dualized/projected (it is off its
        # spectral manifold), and the free-Adam adapters do not use them. Their absence makes the PPO loop
        # fall through to plain grad-clip + Adam on the trainable (adapter) params.
        del self.dualize_gradients
        del self.project_weights

    @staticmethod
    def _concat_dim(obs: TensorDict, groups: list[str]) -> int:
        """Total feature dimension of the concatenated ``groups``."""
        return sum(obs[group].shape[-1] for group in groups)

    def _get_adapter_latent(self, obs: TensorDict) -> torch.Tensor:
        """Concatenate and normalize the adapter observation stream."""
        latent = torch.cat([obs[group] for group in self.adapter_obs_groups], dim=-1)
        return self.adapter_normalizer(latent)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Forward pass: frozen base on the base stream, adapter on the adapter stream."""
        base_latent = self.get_latent(obs, masks, hidden_state)
        adapter_latent = self._get_adapter_latent(obs)
        mlp_output = self.mlp(base_latent, adapter_latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def update_normalization(self, obs: TensorDict) -> None:
        """Update only the adapter normalizer; the base normalizer stays frozen (WBC's trained stats)."""
        if self.obs_normalization:
            latent = torch.cat([obs[group] for group in self.adapter_obs_groups], dim=-1)
            self.adapter_normalizer.update(latent)  # type: ignore

    def as_jit(self) -> torch.nn.Module:
        """JIT export is not yet supported for the two-stream adapter model."""
        raise NotImplementedError("JIT export for MLPWithAdapterModel is not implemented yet.")

    def as_onnx(self, verbose: bool) -> torch.nn.Module:
        """ONNX export is not yet supported for the two-stream adapter model."""
        raise NotImplementedError("ONNX export for MLPWithAdapterModel is not implemented yet.")
