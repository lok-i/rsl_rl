# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO with an auxiliary representation objective (see ``rsl_rl.extensions.aux``)."""

from __future__ import annotations

from typing import Any

from rsl_rl.extensions.aux import AuxObjective
from rsl_rl.utils import resolve_callable

from .ppo import PPO


class PPOAux(PPO):
    """PPO plus one auxiliary objective training the actor's feature encoder.

    The aux objective runs *after* the PPO epochs of each update, so the adaptive-KL schedule
    never sees encoder drift as policy divergence. It reads the rollout storage directly —
    ``storage.clear()`` only resets the write cursor, the tensors stay intact.

    The objective is configured via ``aux_cfg`` (``class_name`` + kwargs of the chosen
    :class:`~rsl_rl.extensions.aux.AuxObjective` subclass) and constructed here from the
    storage template and the actor's encoder handle; the algorithm is otherwise agnostic
    to the objective's nature (SL vs SSL).
    """

    def __init__(self, *args: Any, aux_cfg: dict, **kwargs: Any) -> None:
        """Initialize PPO and construct the auxiliary objective from ``aux_cfg``."""
        super().__init__(*args, **kwargs)
        cfg = dict(aux_cfg)
        aux_class: type[AuxObjective] = resolve_callable(cfg.pop("class_name"))  # type: ignore
        self.aux = aux_class(storage=self.storage, actor=self._raw_actor, device=self.device, **cfg)
        self.aux.to(self.device)

    def update(self) -> tuple[dict[str, float], dict[str, float]]:
        """Run the PPO update, then the auxiliary objective on the same rollout."""
        loss_dict, info_dict = super().update()
        loss_dict.update(self.aux.update(self.storage, self._raw_actor))
        return loss_dict, info_dict

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        super().train_mode()
        self.aux.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        super().eval_mode()
        self.aux.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving, including aux state."""
        saved_dict = super().save()
        saved_dict["aux_state_dict"] = self.aux.state_dict()
        saved_dict["aux_optimizer_state_dict"] = self.aux.optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models and aux state from a saved dict."""
        resume = super().load(loaded_dict, load_cfg, strict)
        if "aux_state_dict" in loaded_dict:
            self.aux.load_state_dict(loaded_dict["aux_state_dict"], strict=strict)
            self.aux.optimizer.load_state_dict(loaded_dict["aux_optimizer_state_dict"])
        return resume
