# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO with an auxiliary representation objective (see ``rsl_rl.extensions.aux``)."""

from __future__ import annotations

import math
import torch
from itertools import chain
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

    If ``encoder_lr`` is set, the actor's encoder parameters are moved to a fixed-LR group in
    PPO's optimizer: the adaptive-KL schedule (which swings 1e-5..1e-2) no longer applies to
    the representation, so PPO cannot shout over the aux objective at 10x its learning rate.
    Encoder gradient RMS norms are logged per source (``aux/enc_grad_{ppo,aux}``) to make the
    PPO-vs-aux tug-of-war on the shared representation observable.
    """

    def __init__(self, *args: Any, aux_cfg: dict, encoder_lr: float | None = None, **kwargs: Any) -> None:
        """Initialize PPO and construct the auxiliary objective from ``aux_cfg``."""
        super().__init__(*args, **kwargs)
        cfg = dict(aux_cfg)
        aux_class: type[AuxObjective] = resolve_callable(cfg.pop("class_name"))  # type: ignore
        self.aux = aux_class(storage=self.storage, actor=self._raw_actor, device=self.device, **cfg)
        self.aux.to(self.device)

        encoder_params = [p for enc in self._raw_actor.encoders.values() for p in enc.parameters()]
        if encoder_lr is not None:
            # Rebuild the PPO optimizer with the encoder in its own fixed-LR group
            # (see the ``fixed_lr`` guard in PPO.update's adaptive-KL block).
            encoder_ids = {id(p) for p in encoder_params}
            rest = [
                p for p in chain(self._raw_actor.parameters(), self._raw_critic.parameters())
                if id(p) not in encoder_ids
            ]
            self.optimizer = type(self.optimizer)(
                [
                    {"params": rest, "lr": self.learning_rate},
                    {"params": encoder_params, "lr": encoder_lr, "fixed_lr": True},
                ]
            )

        # Per-source encoder grad-norm accumulators, fed by post-accumulate-grad hooks. The
        # phase flag attributes each backward pass to PPO or the aux objective; norms are
        # pre-clip (raw signal strength).
        self._grad_phase: str | None = None
        self._grad_sq = {"ppo": torch.zeros((), device=self.device), "aux": torch.zeros((), device=self.device)}
        self._grad_passes = {"ppo": 0, "aux": 0}
        self._pass_marker = encoder_params[0] if encoder_params else None
        for p in encoder_params:
            p.register_post_accumulate_grad_hook(self._encoder_grad_hook)

    def _encoder_grad_hook(self, param: torch.Tensor) -> None:
        if self._grad_phase is None or param.grad is None:
            return
        self._grad_sq[self._grad_phase] += param.grad.detach().square().sum()
        if param is self._pass_marker:
            self._grad_passes[self._grad_phase] += 1

    def update(self) -> tuple[dict[str, float], dict[str, float]]:
        """Run the PPO update, then the auxiliary objective on the same rollout."""
        for phase in ("ppo", "aux"):
            self._grad_sq[phase].zero_()
            self._grad_passes[phase] = 0

        self._grad_phase = "ppo"
        loss_dict, info_dict = super().update()
        self._grad_phase = "aux"
        loss_dict.update(self.aux.update(self.storage, self._raw_actor))
        self._grad_phase = None

        for phase in ("ppo", "aux"):
            if self._grad_passes[phase] > 0:
                loss_dict[f"aux/enc_grad_{phase}"] = math.sqrt(
                    self._grad_sq[phase].item() / self._grad_passes[phase]
                )
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
