# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO with an auxiliary representation objective (see ``rsl_rl.extensions.aux``)."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from itertools import chain
from typing import Any

from rsl_rl.extensions.aux import AuxObjective
from rsl_rl.utils import resolve_callable

from .ppo import PPO


class PPOAux(PPO):
    """PPO plus one auxiliary objective training the actor's feature encoder.

    Two update schemes (``aux_mode``):

    - ``"joint"`` (multimodal_rl style): each PPO minibatch backward is followed by an
      accumulated ``aux_weight * aux_loss`` backward on a freshly-sampled aux minibatch —
      mathematically ``loss_ppo + aux_weight * aux_loss``, one optimizer step, one Adam per
      parameter, and the aux gets a gradient at EVERY PPO step. The optimizer is rebuilt with
      three groups: actor+critic (adaptive-KL LR), encoder (fixed ``encoder_lr``), aux heads
      (fixed aux LR) — the ``fixed_lr`` guard in PPO.update keeps the schedule off the last two.
    - ``"sequential"`` (OpenTrack style): the aux objective runs its own optimizer for
      ``num_epochs x num_mini_batches`` steps *after* the PPO epochs (reads the storage
      directly — ``storage.clear()`` only resets the write cursor). ``encoder_in_ppo`` /
      ``encoder_lr`` control whether PPO co-trains the encoder (fixed-LR group) or not at all.

    Encoder gradient RMS norms are logged per source plus their fraction
    (``enc_grad_frac = aux / (ppo + aux)``, 0.5 = parity) — the tug-of-war on the shared
    representation, comparable across variants and modes. Note the hooks measure gradient
    FLOW: in sequential aux-only mode the ppo entry is PPO's (unapplied) demand, not a bug.

    Metric routing: aux keys ``loss/<name>`` land in the runner's ``Loss/`` section; all other
    aux metrics + the grad diagnostics land under ``Auxiliaries/`` (via info_dict).
    """

    def __init__(
        self,
        *args: Any,
        aux_cfg: dict,
        aux_mode: str = "sequential",
        aux_weight: float = 1.0,
        encoder_lr: float | None = None,
        encoder_in_ppo: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize PPO and construct the auxiliary objective from ``aux_cfg``."""
        super().__init__(*args, **kwargs)
        assert aux_mode in ("joint", "sequential"), f"unknown aux_mode: {aux_mode}"
        self.aux_mode = aux_mode
        self.aux_weight = aux_weight
        cfg = dict(aux_cfg)
        aux_class: type[AuxObjective] = resolve_callable(cfg.pop("class_name"))  # type: ignore
        self.aux = aux_class(storage=self.storage, actor=self._raw_actor, device=self.device, **cfg)
        self.aux.to(self.device)

        encoder_params = [p for enc in self._raw_actor.encoders.values() for p in enc.parameters()]
        encoder_ids = {id(p) for p in encoder_params}
        rest = [
            p for p in chain(self._raw_actor.parameters(), self._raw_critic.parameters())
            if id(p) not in encoder_ids
        ]
        if aux_mode == "joint":
            # One optimizer, one Adam per param; aux head params join it (their own aux Adam
            # from _finalize goes unused). Encoder LR defaults to the aux LR.
            aux_params = [p for p in self.aux.parameters() if p.requires_grad]
            self.optimizer = type(self.optimizer)(
                [
                    {"params": rest, "lr": self.learning_rate},
                    {"params": encoder_params, "lr": encoder_lr or self.aux.learning_rate, "fixed_lr": True},
                    {"params": aux_params, "lr": self.aux.learning_rate, "fixed_lr": True},
                ]
            )
        elif not encoder_in_ppo or encoder_lr is not None:
            groups = [{"params": rest, "lr": self.learning_rate}]
            if encoder_in_ppo:
                groups.append({"params": encoder_params, "lr": encoder_lr, "fixed_lr": True})
            self.optimizer = type(self.optimizer)(groups)

        # Per-source encoder grad-norm accumulators, fed by post-accumulate-grad hooks. The
        # phase flag attributes each backward pass to PPO or the aux objective; norms are
        # pre-clip (raw signal strength).
        self._grad_phase: str | None = None
        self._grad_sq = {"ppo": torch.zeros((), device=self.device), "aux": torch.zeros((), device=self.device)}
        self._grad_passes = {"ppo": 0, "aux": 0}
        self._pass_marker = encoder_params[0] if encoder_params else None
        for p in encoder_params:
            p.register_post_accumulate_grad_hook(self._encoder_grad_hook)
        self._aux_totals: dict[str, float] = {}
        self._aux_calls = 0

    def _encoder_grad_hook(self, param: torch.Tensor) -> None:
        if self._grad_phase is None or param.grad is None:
            return
        self._grad_sq[self._grad_phase] += param.grad.detach().square().sum()
        if param is self._pass_marker:
            self._grad_passes[self._grad_phase] += 1

    def _extra_backward(self) -> None:
        """Joint mode: accumulate ``aux_weight * aux_loss`` gradients into the PPO minibatch step."""
        if self.aux_mode != "joint":
            return
        out = self.aux.sample_loss(self.storage)
        if out is None:
            return
        loss, metrics = out
        self._grad_phase = "aux"
        # PPO's backward hooks accumulated grads into param.grad already; this adds on top —
        # identical to backpropagating the summed loss, but separately phase-tagged.
        (self.aux_weight * loss).backward()
        self._grad_phase = "ppo"
        nn.utils.clip_grad_norm_(
            [p for p in self.aux.parameters() if p.requires_grad], self.aux.max_grad_norm
        )
        # EMA cadence: per gradient step, reference behavior (multimodal_rl updates the target
        # inside compute_loss).
        self.aux._post_step()
        for key, value in metrics.items():
            self._aux_totals[key] = self._aux_totals.get(key, 0.0) + value
        self._aux_calls += 1

    def update(self) -> tuple[dict[str, float], dict[str, float]]:
        """Run the PPO update (+ aux, per ``aux_mode``) and route aux metrics to their sections."""
        for phase in ("ppo", "aux"):
            self._grad_sq[phase].zero_()
            self._grad_passes[phase] = 0
        self._aux_totals, self._aux_calls = {}, 0

        self._grad_phase = "ppo"
        loss_dict, info_dict = super().update()  # joint mode runs aux via _extra_backward
        if self.aux_mode == "joint":
            aux_metrics = {k: v / self._aux_calls for k, v in self._aux_totals.items()} if self._aux_calls else {}
            aux_metrics.update(self.aux.latent_metrics(self.storage.observations[self.aux.feat_group].flatten(0, 1)))
        else:
            self._grad_phase = "aux"
            aux_metrics = self.aux.update(self.storage, self._raw_actor)
        self._grad_phase = None

        for key, value in aux_metrics.items():
            if key.startswith("loss/"):
                loss_dict[key.removeprefix("loss/")] = value
            else:
                info_dict[f"Auxiliaries/{key}"] = value

        norms = {}
        for phase in ("ppo", "aux"):
            if self._grad_passes[phase] > 0:
                norms[phase] = math.sqrt(self._grad_sq[phase].item() / self._grad_passes[phase])
                info_dict[f"Auxiliaries/enc_grad_{phase}"] = norms[phase]
        if len(norms) == 2 and (norms["ppo"] + norms["aux"]) > 0:
            info_dict["Auxiliaries/enc_grad_frac"] = norms["aux"] / (norms["ppo"] + norms["aux"])
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
