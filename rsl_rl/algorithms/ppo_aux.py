# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO with an auxiliary representation objective (see ``rsl_rl.extensions.aux``)."""

from __future__ import annotations

import functools
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
      three groups: actor+critic (adaptive-KL LR), encoder (fixed ``extractor_lr``), aux heads
      (fixed aux LR) — the ``fixed_lr`` guard in PPO.update keeps the schedule off the last two.
    - ``"sequential"`` (OpenTrack style): the aux objective runs its own optimizer for
      ``num_epochs x num_mini_batches`` steps *after* the PPO epochs (reads the storage
      directly — ``storage.clear()`` only resets the write cursor). ``extractor_in_ppo`` /
      ``extractor_lr`` control whether PPO co-trains the encoder (fixed-LR group) or not at all.

    ``ZGradient/*`` reports the tug-of-war on the shared representation, in two currencies:

    - **demand** — per-source pre-clip gradient RMS (``ppo``, ``aux``), their split
      (``frac``, 0.5 = parity) and their alignment (``cos``; < 0 = the two sources are
      pulling the encoder apart).
    - **pull** — ``⟨g_s, -Δθ⟩`` against the REALIZED parameter delta of the step. Since
      ``g_total = g_ppo + g_aux`` and Δθ is shared, this is an exact additive split of the
      first-order loss decrease along the step actually taken, with clipping, the Adam
      preconditioner and the (fixed) encoder LR all already inside Δθ. Demand says who
      shouts loudest; pull says who moved the weights. Signed: a negative entry means that
      source's own loss went UP along the step.

    Note the hooks measure gradient FLOW: in sequential aux-only mode the ppo entry is
    PPO's (unapplied) demand, not a bug.

    Metric routing: aux keys ``loss/<name>`` land in the runner's ``Loss/`` section; every
    other key is passed through verbatim, so the metric's producer owns its section name.
    """

    def __init__(
        self,
        *args: Any,
        aux_cfg: dict,
        aux_mode: str = "sequential",
        aux_weight: float = 1.0,
        extractor_lr: float | None = None,
        extractor_in_ppo: bool = True,
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

        extractor_params = [p for enc in self._raw_actor.extractors.values() for p in enc.parameters()]
        extractor_ids = {id(p) for p in extractor_params}
        rest = [
            p for p in chain(self._raw_actor.parameters(), self._raw_critic.parameters()) if id(p) not in extractor_ids
        ]
        if aux_mode == "joint":
            # One optimizer, one Adam per param; aux head params join it (their own aux Adam
            # from _finalize goes unused). Encoder LR defaults to the aux LR.
            aux_params = [p for p in self.aux.parameters() if p.requires_grad]
            self.optimizer = type(self.optimizer)([
                {"params": rest, "lr": self.learning_rate},
                {"params": extractor_params, "lr": extractor_lr or self.aux.learning_rate, "fixed_lr": True},
                {"params": aux_params, "lr": self.aux.learning_rate, "fixed_lr": True},
            ])
        elif not extractor_in_ppo or extractor_lr is not None:
            groups = [{"params": rest, "lr": self.learning_rate}]
            if extractor_in_ppo:
                groups.append({"params": extractor_params, "lr": extractor_lr, "fixed_lr": True})
            self.optimizer = type(self.optimizer)(groups)

        # Per-source encoder gradient bookkeeping, fed by post-accumulate-grad hooks. The
        # phase flag attributes each backward pass to PPO or the aux objective; norms are
        # pre-clip (raw demand).
        self._extractor_params = extractor_params
        self._param_index = {id(p): i for i, p in enumerate(extractor_params)}
        self._grad_phase: str | None = None
        zero = functools.partial(torch.zeros, (), device=self.device)
        self._grad_sq = {"ppo": zero(), "aux": zero()}
        self._grad_dot = zero()  # <g_ppo, g_aux>, summed over params and passes
        self._pull = {"ppo": zero(), "aux": zero()}
        self._grad_passes = {"ppo": 0, "aux": 0}
        # Live per-source gradients + the pre-step weights of the CURRENT minibatch; consumed
        # and cleared in _post_optimizer_step, so nothing here survives into the next one.
        self._g: dict[str, list[torch.Tensor | None]] = {
            "ppo": [None] * len(extractor_params),
            "aux": [None] * len(extractor_params),
        }
        self._theta_before: list[torch.Tensor | None] = [None] * len(extractor_params)
        self._pass_marker = extractor_params[0] if extractor_params else None
        for p in extractor_params:
            p.register_post_accumulate_grad_hook(self._extractor_grad_hook)
        self._aux_totals: dict[str, float] = {}
        self._aux_calls = 0

    def _extractor_grad_hook(self, param: torch.Tensor) -> None:
        """Record this backward's contribution to ``param``, per source.

        Joint mode accumulates into ONE ``param.grad``: by the time the aux backward's hook
        fires, ``param.grad`` is ``g_ppo + w*g_aux``. Reading it raw therefore reports the
        TOTAL as the aux term — which floors ``frac`` at ~0.5 and was the origin of the
        "suspicious invariant" ``frac ~ 0.54``. Subtracting the snapshot taken during the
        PPO phase recovers the true ``w*g_aux``. Sequential mode needs no subtraction: the
        aux objective zeroes grads before its own backward, so the read is already pure.
        """
        phase = self._grad_phase
        if phase is None or param.grad is None:
            return
        i = self._param_index[id(param)]
        prev = self._g["ppo"][i] if (phase == "aux" and self.aux_mode == "joint") else None
        g = param.grad.detach().clone() if prev is None else param.grad.detach() - prev
        self._g[phase][i] = g
        self._grad_sq[phase] += g.square().sum()
        if prev is not None:
            self._grad_dot += (prev * g).sum()
        if param is self._pass_marker:
            self._grad_passes[phase] += 1

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
        nn.utils.clip_grad_norm_([p for p in self.aux.parameters() if p.requires_grad], self.aux.max_grad_norm)
        # Pre-step weights for the pull attribution — taken here, after both backwards and
        # before the clip + optimizer step, so Delta-theta is the fully realized motion.
        with torch.no_grad():
            for i, p in enumerate(self._extractor_params):
                self._theta_before[i] = p.detach().clone()
        # EMA cadence: per gradient step, reference behavior (multimodal_rl updates the target
        # inside compute_loss).
        self.aux._post_step()
        for key, value in metrics.items():
            self._aux_totals[key] = self._aux_totals.get(key, 0.0) + value
        self._aux_calls += 1

    @torch.no_grad()
    def _post_optimizer_step(self) -> None:
        """Attribute the realized parameter delta of this minibatch to its gradient sources."""
        if self._theta_before[0] is None:
            return  # sequential mode (or an aux call that produced no loss)
        for i, p in enumerate(self._extractor_params):
            d_theta = p.detach() - self._theta_before[i]  # type: ignore[operator]
            for phase in ("ppo", "aux"):
                g = self._g[phase][i]
                if g is not None:
                    self._pull[phase] -= (g * d_theta).sum()  # <g, -dtheta>
            self._theta_before[i] = None
            self._g["ppo"][i] = self._g["aux"][i] = None

    def update(self) -> tuple[dict[str, float], dict[str, float]]:
        """Run the PPO update (+ aux, per ``aux_mode``) and route aux metrics to their sections."""
        for phase in ("ppo", "aux"):
            self._grad_sq[phase].zero_()
            self._pull[phase].zero_()
            self._grad_passes[phase] = 0
        self._grad_dot.zero_()
        self._aux_totals, self._aux_calls = {}, 0

        self._grad_phase = "ppo"
        loss_dict, info_dict = super().update()  # joint mode runs aux via _extra_backward
        if self.aux_mode == "joint":
            aux_metrics = {k: v / self._aux_calls for k, v in self._aux_totals.items()} if self._aux_calls else {}
        else:
            self._grad_phase = "aux"
            aux_metrics = self.aux.update(self.storage, self._raw_actor)
        self._grad_phase = None

        for key, value in aux_metrics.items():
            if key.startswith("loss/"):
                loss_dict[key.removeprefix("loss/")] = value
            else:
                info_dict[key] = value  # the producer owns its section name

        # --- ZGradient: demand (pre-clip RMS per pass) and pull (along the realized step) ---
        sq = {phase: self._grad_sq[phase].item() for phase in ("ppo", "aux")}
        norms = {
            phase: math.sqrt(sq[phase] / self._grad_passes[phase])
            for phase in ("ppo", "aux")
            if self._grad_passes[phase] > 0
        }
        for phase, value in norms.items():
            info_dict[f"ZGradient/{phase}"] = value
        if len(norms) == 2 and (norms["ppo"] + norms["aux"]) > 0:
            info_dict["ZGradient/frac"] = norms["aux"] / (norms["ppo"] + norms["aux"])
        if sq["ppo"] > 0 and sq["aux"] > 0:
            # Cosine in the product space over all params and passes — < 0 means the two
            # sources are asking the encoder to move in opposing directions.
            info_dict["ZGradient/cos"] = self._grad_dot.item() / math.sqrt(sq["ppo"] * sq["aux"])
        pull = {phase: self._pull[phase].item() for phase in ("ppo", "aux")}
        if any(pull.values()):
            info_dict["ZGradient/pull_ppo"] = pull["ppo"]
            info_dict["ZGradient/pull_aux"] = pull["aux"]
            total = pull["ppo"] + pull["aux"]
            if total > 0:  # a negative total means the step raised BOTH losses — no split
                info_dict["ZGradient/pull_frac"] = pull["aux"] / total
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
