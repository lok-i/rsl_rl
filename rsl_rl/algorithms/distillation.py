# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.modules import resolve_amp_dtype, set_model_amp
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import (
    compile_model,
    model_diagnostics,
    resolve_callable,
    resolve_obs_groups,
    resolve_optimizer,
)


class Distillation:
    """Distillation algorithm for training a student model to mimic a teacher model."""

    student: MLPModel
    """The student model."""

    teacher: MLPModel
    """The teacher model."""

    teacher_loaded: bool = False
    """Indicates whether the teacher model parameters have been loaded."""

    def __init__(
        self,
        student: MLPModel,
        teacher: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 1,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        max_grad_norm: float | None = None,
        loss_type: str = "mse",
        kl_direction: str = "reverse",
        optimizer: str = "adam",
        device: str = "cpu",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        **kwargs: dict,  # handle unused config parameters
    ) -> None:
        """Initialize the algorithm with models, storage, and optimization settings."""
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # Distillation components
        self.student = student.to(self.device)
        self.teacher = teacher.to(self.device)

        # Handles to the uncompiled modules for state_dict operations and export. If compilation is disabled, these
        # simply alias ``self.student`` / ``self.teacher``.
        self._raw_student = self.student
        self._raw_teacher = self.teacher

        # Create the optimizer
        self.optimizer = resolve_optimizer(optimizer)(self.student.parameters(), lr=learning_rate)  # type: ignore

        # Add storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()
        self.last_hidden_states = (None, None)

        # Distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        # Initialize the loss function
        if kl_direction not in ("forward", "reverse"):
            raise ValueError(f"kl_direction must be 'forward' or 'reverse', got {kl_direction!r}")
        self.kl_direction = kl_direction
        loss_fn_dict = {
            "mse": nn.functional.mse_loss,
            "huber": nn.functional.huber_loss,
            "kl": self._gaussian_kl,
        }
        if loss_type in loss_fn_dict:
            self.loss_fn = loss_fn_dict[loss_type]
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: {list(loss_fn_dict.keys())}")

        self.num_updates = 0

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        # Compute the actions
        self.transition.actions = self.student(obs, stochastic_output=True).detach()
        self.transition.privileged_actions = self.teacher(obs).detach()
        # Record the observations in the STUDENT's storage layout: a frozen-prefix model
        # swaps raw inputs for the values it just derived from them, so the update reads
        # the cached result instead of recomputing a pure function (SonicBaseModel.
        # storage_obs). Identity for models without the hook — but it must MATCH the
        # layout construct_algorithm allocated the storage with, or add_transition's
        # copy_ fails on the key set. Both or neither.
        self.transition.observations = self._cache_obs(obs)
        return self.transition.actions  # type: ignore

    def _cache_obs(self, obs: TensorDict) -> TensorDict:
        """Storage layout carrying this step's cached derivations (identity without the hook)."""
        hook = getattr(self._raw_student, "cache_obs", None)
        return hook(obs) if hook is not None else obs

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step and update the normalizers."""
        # Update the normalizers
        self.student.update_normalization(obs)
        # Record the rewards and dones
        self.transition.rewards = rewards
        self.transition.dones = dones
        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.student.reset(dones)
        self.teacher.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """No-op since distillation does not use return targets."""
        # Not needed for distillation
        pass

    def update(self) -> tuple[dict[str, float], dict[str, float]]:
        """Run optimization epochs over stored batches; return (losses, diagnostics).

        Two-tuple, matching :meth:`PPO.update` — the runner unpacks both and hands
        ``info_dict`` to the logger. The diagnostics are the MODEL's (AdapterStats/*,
        ZAttention/*, ZCapacity/*), which is what keeps a distilled student readable
        against the PPO row it is measured against.
        """
        self.num_updates += 1
        mean_behavior_loss = 0
        mean_gap_sq = 0.0
        loss = 0
        cnt = 0
        std = self._frozen_std()

        for epoch in range(self.num_learning_epochs):
            self.student.reset(hidden_state=self.last_hidden_states[0])
            self.teacher.reset(hidden_state=self.last_hidden_states[1])
            self.student.detach_hidden_state()
            for batch in self.storage.generator():
                # Inference of the student for gradient computation
                actions = self.student(batch.observations)

                # Behavior cloning loss
                behavior_loss = self.loss_fn(actions, batch.privileged_actions)

                # ZDistill/gap_sigma: the same residual in units of the policy's OWN
                # exploration band. Diagnostic only — it never touches the gradient. The
                # raw loss is in action units (rad^2 for mse), which is not comparable
                # across joints whose frozen sigma differs by ~1.7x, nor across tasks;
                # this one reads as "the student is N exploration bands off the teacher".
                if std is not None:
                    with torch.no_grad():
                        mean_gap_sq += (((actions - batch.privileged_actions) / std) ** 2).mean().item()

                # Total loss
                loss = loss + behavior_loss
                mean_behavior_loss += behavior_loss.item()
                cnt += 1

                # Gradient step
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.student.detach_hidden_state()
                    loss = 0

                # Reset dones
                self.student.reset(batch.dones.view(-1))
                self.teacher.reset(batch.dones.view(-1))
                self.student.detach_hidden_state(batch.dones.view(-1))

        mean_behavior_loss /= cnt
        mean_gap_sq /= cnt
        self.storage.clear()
        self.last_hidden_states = (self.student.get_hidden_state(), self.teacher.get_hidden_state())
        self.student.detach_hidden_state()

        # Construct the loss dictionary
        loss_dict = {"behavior": mean_behavior_loss}

        # Model-owned diagnostics; read off the LAST forward, hence after the loop.
        info_dict = model_diagnostics(self._raw_student)
        if std is not None:
            info_dict["ZDistill/gap_sigma"] = mean_gap_sq**0.5

        return loss_dict, info_dict

    def _gaussian_kl(self, actions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """KL between the student's and teacher's diagonal Gaussians, general form.

            KL(T||S) = sum_j [ log(s_S/s_T) + (s_T^2 + (mu_T - mu_S)^2) / (2 s_S^2) - 1/2 ]
            KL(S||T) = sum_j [ log(s_T/s_S) + (s_S^2 + (mu_S - mu_T)^2) / (2 s_T^2) - 1/2 ]

        Both sigmas are read from their own model every call rather than assumed equal.
        They ARE equal today (adapter agents freeze std, and a distilled student inherits
        its teacher's band -- measured element-wise identical on SONIC), and in that case
        the log term is 0, the trace term is exactly 1/2, and both directions collapse to
        the same quadratic sum_j (mu_T - mu_S)^2 / (2 s^2). Reading both anyway is what
        keeps this correct if a learnable std, a different `std_scale`, or a teacher from
        another band ever shows up -- the collapse is a property of the checkpoint, not of
        distillation, and nothing else in the code asserts it.

        `direction` picks the mode-seeking (reverse, "S||T") or mode-covering (forward,
        "T||S") objective. With equal frozen sigmas they have identical gradients; the
        choice only bites once the sigmas differ.

        Reduced as sum-over-action-dims then mean-over-batch -- KL is a sum over
        independent dims, so this is nats per step. NOTE the gradient scale: against
        `mse` this is ~62x on SONIC (29 dims x 0.5 * mean(1/sigma^2) = 29 x 2.13), so an
        A/B against an `mse` baseline must scale the learning rate down by that factor or
        it is an LR experiment wearing a loss experiment's clothes.
        """
        std_s = self._std_of(self._raw_student)
        std_t = self._std_of(self._raw_teacher)
        if std_s is None or std_t is None:
            raise ValueError("loss_type='kl' needs both models to carry a distribution.")
        sq_err = (target - actions).pow(2)
        if self.kl_direction == "reverse":  # KL(S||T), mode-seeking
            num, den, log_ratio = std_s.pow(2), std_t.pow(2), torch.log(std_t + 1e-8) - torch.log(std_s + 1e-8)
        else:                               # KL(T||S), mode-covering
            num, den, log_ratio = std_t.pow(2), std_s.pow(2), torch.log(std_s + 1e-8) - torch.log(std_t + 1e-8)
        kl = log_ratio + (num + sq_err) / (2.0 * (den + 1e-7)) - 0.5
        return kl.sum(dim=-1).mean()

    @staticmethod
    def _std_of(model: MLPModel) -> torch.Tensor | None:
        """A model's per-dim action std as (1, A), from the distribution PARAMETER."""
        dist = getattr(model, "distribution", None)
        if dist is None:
            return None
        if hasattr(dist, "std_param"):
            return dist.std_param.detach().reshape(1, -1)
        if hasattr(dist, "log_std_param"):
            return dist.log_std_param.detach().exp().reshape(1, -1)
        return None

    def _frozen_std(self) -> torch.Tensor | None:
        """The student's per-dim action std as a (1, A) tensor, or None if it has no head.

        Read from the distribution's PARAMETER, not ``output_std``: the latter is the last
        forward's ``Normal.stddev``, so it carries that batch's shape and would broadcast
        wrong against a minibatch. Adapter agents freeze std (``learn_std=False``), but this
        re-reads it per update so a learnable one stays correct.
        """
        return self._std_of(self._raw_student)

    def train_mode(self) -> None:
        """Set train mode for the student and keep the teacher in eval mode."""
        self.student.train()
        # Teacher is always in eval mode
        self.teacher.eval()

    def eval_mode(self) -> None:
        """Set evaluation mode for student and teacher models."""
        self.student.eval()
        self.teacher.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "student_state_dict": self._raw_student.state_dict(),
            "teacher_state_dict": self._raw_teacher.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        # If no load_cfg is provided, determine what to load automatically
        if load_cfg is None and any("actor_state_dict" in key for key in loaded_dict):  # Load from RL training
            load_cfg = {"teacher": True, "iteration": False}  # Only load teacher by default
        elif load_cfg is None:  # Load from distillation training
            load_cfg = {
                "student": True,
                "teacher": True,
                "optimizer": True,
                "iteration": True,
            }

        # Load the specified models
        if load_cfg.get("student"):
            self._raw_student.load_state_dict(loaded_dict["student_state_dict"], strict=strict)
        if load_cfg.get("teacher"):
            self._raw_teacher.load_state_dict(
                loaded_dict.get("teacher_state_dict") or loaded_dict["actor_state_dict"], strict=strict
            )
            self.teacher_loaded = True
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Get the policy model."""
        return self._raw_student

    def compile(self, mode: str | None = None) -> None:
        """Compile student and teacher with ``torch.compile``.

        See :func:`~rsl_rl.utils.compile_model` for the set of accepted modes.

        Args:
            mode: ``torch.compile`` mode. Defaults to ``None``, in which case compilation is disabled.
        """
        self.student = compile_model(self._raw_student, mode)  # type: ignore
        self.teacher = compile_model(self._raw_teacher, mode)  # type: ignore

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> Distillation:
        """Construct the distillation algorithm."""
        # Resolve class callables
        alg_class: type[Distillation] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        student_class: type[MLPModel] = resolve_callable(cfg["student"].pop("class_name"))  # type: ignore
        teacher_class: type[MLPModel] = resolve_callable(cfg["teacher"].pop("class_name"))  # type: ignore

        # Resolve observation groups
        default_sets = ["student", "teacher"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Distillation is not compatible with RND and symmetry extensions
        if cfg["algorithm"].get("rnd_cfg") is not None:
            raise ValueError("The RND extension is not compatible with Distillation.")
        cfg["algorithm"]["rnd_cfg"] = None
        if cfg["algorithm"].get("symmetry_cfg") is not None:
            raise ValueError("The symmetry extension is not compatible with Distillation.")
        cfg["algorithm"]["symmetry_cfg"] = None

        # Initialize the policy
        student: MLPModel = student_class(obs, cfg["obs_groups"], "student", env.num_actions, **cfg["student"]).to(
            device
        )
        print(f"Student Model: {student}")
        teacher: MLPModel = teacher_class(obs, cfg["obs_groups"], "teacher", env.num_actions, **cfg["teacher"]).to(
            device
        )
        print(f"Teacher Model: {teacher}")

        # Initialize the storage in the STUDENT's layout: a frozen-prefix model may swap raw
        # inputs for cached derivations (SonicBaseModel.storage_obs replaces the 360-float
        # tokenizer window with the 64 floats it encodes to, keeping the frozen encoder out
        # of every update pass). Unlike PPO this needs no dropped-group guard: the STUDENT is
        # the only model that reads storage. The teacher is evaluated on LIVE observations in
        # act() and only its action is stored, so a group the student caches away can never
        # be one the teacher still needs.
        storage_obs = getattr(student, "storage_obs", lambda o: o)(obs)
        dropped = set(obs.keys()) - set(storage_obs.keys())
        if dropped:
            print(f"[storage] {type(student).__name__} cached prefix: dropped {sorted(dropped)}")
        storage = RolloutStorage(
            "distillation", env.num_envs, cfg["num_steps_per_env"], storage_obs, [env.num_actions], device
        )

        # Initialize the algorithm
        alg: Distillation = alg_class(
            student, teacher, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"]
        )

        # Compile the algorithm's models if requested
        alg.compile(cfg.get("torch_compile_mode"))
        alg.set_amp(cfg.get("amp_dtype"))

        return alg

    def set_amp(self, dtype: str | None) -> None:
        """Set the mixed-precision body dtype for the STUDENT — "bfloat16" | "float16" | None.

        The teacher stays fp32 whatever this says, because its action IS the regression
        target: a bf16 body would perturb the labels rather than the learner, and the
        student would then be fitting a slightly different teacher than the one that was
        evaluated. See :meth:`PPO.set_amp` for why the head stays fp32 either way.
        """
        self.amp_dtype = resolve_amp_dtype(dtype)
        set_model_amp(self._raw_student, self.amp_dtype)

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self._raw_student.state_dict(), self._raw_teacher.state_dict()]
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self._raw_student.load_state_dict(model_params[0])
        self._raw_teacher.load_state_dict(model_params[1])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.student.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in self.student.parameters():
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
