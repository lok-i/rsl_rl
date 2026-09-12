# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mixed-precision seam: autocast the model BODY, keep the output head in fp32."""

from __future__ import annotations

import contextlib
import torch
import torch.nn as nn
from contextlib import AbstractContextManager


class AmpMixin:
    """Autocast a model's body while forcing its output back to fp32.

    PPO's surrogate is ``exp(logp - logp_old)``: a difference of two nearly-equal
    logs whose magnitude (~30 for a 29-dim Gaussian) leaves bf16 ~0.12 of absolute
    resolution. Recomputing ``logp`` in bf16 against an fp32 ``logp_old`` therefore
    puts ~12% of noise on a ratio that is clipped at 0.2 -- the surrogate becomes
    noise and mean reward collapses. Two invariants keep bf16 usable:

    1. **fp32 head.** Only the matmul body runs under autocast; :meth:`_head_input`
       lifts the body's output back to fp32, so log-probs, the ratio, the KL and
       advantage normalization are all computed in full precision.
    2. **Same arithmetic on both sides.** The dtype is a property of the MODEL, not
       of ``update()``, so collection, the update and inference all run the same
       kernels. ``logp`` and ``logp_old`` then agree to ~1e-3 (measured as
       ``Diagnostics/logp_drift_mb0``) instead of ~1e-1.

    Ordering matters: a model that autocasts only its update path reintroduces (2)
    and is the exact configuration that collapsed. Set the dtype once, via
    :meth:`set_amp_dtype`, and every call site inherits it.
    """

    _amp_dtype: torch.dtype | None = None
    _amp_device: str = "cuda"

    def set_amp_dtype(self, dtype: torch.dtype | None) -> None:
        """Set the body autocast dtype on this model and every :class:`AmpMixin` submodule."""
        device = next((p.device.type for p in self.parameters()), "cuda")  # type: ignore[attr-defined]
        for module in self.modules():  # type: ignore[attr-defined]
            if isinstance(module, AmpMixin):
                module._amp_dtype = dtype
                module._amp_device = device

    def _amp_body(self) -> AbstractContextManager:
        """Autocast context for the matmul body (a no-op when amp is off)."""
        if self._amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self._amp_device, dtype=self._amp_dtype)

    def _head_input(self, x: torch.Tensor) -> torch.Tensor:
        """Lift the body output back to fp32 before the distribution / value head."""
        return x.float() if self._amp_dtype is not None else x


def resolve_amp_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    """Resolve ``"bfloat16"`` / ``"float16"`` / ``None`` (or a dtype) to a torch dtype."""
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    resolved = getattr(torch, dtype)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"amp_dtype must name a torch dtype, got {dtype!r}")
    return resolved


def set_model_amp(model: nn.Module, dtype: str | torch.dtype | None) -> None:
    """Apply an amp dtype to ``model`` if it participates in the amp seam."""
    if isinstance(model, AmpMixin):
        model.set_amp_dtype(resolve_amp_dtype(dtype))
    elif dtype is not None:
        raise TypeError(
            f"{type(model).__name__} does not inherit AmpMixin; amp_dtype would silently "
            "autocast its loss head and corrupt the PPO ratio. Add the mixin or unset amp_dtype."
        )
