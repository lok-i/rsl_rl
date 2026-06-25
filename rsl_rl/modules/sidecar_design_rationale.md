# Sidecar Output Bounding: Design Rationale

Why the `Sidecar` module uses `tanh`-squashed output (`output_bound`) instead of linear
scaling, and why this differs from standalone policy conventions.

## Context

Residual Policy Learning (RPL; Silver et al. 2018) adds a trainable sidecar network on top
of a frozen base policy:

```
action = base(obs) + sidecar(sidecar_obs)
```

The sidecar must start at near-zero output (small Xavier head init) so the composed policy
reproduces the base behavior at step 0. This is the defining property of RPL — and the
source of its optimization pathology.

## The two output-bounding strategies

| | **tanh + scale** (`bound * tanh(raw)`) | **just scale** (`scale * raw`) |
|---|---|---|
| range | `[-bound, bound]` hard | `(-inf, +inf)` soft |
| gradient at extremes | vanishes (saturation) | constant |
| bound depends on | architecture | training dynamics |
| stability guarantee | yes | no |

## Why just-scale works for standalone policies

In standard locomotion / whole-body-control with `action_scale=0.25`:

1. Policy outputs non-trivial actions from step 0 (random init, `init_std=1.0`)
2. KL divergence is immediately non-trivial => adaptive LR gets honest signal
3. Reward shaping (action_rate, joint_limits) provides implicit bounding
4. The policy *discovers* its own operating range — no architectural ceiling

This is the reigning convention in locomotion RL. No architectural constraint means no
conservatism, no gradient starvation, full reachability. The policy IS the whole controller.

## Why just-scale fails specifically in sidecar / RPL

The sidecar has a pathological initial condition that standalone policies do not:

1. **Near-zero init is mandatory** — RPL's whole point is starting from the base behavior
2. **Sidecar contribution ~ 0 initially** => total action ~ base action
3. **As `std` decreases through training**, mean changes start dominating KL
4. But the sidecar mean is still small => KL stays artificially low => LR inflates
5. At some point, the accumulated LR is high enough that one update makes the sidecar
   output jump
6. **No bound on the jump size** => catastrophic KL spike => gradient explosion => NaN

Linear scaling (`scale * raw`) only reduces the *rate* of growth — `raw` grows to
compensate. It delays the explosion, does not prevent it.

### Empirical evidence (fcrl, G1 humanoid loco-manipulation)

| variant | output strategy | crash step |
|---|---|---|
| zero-init head, xavier-0.01 trunk | unbounded | ~1k |
| xavier-0.01 head, default trunk | unbounded | ~5k |
| xavier-0.01 head, default trunk, fixed LR + init_std=0.1 | unbounded | ~300 |
| xavier-0.01 head, default trunk, `tanh`, `output_bound=1.0` | bounded | stable (15k+) |

Note: removing adaptive LR (`schedule=fixed`) made things *worse* — the adaptive schedule
was a safety net (reducing LR on KL spikes), not the cause. The cause is the unbounded
output-to-KL sensitivity.

## Lyapunov reachability: why bounding is justified for RPL

**Counterargument**: from classic Lyapunov analysis, to reach a target state `q*`, the
required control input `u` may exceed any predetermined bound. Hard-clamping control
authority limits reachability.

**Rebuttal**: in RPL, the base policy already provides `q*` (approximately). The sidecar
learns only the residual `eps` — the small correction that the base misses. If `eps` needs
to be larger than `output_bound`, the base trajectory is fundamentally wrong for the task
and RPL is the wrong tool (retrain from scratch). `output_bound` encodes the RPL
assumption: "the optimal policy is within `bound` of the base."

## tanh conservatism: acceptable for RPL

The gradient dynamics of `tanh`:

| `raw` | `tanh(raw)` | gradient `(1 - tanh^2)` | note |
|---|---|---|---|
| 0.0 | 0.00 | 1.00 | full gradient |
| 1.0 | 0.76 | 0.42 | moderate |
| 2.0 | 0.96 | 0.07 | 14x weaker |
| 3.0 | 0.99 | 0.01 | gradient-starved |

The effective operating range is ~`[-0.9*bound, 0.9*bound]`. Past 90% of the bound, the
gradient is starved and the policy gets weak signal for pushing further.

**For RPL, this is acceptable.** The residual should be moderate by definition. If the
sidecar consistently saturates at `0.9 * bound`, increase `output_bound` — that is the
signal, not a reason to remove the bound. For standalone locomotion policies, this gradient
starvation *would* be a problem, because the policy needs full authority without
architectural penalty.

## The design bifurcation

| | standalone policy | sidecar / RPL |
|---|---|---|
| init condition | random (non-trivial from step 0) | near-zero (mandatory) |
| KL-LR dynamics | well-conditioned | pathological without bounding |
| reachability needs | full (policy IS the controller) | moderate (small corrections) |
| conservatism tolerance | low (need full authority) | high (residuals should be small) |
| **right choice** | **just-scale** | **tanh** |

This bifurcation follows from the **initialization regime**, not aesthetic preference.

## Implementation

```python
class Sidecar(nn.Module):
    def __init__(self, ..., output_bound: float | None = None):
        self.output_bound = output_bound
        # head: Xavier-uniform with small gain (default 0.01)
        # trunk: PyTorch default init

    def forward(self, x):
        raw = self.head(self.trunk(x))
        if self.output_bound is not None:
            return self.output_bound * torch.tanh(raw)
        return raw
```

- `output_bound = None`: unbounded (standalone-policy convention, default in rsl_rl)
- `output_bound = 1.0`: bounded residual in `[-1, 1]` (RPL convention, set in task cfg)

The bound is set in the **task agent cfg**, not as a library default, so the user
explicitly declares their intent.

## References

- Silver, T. et al. (2018). *Residual Policy Learning.* arXiv:1812.06298
- Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685 (alpha/rank scaling analogy)
- Luo, J. et al. (2023). *Perpetual Humanoid Control for Real-time Simulated Avatars.* (ResMimic, xavier-0.01 head init)
