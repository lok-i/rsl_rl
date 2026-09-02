# fork surface

This branch extends upstream `rsl_rl` 5.4.1 at
[`016c7ed`](https://github.com/leggedrobotics/rsl_rl/commit/016c7ede710e358b7d6c205642e2540804d6281f).
It carries the policy stack used by `orcs`, `vibe`, and `mocke`; it is not a second
general-purpose model zoo.

## supported

The release path is:

```text
frozen SONIC -> decoder LoRA -> optional cross-attention extractor -> PPO -> one ONNX graph
```

The supported pieces are:

- `SonicBaseModel`: checkpoint-compatible encoder, FSQ tokenization, decoder, per-joint
  exploration scaling, and frozen-token caching
- `SonicWithAdapterModel`: zero-initialized LoRA over the SONIC decoder
- `CrossAttentionExtractor`: one frozen-backbone token term pooled by one or more explicit
  query groups
- `ExtractorSonicAdapterModel`: cross-attention latent plus optional plain conditioning groups
  feeding the SONIC adapter
- monolithic ONNX export: folded LoRA weights, folded FSQ constants, named inputs, optional
  attention outputs, and no Python-side policy wrapper
- the PPO plumbing needed by that stack: nested `TensorDict` storage, model-owned cache hooks,
  model-scoped AMP, and adapter/extractor diagnostics

The configuration exercised by `orcs` and `vibe` is deliberately narrower than every shape
the constructors can express:

- the SONIC base is frozen
- LoRA is attached to the decoder, not the encoder
- one scalar rank is applied to every decoder layer
- the encoder token cache is enabled
- cross-attention reads one token term
- at least one explicit query group is present
- exported observation ports have distinct names and fixed shapes

A plain adapter actor has this shape:

```python
actor = {
    "class_name": "rsl_rl.models.SonicWithAdapterModel",
    "base_checkpoint": "path/to/last_ported.pt",
    "freeze_base": True,
    "adapt_encoder": False,
    "adapt_decoder": True,
    "cache_tokens": True,
    "adapter_obs_group": "augmentation",
    "rank": 16,
    "alpha": 1.0,
}
```

The vision row changes the host and makes the extractor explicit:

```python
actor |= {
    "class_name": "rsl_rl.models.ExtractorSonicAdapterModel",
    "adapter_obs_group": ["augmentation", "kv_tokens"],
    "extractor_cfg": {
        "kv_tokens": {
            "class_name": "rsl_rl.modules.CrossAttentionExtractor",
            "token_terms": ["img_tokens"],
            "query_groups": ["q_task_cmd", "q_proprio"],
            "latent_dim": 128,
            "attn_dim": 64,
            "num_heads": 1,
            "num_learned_queries": 0,
            "layer_norm": True,
        }
    },
}
```

`alpha` is the adapter scale numerator: low-rank adapters use `alpha / rank`. Changing rank
at fixed alpha therefore changes both capacity and update scale.

## bleeding edge

These pieces stay available, but are not the release-critical path.

### sidecar

`Sidecar`, `MLPWithSidecar`, and their model and ModularNorm variants implement a bounded
action residual over a frozen base. The path has been trained and its construction, gradient,
normalization, and composition contracts are tested. It remains experimental.

### textop compatibility

`ModularNormMLP` and `ModularNormMLPWithAdapterModel` remain for frozen TextOp checkpoint
inference. Mocke's compatibility smoke is:

```bash
python scripts/play_textop.py --num_envs 1
```

The command loads the base through an adapter model with every adapter skipped. Frozen
checkpoint inference is supported; reproducing TextOp training is not claimed.

### auxiliary objectives

`PPOAux`, `StateFdAux`, and `LatentFdAux` support Vibe's `-Sfd` and `-Lfd` research rows. They
are usable, but remain experimental and are not required by the base extractor row.

## removed

The retired MLP feature extractor and its generic MLP hosts are not part of this fork:

- `MlpExtractor`
- `ExtractorMLPModel`
- `ExtractorAdapterModel`

The supported extractor is the multi-query cross-attention module hosted by the SONIC adapter.

## feature parity with upstream

Most upstream configuration remains valid. The intentional differences are:

- `PPO.update()` returns `(loss_dict, info_dict)`; the runner and logger consume both
- models may expose `storage_obs()` and `cache_obs()` to replace a frozen input with its cached
  derivation during updates
- rollout storage accepts nested `TensorDict` observation groups
- PPO can set body AMP on models while leaving distribution heads and losses in fp32
- PPO drains adapter, extractor, and precision diagnostics into `info_dict`
- PPO exposes pre/post optimizer hooks used by `PPOAux`
- models with `dualize_gradients()` and `project_weights()` replace ordinary gradient clipping
- adaptive KL leaves optimizer parameter groups marked `fixed_lr` unchanged
- NaN checks recurse through nested observation groups

These changes intentionally live in the fork's PPO path because the current `orcs` and `vibe`
runners depend on them.

## limits

The following constructor combinations are outside the supported release contract:

- adapted SONIC encoder export
- per-layer rank lists that skip the first adapter during ONNX folding
- several token terms whose source identity must remain distinct
- learned-query-only ONNX extraction with no explicit query group
- repeated or overlapping ONNX input names
- dynamic patch counts or other dynamic ONNX axes
- training `MLPWithAdapterModel` or reproducing ModularNorm/TextOp optimization

They are not used by the released Orcs/Vibe configurations.

## checks

Run the fork gates from this directory:

```bash
ruff check .
ruff format --check .
pytest tests/
pytest tests/models/test_sonic_onnx_export.py
```

Mocke owns the real-checkpoint SONIC and TextOp environment smokes. Orcs and Vibe own their
registered task and fresh-install checks.

## upstream

The upstream license, citation, and acknowledgements remain authoritative. Keep fork changes
focused: pull upstream fixes forward, and put task mechanics in `orcs` or perception code in
`vibe` when they do not need to alter the generic learning loop.
