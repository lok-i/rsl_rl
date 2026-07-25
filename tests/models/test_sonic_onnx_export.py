# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ONNX-export parity for the SONIC deploy family (base / +LoRA / +extractor).

Every fixture RANDOMIZES the adapter weights and the normalizer statistics before exporting:
adapters are zero-initialized and a fresh normalizer is the identity, so a test on a freshly
built model would pass while merging (or skipping) either one — vacuously green.
"""

from __future__ import annotations

import numpy as np
import tempfile
import torch
from tensordict import TensorDict

import onnx
import onnxruntime as ort
import pytest

from rsl_rl.models import (
    ExtractorSonicAdapterModel,
    SonicBaseModel,
    SonicWithAdapterModel,
)
from rsl_rl.modules import EmpiricalNormalization

NUM_ENVS = 4
PROPRIO_DIM = 12
TOKENIZER_DIM = 10
NUM_ACTIONS = 5
NUM_PATCHES = 6
TOKEN_CHANNELS = 8
AUG_DIM = 7
QUERY_DIM = 4
LATENT_DIM = 6

DIST_CFG = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}
BASE_KWARGS = dict(
    num_tokens=2,
    token_dim=4,
    fsq_levels=[8, 5, 4, 3],  # mixed even/odd -> exercises both FSQ offset branches
    encoder_hidden_dims=[16, 16],
    decoder_hidden_dims=[16, 16],
    activation="SiLU",
)
EXTRACTOR_CFG = {
    "class_name": "rsl_rl.modules.CrossAttentionExtractor",
    "token_terms": ["img_tokens"],
    "query_groups": ["q_task_cmd", "q_proprio"],
    "latent_dim": LATENT_DIM,
    "attn_dim": 8,
    "num_heads": 1,
    "num_learned_queries": 0,
    "layer_norm": True,
}


def _obs(batch: int = NUM_ENVS) -> TensorDict:
    """Observation template covering every group the SONIC family can consume."""
    return TensorDict(
        {
            "policy": torch.randn(batch, PROPRIO_DIM),
            "tokenizer": torch.randn(batch, TOKENIZER_DIM),
            "augmentation": torch.randn(batch, AUG_DIM),
            "kv_tokens": TensorDict(
                {"img_tokens": torch.randn(batch, NUM_PATCHES, TOKEN_CHANNELS)},
                batch_size=[batch],
            ),
            "q_task_cmd": torch.randn(batch, QUERY_DIM),
            "q_proprio": torch.randn(batch, QUERY_DIM),
        },
        batch_size=[batch],
    )


def _randomize(model: torch.nn.Module) -> None:
    """Break the zero-init / identity defaults that would make parity vacuous."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "adapters" in name or "learned_queries" in name:
                param.normal_(0.0, 0.3)
        for module in model.modules():
            if isinstance(module, EmpiricalNormalization):
                module._mean.normal_(0.0, 1.0)
                module._var.uniform_(0.5, 2.0)
                module._std.copy_(module._var.sqrt())
    model.eval()


def _make(kind: str) -> tuple[torch.nn.Module, TensorDict]:
    """Build one member of the SONIC deploy family with randomized weights."""
    obs = _obs()
    common = dict(
        obs=obs,
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=NUM_ACTIONS,
        distribution_cfg=dict(DIST_CFG),
        **BASE_KWARGS,
    )
    if kind == "base":
        model = SonicBaseModel(**common)
    elif kind == "adapter":
        model = SonicWithAdapterModel(**common, adapter_obs_group="augmentation", rank=2)
    elif kind == "extractor":
        model = ExtractorSonicAdapterModel(
            **common,
            adapter_obs_group=["augmentation", "kv_tokens"],
            rank=2,
            extractor_cfg={"kv_tokens": dict(EXTRACTOR_CFG)},
        )
    else:
        raise ValueError(kind)
    _randomize(model)
    return model, obs


def _export(onnx_model: torch.nn.Module, path: str) -> None:
    """Write the ONNX file the same way the runner does (static shapes, opset 18)."""
    torch.onnx.export(
        onnx_model,
        onnx_model.get_dummy_inputs(),
        path,
        export_params=True,
        opset_version=18,
        input_names=onnx_model.input_names,
        output_names=onnx_model.output_names,
        dynamic_axes={},
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(path))


def _feed(onnx_model: torch.nn.Module, obs: TensorDict, index: int) -> dict[str, np.ndarray]:
    """Build the ORT input dict for one environment row, straight from the port table."""
    feed = {}
    for port in onnx_model.layout:
        group = port["groups"][0]
        value = obs[group][port["term"]] if port["term"] else obs[group]
        feed[port["name"]] = value[index : index + 1].numpy().astype(np.float32)
    return feed


def _vectors(onnx_model: torch.nn.Module) -> list[dict[str, torch.Tensor]]:
    """Deterministic probe vectors: zeros, arange, saturating, random."""
    ports = onnx_model.layout
    rows = []
    for kind_index in range(4):
        row = {}
        for port in ports:
            shape = tuple(port["shape"])
            size = int(np.prod(shape))
            if kind_index == 0:  # zeros: biases, normalizer offsets, FSQ constants
                flat = torch.zeros(size)
            elif kind_index == 1:  # arange: ordering / offsets within every port
                flat = torch.arange(size, dtype=torch.float32) / max(size - 1, 1) * 2 - 1
            elif kind_index == 2:  # saturating: FSQ tanh bounds + Round tie-breaking
                flat = torch.full((size,), 6.0)
            else:
                flat = torch.randn(size, generator=torch.Generator().manual_seed(7))
            row[port["name"]] = flat.reshape(shape)
        rows.append(row)
    return rows


@pytest.mark.parametrize("kind", ["base", "adapter", "extractor"])
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestSonicOnnxExport:
    """Torch-vs-ONNX parity across the three deployable SONIC model classes."""

    def test_ports_match_graph(self, kind: str) -> None:
        """Declared ports must be exactly the ONNX graph inputs, in order."""
        model, _ = _make(kind)
        onnx_model = model.as_onnx(verbose=False)
        onnx_model.eval()
        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            _export(onnx_model, f.name)
            graph = onnx.load(f.name).graph
            assert [i.name for i in graph.input] == onnx_model.input_names
            assert [o.name for o in graph.output] == onnx_model.output_names
        assert onnx_model.input_names[0] == "tokenizer"
        assert "policy" in onnx_model.input_names

    def test_parity_on_probe_vectors(self, kind: str) -> None:
        """Zero / arange / saturating / random vectors: torch class == exported ONNX."""
        model, _ = _make(kind)
        onnx_model = model.as_onnx(verbose=False)
        onnx_model.eval()
        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            _export(onnx_model, f.name)
            session = ort.InferenceSession(f.name, providers=["CPUExecutionProvider"])
            for row in _vectors(onnx_model):
                feed = {k: v.unsqueeze(0).numpy() for k, v in row.items()}
                ort_actions = session.run(None, feed)[0]
                with torch.no_grad():
                    torch_actions = onnx_model(*(row[n].unsqueeze(0) for n in onnx_model.input_names))
                if isinstance(torch_actions, tuple):
                    torch_actions = torch_actions[0]
                err = np.abs(torch_actions.numpy() - ort_actions).max()
                assert err < 1e-5, f"{kind}: ONNX/torch mismatch {err:.2e}"

    def test_merge_matches_unmerged_model(self, kind: str) -> None:
        """The exported (adapter-merged, FSQ-folded) graph must match the trained class."""
        model, obs = _make(kind)
        onnx_model = model.as_onnx(verbose=False)
        onnx_model.eval()
        with torch.no_grad():
            reference = model(obs)
            for i in range(NUM_ENVS):
                inputs = [
                    torch.as_tensor(_feed(onnx_model, obs, i)[n]) for n in onnx_model.input_names
                ]
                exported = onnx_model(*inputs)
                if isinstance(exported, tuple):
                    exported = exported[0]
                err = (reference[i : i + 1] - exported).abs().max().item()
                assert err < 1e-5, f"{kind}: merged export diverges from the model, {err:.2e}"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_alias_ports_dedupes_identical_groups() -> None:
    """Aliasing two content-identical groups leaves one input that feeds both slots."""
    obs = _obs()
    obs["q_task_cmd"] = obs["augmentation"].clone()  # same content, same width
    cfg = dict(EXTRACTOR_CFG)
    model = ExtractorSonicAdapterModel(
        obs=obs,
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=NUM_ACTIONS,
        distribution_cfg=dict(DIST_CFG),
        adapter_obs_group=["augmentation", "kv_tokens"],
        rank=2,
        extractor_cfg={"kv_tokens": cfg},
        **BASE_KWARGS,
    )
    _randomize(model)
    onnx_model = model.as_onnx(verbose=False)
    onnx_model.eval()
    before = [torch.as_tensor(_feed(onnx_model, obs, 0)[n]) for n in onnx_model.input_names]
    with torch.no_grad():
        expected = onnx_model(*before)[0]

    onnx_model.alias_ports({"q_task_cmd": "augmentation"})
    assert "q_task_cmd" not in onnx_model.input_names
    merged = next(p for p in onnx_model.layout if p["name"] == "augmentation")
    assert set(merged["groups"]) == {"augmentation", "q_task_cmd"}
    after = [torch.as_tensor(_feed(onnx_model, obs, 0)[n]) for n in onnx_model.input_names]
    with torch.no_grad():
        got = onnx_model(*after)[0]
    assert torch.allclose(expected, got, atol=1e-6)
