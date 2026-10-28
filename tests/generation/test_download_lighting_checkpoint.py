import json
from pathlib import Path

import pytest

from tools.generation.download_lighting_checkpoint import finalize_shard_indices, sha256


def test_derived_index_normalizes_relative_shards(tmp_path: Path):
    index = tmp_path / "transformer/model.safetensors.index.json"
    index.parent.mkdir()
    index.write_text(json.dumps({"metadata": {"total_size": 80},
                                "weight_map": {"a": "./model-00001.safetensors",
                                               "b": "model-00002.safetensors"}}))
    manifest = {"files": {
        "transformer/model.safetensors.index.json": {},
        "transformer/model-00001.safetensors": {
            "conversion": "float32_to_bfloat16", "tensor_data_bytes": 12},
        "transformer/model-00002.safetensors": {
            "conversion": "float32_to_bfloat16", "tensor_data_bytes": 28},
    }}
    finalize_shard_indices(tmp_path, manifest)
    assert json.loads(index.read_text())["metadata"]["total_size"] == 40
    stored = manifest["files"]["transformer/model.safetensors.index.json"]
    assert stored["stored_sha256"] == sha256(index)
    assert stored["conversion"] == "native_bfloat16_shard_index_metadata"


def test_original_checkpoint_index_remains_unchanged(tmp_path: Path):
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"metadata": {"total_size": 80},
                                "weight_map": {"a": "./model.safetensors"}}))
    original = index.read_bytes()
    manifest = {"files": {"model.safetensors.index.json": {},
                         "model.safetensors": {"conversion": "none"}}}
    finalize_shard_indices(tmp_path, manifest)
    assert index.read_bytes() == original


def test_index_rejects_parent_shard_path(tmp_path: Path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "../model.safetensors"}}))
    with pytest.raises(ValueError, match="unsafe"):
        finalize_shard_indices(tmp_path, {"files": {"model.safetensors.index.json": {}}})
