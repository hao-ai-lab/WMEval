from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

LOCK = json.loads(
    (Path(__file__).resolve().parents[2] / "experiments/protocol/evaluator_assets.lock.json").read_text(
        encoding="utf-8"
    )
)["assets"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_raft_weights_are_pinned_by_name_and_match_the_lock() -> None:
    from torchvision.models.optical_flow import Raft_Large_Weights

    from evaluation.modules.flow_utils import RAFT_WEIGHTS_NAME

    locked = LOCK["raft_large"]
    assert RAFT_WEIGHTS_NAME == locked["weights_enum"]
    # torchvision names checkpoints <model>_<enum>-<first 8 hex of sha256>.pth
    filename = Raft_Large_Weights[RAFT_WEIGHTS_NAME].url.rsplit("/", 1)[-1]
    assert filename == locked["weight"]
    assert locked["sha256"].startswith(filename.rsplit("-", 1)[-1].removesuffix(".pth"))


def test_cached_raft_checkpoint_matches_the_locked_hash() -> None:
    path = Path(os.path.expanduser("~/.cache/torch/hub/checkpoints")) / LOCK["raft_large"]["weight"]
    if not path.is_file():
        pytest.skip("RAFT checkpoint not cached on this machine")
    assert _sha256(path) == LOCK["raft_large"]["sha256"]


def test_lpips_linear_weights_match_the_locked_hash() -> None:
    lpips = pytest.importorskip("lpips")
    path = Path(lpips.__file__).parent / "weights" / "v0.1" / "vgg.pth"
    assert path.as_posix().endswith(LOCK["lpips_vgg"]["linear_weight"])
    assert _sha256(path) == LOCK["lpips_vgg"]["linear_sha256"]


def test_cached_vgg16_backbone_matches_the_locked_hash() -> None:
    locked = LOCK["lpips_vgg"]
    path = Path(os.path.expanduser("~/.cache/torch/hub/checkpoints")) / locked["backbone_weight"]
    if not path.is_file():
        pytest.skip("VGG16 backbone not cached on this machine")
    assert _sha256(path) == locked["backbone_sha256"]
    assert locked["backbone_sha256"].startswith(locked["backbone_weight"].rsplit("-", 1)[-1].removesuffix(".pth"))


def _hf_cached_weight(locked: dict) -> Path:
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    repo = "models--" + locked["model_id"].replace("/", "--")
    return Path(hub) / repo / "snapshots" / locked["revision"] / locked["weight"]


def test_depth_model_ids_match_the_evaluator_constants() -> None:
    from evaluation.modules.environment_object_consistency import (
        DEPTH_AUTO_SELECT_CANDIDATES,
        DEPTH_DEFAULT_MODEL,
        DEPTH_METRIC_INDOOR_MODEL,
    )

    assert LOCK["depth_anything_metric_outdoor"]["model_id"] == DEPTH_DEFAULT_MODEL
    assert LOCK["depth_anything_metric_indoor"]["model_id"] == DEPTH_METRIC_INDOOR_MODEL
    assert set(DEPTH_AUTO_SELECT_CANDIDATES) == {DEPTH_DEFAULT_MODEL, DEPTH_METRIC_INDOOR_MODEL}


@pytest.mark.parametrize(
    "name",
    [
        "depth_anything_metric_outdoor",
        "depth_anything_metric_indoor",
        "depth_anything_relative_small",
        "sam2_small",
    ],
)
def test_hf_pins_are_complete_and_cached_weights_match(name: str) -> None:
    locked = LOCK[name]
    assert len(locked["revision"]) == 40 and len(locked["sha256"]) == 64
    path = _hf_cached_weight(locked)
    if not path.is_file():
        pytest.skip(f"{locked['model_id']}@{locked['revision'][:8]} not cached on this machine")
    assert _sha256(path) == locked["sha256"]


def test_relative_depth_and_sam2_small_pins_match_code_defaults() -> None:
    root = Path(__file__).resolve().parents[2] / "evaluation" / "modules"
    synthetic = (root / "vendor" / "ptlflow_synthetic" / "synthetic_flow.py").read_text(encoding="utf-8")
    assert f'model_name: str = "{LOCK["depth_anything_relative_small"]["model_id"]}"' in synthetic
    for module in ("liquid_velocity", "liquid_volume_conservation", "liquid_surface_orientation", "slosh_spectrum"):
        source = (root / f"{module}.py").read_text(encoding="utf-8")
        assert f'"liquid_sam2_model", "{LOCK["sam2_small"]["model_id"]}"' in source


def test_vendored_ptlflow_files_match_the_lock() -> None:
    locked = LOCK["ptlflow_synthetic"]
    root = Path(__file__).resolve().parents[2] / locked["vendored_path"]
    vendored = {p.name for p in root.iterdir() if p.is_file()}
    assert vendored == set(locked["files"]) | {locked["notice"]}
    for name, pins in locked["files"].items():
        data = (root / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == pins["sha256"]
        # Upstream blob id ties the vendored bytes to source_revision's tree.
        assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == pins["blob"]
    assert _sha256(root / locked["notice"]) == locked["notice_sha256"]
    assert locked["source"] in (root / locked["notice"]).read_text(encoding="utf-8")


def test_openpiv_pin_matches_the_macos_lock_and_the_installed_package() -> None:
    import importlib.metadata as md

    locked = LOCK["openpiv"]
    lock_file = Path(__file__).resolve().parents[2] / "requirements/macos.lock"
    pinned = f"{locked['import_name']}=={locked['version']}"
    assert pinned in lock_file.read_text(encoding="utf-8").splitlines()
    try:
        installed = md.version(locked["package"])
    except md.PackageNotFoundError:
        pytest.skip("OpenPIV not installed")
    assert installed == locked["version"]
