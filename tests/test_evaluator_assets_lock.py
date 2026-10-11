from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

LOCK = json.loads(
    (Path(__file__).resolve().parents[1] / "experiments/tables_4_7/evaluator_assets.lock.json").read_text(
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
