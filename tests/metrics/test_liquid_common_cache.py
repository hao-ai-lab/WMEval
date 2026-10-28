# WMEval/tests/test_liquid_common_cache.py
"""Cache-key correctness for `get_piv_field`.

Without these checks, a regression in `_config_hash` (e.g. silently
dropping `mask_hash`) would let stale PIV results survive a SAM2 model
upgrade or a non-deterministic mask refresh. We monkey-patch OpenPIV so
the test runs in milliseconds and stays focused on the cache key — the
numerics of the PIV call itself are covered elsewhere.
"""
from __future__ import annotations
import sys
import types
import pickle
from pathlib import Path
import unittest.mock as mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_mask_result(
    masks: np.ndarray,
    *,
    sam2_model: str = "facebook/sam2.1-hiera-small",
    prompt: str = "the liquid inside the bottle",
    fps: float = 4.0,
):
    from evaluation.modules._liquid_common import LiquidMaskResult
    return LiquidMaskResult(masks=masks, fps=fps, sam2_model=sam2_model, prompt=prompt)


class _StubPyProcess:
    """Records how many times PIV was actually invoked."""

    def __init__(self) -> None:
        self.calls = 0

    def extended_search_area_piv(self, a, b, **_kw):
        self.calls += 1
        H_p = max(1, a.shape[0] // 16)
        W_p = max(1, a.shape[1] // 16)
        u = np.zeros((H_p, W_p), dtype=np.float32)
        v = np.zeros((H_p, W_p), dtype=np.float32)
        s2n = np.full((H_p, W_p), 2.0, dtype=np.float32)
        return u, v, s2n

    def get_coordinates(self, image_size, search_area_size, overlap):
        H, W = image_size
        H_p = max(1, H // 16)
        W_p = max(1, W // 16)
        xs, ys = np.meshgrid(np.arange(W_p) * 16, np.arange(H_p) * 16)
        return xs.astype(np.float32), ys.astype(np.float32)


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    # Cache key uses the file's content hash — bytes are enough; no decode.
    p = tmp_path / "tiny.mp4"
    p.write_bytes(b"fake video bytes for content hash test")
    return p


def _run(stub: _StubPyProcess, **kwargs):
    """Invoke get_piv_field with OpenPIV + frame decode mocked out."""
    from evaluation.modules import _liquid_common as _lc

    fake_openpiv = types.ModuleType("openpiv")
    fake_openpiv.pyprocess = stub  # type: ignore[attr-defined]

    gray = np.zeros((4, 64, 64), dtype=np.uint8)
    with mock.patch.object(_lc, "_gray_uint8_video", return_value=gray), \
         mock.patch.dict(sys.modules, {"openpiv": fake_openpiv, "openpiv.pyprocess": stub}):
        return _lc.get_piv_field(window_size=16, overlap=0, fps=4.0, **kwargs)


def test_second_call_with_identical_inputs_hits_cache(synthetic_video, tmp_path):
    masks = np.ones((4, 64, 64), dtype=bool)
    mask = _make_mask_result(masks)
    stub = _StubPyProcess()
    cache_dir = tmp_path / "cache"

    _run(stub, video_path=str(synthetic_video), mask=mask, cache_dir=cache_dir)
    after_first = stub.calls
    assert after_first > 0, "first call must actually invoke PIV"

    _run(stub, video_path=str(synthetic_video), mask=mask, cache_dir=cache_dir)
    assert stub.calls == after_first, "second call with identical inputs should hit cache"


def test_different_sam2_model_misses_cache(synthetic_video, tmp_path):
    masks = np.ones((4, 64, 64), dtype=bool)
    mask_small = _make_mask_result(masks, sam2_model="facebook/sam2.1-hiera-small")
    mask_large = _make_mask_result(masks, sam2_model="facebook/sam2.1-hiera-large")
    stub = _StubPyProcess()
    cache_dir = tmp_path / "cache"

    _run(stub, video_path=str(synthetic_video), mask=mask_small, cache_dir=cache_dir)
    after_first = stub.calls
    _run(stub, video_path=str(synthetic_video), mask=mask_large, cache_dir=cache_dir)
    assert stub.calls > after_first, "different sam2_model must produce a cache miss"


def test_different_mask_content_misses_cache(synthetic_video, tmp_path):
    """Locks the N3 fix: same sam2_model + prompt strings, different mask bytes ⇒ miss."""
    masks_a = np.ones((4, 64, 64), dtype=bool)
    masks_b = masks_a.copy()
    masks_b[0, 0, 0] = False  # single-bit content change
    mask_a = _make_mask_result(masks_a)
    mask_b = _make_mask_result(masks_b)
    stub = _StubPyProcess()
    cache_dir = tmp_path / "cache"

    _run(stub, video_path=str(synthetic_video), mask=mask_a, cache_dir=cache_dir)
    after_first = stub.calls
    _run(stub, video_path=str(synthetic_video), mask=mask_b, cache_dir=cache_dir)
    assert stub.calls > after_first, (
        "mask CONTENT change must bust the PIV cache even when sam2_model/prompt strings match"
    )


def test_pickle_cache_write_is_atomic_on_serialization_failure(
    synthetic_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluation.modules import _liquid_common as _lc

    cache_dir = tmp_path / "cache"
    mask = _make_mask_result(np.ones((2, 2, 2), dtype=bool), fps=4.0)
    path = _lc.liquid_mask_cache_path(
        video_path=synthetic_video,
        sam2_model=mask.sam2_model,
        prompt=mask.prompt,
        fps=mask.fps,
        max_pairs=None,
        cache_dir=cache_dir,
    )
    path.parent.mkdir(parents=True)
    original = _make_mask_result(np.zeros((2, 2, 2), dtype=bool), fps=4.0)
    with path.open("wb") as handle:
        pickle.dump(original, handle)

    def fail_after_partial_write(payload, handle):
        handle.write(b"partial replacement")
        raise RuntimeError("simulated pickle failure")

    monkeypatch.setattr(_lc.pickle, "dump", fail_after_partial_write)

    with pytest.raises(RuntimeError, match="simulated pickle failure"):
        _lc.store_liquid_mask_cache(
            video_path=synthetic_video,
            result=mask,
            max_pairs=None,
            cache_dir=cache_dir,
        )

    with path.open("rb") as handle:
        loaded = pickle.load(handle)
    np.testing.assert_array_equal(loaded.masks, original.masks)
    assert list(path.parent.iterdir()) == [path]


def test_empty_prompt_uses_raw_prompt_key_with_final_prompt_payload_once(
    synthetic_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluation.modules import _liquid_common as _lc

    class _FakeMaskLogits:
        def __getitem__(self, _index):
            return self

        def __gt__(self, _other):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.ones((1, 2, 2), dtype=np.float32)

    class _FakePredictor:
        def init_state(self, *, video_path):
            return {"video_path": video_path}

        def add_new_points_or_box(self, **_kwargs):
            return None

        def propagate_in_video(self, _state):
            yield 0, [0], _FakeMaskLogits()
            yield 1, [0], _FakeMaskLogits()

    fake_build_sam = types.ModuleType("sam2.build_sam")
    fake_build_sam.build_sam2_video_predictor_hf = lambda *args, **kwargs: _FakePredictor()
    fake_build_sam.build_sam2_video_predictor = lambda *args, **kwargs: _FakePredictor()
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "sam2", types.ModuleType("sam2"))
    monkeypatch.setitem(sys.modules, "sam2.build_sam", fake_build_sam)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        _lc,
        "_read_video_frames_uint8",
        lambda *args, **kwargs: np.zeros((2, 2, 2, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        _lc,
        "_grounding_dino_box_xyxy",
        lambda *args, **kwargs: np.array([0, 0, 1, 1], dtype=np.float32),
    )
    real_pickle_dump = pickle.dump
    dumped_prompts: list[str] = []

    def record_pickle_dump(payload, handle):
        dumped_prompts.append(payload.prompt)
        return real_pickle_dump(payload, handle)

    monkeypatch.setattr(_lc.pickle, "dump", record_pickle_dump)

    cache_dir = tmp_path / "cache"
    result = _lc.get_liquid_mask_video(
        video_path=str(synthetic_video),
        sam2_model="facebook/sam2.1-hiera-small",
        prompt="",
        fps=4.0,
        max_pairs=None,
        cache_dir=cache_dir,
    )

    assert result.prompt == "the liquid inside the bottle"
    assert dumped_prompts == [result.prompt]
    raw_prompt_path = _lc.liquid_mask_cache_path(
        video_path=synthetic_video,
        sam2_model="facebook/sam2.1-hiera-small",
        prompt="",
        fps=4.0,
        max_pairs=None,
        cache_dir=cache_dir,
    )
    final_prompt_path = _lc.liquid_mask_cache_path(
        video_path=synthetic_video,
        sam2_model="facebook/sam2.1-hiera-small",
        prompt=result.prompt,
        fps=4.0,
        max_pairs=None,
        cache_dir=cache_dir,
    )
    assert raw_prompt_path.exists()
    assert not final_prompt_path.exists()
    assert list(raw_prompt_path.parent.iterdir()) == [raw_prompt_path]
    with raw_prompt_path.open("rb") as handle:
        cached = pickle.load(handle)
    assert cached.prompt == result.prompt

    monkeypatch.setattr(
        _lc,
        "_read_video_frames_uint8",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("inference ran")),
    )
    cached_result = _lc.get_liquid_mask_video(
        video_path=str(synthetic_video),
        sam2_model="facebook/sam2.1-hiera-small",
        prompt="",
        fps=4.0,
        max_pairs=None,
        cache_dir=cache_dir,
    )
    assert cached_result.prompt == result.prompt
