from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pytest

from evaluation.lite.case_loader import LiteCase
from evaluation.modules import _liquid_common as _lc
from evaluation.modules._liquid_common import BottleTrace, LiquidMaskResult, PIVField


SAM2_MODEL = "facebook/sam2.1-hiera-small"
LIQUID_PROMPT = "the liquid inside the bottle"
BOTTLE_PROMPT = "the glass bottle"


def _mask(prompt: str = LIQUID_PROMPT) -> LiquidMaskResult:
    return LiquidMaskResult(
        masks=np.array([[[True, False], [False, True]]], dtype=bool),
        fps=16.0,
        sam2_model=SAM2_MODEL,
        prompt=prompt,
    )


def _field() -> PIVField:
    return PIVField(
        u=np.array([[[1.0]]], dtype=np.float32),
        v=np.array([[[2.0]]], dtype=np.float32),
        valid=np.array([[[True]]], dtype=bool),
    )


def _trace() -> BottleTrace:
    return BottleTrace(
        centroid_xy=np.array([[1.25, 2.5], [3.75, 4.0]], dtype=np.float64),
        bbox_xyxy=np.array(
            [[0.0, 1.0, 4.0, 5.0], [1.0, 2.0, 5.0, 6.0]], dtype=np.float64
        ),
        valid=np.array([True, False], dtype=bool),
    )


def _case(tmp_path: Path, *, fps: float = 16.0) -> LiteCase:
    root = tmp_path / "case"
    reference_video = root / "reference" / "reference.mp4"
    reference_video.parent.mkdir(parents=True)
    reference_video.write_bytes(b"stable reference bytes")
    return LiteCase(
        root=root,
        schema_version=1,
        case_id="liquid_container/rest",
        scene="liquid_container",
        display_name="Liquid Rest",
        modules=(
            "liquid_velocity",
            "liquid_volume_conservation",
            "liquid_surface_orientation",
            "slosh_spectrum",
        ),
        reference_video=reference_video,
        baseline_path=root / "reference" / "baseline.json",
        checksums_path=root / "reference" / "checksums.sha256",
        protocol={
            "candidate_normalization": "liquid_832x464_16fps",
            "min_duration_sec": 1.0,
            "min_duration_ratio": 0.5,
            "duration_threshold_mode": "max",
        },
        evaluator_lock={
            "release_id": "wmeval-lite-v1",
            "code_revision": "test",
            "evaluator_fingerprint": "test",
            "module_config": {
                "liquid": {
                    "normalization_fps": fps,
                    "sam2_model": SAM2_MODEL,
                    "liquid_prompt": LIQUID_PROMPT,
                    "bottle_prompt": BOTTLE_PROMPT,
                    "max_pairs": 512,
                    "liquid_max_pairs": 0,
                    "raft": {
                        "window_size": 32,
                        "overlap": 16,
                        "min_mask_frac": 0.5,
                    },
                }
            },
            "artifact_schema_versions": {},
            "dependency_versions": {},
        },
    )


def _legacy_cache_path(
    cache_dir: Path, video: Path, kind: str, config: dict[str, object]
) -> Path:
    content_hash = hashlib.sha256(video.read_bytes()).hexdigest()[:16]
    items = sorted((str(key), str(value)) for key, value in config.items())
    config_hash = hashlib.sha256(repr(items).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{video.stem}.{kind}.{content_hash}.{config_hash}.pkl"


def test_portable_artifacts_round_trip_without_pickle(tmp_path: Path) -> None:
    from evaluation.lite.reference_artifacts import (
        read_bottle_trace,
        read_liquid_mask,
        read_piv_field,
        write_bottle_trace,
        write_liquid_mask,
        write_piv_field,
    )

    mask = _mask()
    trace = _trace()
    field = _field()
    paths = {
        "mask": tmp_path / "nested" / "mask.npz",
        "trace": tmp_path / "nested" / "trace.npz",
        "field": tmp_path / "nested" / "field.npz",
    }

    write_liquid_mask(paths["mask"], mask)
    write_bottle_trace(paths["trace"], trace)
    write_piv_field(paths["field"], field)

    loaded_mask = read_liquid_mask(paths["mask"])
    loaded_trace = read_bottle_trace(paths["trace"])
    loaded_field = read_piv_field(paths["field"])
    np.testing.assert_array_equal(loaded_mask.masks, mask.masks)
    assert (loaded_mask.fps, loaded_mask.sam2_model, loaded_mask.prompt) == (
        mask.fps,
        mask.sam2_model,
        mask.prompt,
    )
    np.testing.assert_array_equal(loaded_trace.centroid_xy, trace.centroid_xy)
    np.testing.assert_array_equal(loaded_trace.bbox_xyxy, trace.bbox_xyxy)
    np.testing.assert_array_equal(loaded_trace.valid, trace.valid)
    np.testing.assert_array_equal(loaded_field.u, field.u)
    np.testing.assert_array_equal(loaded_field.v, field.v)
    np.testing.assert_array_equal(loaded_field.valid, field.valid)
    assert loaded_mask.masks.dtype == np.bool_
    assert loaded_trace.centroid_xy.dtype == np.float64
    assert loaded_trace.bbox_xyxy.dtype == np.float64
    assert loaded_trace.valid.dtype == np.bool_
    assert loaded_field.u.dtype == np.float32
    assert loaded_field.v.dtype == np.float32
    assert loaded_field.valid.dtype == np.bool_

    for path in paths.values():
        with np.load(path, allow_pickle=False) as archive:
            assert all(array.dtype.kind != "O" for array in archive.values())


@pytest.mark.parametrize(
    ("reader_name", "arrays", "message"),
    [
        ("read_liquid_mask", {"masks": np.ones((1, 2, 2), dtype=bool)}, "liquid mask"),
        (
            "read_bottle_trace",
            {
                "centroid_xy": np.ones((1, 2), dtype=np.float64),
                "bbox_xyxy": np.ones((1, 4), dtype=np.float64),
            },
            "bottle trace",
        ),
        (
            "read_piv_field",
            {
                "u": np.ones((1, 1, 1), dtype=np.float32),
                "v": np.ones((1, 1, 1), dtype=np.float32),
            },
            "PIV field",
        ),
    ],
)
def test_portable_artifacts_reject_missing_keys(
    tmp_path: Path, reader_name: str, arrays: dict[str, np.ndarray], message: str
) -> None:
    from evaluation.lite import reference_artifacts

    path = tmp_path / "bad.npz"
    np.savez_compressed(path, **arrays)
    with pytest.raises(reference_artifacts.ReferenceArtifactError, match=message):
        getattr(reference_artifacts, reader_name)(path)


@pytest.mark.parametrize(
    ("reader_name", "arrays", "artifact"),
    [
        (
            "read_liquid_mask",
            {
                "masks": np.ones((1, 2, 2), dtype=bool),
                "fps": np.array(16.0, dtype=np.float64),
                "sam2_model": np.array("sam2"),
                "prompt": np.array("liquid"),
            },
            "liquid mask",
        ),
        (
            "read_bottle_trace",
            {
                "centroid_xy": np.ones((1, 2), dtype=np.float64),
                "bbox_xyxy": np.ones((1, 4), dtype=np.float64),
                "valid": np.ones((1,), dtype=bool),
            },
            "bottle trace",
        ),
        (
            "read_piv_field",
            {
                "u": np.ones((1, 1, 1), dtype=np.float32),
                "v": np.ones((1, 1, 1), dtype=np.float32),
                "valid": np.ones((1, 1, 1), dtype=bool),
            },
            "PIV field",
        ),
    ],
)
def test_portable_artifacts_reject_unexpected_keys(
    tmp_path: Path,
    reader_name: str,
    arrays: dict[str, np.ndarray],
    artifact: str,
) -> None:
    from evaluation.lite import reference_artifacts

    path = tmp_path / "extra.npz"
    np.savez_compressed(path, **arrays, unexpected=np.array(1, dtype=np.int64))
    with pytest.raises(
        reference_artifacts.ReferenceArtifactError,
        match=rf"{artifact} artifact .*unexpected keys \['unexpected'\]",
    ):
        getattr(reference_artifacts, reader_name)(path)


def test_portable_artifacts_reject_unexpected_object_array_before_loading(
    tmp_path: Path,
) -> None:
    from evaluation.lite.reference_artifacts import ReferenceArtifactError, read_liquid_mask

    path = tmp_path / "hidden-object.npz"
    np.savez_compressed(
        path,
        masks=np.ones((1, 2, 2), dtype=bool),
        fps=np.array(16.0, dtype=np.float64),
        sam2_model=np.array("sam2"),
        prompt=np.array("liquid"),
        hidden=np.array([{"unsafe": True}], dtype=object),
    )
    with pytest.raises(
        ReferenceArtifactError,
        match=r"liquid mask artifact .*unexpected keys \['hidden'\]",
    ):
        read_liquid_mask(path)


@pytest.mark.parametrize(
    ("reader_name", "arrays", "detail"),
    [
        (
            "read_liquid_mask",
            {
                "masks": np.ones((1, 2, 2), dtype=np.uint8),
                "fps": np.array(16.0, dtype=np.float64),
                "sam2_model": np.array("sam2"),
                "prompt": np.array("liquid"),
            },
            "masks",
        ),
        (
            "read_bottle_trace",
            {
                "centroid_xy": np.ones((1, 3), dtype=np.float64),
                "bbox_xyxy": np.ones((1, 4), dtype=np.float64),
                "valid": np.ones((1,), dtype=bool),
            },
            "centroid_xy",
        ),
        (
            "read_piv_field",
            {
                "u": np.ones((1, 1, 1), dtype=np.float64),
                "v": np.ones((1, 1, 1), dtype=np.float32),
                "valid": np.ones((1, 1, 1), dtype=bool),
            },
            "u",
        ),
    ],
)
def test_portable_artifacts_reject_wrong_dtype_or_shape(
    tmp_path: Path, reader_name: str, arrays: dict[str, np.ndarray], detail: str
) -> None:
    from evaluation.lite import reference_artifacts

    path = tmp_path / "bad.npz"
    np.savez_compressed(path, **arrays)
    with pytest.raises(reference_artifacts.ReferenceArtifactError, match=detail):
        getattr(reference_artifacts, reader_name)(path)


def test_portable_artifacts_reject_object_arrays(tmp_path: Path) -> None:
    from evaluation.lite.reference_artifacts import ReferenceArtifactError, read_liquid_mask

    path = tmp_path / "unsafe.npz"
    np.savez_compressed(
        path,
        masks=np.ones((1, 2, 2), dtype=bool),
        fps=np.array(16.0, dtype=np.float64),
        sam2_model=np.array(["sam2"], dtype=object),
        prompt=np.array("liquid"),
    )
    with pytest.raises(ReferenceArtifactError, match="liquid mask"):
        read_liquid_mask(path)


def test_portable_artifacts_reject_corrupt_npz_with_specific_error(tmp_path: Path) -> None:
    from evaluation.lite.reference_artifacts import ReferenceArtifactError, read_piv_field

    path = tmp_path / "corrupt.npz"
    path.write_bytes(b"PK\x03\x04not a valid npz archive")
    with pytest.raises(ReferenceArtifactError, match="PIV field"):
        read_piv_field(path)


def test_cache_path_helpers_match_legacy_keys(tmp_path: Path) -> None:
    video = tmp_path / "reference.mp4"
    video.write_bytes(b"stable reference bytes")
    mask = _mask()
    cache_dir = tmp_path / "cache"

    expected_mask = _legacy_cache_path(
        cache_dir,
        video,
        "sam2_mask",
        {
            "sam2_model": mask.sam2_model,
            "prompt": mask.prompt,
            "fps": mask.fps,
            "max_pairs": None,
            "gdino_model": _lc._GDINO_MODEL_ID,
            "gdino_box_threshold": _lc._GDINO_BOX_THRESHOLD,
            "gdino_text_threshold": _lc._GDINO_TEXT_THRESHOLD,
        },
    )
    expected_raft = _legacy_cache_path(
        cache_dir,
        video,
        "raft_pooled",
        {
            "estimator": "raft_pooled_v1",
            "window_size": 32,
            "overlap": 16,
            "fps": 16.0,
            "max_frames": None,
            "sam2_model": mask.sam2_model,
            "prompt": mask.prompt,
            "mask_hash": hashlib.sha256(
                np.ascontiguousarray(mask.masks).tobytes()
            ).hexdigest()[:16],
            "min_mask_frac": 0.5,
        },
    )

    assert _lc.liquid_mask_cache_path(
        video_path=video,
        sam2_model=mask.sam2_model,
        prompt=mask.prompt,
        fps=mask.fps,
        max_pairs=None,
        cache_dir=cache_dir,
    ) == expected_mask
    assert _lc.raft_flow_cache_path(
        video_path=video,
        mask=mask,
        window_size=32,
        overlap=16,
        fps=16.0,
        max_frames=None,
        cache_dir=cache_dir,
    ) == expected_raft


def test_store_helpers_hit_existing_getters_without_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "reference.mp4"
    video.write_bytes(b"stable reference bytes")
    cache_dir = tmp_path / "cache"
    mask = _mask()
    field = _field()

    mask_path = _lc.store_liquid_mask_cache(
        video_path=video, result=mask, max_pairs=None, cache_dir=cache_dir
    )
    flow_path = _lc.store_raft_flow_cache(
        video_path=video,
        mask=mask,
        result=field,
        window_size=32,
        overlap=16,
        fps=16.0,
        max_frames=None,
        cache_dir=cache_dir,
        min_mask_frac=0.5,
    )
    monkeypatch.setattr(
        _lc,
        "_read_video_frames_uint8",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("inference ran")),
    )

    assert mask_path.exists()
    assert flow_path.exists()
    loaded_mask = _lc.get_liquid_mask_video(
        video_path=str(video),
        sam2_model=mask.sam2_model,
        prompt=mask.prompt,
        fps=mask.fps,
        max_pairs=None,
        cache_dir=cache_dir,
    )
    loaded_field = _lc.get_raft_flow_field(
        video_path=str(video),
        mask=mask,
        window_size=32,
        overlap=16,
        fps=16.0,
        max_frames=None,
        cache_dir=cache_dir,
        min_mask_frac=0.5,
    )
    np.testing.assert_array_equal(loaded_mask.masks, mask.masks)
    np.testing.assert_array_equal(loaded_field.u, field.u)


def test_relevant_cache_key_changes_miss_and_same_inputs_hit(tmp_path: Path) -> None:
    video = tmp_path / "reference.mp4"
    video.write_bytes(b"stable reference bytes")
    cache_dir = tmp_path / "cache"
    mask = _mask()
    field = _field()

    stored_mask = _lc.store_liquid_mask_cache(
        video_path=video, result=mask, max_pairs=None, cache_dir=cache_dir
    )
    stored_flow = _lc.store_raft_flow_cache(
        video_path=video,
        mask=mask,
        result=field,
        window_size=32,
        overlap=16,
        fps=16.0,
        max_frames=None,
        cache_dir=cache_dir,
    )
    assert stored_mask == _lc.liquid_mask_cache_path(
        video_path=video,
        sam2_model=mask.sam2_model,
        prompt=mask.prompt,
        fps=16.0,
        max_pairs=None,
        cache_dir=cache_dir,
    )
    assert stored_flow == _lc.raft_flow_cache_path(
        video_path=video,
        mask=mask,
        window_size=32,
        overlap=16,
        fps=16.0,
        max_frames=None,
        cache_dir=cache_dir,
    )
    assert stored_mask != _lc.liquid_mask_cache_path(
        video_path=video,
        sam2_model=mask.sam2_model,
        prompt="different",
        fps=16.0,
        max_pairs=None,
        cache_dir=cache_dir,
    )
    changed_masks = mask.masks.copy()
    changed_masks[0, 0, 0] = ~changed_masks[0, 0, 0]
    changed_mask = LiquidMaskResult(
        masks=changed_masks,
        fps=mask.fps,
        sam2_model=mask.sam2_model,
        prompt=mask.prompt,
    )
    assert stored_flow != _lc.raft_flow_cache_path(
        video_path=video,
        mask=changed_mask,
        window_size=32,
        overlap=16,
        fps=16.0,
        max_frames=None,
        cache_dir=cache_dir,
    )


def test_install_reference_artifacts_populates_existing_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluation.lite.reference_artifacts import (
        install_liquid_reference_artifacts,
        write_liquid_mask,
        write_piv_field,
    )

    case = _case(tmp_path)
    liquid = _mask()
    bottle = _mask(BOTTLE_PROMPT)
    field = _field()
    write_liquid_mask(case.artifacts_dir / "liquid_mask.npz", liquid)
    write_liquid_mask(case.artifacts_dir / "bottle_mask.npz", bottle)
    write_piv_field(case.artifacts_dir / "raft_pooled_flow.npz", field)
    cache_dir = tmp_path / "cache"

    install_liquid_reference_artifacts(case, cache_dir)
    monkeypatch.setattr(
        _lc,
        "_read_video_frames_uint8",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("inference ran")),
    )
    for prompt, expected in ((LIQUID_PROMPT, liquid), (BOTTLE_PROMPT, bottle)):
        loaded = _lc.get_liquid_mask_video(
            video_path=str(case.reference_video),
            sam2_model=SAM2_MODEL,
            prompt=prompt,
            fps=16.0,
            max_pairs=None,
            cache_dir=cache_dir,
        )
        np.testing.assert_array_equal(loaded.masks, expected.masks)
    loaded_field = _lc.get_raft_flow_field(
        video_path=str(case.reference_video),
        mask=liquid,
        window_size=32,
        overlap=16,
        fps=16.0,
        max_frames=None,
        cache_dir=cache_dir,
        min_mask_frac=0.5,
    )
    np.testing.assert_array_equal(loaded_field.u, field.u)


def test_install_reference_artifacts_rejects_metadata_mismatch(tmp_path: Path) -> None:
    from evaluation.lite.reference_artifacts import (
        ReferenceArtifactError,
        install_liquid_reference_artifacts,
        write_liquid_mask,
        write_piv_field,
    )

    case = _case(tmp_path, fps=15.0)
    write_liquid_mask(case.artifacts_dir / "liquid_mask.npz", _mask())
    write_liquid_mask(case.artifacts_dir / "bottle_mask.npz", _mask(BOTTLE_PROMPT))
    write_piv_field(case.artifacts_dir / "raft_pooled_flow.npz", _field())

    with pytest.raises(ReferenceArtifactError, match=r"normalization_fps.*15.0.*16.0"):
        install_liquid_reference_artifacts(case, tmp_path / "cache")


def test_legacy_pickle_at_helper_path_is_read_without_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "reference.mp4"
    video.write_bytes(b"stable reference bytes")
    mask = _mask()
    path = _lc.liquid_mask_cache_path(
        video_path=video,
        sam2_model=mask.sam2_model,
        prompt=mask.prompt,
        fps=mask.fps,
        max_pairs=None,
        cache_dir=tmp_path / "cache",
    )
    path.parent.mkdir(parents=True)
    with path.open("wb") as handle:
        pickle.dump(mask, handle)
    monkeypatch.setattr(
        _lc,
        "_read_video_frames_uint8",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("inference ran")),
    )
    loaded = _lc.get_liquid_mask_video(
        video_path=str(video),
        sam2_model=mask.sam2_model,
        prompt=mask.prompt,
        fps=mask.fps,
        max_pairs=None,
        cache_dir=tmp_path / "cache",
    )
    np.testing.assert_array_equal(loaded.masks, mask.masks)
