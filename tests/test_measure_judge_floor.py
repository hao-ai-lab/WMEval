from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tools.dataset import measure_judge_floor as mjf
def _row(dim, case, seed, status="completed", model="m", **metrics):
    return {"dimension": dim, "case_id": case, "model": model, "seed": seed,
            "status": status, "metrics": metrics}


def test_repeat_seeds_round_trip() -> None:
    assert mjf.repeat_seed("run3", 2) == "run3@r2"
    assert mjf.base_seed("run3@r2") == "run3"
    assert mjf.base_seed("run3") == "run3"
    assert mjf.base_seed("a@rb") == "a@rb"  # not a repeat suffix


def test_repeat_rows_gives_causal_rows_a_fresh_cache_per_repeat(tmp_path: Path) -> None:
    rows = [{"model": "unity", "case_id": "BeamDomino/causal", "seed": "self", "candidate_video": "/v.mp4"}]
    out = mjf.repeat_rows(rows, 3, dimension="physical_causality", cache_root=tmp_path)
    assert [r["seed"] for r in out] == ["self@r0", "self@r1", "self@r2"]
    caches = {r["params"][mjf.CAUSAL_CACHE_PARAM] for r in out}
    assert len(caches) == 3 and all(c.startswith(str(tmp_path)) for c in caches)
    # Other dimensions carry no cache param and keep existing params.
    plain = mjf.repeat_rows([{**rows[0], "params": {"x": 1}}], 2, dimension="lpips")
    assert all(r["params"] == {"x": 1} for r in plain)
    with pytest.raises(ValueError):
        mjf.repeat_rows(rows, 2, dimension="physical_causality")
    with pytest.raises(ValueError):
        mjf.repeat_rows(rows, 0, dimension="lpips")


def test_select_synthetic_takes_first_n_judge_trajectories_per_case() -> None:
    def t(family, case, slug, dims=("lpips",), status="unity_rendered"):
        return {"family": family, "conditioning_case": case, "synthetic_id": f"{family}/{slug}",
                "dimensions": list(dims), "reference_status": status}

    trajectories = [
        t("ball_force", "A", "s0"), t("ball_force", "A", "s1"), t("ball_force", "A", "s2"),
        t("ball_force", "B", "s3", status="pending_unity_render"), t("ball_force", "B", "s4"),
        t("camera_yaw", "C", "s5", dims=("environment_object_consistency",)),  # no judge dim
        t("character", "D", "s6"),  # family without a judge dimension in scope
    ]
    picked = [x["synthetic_id"] for x in mjf.select_synthetic(trajectories, per_case=2)]
    assert picked == ["ball_force/s0", "ball_force/s1", "ball_force/s4"]


def test_mdd_formula() -> None:
    assert mjf.mdd(None, 5) is None
    assert mjf.mdd(0.1, 1) == pytest.approx(1.96 * math.sqrt(2) * 0.1, abs=1e-3)
    assert mjf.mdd(0.1, 5) == pytest.approx(1.96 * math.sqrt(0.4) * 0.1, abs=1e-3)


def test_pooled_std_ignores_single_samples() -> None:
    assert mjf.pooled_std([0.1, 0.3, None], [3, 3, 1]) == pytest.approx(math.sqrt((0.01 + 0.09) / 2), abs=1e-4)
    assert mjf.pooled_std([None], [1]) is None


def test_summarize_selfmatch_candidates_and_stored() -> None:
    dim = "prompt_scene_consistency"
    selfmatch = [_row(dim, "X/prompt_first", f"self@r{k}", model="unity", overall_score=v)
                 for k, v in enumerate([1.0, 0.9, 0.8])]
    selfmatch.append(_row(dim, "X/prompt_first", "self@r3", status="error", model="unity"))
    candidates = [
        _row(dim, "X/prompt_first", "run0@r0", overall_score=0.6),
        _row(dim, "X/prompt_first", "run0@r1", overall_score=0.8),
        _row(dim, "X/prompt_first", "run1@r0", overall_score=0.5),
        _row(dim, "X/prompt_first", "run1@r1", overall_score=0.5),
    ]
    stored = [_row(dim, "X/prompt_first", "run0", overall_score=0.7),
              _row(dim, "X/prompt_first", "run1", overall_score=0.9)]
    report = mjf.summarize(selfmatch, candidates, {"old": stored}, seeds_for_mdd=5)

    floor = report["selfmatch"][dim]["X/prompt_first"]
    assert floor["status"] == {"completed": 3, "error": 1}
    assert floor["metrics"]["overall_score"]["mean"] == pytest.approx(0.9)
    assert floor["metrics"]["overall_score"]["std"] == pytest.approx(0.1)
    assert floor["mdd"]["n5"] == pytest.approx(1.96 * math.sqrt(0.4) * 0.1, abs=1e-3)

    case = report["candidates"][dim]["m"]["X/prompt_first"]
    summary = case["summary"]["overall_score"]
    assert summary["videos"] == 2
    # run0 std = 0.1414, run1 std = 0 -> pooled sqrt(0.02 / 2) = 0.1
    assert summary["judge_std_pooled"] == pytest.approx(0.1, abs=1e-4)
    assert summary["mean_of_video_means"] == pytest.approx(0.6)
    vs = case["vs_stored"]["old"]
    assert vs["videos"] == 2
    assert vs["stored_mean"] == pytest.approx(0.8)
    assert vs["rescored_mean"] == pytest.approx(0.6)
    assert vs["stored_within_rescored_range"] == "1/2"
    floor_row = report["floors"][dim]["X/prompt_first"]
    assert floor_row["self_std"] == pytest.approx(0.1)
    assert floor_row["candidate_judge_std"] == pytest.approx(0.1, abs=1e-4)
    assert floor_row["sigma_judge"] == pytest.approx(0.1, abs=1e-4)
    slim = mjf.compact(report)["candidates"][dim]["m"]["X/prompt_first"]
    assert "videos" not in slim and "per_seed" not in slim["vs_stored"]["old"]


def test_cli_plan_candidates_and_summarize(tmp_path: Path) -> None:
    manifest = tmp_path / "cands.jsonl"
    manifest.write_text("".join(json.dumps({"model": "m", "case_id": c, "seed": "run0",
                                            "candidate_video": f"/{c}.mp4"}) + "\n"
                                for c in ("A/prompt_first", "B/prompt_first")))
    out = tmp_path / "plan"
    assert mjf.main(["plan", "--set", "candidates", "--from", str(manifest), "--repeats", "2",
                     "--dimensions", "prompt_scene_consistency", "--cases", "A/prompt_first",
                     "--out", str(out)]) == 0
    rows = [json.loads(x) for x in (out / "prompt_scene_consistency.jsonl").read_text().splitlines()]
    assert [(r["case_id"], r["seed"]) for r in rows] == [("A/prompt_first", "run0@r0"), ("A/prompt_first", "run0@r1")]

    results = tmp_path / "per_case_results.jsonl"
    results.write_text("".join(json.dumps(_row("prompt_scene_consistency", r["case_id"], r["seed"],
                                               overall_score=0.5 + 0.1 * i)) + "\n"
                               for i, r in enumerate(rows)))
    summary_path = tmp_path / "floor.json"
    assert mjf.main(["summarize", "--candidates", str(results), "--output", str(summary_path)]) == 0
    summary = json.loads(summary_path.read_text())
    entry = summary["candidates"]["prompt_scene_consistency"]["m"]["A/prompt_first"]
    assert entry["videos"]["run0"]["metrics"]["overall_score"]["n"] == 2


def test_lpips_event_offset_is_reported_in_seconds() -> None:
    def lp_row(seed, e_cand):
        row = _row("lpips", "C/x", seed, lpips_mean=0.1, lpips_primary_source="causal")
        row["full_report"] = {"results": {"lpips": {
            "reference_metadata": {"fps": 10.0},
            "lpips_per_matched_event": [{"e_ref": 20, "e_cand": e_cand}]}}}
        return row

    report = mjf.summarize([lp_row("self@r0", 25), lp_row("self@r1", 10)])
    entry = report["selfmatch"]["lpips"]["C/x"]
    assert entry["lpips_primary_source"] == {"causal": 2}
    assert entry["metrics"]["event_offset_sec"]["mean"] == pytest.approx(0.75)
    assert mjf.matched_event_offset_sec({"full_report": {}}) is None


def test_synthetic_floors_pool_per_conditioning_case() -> None:
    dim = "prompt_scene_consistency"
    rows = []
    for slug, values in (("s00_a", [0.8, 0.9]), ("s09_b", [0.7, 0.7])):
        for k, v in enumerate(values):
            rows.append(_row(dim, f"synthetic/camera_yaw/Garden__scene_in/{slug}", f"self@r{k}",
                             model="unity", overall_score=v))
    floor = mjf.summarize(rows)["floors"][dim]["synthetic/camera_yaw/Garden__scene_in"]
    assert floor["trajectories"] == 2
    assert floor["self_mean"] == pytest.approx(0.775)
    # stds 0.0707 and 0 with one dof each -> pooled sqrt(0.005 / 2) = 0.05
    assert floor["self_std_pooled"] == pytest.approx(0.05, abs=1e-3)
    assert mjf.synthetic_group("BeamDomino/causal") is None
