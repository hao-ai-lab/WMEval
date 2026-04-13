"""Unit tests for causal LPIPS event pairing and diagnostics."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evaluation.path_resolution import resolve_ptlflow_root
from evaluation.modules.flow_utils import (
    build_flow_proposal_windows_from_scores,
    compute_flow_transition_saliency,
)
from evaluation.modules.lpips_causal_vlm import (
    build_windowed_keyframe_groups,
    build_detection_user_prompt,
    build_judge_user_prompt,
    build_side_relabel_user_prompt,
    compute_first_divergence,
    filter_events_by_contact_types,
    merge_fine_and_coarse_events,
    merge_nearby_events,
    parse_or_repair_judge_output,
    repair_json_object_with_model,
    resolve_detection_max_events,
    split_events_by_evidence_gaps,
    pair_collision_events_monotonic_with_log,
    pair_collision_events_order_with_log,
    pair_collision_events_strict_with_log,
    resolve_reasoning_effort,
    sanitize_relabel_payload,
)
from evaluation.scene_prompt_registry import infer_scene_key, resolve_scene_config, resolve_scene_hints


def _ev(frame: int, side: str) -> dict:
    return {
        "type": "collision",
        "side": side,
        "contact_class": "unknown",
        "frame": frame,
        "evidence_frame_indices": [frame],
        "evidence_summary": "test",
    }


class TestCausalPairing(unittest.TestCase):
    def test_parse_or_repair_judge_output_uses_retry_after_failed_repair(self) -> None:
        calls: list[str] = []

        def _repair(_: str) -> dict:
            calls.append("repair")
            raise ValueError("repair failed")

        def _retry(_: str, parse_exc: Exception, repair_exc: Exception | None) -> dict:
            calls.append("retry")
            self.assertIn("delimiter", str(parse_exc))
            self.assertEqual(str(repair_exc), "repair failed")
            return {"reasoning_steps": ["ok"], "events": []}

        payload = parse_or_repair_judge_output(
            '{"reasoning_steps":["bad"] "events":[]}',
            repair_json=_repair,
            retry_json=_retry,
        )
        self.assertEqual(payload["events"], [])
        self.assertEqual(calls, ["repair", "retry"])

    def test_repair_json_object_with_model_returns_fixed_object(self) -> None:
        repaired_text = '{"reasoning_steps":["ok"],"events":[]}'

        class _FakeCompletions:
            def create(self, **_: object) -> object:
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=repaired_text)
                        )
                    ]
                )

        class _FakeClient:
            def with_options(self, **_: object) -> "_FakeClient":
                return self

            @property
            def chat(self) -> object:
                return SimpleNamespace(completions=_FakeCompletions())

        payload = repair_json_object_with_model(
            client=_FakeClient(),
            model="gpt-5.4-mini",
            raw_text='{"reasoning_steps":["bad"] "events":[]}',
            timeout_sec=30.0,
        )
        self.assertEqual(payload["events"], [])
        self.assertEqual(payload["reasoning_steps"], ["ok"])

    def test_strict_log_side_mismatch_no_match(self) -> None:
        ref = [_ev(10, "left"), _ev(20, "left")]
        cand = [_ev(12, "right")]
        matched, ur, uc, log = pair_collision_events_strict_with_log(ref, cand)
        self.assertEqual(len(matched), 0)
        self.assertEqual(len(ur), 2)
        self.assertEqual(log[0]["status"], "unmatched")
        self.assertIn("no_unused_candidate_with_side_left", log[0]["reason"])

    def test_strict_match_same_side(self) -> None:
        ref = [_ev(10, "left")]
        cand = [_ev(12, "left")]
        matched, ur, uc, log = pair_collision_events_strict_with_log(ref, cand)
        self.assertEqual(len(matched), 1)
        self.assertEqual(len(ur), 0)
        self.assertEqual(log[0]["status"], "matched")
        self.assertEqual(log[0]["match_rule"], "strict_side")

    def test_strict_match_front(self) -> None:
        ref = [_ev(10, "front")]
        cand = [_ev(12, "front")]
        matched, ur, uc, log = pair_collision_events_strict_with_log(ref, cand)
        self.assertEqual(len(matched), 1)
        self.assertEqual(log[0]["match_rule"], "strict_side")

    def test_order_pairs_despite_side(self) -> None:
        ref = [_ev(10, "left"), _ev(30, "right")]
        cand = [_ev(11, "right"), _ev(31, "left")]
        matched, log = pair_collision_events_order_with_log(ref, cand)
        self.assertEqual(len(matched), 2)
        self.assertEqual(log[0]["match_rule"], "order_index")

    def test_monotonic_pairing_preserves_time_order(self) -> None:
        ref = [_ev(199, "front"), _ev(247, "right"), _ev(275, "back")]
        cand = [_ev(20, "front"), _ev(60, "right"), _ev(120, "back")]
        matched, ur, uc, log = pair_collision_events_monotonic_with_log(
            ref,
            cand,
            ref_frame_count=496,
            cand_frame_count=172,
        )
        self.assertEqual(len(matched), 3)
        self.assertEqual(
            [(r["side"], c["side"]) for r, c in matched],
            [("front", "front"), ("right", "right"), ("back", "back")],
        )
        self.assertEqual(len(ur), 0)
        self.assertEqual(len(uc), 0)
        self.assertEqual(log[0]["match_rule"], "monotonic_chronology")
        self.assertIn("pair_cost_breakdown", log[0])

    def test_monotonic_pairing_can_leave_mismatch_unmatched(self) -> None:
        ref = [_ev(10, "left"), _ev(20, "right")]
        cand = [_ev(12, "left")]
        matched, ur, uc, log = pair_collision_events_monotonic_with_log(
            ref,
            cand,
            ref_frame_count=100,
            cand_frame_count=100,
        )
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0][0]["side"], "left")
        self.assertEqual(matched[0][1]["side"], "left")
        self.assertEqual(len(ur), 1)
        self.assertEqual(ur[0]["side"], "right")
        self.assertEqual(len(uc), 0)
        self.assertEqual(log[-1]["reason"], "monotonic_skip_reference")

    def test_monotonic_pairing_allows_label_mismatch_when_time_aligned(self) -> None:
        ref = [_ev(10, "front") | {"contact_class": "depth"}]
        cand = [_ev(12, "right") | {"contact_class": "lateral", "side_confidence": 0.1}]
        matched, ur, uc, _ = pair_collision_events_monotonic_with_log(
            ref,
            cand,
            ref_frame_count=100,
            cand_frame_count=100,
        )
        self.assertEqual(len(matched), 1)
        self.assertEqual(len(ur), 0)
        self.assertEqual(len(uc), 0)

    def test_resolve_reasoning_effort_defaults_by_stage(self) -> None:
        self.assertEqual(
            resolve_reasoning_effort({}, stage="coarse", model="gpt-5.4"),
            "low",
        )
        self.assertEqual(
            resolve_reasoning_effort({}, stage="fine", model="gpt-5.4"),
            "medium",
        )
        self.assertEqual(
            resolve_reasoning_effort({}, stage="side", model="gpt-5.4"),
            "medium",
        )
        self.assertIsNone(
            resolve_reasoning_effort({}, stage="fine", model="gpt-4o")
        )

    def test_resolve_reasoning_effort_honors_artifacts(self) -> None:
        artifacts = {
            "lpips_vlm_coarse_reasoning_effort": "low",
            "lpips_vlm_fine_reasoning_effort": "high",
            "lpips_vlm_side_reasoning_effort": "minimal",
        }
        self.assertEqual(
            resolve_reasoning_effort(artifacts, stage="coarse", model="gpt-5.4"),
            "low",
        )
        self.assertEqual(
            resolve_reasoning_effort(artifacts, stage="fine", model="gpt-5.4"),
            "high",
        )
        self.assertEqual(
            resolve_reasoning_effort(artifacts, stage="side", model="gpt-5.4"),
            "minimal",
        )

    def test_flow_transition_saliency_camera_compensation_reduces_global_drift(self) -> None:
        flow_a = np.zeros((8, 8, 2), dtype=np.float32)
        flow_b = np.zeros((8, 8, 2), dtype=np.float32)
        flow_a[..., 0] = 5.0
        flow_b[..., 0] = 5.0
        without_comp = compute_flow_transition_saliency(
            [flow_a, flow_b],
            camera_compensation=False,
            global_motion_mode="median_translation",
        )
        with_comp = compute_flow_transition_saliency(
            [flow_a, flow_b],
            camera_compensation=True,
            global_motion_mode="median_translation",
        )
        self.assertGreater(without_comp[0], 1.0)
        self.assertLess(with_comp[0], 1e-3)
        self.assertLess(with_comp[1], 1e-3)

    def test_build_flow_proposal_windows_from_scores_keeps_local_peaks(self) -> None:
        groups, logs = build_flow_proposal_windows_from_scores(
            [0, 10, 20, 30, 40],
            [0.1, 0.9, 0.2, 0.8],
            frame_count=50,
            fps=10.0,
            window_sec=0.5,
            max_frames_per_window=6,
            topk_windows=2,
            peak_threshold=0.3,
        )
        self.assertEqual(len(groups), 2)
        anchors = [row["anchor_frame"] for row in logs]
        self.assertEqual(anchors, [15, 35])

    def test_resolve_ptlflow_root_prefers_sibling_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            sibling = repo_root / "ptlflow"
            sibling.mkdir()
            (sibling / "synthetic_flow.py").write_text("# test\n", encoding="utf-8")
            self.assertEqual(
                resolve_ptlflow_root(repo_root=repo_root),
                str(sibling.resolve()),
            )

    def test_monotonic_pairing_rejects_bad_temporal_match(self) -> None:
        ref = [_ev(10, "front") | {"contact_class": "depth", "side_confidence": 1.0}]
        cand = [_ev(90, "left") | {"contact_class": "lateral", "side_confidence": 1.0}]
        matched, ur, uc, _ = pair_collision_events_monotonic_with_log(
            ref,
            cand,
            ref_frame_count=100,
            cand_frame_count=100,
        )
        self.assertEqual(len(matched), 0)
        self.assertEqual(len(ur), 1)
        self.assertEqual(len(uc), 1)

    def test_first_divergence_temporal_spike(self) -> None:
        temporal = {
            "lpips_per_pair": [0.1, 0.11, 0.9, 0.12],
            "timestamps_sec": [0.0, 0.25, 0.5, 0.75],
            "reference_frame_indices": [0, 1, 2, 3],
            "candidate_frame_indices": [0, 1, 2, 3],
        }
        div = compute_first_divergence([], [], 100, 100, temporal)
        self.assertIsNotNone(div["temporal_stream"]["first_lpips_spike_index"])
        self.assertEqual(div["temporal_stream"]["lpips"], 0.9)

    def test_first_divergence_event_timeline(self) -> None:
        ref = [_ev(50, "left")]
        cand = [_ev(10, "right")]
        div = compute_first_divergence(ref, cand, 101, 101, None)
        et = div["event_timeline"]
        self.assertTrue(et["first_event_side_mismatch"])
        self.assertAlmostEqual(et["first_event_time_delta_cand_minus_ref"], -0.4, places=5)

    def test_filter_events_by_contact_types(self) -> None:
        evs = [
            {"type": "collision", "side": "left", "frame": 1},
            {"type": "other", "side": "right", "frame": 2},
        ]
        kept = filter_events_by_contact_types(evs, ("collision",))
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["frame"], 1)

    def test_strict_pairing_respects_directional_sides_set(self) -> None:
        ref = [_ev(10, "front")]
        cand = [_ev(12, "unknown")]
        matched, ur, uc, _ = pair_collision_events_strict_with_log(
            ref, cand, directional_sides=frozenset({"left", "right"})
        )
        self.assertEqual(len(matched), 1)
        self.assertEqual(len(ur), 0)

    def test_build_judge_user_prompt_includes_scene_hints(self) -> None:
        text = build_judge_user_prompt(
            [0, 5, 10], {"lpips_vlm_scene_hints": "Describe the rolling sphere."}
        )
        self.assertIn("Scene-specific instructions", text)
        self.assertIn("rolling sphere", text)
        self.assertIn("0, 5, 10", text)

    def test_build_detection_user_prompt_deprioritizes_side_labels(self) -> None:
        text = build_detection_user_prompt([0, 5, 10], {})
        self.assertIn('side="unknown"', text)
        self.assertIn("distinct contact/impact episodes", text)

    def test_build_detection_user_prompt_high_recall_prefers_recall(self) -> None:
        text = build_detection_user_prompt([0, 5, 10], {}, max_events=12, high_recall=True)
        self.assertIn("Prefer recall over precision", text)
        self.assertIn("Return at most 12 events", text)
        self.assertIn("grazing", text)

    def test_build_side_relabel_user_prompt_mentions_all_directional_sides(self) -> None:
        text = build_side_relabel_user_prompt(
            [8, 10, 12],
            _ev(10, "unknown"),
            {"lpips_vlm_scene_hints": "Use motion cues."},
        )
        self.assertIn('"left"', text)
        self.assertIn('"right"', text)
        self.assertIn('"front"', text)
        self.assertIn('"back"', text)
        self.assertIn('"contact_class"', text)
        self.assertIn("Event anchor frame: 10", text)

    def test_sanitize_relabel_payload_keeps_contact_class_and_side_confidence(self) -> None:
        payload = sanitize_relabel_payload(
            {
                "contact_class": "lateral",
                "side": "left",
                "confidence": 0.8,
                "side_confidence": 0.6,
                "evidence_frame_indices": [8, 10],
                "evidence_summary": "test",
                "reasoning_steps": ["ok"],
            },
            allowed={8, 10, 12},
        )
        self.assertEqual(payload["contact_class"], "lateral")
        self.assertEqual(payload["side"], "left")
        self.assertEqual(payload["side_confidence"], 0.6)

    def test_build_windowed_keyframe_groups_overlaps(self) -> None:
        groups = build_windowed_keyframe_groups(
            frame_count=100,
            fps=10.0,
            window_sec=2.0,
            overlap_sec=1.0,
            max_frames_per_window=5,
        )
        self.assertGreaterEqual(len(groups), 2)
        self.assertLess(groups[1][0], groups[0][-1])

    def test_merge_nearby_events_keeps_higher_confidence(self) -> None:
        merged = merge_nearby_events(
            [
                _ev(10, "unknown") | {"confidence": 0.2},
                _ev(14, "unknown") | {"confidence": 0.9},
                _ev(40, "unknown") | {"confidence": 0.5},
            ],
            frame_gap=6,
        )
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["frame"], 14)

    def test_split_events_by_evidence_gaps_emits_multiple_episodes(self) -> None:
        split = split_events_by_evidence_gaps(
            [
                _ev(20, "unknown")
                | {"evidence_frame_indices": [10, 12, 14, 40, 42, 44], "confidence": 0.8}
            ],
            frame_gap=10,
            min_cluster_size=2,
        )
        self.assertEqual(len(split), 2)
        self.assertEqual([event["frame"] for event in split], [12, 42])
        self.assertEqual(
            [event["evidence_frame_indices"] for event in split],
            [[10, 12, 14], [40, 42, 44]],
        )

    def test_split_events_by_evidence_gaps_drops_tiny_clusters(self) -> None:
        split = split_events_by_evidence_gaps(
            [
                _ev(20, "unknown")
                | {"evidence_frame_indices": [10, 12, 14, 40], "confidence": 0.8}
            ],
            frame_gap=10,
            min_cluster_size=2,
        )
        self.assertEqual(len(split), 1)
        self.assertEqual(split[0]["evidence_frame_indices"], [10, 12, 14])

    def test_merge_fine_and_coarse_events_preserves_unmatched_coarse(self) -> None:
        merged = merge_fine_and_coarse_events(
            [_ev(100, "unknown") | {"confidence": 0.6}],
            [
                _ev(20, "unknown") | {"confidence": 0.7},
                _ev(104, "unknown") | {"confidence": 0.9},
            ],
            frame_gap=6,
        )
        self.assertEqual([event["frame"] for event in merged], [20, 104])
        self.assertEqual(merged[1]["confidence"], 0.9)

    def test_resolve_detection_max_events_uses_fine_default(self) -> None:
        self.assertEqual(
            resolve_detection_max_events({}, prompt_mode="detection_high_recall"),
            12,
        )
        self.assertEqual(
            resolve_detection_max_events({}, prompt_mode="detection_only"),
            6,
        )

    def test_infer_scene_key_from_paths(self) -> None:
        key = infer_scene_key(
            "/tmp/out/unity_runs/sample_scene/unity_rollout.mp4",
            "/tmp/out/genie_runs/sample_scene/run_02/sample_scene_genie_rollout.mp4",
        )
        self.assertEqual(key, "sample_scene")

    def test_resolve_scene_hints_uses_registry(self) -> None:
        key, hints = resolve_scene_hints(
            scene_key=None,
            reference_video="/tmp/out/unity_runs/sample_scene/unity_rollout.mp4",
            candidate_video="/tmp/out/genie_runs/sample_scene/run_02/sample_scene_genie_rollout.mp4",
            explicit_hints=None,
        )
        self.assertEqual(key, "sample_scene")
        self.assertIsNotNone(hints)
        assert hints is not None
        self.assertIn("full collision timeline", hints)

    def test_resolve_scene_config_uses_registry(self) -> None:
        key, config = resolve_scene_config(
            scene_key=None,
            reference_video="/tmp/out/unity_runs/sample_scene/unity_rollout.mp4",
            candidate_video="/tmp/out/genie_runs/sample_scene/run_02/sample_scene_genie_rollout.mp4",
        )
        self.assertEqual(key, "sample_scene")
        self.assertEqual(config["match_bias"], "chronology")


if __name__ == "__main__":
    unittest.main()
