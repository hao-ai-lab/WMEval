"""Unit tests for causal LPIPS event pairing and diagnostics."""

from __future__ import annotations

import unittest

from evaluation.modules.lpips_causal_vlm import (
    build_windowed_keyframe_groups,
    build_detection_user_prompt,
    build_judge_user_prompt,
    build_side_relabel_user_prompt,
    compute_first_divergence,
    filter_events_by_contact_types,
    merge_nearby_events,
    split_events_by_evidence_gaps,
    pair_collision_events_monotonic_with_log,
    pair_collision_events_order_with_log,
    pair_collision_events_strict_with_log,
)
from evaluation.scene_prompt_registry import infer_scene_key, resolve_scene_hints


def _ev(frame: int, side: str) -> dict:
    return {
        "type": "collision",
        "side": side,
        "frame": frame,
        "evidence_frame_indices": [frame],
        "evidence_summary": "test",
    }


class TestCausalPairing(unittest.TestCase):
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
        self.assertEqual(log[0]["match_rule"], "monotonic_side_time")

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
        self.assertIn("Event anchor frame: 10", text)

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


if __name__ == "__main__":
    unittest.main()
