"""Unit tests for causal LPIPS event pairing and diagnostics."""

from __future__ import annotations

import unittest

from evaluation.modules.lpips_causal_vlm import (
    build_judge_user_prompt,
    compute_first_divergence,
    filter_events_by_contact_types,
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
