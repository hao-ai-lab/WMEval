"""Tests for video_io helpers used by causal VLM."""

from __future__ import annotations

import unittest

from evaluation.video_io import build_vlm_keyframe_indices


class TestVlmKeyframeGrid(unittest.TestCase):
    def test_evenly_spaced_default(self) -> None:
        idx = build_vlm_keyframe_indices(100, 5)
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[-1], 99)
        self.assertEqual(len(idx), 5)

    def test_all_frames_when_non_positive_cap(self) -> None:
        idx = build_vlm_keyframe_indices(10, 0)
        self.assertEqual(idx, list(range(10)))
        idx2 = build_vlm_keyframe_indices(10, -1)
        self.assertEqual(idx2, list(range(10)))

    def test_different_lengths_per_video_semantics(self) -> None:
        """Each video calls build_vlm_keyframe_indices with its own frame_count."""
        u = build_vlm_keyframe_indices(496, 0)
        g = build_vlm_keyframe_indices(176, 0)
        self.assertEqual(len(u), 496)
        self.assertEqual(len(g), 176)


if __name__ == "__main__":
    unittest.main()
