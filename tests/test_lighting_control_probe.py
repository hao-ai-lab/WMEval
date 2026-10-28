import importlib.util
from pathlib import Path

import cv2
import numpy as np

spec = importlib.util.spec_from_file_location('probe', Path(__file__).parents[1] / 'tools/analysis/analyze_lighting_control_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_pnp_recovers_metric_camera_despite_outliers():
    rng = np.random.default_rng(5)
    world = rng.uniform([-1, -.5, 3], [1, .5, 6], size=(80, 3))
    rotation = np.array([.02, -.1, 0], np.float64)
    matrix, _ = cv2.Rodrigues(rotation)
    centre = np.array([.3, .1, 0])
    translation = -matrix @ centre
    k = np.array([[500, 0, 320], [0, 500, 176], [0, 0, 1]], np.float64)
    pixels, _ = cv2.projectPoints(world, rotation, translation, k, None)
    pixels = pixels[:, 0]
    pixels[:15] += 100
    result = probe.estimate_pose(world, pixels, k)
    assert result is not None
    np.testing.assert_allclose(np.array(result['c2w'])[:3, 3], centre, atol=1e-6)
    np.testing.assert_allclose(np.array(result['c2w'])[:3, :3], matrix.T, atol=1e-6)
    assert result['inliers'] == 65


def test_insufficient_correspondences_stay_unresolved():
    assert probe.estimate_pose(np.zeros((11, 3)), np.zeros((11, 2)), np.eye(3)) is None
