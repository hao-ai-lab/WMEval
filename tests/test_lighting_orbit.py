import numpy as np
import pytest

from tools.trajectories.prepare_lighting_orbit import orbit_poses


def test_orbit_translates_returns_home_and_keeps_table_on_optical_axis():
    times, poses, intrinsics = orbit_poses()
    assert times.shape == (161,)
    assert times[-1] == 10
    np.testing.assert_allclose(poses[0], poses[-1], atol=1e-12)
    np.testing.assert_allclose(poses[40, :3, 3], [2, 1.6, 0], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(poses[:, [0,2], 3], axis=1), 2)
    rotation = poses[:, :3, :3]
    np.testing.assert_allclose(np.linalg.det(rotation), 1)
    np.testing.assert_allclose(rotation.transpose(0,2,1) @ rotation,
                               np.tile(np.eye(3), (161,1,1)), atol=1e-12)
    target_camera = np.einsum('tji,tj->ti', rotation, np.array([0,.75,0])-poses[:,:3,3])
    np.testing.assert_allclose(target_camera[:,:2], 0, atol=1e-12)
    assert (target_camera[:,2] > 0).all()
    assert intrinsics.shape == (161,4)
    # Upstream rescales the canonical 832x480 K to the square seed's output.
    assert intrinsics[0,0] * 624/832 == pytest.approx(intrinsics[0,1] * 624/480)


@pytest.mark.parametrize("kwargs", [{"radius":0},{"fps":float('nan')}, {"vertical_fov":180}, {"duration":.03}])
def test_reject_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        orbit_poses(**kwargs)
