import json
from pathlib import Path

import numpy as np

from tools.prepare_lighting_backend_inputs import prepare

CASE = Path(__file__).resolve().parents[1]/'eval_sets/wmeval_lite_v1/cases/SunRisesAndSets/lighting_orbit_v2'


def test_lingbot_native_intrinsics_preserve_square_seed_fov(tmp_path):
    receipt=prepare(CASE,tmp_path,'lingbot')
    poses=np.load(tmp_path/'poses.npy'); k=np.load(tmp_path/'intrinsics.npy')
    assert len(poses)==161 and receipt['last_control_timestamp_sec']==10
    # Apply the pinned upstream's per-axis resize to its actual square output.
    transformed=k[0]*np.array([624/832,624/480,624/832,624/480])
    np.testing.assert_allclose(transformed[:2],624/(2*np.tan(np.pi/6)),rtol=1e-7)
    np.testing.assert_allclose(transformed[2:],[312,312])
    np.testing.assert_allclose(poses[0],poses[-1],atol=1e-6)
    np.testing.assert_allclose(poses[40,:3,3],[2,1.6,0],atol=1e-6)


def test_hy_latent_spacing_and_terminal_hold_are_not_time_stretched(tmp_path):
    receipt=prepare(CASE,tmp_path,'hy-worldplay')
    times=json.loads((tmp_path/'native_control_times.json').read_text())
    rows=json.loads((tmp_path/'poses.json').read_text())
    assert len(rows)==64 and receipt['native_frames']==253
    assert times[15]==2.5 and times[60]==10 and times[-1]==10.5
    np.testing.assert_allclose(np.array(rows['15']['extrinsic'])[:3,3],[2,1.6,0],atol=1e-7)
    np.testing.assert_allclose(rows['60']['extrinsic'],rows['63']['extrinsic'],atol=1e-14)
    assert receipt['terminal_hold_after_sec']==10
