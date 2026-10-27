import copy
import json

import numpy as np
import pytest
import yaml

from evaluation.camera_orbit import camera_intrinsics, orbit_poses
from evaluation.trajectories import TrajectoryError, compile_trajectory, load_actions_jsonl, validate_trajectory, write_compiled_trajectory


@pytest.fixture
def spec():
    return {"schema_version":1,"trajectory_id":"lighting.camera.look_at_orbit",
            "kind":"camera_look_at_orbit","target":"camera", "coordinates":"right_handed_y_up_opencv_c2w",
            "duration_sec":10,"control_hz":60,
            "geometry":{"origin_m":[0,0,0],"look_at_m":[0,1.2735698223114014,0],
                        "radius_m":2,"eye_height_m":1.6,"initial_phase_deg":0,"vertical_fov_deg":60},
            "channels":{"orbit_phase_rate_deg_s":{"kind":"continuous","min":-360,"max":360}},
            "segments":[{"start_sec":0,"end_sec":10,"values":{"orbit_phase_rate_deg_s":36}}],
            "paper_alignment":{"status":"reconstructed"}}


def test_true_orbit_closes_and_looks_at_fixed_target(spec):
    poses=orbit_poses(spec,np.linspace(0,10,601))
    np.testing.assert_allclose(poses[0],poses[-1],atol=1e-14)
    np.testing.assert_allclose(poses[[0,150,300,450],:3,3],
                               [[0,1.6,2],[2,1.6,0],[0,1.6,-2],[-2,1.6,0]],atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(poses[:,:3,:3]),1,atol=1e-14)
    rotations=poses[:,:3,:3]
    np.testing.assert_allclose(rotations.transpose(0,2,1)@rotations,np.broadcast_to(np.eye(3),(601,3,3)),atol=1e-14)
    target=np.array(spec['geometry']['look_at_m'])
    camera_target=np.einsum('nij,nj->ni', rotations.transpose(0,2,1),target-poses[:,:3,3])
    np.testing.assert_allclose(camera_target[:,:2],0,atol=1e-14)
    assert np.all(camera_target[:,2]>0)


def test_native_samples_are_analytic_and_compilation_is_deterministic(tmp_path,spec):
    source=tmp_path/'trajectory.yaml'; source.write_text(yaml.safe_dump(spec))
    compiled=compile_trajectory(source)
    assert compiled.values.shape==(600,1) and compiled.camera_to_world.shape==(601,4,4)
    manifest=write_compiled_trajectory(source,tmp_path/'artifacts')
    again=write_compiled_trajectory(source,tmp_path/'artifacts')
    assert manifest==again
    records=[json.loads(s) for s in (tmp_path/'artifacts'/manifest['camera_pose_samples']).read_text().splitlines()]
    assert len(records)==601 and records[-1]['timestamp_sec']==10
    assert records[0]['record_type']=='camera_pose' and 'project_genie_actions' not in manifest
    with pytest.raises(TrajectoryError,match='browser action'):
        load_actions_jsonl(tmp_path/'artifacts'/manifest['camera_pose_samples'])
    native=orbit_poses(spec,np.arange(161)/16)
    np.testing.assert_allclose(native[40],compiled.camera_to_world[150],atol=1e-14)
    k=camera_intrinsics(spec,1024,1024)
    assert k[0,0]==k[1,1] and k[0,2]==512


@pytest.mark.parametrize('mutation', ['yaw','target','coordinates','off_axis','nan'])
def test_invalid_orbit_is_rejected(spec,mutation):
    raw=copy.deepcopy(spec)
    if mutation=='yaw': raw['channels']={'yaw_rate_deg_s':raw['channels']['orbit_phase_rate_deg_s']}; raw['segments'][0]['values']={'yaw_rate_deg_s':36}
    elif mutation=='target': raw['target']='ego_vehicle'
    elif mutation=='coordinates': raw['coordinates']='unity'
    elif mutation=='off_axis': raw['geometry']['look_at_m'][0]=1
    else: raw['geometry']['radius_m']=float('nan')
    with pytest.raises(TrajectoryError): validate_trajectory(raw)
