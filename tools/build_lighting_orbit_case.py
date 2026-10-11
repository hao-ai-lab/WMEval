#!/usr/bin/env python3
"""Build a portable lighting orbit case from verified Unity editor capture.

Run build_lite_case.py --relock AFTER committing the implementation. This
builder never declares generation coverage or evaluator completion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.camera_orbit import camera_intrinsics, orbit_poses
from evaluation.trajectories import write_compiled_trajectory


def vector(value):
    return np.array([value[k] for k in ('x', 'y', 'z')], dtype=np.float64)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def build(capture: Path, destination: Path, revision: str):
    if destination.exists():
        raise FileExistsError(f'refusing to replace case: {destination}')
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('scene source revision must be a full Git SHA')
    calibration = json.loads((capture/'calibration.json').read_text())
    frames = [json.loads(s) for s in (capture/'frames.jsonl').read_text().splitlines()]
    if len(frames) != 121 or len({json.dumps(f['sun'], sort_keys=True) for f in frames}) != 1 or len({f['staticStateSha256'] for f in frames}) != 1:
        raise ValueError('capture requires 121 frames with fixed sun and static objects')
    origin = vector(calibration['originUnity'])
    height = vector(calibration['targetUnity'])[1] - origin[1]
    spec = {
        'schema_version': 1, 'trajectory_id': 'lighting.camera.look_at_orbit_v2',
        'kind': 'camera_look_at_orbit', 'target': 'camera',
        'coordinates': 'right_handed_y_up_opencv_c2w', 'duration_sec': 10.0, 'control_hz': 60,
        'geometry': {'origin_m': [0,0,0], 'look_at_m': [0,float(height),0],
                     'radius_m': 2.0, 'eye_height_m': 1.6, 'initial_phase_deg': 0.0,
                     'vertical_fov_deg': float(calibration['verticalFovDeg'])},
        'channels': {'orbit_phase_rate_deg_s': {'kind':'continuous','min':-360,'max':360}},
        'segments': [{'start_sec':0.0,'end_sec':10.0,'values':{'orbit_phase_rate_deg_s':36.0}}],
        'environment': {'sun':'fixed world direction, intensity and colour temperature', 'objects':'static'},
        'paper_alignment': {'status':'reconstructed', 'note':'New explicit orbit protocol; historical yaw inputs are preserved separately.'}
    }
    times = np.array([f['canonicalTimeSec'] for f in frames])
    np.testing.assert_allclose(times, np.arange(121)/12, atol=1e-12, rtol=0)
    expected = orbit_poses(spec, times)
    actual = np.tile(np.eye(4), (121,1,1))
    world_reflection = np.diag([1.,1.,-1.])
    for i,f in enumerate(frames):
        actual[i,:3,3] = world_reflection @ (vector(f['positionUnity']) - origin)
        actual[i,:3,:3] = world_reflection @ np.column_stack((vector(f['rightUnity']), -vector(f['upUnity']), vector(f['forwardUnity'])))
    # Unity single-precision transforms, Python analytic double precision.
    error = float(np.max(np.abs(actual-expected)))
    if error > 2e-6:
        raise ValueError(f'Unity/C2W mismatch: {error}')
    from PIL import Image
    first = np.asarray(Image.open(capture/'frames/frame_000000.png'), dtype=float)
    last = np.asarray(Image.open(capture/'frames/frame_000120.png'), dtype=float)
    closure_mae = float(np.abs(first-last).mean())
    if closure_mae > 2.0:
        raise ValueError(f'capture image closure MAE > 2/255: {closure_mae}')
    ref = destination/'reference'; artifacts = ref/'artifacts'
    artifacts.mkdir(parents=True)
    shutil.copy2(capture/'ref.png',ref/'ref.png')
    for name in ['calibration.json','frames.jsonl','scene_dependencies.json']:
        shutil.copy2(capture/name,artifacts/name)
    (artifacts/'trajectory.yaml').write_text(yaml.safe_dump(spec,sort_keys=False))
    manifest=write_compiled_trajectory(artifacts/'trajectory.yaml',artifacts)
    for key,newname in [('action_npy','action.npy'),('action_npz','action.npz'),('actions_txt','actions.txt'),
                        ('camera_pose_samples','actions.jsonl'),('poses','poses.npy')]:
        (artifacts/manifest[key]).rename(artifacts/newname); manifest[key]=newname
    (artifacts/'trajectory.manifest.json').unlink()
    (artifacts/'compile_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    np.save(artifacts/'intrinsics.npy',camera_intrinsics(spec,calibration['width'],calibration['height']),allow_pickle=False)
    subprocess.run(['ffmpeg','-v','error','-framerate','12','-i',str(capture/'frames/frame_%06d.png'),
                    '-frames:v','120','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(ref/'unity_orbit.mp4')],check=True)
    qa={'schema_version':1,'status':'unity_geometry_verified','scene_source_revision':revision,
        'unity_version':calibration['unityVersion'], 'unity_to_local_world_linear':world_reflection.tolist(),
        'unity_origin_m':origin.tolist(),'opencv_camera_axes':['right','down','forward'],
        'unity_c2w_max_absolute_error':error,'geometry_tolerance':2e-6,
        'static_state_count':1,'sun_state_count':1,'image_closure_mae_8bit':closure_mae,
        'image_closure_tolerance_8bit':2.0,'reference_scoring_frames':120,'closure_frame_in_video':False,
        'conditioning_sha256':sha(ref/'ref.png'),'qa_video_sha256':sha(ref/'unity_orbit.mp4'),
        'scene_dependencies_sha256':sha(artifacts/'scene_dependencies.json'),
        'capture_script_sha256':sha(ROOT/'tools/unity/LightingOrbitCapture.cs'),
        'rendering':'original materials, no postprocessing; editor static capture; source sun world transform frozen',
        'generation_coverage':'not generated; no model results claimed'}
    (artifacts/'unity_qa.json').write_text(json.dumps(qa,indent=2,sort_keys=True)+'\n')
    raw={
        'schema_version':2,'case_id':'SunRisesAndSets/lighting_orbit_v2','scene':'SunRisesAndSets',
        'display_name':'SunRisesAndSets complete camera orbit',
        'prompt': 'First-person viewpoint, eye height 1.6 metres above a wooden floor. No body or hands visible. '
                  'Move counter-clockwise around the round wooden table at a constant 2-metre radius, completing '
                  'one full circle in exactly 10 seconds, continuously looking at the tabletop centre. '
                  'Constant speed, no head bobbing or roll. The table, chairs, red toaster, dark mug, white plate, '
                  'white bowl, knife and tall table lamp remain static. One fixed bright sun lights the scene; '
                  'its world direction, intensity and colour temperature remain constant throughout.',
        'perspective':'first_person','modules':['shadow_consistency'],
        'reference':{'first_frame':'reference/ref.png','video':'reference/unity_orbit.mp4',
                     'checksums':'reference/checksums.sha256','artifacts':{
                         'trajectory':'reference/artifacts/trajectory.yaml','action_npz':'reference/artifacts/action.npz',
                         'camera_pose_samples':'reference/artifacts/actions.jsonl','camera_poses':'reference/artifacts/poses.npy',
                         'camera_intrinsics':'reference/artifacts/intrinsics.npy','compile_manifest':'reference/artifacts/compile_manifest.json',
                         'scene_calibration':'reference/artifacts/calibration.json','unity_qa':'reference/artifacts/unity_qa.json'}},
        'protocol':{'candidate_normalization':'lighting_orbit_10s_fps12','min_duration_sec':10.0,
                    'min_duration_ratio':1.0,'duration_threshold_mode':'max', 'scoring_window_sec':[0.0,10.0],
                    'scoring_fps':12,'scoring_frames':120,'time_mapping':'receipt-derived native frame offset; no stretch'},
        'evaluator_lock':{'release_id':'wmeval-lite-v1','code_revision':'','evaluator_fingerprint':'',
                          'module_config':{},'artifact_schema_versions':{'camera_pose_samples':1},'dependency_versions':{}},
        'module_params':{'shadow_consistency':{'shadow_text_prompt':'table. chair. lamp.','shadow_anchor_label':'table'}}}
    (destination/'case.yaml').write_text(yaml.safe_dump(raw,sort_keys=False,allow_unicode=True))
    return qa


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scene-source-revision',required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.capture,args.output,args.scene_source_revision),indent=2))


if __name__=='__main__': main()
