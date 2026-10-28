#!/usr/bin/env python3
"""Verify and pin the paper's 30 stored lighting videos using a CPU container."""
from __future__ import annotations
import json
from pathlib import Path
import modal

app = modal.App("wmeval-lighting-baseline-pin")
volume = modal.Volume.from_name("WMEval-lighting-repro", create_if_missing=False)

MODEL_NAMES = {"genie": "project_genie", "lingbot": "lingbot_world",
               "hy": "hy_worldplay", "mg2": "matrix_game_2",
               "mg3": "matrix_game_3", "waypoint15": "waypoint15"}
METRICS = {"scene_retention_rate_pct", "shadow_detection_rate_pct",
           "multi_object_sun_disagreement_deg", "shadow_direction_jitter_deg_per_frame",
           "shadow_area_stability_pct", "shadow_shape_drift"}


def attach_scores(result: dict, path: Path) -> dict:
    """Join completed native-FPS score rows by video identity, not seed labels.

    The hosted Genie runs have run indexes, not numeric seeds. In particular,
    do not copy the Lite runner's seed labels into their provenance records.
    """
    import hashlib
    import math
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows) != 30 or any(r['status'] != 'completed' for r in rows):
        raise ValueError('expected 30 completed native score rows')
    lookup = {Path(r['candidate_video']).name: r for r in rows}
    if len(lookup) != 30:
        raise ValueError('duplicate scored video')
    expected_names = {Path(v['video']).name for v in result['videos']}
    if set(lookup) != expected_names:
        raise ValueError('scored videos do not match the verified baseline')
    if len({r['run_id'] for r in rows}) != 1:
        raise ValueError('score rows must come from one run')
    for video in result['videos']:
        row = lookup[Path(video['video']).name]
        suite_model = MODEL_NAMES[video['model']]
        if row['model'] != suite_model or row['dimension'] != 'lighting_shading':
            raise ValueError(f'model/dimension mismatch: {video["video"]}')
        if row['case_id'] != 'SunRisesAndSets/lighting' or not (
            row['candidate_video'] == video['video'] or
            row['candidate_video'].endswith('/' + video['video'])
        ):
            raise ValueError(f'score is not for the original native video: {video["video"]}')
        if set(row['metrics']) != METRICS or not all(math.isfinite(x) for x in row['metrics'].values()):
            raise ValueError(f'invalid metrics: {video["video"]}')
        video.update(suite_model=suite_model, native_metrics=row['metrics'],
                     evaluator_lock=row['evaluator_lock'])
    result.update(protocol='historical_paper_native_fps', candidate_normalization='none',
                  score_source={'run_id': rows[0]['run_id'],
                                'jsonl_sha256': hashlib.sha256(raw).hexdigest(),
                                'volume': 'WMEval-lite-candidates',
                                'path': f'eval/lite_runs/{rows[0]["run_id"]}/per_case_results.jsonl'},
                  limitations=['These historical runs have different actions and durations; they are not a matched-orbit baseline.',
                               'Genie has five independent hosted runs; its numeric generation seeds are unavailable.',
                               'FPS-normalized and new-action scores require separate baselines.'])
    return result


@app.function(image=modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg"),
              timeout=300, volumes={"/vol": volume})
def verify() -> dict:
    import hashlib
    import subprocess
    from fractions import Fraction
    root = Path('/vol')
    data = (root/'MANIFEST.json').read_bytes()
    manifest = json.loads(data)
    rows=[]
    for item in manifest['lighting_runs']:
        relative=Path(item['video'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[:2] != ('rollouts','lighting'):
            raise ValueError(f'invalid video path: {relative}')
        path=root/relative
        expected=manifest['files'][str(relative)]['sha256']
        h=hashlib.sha256()
        with path.open('rb') as f:
            while chunk := f.read(1024*1024): h.update(chunk)
        actual=h.hexdigest()
        if actual != expected: raise ValueError(f'video changed: {relative}')
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                         '-show_entries','stream=width,height,avg_frame_rate,nb_frames,duration',
                         '-of','json',str(path)],text=True))['streams'][0]
        fps=float(Fraction(probe['avg_frame_rate']))
        rows.append({**item,'sha256':actual,'bytes':path.stat().st_size,
                     'fps':fps,'frames':int(probe['nb_frames']),
                     'width':int(probe['width']),'height':int(probe['height']),
                     'duration_sec':float(probe['duration'])})
    from collections import Counter
    if Counter(r['model'] for r in rows) != Counter({name:5 for name in MODEL_NAMES}):
        raise ValueError('expected six models times five videos')
    return {'schema_version':1,'volume':'WMEval-lighting-repro',
            'manifest_sha256':hashlib.sha256(data).hexdigest(),
            'verification':'all 30 video bytes match the original volume MANIFEST.json',
            'videos':rows}


@app.local_entrypoint()
def main(output: str, scores: str = ""):
    result=verify.remote()
    if scores:
        result=attach_scores(result, Path(scores))
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(f"Verified {len(result['videos'])} videos; wrote {path}")
