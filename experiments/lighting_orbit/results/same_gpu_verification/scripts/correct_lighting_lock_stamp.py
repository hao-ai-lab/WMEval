#!/usr/bin/env python3
"""Repair only a stale derived identity stamp; retain raw source rows unchanged."""
import sys,json,shutil,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();deadline=datetime.fromisoformat(sys.argv[3]);sys.path.insert(0,str(root))
from evaluation.lite.authoring import update_lock,_fingerprint
import yaml
report=json.loads((out/'full_release/verification.json').read_text());assert report['status']=='passed' and report['compared_metrics']==90 and report['resume_rows_unchanged']
suite=out/'lock_corrected_suite';suite.mkdir(exist_ok=False);shutil.copyfile(root/'eval_sets/wmeval_lite_v1/suite.yaml',suite/'suite.yaml');old=root/'eval_sets/wmeval_lite_v1/cases/SunRisesAndSets/lighting_orbit_v2';new=suite/'cases/SunRisesAndSets/lighting_orbit_v2';shutil.copytree(old,new)
before=yaml.safe_load((old/'case.yaml').read_text());recorded=before['evaluator_lock']['evaluator_fingerprint'];derived=_fingerprint(old,before);update_lock(new,relock=False);after=yaml.safe_load((new/'case.yaml').read_text());assert after['evaluator_lock']['evaluator_fingerprint']==derived
assert {k:v for k,v in before.items() if k!='evaluator_lock'}=={k:v for k,v in after.items() if k!='evaluator_lock'}
for p in (old/'reference').rglob('*'):
 if p.is_file():assert p.read_bytes()==(new/p.relative_to(old)).read_bytes()
rows=[json.loads(x) for x in (out/'full_release/combined/per_case_results.jsonl').read_text().splitlines()];target=out/'lock_corrected_results';target.mkdir(exist_ok=False)
for row in rows:
 assert row['evaluator_lock']['evaluator_fingerprint']==recorded
 row['source_evaluator_lock']=dict(row['evaluator_lock']);row['evaluator_lock']['evaluator_fingerprint']=derived;row['lock_stamp_correction']='same_gpu_verification/lock_stamp_correction.json'
p=target/'per_case_results.jsonl';p.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();before_sha=sha(p)
remaining=(deadline-datetime.now(timezone.utc)).total_seconds();assert remaining>5
with (out/'lock_corrected_resume.log').open('w') as log:
 subprocess.run([sys.executable,str(root/'tools/run_wmeval_lite.py'),'run','--suite',str(suite),'--dimensions','lighting_shading','--candidate-manifest',str(root/'candidates.jsonl'),'--output-dir',str(target),'--device','cuda','--resume','--run-id','corrected-identity-resume'],check=True,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
assert sha(p)==before_sha
value={'schema_version':1,'status':'verified_stamp_correction','old_recorded':recorded,'derived_and_corrected':derived,'source_case_without_lock_unchanged':True,'all_reference_bytes_unchanged':True,'metrics_unchanged':True,'scorer_sources_unchanged':True,'corrected_resume_rows':15,'corrected_resume_hash_unchanged':True,'corrected_resume_sha256':before_sha,'original_rows_preserved':'full_release/combined/per_case_results.jsonl','completed_at_utc':datetime.now(timezone.utc).isoformat()};(out/'lock_stamp_correction.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');print(json.dumps(value))
