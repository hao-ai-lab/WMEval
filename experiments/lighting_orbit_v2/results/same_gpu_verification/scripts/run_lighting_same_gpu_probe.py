#!/usr/bin/env python3
"""Bounded, fresh-process same-hardware diagnostic; never replaces expected scores."""
import argparse,json,os,sys,time,subprocess,signal,hashlib
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline-utc',required=True);p.add_argument('--verifier',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();out=a.output.resolve();out.mkdir(exist_ok=False,parents=True);deadline=datetime.fromisoformat(a.deadline_utc)
report=out/'probe.json';state={'schema_version':1,'status':'running','phase':'inventory','case':'SunRisesAndSets/lighting_orbit_v2','model':'LingBot','seed':'42','runs':[]}
def save():report.write_text(json.dumps(state,indent=2,sort_keys=True)+'\n')
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def execute(cmd,log,env,max_seconds):
 remaining=(deadline-datetime.now(timezone.utc)).total_seconds()
 if remaining<=5:raise TimeoutError('worker deadline reached')
 with log.open('w') as stream:
  job=subprocess.Popen(cmd,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=job.wait(timeout=min(max_seconds,remaining))
  except BaseException:
   os.killpg(job.pid,signal.SIGTERM)
   try:job.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL)
   raise
 if code:raise RuntimeError(f'worker exit {code}: {log.name}')
def score(index,deterministic=False):
 name=('deterministic_' if deterministic else 'original_')+str(index);target=out/name;state['phase']=name;save()
 cmd=[sys.executable,str(root/'tools/run_wmeval_lite.py'),'run','--suite',str(root/'eval_sets/wmeval_lite_v1'),'--dimensions','lighting_shading','--candidate-manifest',str(out/'candidate.jsonl'),'--output-dir',str(target),'--device','cuda','--run-id',name]
 env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1'}
 if deterministic:
  env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
  code="import random,numpy as np,torch,runpy,sys; random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.use_deterministic_algorithms(True); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
  cmd=[sys.executable,'-c',code,*cmd[1:]]
 execute(cmd,out/(name+'.log'),env,150)
 rows=[json.loads(x) for x in (target/'per_case_results.jsonl').read_text().splitlines()]
 if len(rows)!=1 or rows[0]['status']!='completed':raise ValueError('diagnostic requires one completed row')
 row=rows[0]; norm=Path(row['full_report']['candidate_metadata']['path']);metrics=row['metrics'];diff={k:metrics[k]-expected[k] for k in expected if abs(metrics[k]-expected[k])>1e-6}
 state['runs'].append({'name':name,'metrics':metrics,'differences_from_original':diff,'normalized_sha256':sha(norm),'result_jsonl_sha256':sha(target/'per_case_results.jsonl')});save()
 return metrics
save()
try:
 sys.path.insert(0,str(root));from tools.build_lighting_release_package import verify;verify(root)
 rows=[json.loads(x) for x in (root/'candidates.jsonl').read_text().splitlines()];row=next(r for r in rows if r['model']=='LingBot' and r['seed']=='42');row['candidate_video']=str(root/row['candidate_video']);(out/'candidate.jsonl').write_text(json.dumps(row,sort_keys=True)+'\n')
 expected=next(r['metrics'] for r in json.loads((root/'expected_scores.json').read_text())['rows'] if r['model']=='LingBot' and r['seed']=='42');state['expected_metrics']=expected;state['native_sha256']=sha(row['candidate_video']);state['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip();save()
 first=score(1);second=score(2);state['original_repeats_equal']=first==second;state['repeats_match_original']=all(not r['differences_from_original'] for r in state['runs']);save()
 if state['repeats_match_original']:
  state['phase']='full_fifteen';save();execute([sys.executable,'-c',"import sys,runpy;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name='__main__')",str(root),str(a.verifier.resolve()),'--root',str(root),'--output',str(out/'full_release'),'--deadline-utc',a.deadline_utc,'--gpu-count','1'],out/'full_release.log',dict(os.environ),10000)
  state.update(status='passed',phase='complete',full_release_verified=True)
 elif not state['original_repeats_equal']:
  first=score(1,True);second=score(2,True);state.update(status='same_gpu_nondeterminism',phase='complete',deterministic_repeats_equal=first==second)
 else:state.update(status='stable_same_gpu_but_different_original',phase='complete')
except BaseException as error:state.update(status='failed',error=f'{type(error).__name__}: {error}')
state['completed_at_utc']=datetime.now(timezone.utc).isoformat();save();print(json.dumps(state),flush=True)
