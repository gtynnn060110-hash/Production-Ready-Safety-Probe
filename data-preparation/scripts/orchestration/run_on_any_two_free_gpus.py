"""Run a rollout on two suitable GPUs, reserving them against peer jobs."""
import sys as _prep_sys; from pathlib import Path as _PrepPath; _prep_sys.path.insert(0, str(_PrepPath(__file__).resolve().parents[2])); from scripts._bootstrap import setup as _prep_setup; _prep_setup()
import argparse,fcntl,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--python',required=True);p.add_argument('--script',required=True);p.add_argument('--log',required=True);p.add_argument('--smoke-huihui',action='store_true');p.add_argument('--min-free-fraction',type=float,default=None);p.add_argument('--headroom-mib',type=int,default=2048);a=p.parse_args()
locks=Path('data/.rollout-gpu-locks');locks.mkdir(exist_ok=True); chosen=[]; handles=[]
def gpu_memory():
 r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.free,memory.total','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
 return {int(i.strip()):(int(used.strip()),int(free.strip()),int(total.strip())) for i,used,free,total in (line.split(',') for line in r.stdout.splitlines())}
def eligible(stat):
 used,free,total=stat
 return used<512 if a.min_free_fraction is None else free>=total*a.min_free_fraction+a.headroom_mib
print('Waiting for two GPUs with sufficient free memory',flush=True)
while True:
 chosen=[];handles=[];mem=gpu_memory()
 for idx in sorted(mem):
  if not eligible(mem[idx]):continue
  handle=(locks/f'gpu-{idx}.lock').open('w')
  try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:handle.close();continue
  chosen.append(idx);handles.append(handle)
  if len(chosen)==2:break
 if len(chosen)==2:
  time.sleep(10);now=gpu_memory()
  if all(eligible(now[x]) for x in chosen):break
 for h in handles:h.close()
 time.sleep(30)
env=os.environ.copy();env['PATH']=str(Path(a.python).parent)+os.pathsep+env.get('PATH','');env['CUDA_VISIBLE_DEVICES']=','.join(map(str,chosen));env['CC']='/workspace/lab-wyc/users/gty/.conda/envs/vllm-compiler/bin/x86_64-conda-linux-gnu-gcc';env['JUDGE_API_KEY']=Path('data/.qwen2vl-judge-key').read_text().strip()
print('Reserved GPUs',env['CUDA_VISIBLE_DEVICES'],flush=True)
if a.smoke_huihui:
 print('Running Huihui 27B smoke test',flush=True)
 with open(a.log,'a') as out: code=subprocess.run([a.python,'-u','scripts/tools/smoke_huihui_vllm.py'],env=env,stdout=out,stderr=subprocess.STDOUT).returncode
 if code:raise SystemExit(f'Huihui smoke test failed: {code}; rollout not started')
print('Starting',a.script,flush=True)
with open(a.log,'a') as out:code=subprocess.run([a.python,'-u',a.script],env=env,stdout=out,stderr=subprocess.STDOUT).returncode
print('Workflow exit',code,flush=True);sys.exit(code)
