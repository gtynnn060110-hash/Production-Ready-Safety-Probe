"""Resume Huihui rollout after unexpected failures, with conservative GPU reservations."""
import sys as _prep_sys; from pathlib import Path as _PrepPath; _prep_sys.path.insert(0, str(_PrepPath(__file__).resolve().parents[2])); from scripts._bootstrap import setup as _prep_setup; _prep_setup()
import fcntl, os, re, signal, subprocess, sys, time
from pathlib import Path

ROOT=Path('data/total-text-prompts-v1/candidates-v1/production-v1')
LOG=ROOT/'huihui-unsafe.log'
STOP=ROOT/'huihui-watchdog.stop'
PYTHON='/workspace/lab-wyc/users/gty/.venvs/vllm-qwen35/bin/python'
SCRIPT='scripts/rollout/rollout_total_huihui_unsafe_three_rounds.py'

def identity(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text()
        fields=stat[stat.rfind(')')+2:].split()
        return (fields[0],fields[19])
    except (FileNotFoundError,ProcessLookupError): return None

def outcome(text):
    s=text.lower()
    if 'three-round workflow complete' in s or re.search(r'round [123]: no eligible prompts; finished',s): return 'complete'
    if 'unresolved after retries' in s or 'existing manifest differs' in s or 'existing rollout mismatches' in s or 'existing judgment mismatch' in s or 'invalid input' in s: return 'manual'
    if any(x in s for x in ('out of memory','outofmemory','no available memory for the cache blocks','free memory on device','cuda_error_out_of_memory','insufficient memory')): return 'memory'
    return 'unexpected'

def excerpt(offset):
    with LOG.open('rb') as f:
        f.seek(offset); return f.read().decode('utf8','replace')

def main():
    pid=int(sys.argv[1]); ident=identity(pid)
    if not ident or ident[0]=='Z': raise SystemExit('Initial rollout wrapper is not running')
    lock=(ROOT/'huihui-watchdog.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if STOP.exists(): raise SystemExit(f'Remove stop file first: {STOP}')
    offset=LOG.stat().st_size; fraction=.6; failures=0
    print(f'Monitoring initial wrapper pid={pid}, start={ident[1]}, gpu_fraction={fraction}',flush=True)
    while True:
        while True:
            if STOP.exists(): print('Stop file detected; watchdog exiting',flush=True); return
            current=identity(pid)
            if current is None or current[0]=='Z' or current[1]!=ident[1]: break
            time.sleep(15)
        text=excerpt(offset); status=outcome(text)
        print(f'Wrapper exited; classification={status}',flush=True)
        if status in ('complete','manual'):
            print('Workflow finished' if status=='complete' else 'Non-GPU data or judge failure; stop for review',flush=True); return
        failures+=1
        if failures>5: print('Stopped after 5 automatic restarts; review logs',flush=True); return
        if status=='memory': fraction=min(.85,round(fraction+.1,2))
        print(f'Restart {failures}/5; waiting for two GPUs with {fraction:.0%} reserved plus 8192 MiB free headroom',flush=True)
        time.sleep(30)
        if STOP.exists(): print('Stop file detected; watchdog exiting',flush=True); return
        offset=LOG.stat().st_size
        env=os.environ.copy();env['HUIHUI_GPU_MEMORY_UTILIZATION']=str(fraction)
        args=[sys.executable,'-u','scripts/orchestration/run_on_any_two_free_gpus.py','--python',PYTHON,'--script',SCRIPT,'--log',str(LOG),'--min-free-fraction',str(fraction),'--headroom-mib','8192']
        child=subprocess.Popen(args,env=env,start_new_session=True)
        pid=child.pid; ident=identity(pid)
        if not ident: raise RuntimeError('Failed to monitor restarted wrapper')
        while child.poll() is None:
            if STOP.exists():
                print('Stop file detected; terminating queued/managed wrapper',flush=True)
                os.killpg(child.pid, signal.SIGTERM); child.wait(timeout=30); return
            time.sleep(15)
        print(f'Restarted wrapper returncode={child.returncode}',flush=True)
        # Reap now, then evaluate the log for this attempt on the next loop.
        ident=('Z',ident[1])

if __name__=='__main__':main()
