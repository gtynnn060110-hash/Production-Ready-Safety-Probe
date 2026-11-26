"""Classify currently unlabeled prompts, writing only validated results to the DB."""
import collections,fcntl,json,os,sqlite3,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from screen_response_gaps import judge_labels

DB='/workspace/models/gty/probe-generated-gty-1125.sqlite'
ROOT=Path('data/total-text-prompts-v1/candidates-v1/production-v1')
ORIGIN='deepseek_v41_flash_prompt_judge_20261126'
SETTINGS={'judge':{'base_url':'https://llmapi.paratera.com','model':'DeepSeek-V4.1-Flash','api_key_env':'JUDGE_API_KEY','temperature':0,'max_tokens':1024,'timeout':180,'retries':2,'json_mode':True,'extra_body':{'thinking':{'type':'disabled'}}}}

def classify(payload):
 value=json.loads(payload)
 if value.get('images'):raise ValueError('Unexpected images in text-only prompt database')
 if not isinstance(value.get('prompt'),str) or not value['prompt'].strip():raise ValueError('Empty prompt')
 result=judge_labels(SETTINGS,{'prompt':value['prompt'],'context':value.get('context',[]),'images':[]})
 if result['safety'] not in ('safe','unsafe'):return 'uncertain',result
 return 'judged',result

def main():
 if not os.environ.get('JUDGE_API_KEY'):raise SystemExit('JUDGE_API_KEY required')
 lock=(ROOT/'gty1125-prompt-judge.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 db=sqlite3.connect('file:'+DB+'?mode=rw',uri=True,timeout=60);db.execute('pragma foreign_keys=on');db.execute('pragma busy_timeout=60000')
 db.execute('''create table if not exists prompt_judge_attempts (prompt_id text primary key references prompts(id), status text not null, result text, error_type text, updated real not null)''');db.commit()
 total=db.execute('select count(*) from prompts where label is null').fetchone()[0]
 pending=db.execute("select count(*) from prompts p left join prompt_judge_attempts a on a.prompt_id=p.id where p.label is null and a.prompt_id is null").fetchone()[0]
 print(f'start total_unlabeled={total} pending_new={pending} model={SETTINGS["judge"]["model"]} workers=4 batch=64',flush=True)
 started=time.monotonic();attempted=failed=0;counts=collections.Counter();cursor=''
 with ThreadPoolExecutor(max_workers=4) as pool:
  while True:
   rows=db.execute("select p.id,p.payload from prompts p left join prompt_judge_attempts a on a.prompt_id=p.id where p.id>? and p.label is null and a.prompt_id is null order by p.id limit 64",(cursor,)).fetchall()
   if not rows:break
   jobs={pool.submit(classify,payload):pid for pid,payload in rows};results=[]
   for fut in as_completed(jobs):
    pid=jobs[fut]
    try:status,result=fut.result();results.append((pid,status,result,None))
    except Exception as e:results.append((pid,'error',None,type(e).__name__))
   with db:
    for pid,status,result,error in results:
     attempted+=1;counts[status]+=1
     if status=='judged':
      label={'task':'input',**result};raw=json.dumps(result,ensure_ascii=False)
      changed=db.execute('update prompts set label=? where id=? and label is null',(json.dumps(label,ensure_ascii=False),pid)).rowcount
      if changed!=1:raise ValueError('Prompt label changed concurrently: '+pid)
      db.execute('insert into prompt_judgment_provenance(prompt_id,origin,result) values(?,?,?)',(pid,ORIGIN,raw))
     else:failed+=1
     db.execute('insert into prompt_judge_attempts values(?,?,?,?,?)',(pid,status,json.dumps(result,ensure_ascii=False) if result is not None else None,error,time.time()))
   cursor=rows[-1][0];elapsed=time.monotonic()-started
   print(f'attempted={attempted}/{pending} judged={counts["judged"]} safe={sum(1 for _,s,r,_ in results if s=="judged" and r["safety"]=="safe")} unsafe={sum(1 for _,s,r,_ in results if s=="judged" and r["safety"]=="unsafe")} uncertain={counts["uncertain"]} errors={counts["error"]} elapsed_s={elapsed:.1f} rate_per_hour={attempted*3600/elapsed:.1f}',flush=True)
   if attempted>=64 and failed/attempted>.20:raise SystemExit(f'Stopped: unresolved or failed rate {failed/attempted:.1%} exceeds 20%; progress saved')
 print('PROMPT JUDGE PASS COMPLETE',dict(counts),flush=True);db.close()
if __name__=='__main__':main()
