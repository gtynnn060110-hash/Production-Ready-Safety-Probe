"""Classify missing-unsafe text prompts; no response generation."""
import sys as _prep_sys; from pathlib import Path as _PrepPath; _prep_sys.path.insert(0, str(_PrepPath(__file__).resolve().parents[2])); from scripts._bootstrap import setup as _prep_setup; _prep_setup()
import json,os,sqlite3,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from screen_response_gaps import judge_labels

root=Path('data/remaining-text-response-gaps-v1'); src=root/'missing_unsafe.jsonl'
if not os.getenv('JUDGE_API_KEY'): raise SystemExit('JUDGE_API_KEY not set')
settings={'judge':{'base_url':'https://llmapi.paratera.com','model':'DeepSeek-V4.1-Flash','api_key_env':'JUDGE_API_KEY','temperature':0,'max_tokens':1024,'timeout':180,'retries':2,'json_mode':True,'extra_body':{'thinking':{'type':'disabled'}}}}
db=sqlite3.connect(root/'prompt_judge.sqlite');db.execute('CREATE TABLE IF NOT EXISTS decisions (prompt_id TEXT PRIMARY KEY, safety TEXT NOT NULL, result TEXT NOT NULL)');db.execute('CREATE TABLE IF NOT EXISTS errors (prompt_id TEXT PRIMARY KEY, error TEXT NOT NULL)');db.commit()
def batch_process(batch):
 with ThreadPoolExecutor(max_workers=4) as pool:
  jobs={pool.submit(judge_labels,settings,{'prompt':r['prompt'],'context':[],'images':[]}):r for r in batch}
  for fut in as_completed(jobs):
   r=jobs[fut]
   try:
    decision=fut.result()
    if decision['safety'] not in ('safe','unsafe'): raise ValueError('uncertain')
    db.execute('INSERT OR REPLACE INTO decisions VALUES(?,?,?)',(r['prompt_id'],decision['safety'],json.dumps(decision,ensure_ascii=False)));db.execute('DELETE FROM errors WHERE prompt_id=?',(r['prompt_id'],))
   except Exception as e:db.execute('INSERT OR REPLACE INTO errors VALUES(?,?)',(r['prompt_id'],type(e).__name__))
  db.commit()
count=0; batch=[]
with src.open(encoding='utf8') as f:
 for line in f:
  row=json.loads(line); count+=1
  if db.execute('SELECT 1 FROM decisions WHERE prompt_id=?',(row['prompt_id'],)).fetchone():continue
  batch.append(row)
  if len(batch)>=64:
   batch_process(batch); batch=[]
   print(f'Processed first {count} prompts; classified={db.execute("SELECT count(*) FROM decisions").fetchone()[0]}; unresolved={db.execute("SELECT count(*) FROM errors").fetchone()[0]}',flush=True)
if batch:batch_process(batch)
print('Initial pass done; classified',db.execute('SELECT count(*) FROM decisions').fetchone()[0],'unresolved',db.execute('SELECT count(*) FROM errors').fetchone()[0],flush=True)
for label in ('safe','unsafe'):
 with (root/f'missing_unsafe_prompt_{label}.jsonl').open('w',encoding='utf8') as out, src.open(encoding='utf8') as f:
  for line in f:
   row=json.loads(line); decision=db.execute('SELECT result FROM decisions WHERE prompt_id=? AND safety=?',(row['prompt_id'],label)).fetchone()
   if decision: row['prompt_label']=json.loads(decision[0]);out.write(json.dumps(row,ensure_ascii=False)+'\n')
print('Finished; unresolved prompts can be retried by re-running',flush=True)
