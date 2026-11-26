import sys as _prep_sys; from pathlib import Path as _PrepPath; _prep_sys.path.insert(0, str(_PrepPath(__file__).resolve().parents[2])); from scripts._bootstrap import setup as _prep_setup; _prep_setup()
from pathlib import Path
from collections import Counter
import json,sqlite3
from screen_response_gaps import digest,normalize
ROOT=Path('data'); OTHER=ROOT/'remaining-text-response-gaps-v1'; SAFE=ROOT/'saferlhf-response-gaps-v1'; OUT=ROOT/'total-text-prompts-v1'
old=json.loads((OUT/'summary.json').read_text()); counts=Counter(old['input_counts']); counts['generated_judged_complete_records']=0; counts['generated_excluded_incomplete']=0; counts['generated_unsafe_excluded']=0
db=sqlite3.connect(OUT/'index.sqlite'); db.execute('DELETE FROM generated');db.commit()
groups=[SAFE/'qwen3-original-safe-refusal-three-rounds-v1',OTHER/'qwen3-original-safe-three-rounds-v1',OTHER/'missing-both-qwen3-safe-three-rounds-v1',]
for group in groups:
 for rnd in (1,2,3):
  j=group/f'round-{rnd}-judged.jsonl';r=group/f'round-{rnd}-rollout.jsonl'
  if not j.exists() or not r.exists():continue
  labels={}
  with j.open(encoding='utf8') as f:
   for line in f:
    x=json.loads(line);pid=x['prompt_id']
    if pid in labels:raise ValueError(f'duplicate judgment {j} {pid}')
    labels[pid]=x['safety']
  seen=set()
  with r.open(encoding='utf8') as f:
   for line in f:
    x=json.loads(line);pid=x['prompt_id']
    if pid not in labels:continue
    if pid in seen:raise ValueError(f'duplicate rollout {r} {pid}')
    seen.add(pid)
    if x.get('finish_reason')!='stop' or not isinstance(x.get('response'),str) or not x['response'].strip():counts['generated_excluded_incomplete']+=1;continue
    label=labels[pid]
    if label not in ('safe','unsafe'):raise ValueError('Invalid generated label')
    if label != 'safe':
     counts['generated_unsafe_excluded']+=1
     continue
    rid=digest([pid,normalize(x['response'])]);mask=1
    db.execute('INSERT INTO generated VALUES(?,?,?) ON CONFLICT(pid,rid) DO UPDATE SET mask=generated.mask | excluded.mask',(pid,rid,mask));counts['generated_judged_complete_records']+=1
  if seen!=set(labels):raise ValueError(f'missing rollout for judgments: {j}')
  db.commit();print('Loaded generated',group.name,rnd,len(seen),flush=True)
db.executescript('''CREATE INDEX IF NOT EXISTS response_pid_idx ON responses(pid); CREATE INDEX IF NOT EXISTS gen_pid_idx ON generated(pid);
CREATE TEMP TABLE agg_response AS SELECT r.pid,COUNT(*) AS total,SUM(CASE WHEN (r.mask & 3)=1 OR (r.mask & 3)=3 AND a.safety='safe' THEN 1 ELSE 0 END) AS safe,SUM(CASE WHEN (r.mask & 3)=2 OR (r.mask & 3)=3 AND a.safety='unsafe' THEN 1 ELSE 0 END) AS unsafe,SUM(CASE WHEN (r.mask & 3)=0 OR (r.mask & 3)=3 AND a.safety IS NULL THEN 1 ELSE 0 END) AS unresolved FROM responses r LEFT JOIN response_adjudications a ON a.rid=r.rid GROUP BY r.pid;
CREATE UNIQUE INDEX agg_response_pid ON agg_response(pid);
CREATE TEMP TABLE agg_generated AS SELECT pid,COUNT(*) AS total,SUM(CASE WHEN mask=1 THEN 1 ELSE 0 END) AS safe,SUM(CASE WHEN mask=2 THEN 1 ELSE 0 END) AS unsafe,SUM(CASE WHEN mask=3 THEN 1 ELSE 0 END) AS conflicting FROM generated GROUP BY pid;
CREATE UNIQUE INDEX agg_generated_pid ON agg_generated(pid);
''')
def status(safe,unsafe,unresolved):
 if safe and unsafe:return 'both'
 if unresolved:return 'unresolved'
 if safe:return 'missing_unsafe'
 if unsafe:return 'missing_safe'
 return 'missing_both'
summary=Counter(); bysrc=Counter(); out=OUT/'prompts.jsonl';tmp=OUT/'prompts.jsonl.tmp'
query='''SELECT p.id,p.prompt,COALESCE(r.total,0),COALESCE(r.safe,0),COALESCE(r.unsafe,0),COALESCE(r.unresolved,0),COALESCE(g.total,0),COALESCE(g.safe,0),COALESCE(g.unsafe,0),COALESCE(g.conflicting,0) FROM prompts p LEFT JOIN agg_response r ON r.pid=p.id LEFT JOIN agg_generated g ON g.pid=p.id ORDER BY p.id'''
with tmp.open('x',encoding='utf8') as f:
 for pid,prompt,ot,osafe,ounsafe,ou,gt,gsafe,gunsafe,gconf in db.execute(query):
  sources=[x[0] for x in db.execute('SELECT source FROM sources WHERE pid=? ORDER BY source',(pid,))]
  judgments=[{'origin':origin,**json.loads(result)} for origin,result in db.execute('SELECT origin,result FROM prompt_judgments WHERE pid=? ORDER BY origin',(pid,))]
  jvals={j['safety'] for j in judgments}; jsafety=next(iter(jvals)) if len(jvals)==1 else 'conflict' if jvals else 'unjudged'
  original=status(osafe,ounsafe,ou); combined=status(osafe+gsafe,ounsafe+gunsafe,ou+gconf)
  obj={'prompt_id':pid,'prompt':prompt,'sources':sources,'prompt_judge':{'safety':jsafety,'decisions':judgments},'original_responses':{'total':ot,'safe':osafe,'unsafe':ounsafe,'unresolved':ou,'status':original},'generated_responses':{'total':gt,'safe':gsafe,'unsafe':gunsafe,'conflicting':gconf},'combined_status':combined}
  f.write(json.dumps(obj,ensure_ascii=False)+'\n');summary['total_prompts']+=1;summary['original_status_'+original]+=1;summary['combined_status_'+combined]+=1;summary['prompt_judge_'+jsafety]+=1
  for source in sources:bysrc[source]+=1
  if summary['total_prompts']%50000==0:print('Exported',summary['total_prompts'],flush=True)
tmp.replace(out)
summary_obj={'counts':dict(summary),'source_prompt_counts_overlapping':dict(bysrc),'input_counts':dict(counts),'deduplication':'NFC normalized text, CRLF normalized and whitespace trimmed; prompt_id sha256 of JSON-encoded normalized string','scope':'original response labels and existing prompt judgments; generated responses retained only from original Qwen3-8B when judge says safe and finish_reason=stop','original_response_conflict_rule':'same-response original label disagreements are unresolved unless SafeRLHF adjudication exists','prompt_judge_rule':'decisions preserved separately; conflicting labels marked conflict','limitations':['prompt judge only covers previously judged subsets','no medical or biosecurity review added','labels reflect source annotations or existing judges, not verified ground truth','source counts overlap','only judge-safe generated responses from original Qwen3-8B are counted; mixed-adapter outputs are excluded']}
(OUT/'summary.json').write_text(json.dumps(summary_obj,ensure_ascii=False,indent=2)+'\n');db.commit();db.close();print(json.dumps(summary_obj,ensure_ascii=False,indent=2),flush=True)
