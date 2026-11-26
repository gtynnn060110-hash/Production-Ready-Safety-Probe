"""Merge SafeRLHF/QAs and remaining text prompts into a deduplicated prompt index.
Original-answer labels, prompt judgments, and rollout judgments stay separate.
No external calls; source datasets are not modified.
"""
import sys as _prep_sys; from pathlib import Path as _PrepPath; _prep_sys.path.insert(0, str(_PrepPath(__file__).resolve().parents[2])); from scripts._bootstrap import setup as _prep_setup; _prep_setup()
from pathlib import Path
from collections import Counter
import glob, hashlib, json, os, sqlite3, time
from screen_response_gaps import load_inputs, digest, normalize

ROOT=Path('data'); OTHER=ROOT/'remaining-text-response-gaps-v1'; SAFE=ROOT/'saferlhf-response-gaps-v1'; OUT=ROOT/'total-text-prompts-v1';OUT.mkdir(parents=True,exist_ok=True)
DB=OUT/'index.sqlite'
if DB.exists():raise SystemExit(f'{DB} exists; refusing to overwrite')
db=sqlite3.connect(DB);db.executescript('''PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
CREATE TABLE prompts(id TEXT PRIMARY KEY,prompt TEXT NOT NULL);
CREATE TABLE sources(pid TEXT NOT NULL,source TEXT NOT NULL,PRIMARY KEY(pid,source));
CREATE TABLE responses(rid TEXT PRIMARY KEY,pid TEXT NOT NULL,mask INTEGER NOT NULL);
CREATE TABLE response_adjudications(rid TEXT PRIMARY KEY,safety TEXT NOT NULL);
CREATE TABLE prompt_judgments(pid TEXT NOT NULL,origin TEXT NOT NULL,result TEXT NOT NULL,PRIMARY KEY(pid,origin));
CREATE TABLE generated(pid TEXT NOT NULL,rid TEXT NOT NULL,mask INTEGER NOT NULL,PRIMARY KEY(pid,rid));
''')
counts=Counter()
def add_prompt(pid,prompt):
 if digest(normalize(prompt))!=pid:raise ValueError(f'Prompt hash mismatch: {pid}')
 db.execute('INSERT OR IGNORE INTO prompts VALUES(?,?)',(pid,normalize(prompt)))
def add_resp(pid,rid,labels):
 mask=sum(bit for label,bit in [('safe',1),('unsafe',2),('unknown',4)] if label in labels)
 db.execute('INSERT INTO responses VALUES(?,?,?) ON CONFLICT(rid) DO UPDATE SET mask=responses.mask | excluded.mask',(rid,pid,mask))
for name in ['missing_safe','missing_unsafe','missing_both','unresolved','both']:
 path=OTHER/(name+'.jsonl')
 with path.open(encoding='utf8') as f:
  for i,line in enumerate(f,1):
   r=json.loads(line);pid=r['prompt_id'];add_prompt(pid,r['prompt'])
   for source in {x['source'] for x in r['sources']}: db.execute('INSERT OR IGNORE INTO sources VALUES(?,?)',(pid,source))
   for x in r['responses']:add_resp(pid,x['response_id'],x['original_labels'])
   if i%20000==0:db.commit()
 counts['remaining_'+name]=i;db.commit();print('Loaded remaining',name,i,flush=True)
paths=[]
for source in ['saferlhf','saferlhf_qa']:
 for file in sorted(glob.glob(str(ROOT/'raw'/source/'data'/'*'/'train.jsonl'))):paths.append((Path(file).resolve(),None,Path.cwd()))
prompts,stats=load_inputs(paths)
print('Loaded SafeRLHF prompts',len(prompts),flush=True)
for i,(pid,item) in enumerate(prompts.items(),1):
 add_prompt(pid,item['prompt'])
 for source in item['sources']:
  tag='saferlhf_qa' if '/saferlhf_qa/' in source else 'saferlhf'
  db.execute('INSERT OR IGNORE INTO sources VALUES(?,?)',(pid,tag))
 for rid,response in item['responses'].items():add_resp(pid,rid,response['labels']|({'unknown'} if response['has_unknown'] else set()))
 if i%10000==0:db.commit()
db.commit();del prompts
cache=sqlite3.connect('file:'+str(SAFE/'judge_cache.sqlite')+'?mode=ro',uri=True)
for rid,result in cache.execute('select response_id,result from decisions'):
 x=json.loads(result)
 if x.get('safety') in ('safe','unsafe') and x.get('usable') is True:db.execute('INSERT OR REPLACE INTO response_adjudications VALUES(?,?)',(rid,x['safety']))
for pid,result in cache.execute('select prompt_id,result from prompt_decisions'):
 x=json.loads(result)
 if x.get('safety') in ('safe','unsafe') and x.get('usable') is True:
  db.execute('INSERT OR REPLACE INTO prompt_judgments VALUES(?,?,?)',(pid,'saferlhf_prompt_judge',json.dumps({'safety':x['safety'],'categories':x.get('categories',[]),'judge_model':x.get('judge_model'),'taxonomy_version':x.get('taxonomy_version')},ensure_ascii=False)))
cache.close();db.commit()
cache=sqlite3.connect('file:'+str(OTHER/'prompt_judge.sqlite')+'?mode=ro',uri=True)
for pid,result in cache.execute('select prompt_id,result from decisions'):
 x=json.loads(result);db.execute('INSERT OR REPLACE INTO prompt_judgments VALUES(?,?,?)',(pid,'remaining_prompt_judge',json.dumps(x,ensure_ascii=False)))
cache.close();db.commit(); print('Loaded prompt judgments',flush=True)
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
db.executescript('''CREATE INDEX response_pid_idx ON responses(pid); CREATE INDEX gen_pid_idx ON generated(pid);
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
