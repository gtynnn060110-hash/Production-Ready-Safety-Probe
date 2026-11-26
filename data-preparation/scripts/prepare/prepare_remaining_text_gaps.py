"""Organize remaining text sources using original response labels only.

No model or API calls. Unknown/conflicting responses remain unresolved.
"""
import sys as _prep_sys; from pathlib import Path as _PrepPath; _prep_sys.path.insert(0, str(_PrepPath(__file__).resolve().parents[2])); from scripts._bootstrap import setup as _prep_setup; _prep_setup()
import collections, csv, hashlib, json, sqlite3, unicodedata, sys
csv.field_size_limit(sys.maxsize)
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.ipc as ipc
from screen_response_gaps import read_records

ROOT=Path('data/raw'); OUT=Path('data/remaining-text-response-gaps-v1'); OUT.mkdir(parents=True,exist_ok=True)
DB=OUT/'organized.sqlite'
if DB.exists(): raise SystemExit(f'{DB} already exists; choose a fresh output directory')
db=sqlite3.connect(DB)
db.executescript('''PRAGMA journal_mode=WAL; CREATE TABLE prompts(id TEXT PRIMARY KEY,prompt TEXT NOT NULL); CREATE TABLE responses(id TEXT PRIMARY KEY,prompt_id TEXT NOT NULL,response TEXT NOT NULL); CREATE TABLE response_labels(response_id TEXT NOT NULL,label TEXT NOT NULL,source TEXT NOT NULL,source_ref TEXT NOT NULL, UNIQUE(response_id,label,source,source_ref)); CREATE TABLE provenance(prompt_id TEXT NOT NULL,source TEXT NOT NULL,source_ref TEXT NOT NULL, UNIQUE(prompt_id,source,source_ref));''')
counts=collections.Counter()
def norm(s): return unicodedata.normalize('NFC',s).replace('\r\n','\n').strip()
def digest(x): return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def add(source,path,idx,prompt,response=None,label=None):
 if not isinstance(prompt,str) or not prompt.strip() or prompt.strip()=='REDACTED': counts[source+':invalid_prompt']+=1; return
 prompt=norm(prompt); pid=digest(prompt); ref=f'{path}:{idx}'
 db.execute('INSERT OR IGNORE INTO prompts VALUES(?,?)',(pid,prompt)); db.execute('INSERT OR IGNORE INTO provenance VALUES(?,?,?)',(pid,source,ref)); counts[source+':records']+=1
 if isinstance(response,str) and response.strip() and response.strip().lower() not in ('nan','none','null'):
  text=norm(response); rid=digest([pid,text]); db.execute('INSERT OR IGNORE INTO responses VALUES(?,?,?)',(rid,pid,text)); db.execute('INSERT OR IGNORE INTO response_labels VALUES(?,?,?,?)',(rid,label or 'unknown',source,ref)); counts[source+':response_'+(label or 'unknown')]+=1

def run(source,paths,extract):
 for path in paths:
  for i,row in enumerate(read_records(path),1):
   p,r,l=extract(row); add(source,str(path),i,p,r,l)
   if i%20000==0: db.commit(); print(source,path.name,i,flush=True)
  db.commit(); print(source,path.name,'done',flush=True)
run('beavertails',[ROOT/'beavertails/round0/330k/train.jsonl.xz'],lambda r:(r.get('prompt'),r.get('response'),('safe' if r.get('is_safe') is True else 'unsafe' if r.get('is_safe') is False else None)))
run('wildguard',sorted((ROOT/'wildguard/train').glob('*.parquet')),lambda r:(r.get('prompt'),r.get('response'),{'unharmful':'safe','harmful':'unsafe'}.get(r.get('response_harm_label'))))
run('aegis2',[ROOT/'aegis2/train.json',ROOT/'aegis2/refusals_train.json'],lambda r:(None if r.get('reconstruction_id_if_redacted') else r.get('prompt'),r.get('response'),{'safe':'safe','unsafe':'unsafe'}.get(r.get('response_label'))))
run('toxicchat',[ROOT/'toxicchat/data/0124/toxic-chat_annotation_train.csv'],lambda r:(r.get('user_input'),r.get('model_output'),None))
def wild(row):
 t=row.get('data_type',''); prompt=row.get('adversarial') if t.startswith('adversarial_') else row.get('vanilla'); label='safe' if t in ('vanilla_harmful','vanilla_benign','adversarial_harmful','adversarial_benign') else None
 return prompt,row.get('completion'),label
run('wildjailbreak',[ROOT/'wildjailbreak/train/train.tsv'],wild)
run('seval',sorted((ROOT/'seval').glob('S-Eval_*_full.jsonl')),lambda r:(r.get('prompt'),None,None))
paths=sorted((ROOT/'streamguard/wildguard').glob('*/train/*.arrow'))+sorted((ROOT/'streamguard/s_eval').glob('*/train/*.arrow'))
run('streamguard_text',paths,lambda r:(r.get('prompt'),r.get('response'),{0:'safe',1:'unsafe'}.get(r.get('label')) if type(r.get('label')) is int else None))
db.executescript('CREATE INDEX responses_prompt_idx ON responses(prompt_id); CREATE INDEX response_labels_id_idx ON response_labels(response_id); CREATE INDEX provenance_prompt_idx ON provenance(prompt_id);');db.commit()
outputs={k:(OUT/(k+'.jsonl')).open('x',encoding='utf8') for k in ('missing_safe','missing_unsafe','missing_both','unresolved','both')}; categories=collections.Counter()
q='''SELECT p.id,p.prompt,r.id,r.response,GROUP_CONCAT(DISTINCT l.label) FROM prompts p LEFT JOIN responses r ON r.prompt_id=p.id LEFT JOIN response_labels l ON l.response_id=r.id GROUP BY p.id,r.id ORDER BY p.id'''
cur=db.execute(q); current=None; rows=[]
def emit(pid,prompt,rows):
 if pid is None:return
 labels=set(); unresolved=False; responses=[]
 for rid,text,joined in rows:
  if rid is None:continue
  ls=set((joined or '').split(',')) - {''}; definite=ls & {'safe','unsafe'}
  if len(definite)>1 or not definite: unresolved=True
  labels.update(definite if len(definite)==1 else set()); responses.append({'response_id':rid,'response':text,'original_labels':sorted(ls)})
 cat='both' if labels=={'safe','unsafe'} else 'unresolved' if unresolved else 'missing_unsafe' if labels=={'safe'} else 'missing_safe' if labels=={'unsafe'} else 'missing_both'
 sources=[{'source':x[0],'source_ref':x[1]} for x in db.execute('SELECT source,source_ref FROM provenance WHERE prompt_id=? ORDER BY source,source_ref',(pid,))]
 outputs[cat].write(json.dumps({'prompt_id':pid,'prompt':prompt,'sources':sources,'responses':responses,'existing_labels':sorted(labels)},ensure_ascii=False)+'\n'); categories[cat]+=1
for pid,prompt,rid,text,joined in cur:
 if current!=pid:
  if current is not None: emit(current,oldprompt,rows)
  current,oldprompt,rows=pid,prompt,[]
 rows.append((rid,text,joined))
emit(current,oldprompt,rows)
for f in outputs.values():f.close()
summary={'source_counts':dict(counts),'categories':dict(categories),'label_note':'wildjailbreak completion labeled safe according to dataset card; toxicchat response label unknown; seval prompt-only; streamguard only text s_eval and wildguard','exclusions':['saferlhf','saferlhf_qa','multimodal sources'], 'dedup':'normalized exact prompt and response across included sources only','risk_review':'unreviewed; unsafe generation requires independent biosafety review'}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
db.close()
