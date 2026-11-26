import sys as _prep_sys; from pathlib import Path as _PrepPath; _prep_sys.path.insert(0, str(_PrepPath(__file__).resolve().parents[2])); from scripts._bootstrap import setup as _prep_setup; _prep_setup()
import csv,hashlib,json,sqlite3,time
from pathlib import Path
from common import LABELS

P=Path('data/total-text-prompts-v1/candidates-v1/production-v1')
src=P/'review-all-qwen3-safe-huihui-unsafe.jsonl'
out=P/'probe-generated-gty-1125.sqlite'
if out.exists():raise SystemExit(f'Output already exists: {out}')
c=sqlite3.connect(out)
c.executescript('''PRAGMA journal_mode=DELETE; PRAGMA foreign_keys=ON;
CREATE TABLE prompts(id TEXT PRIMARY KEY,payload TEXT NOT NULL,label TEXT,state TEXT NOT NULL DEFAULT 'pending',attempts INTEGER NOT NULL DEFAULT 0,error TEXT,updated REAL NOT NULL);
CREATE TABLE responses(id TEXT PRIMARY KEY,prompt_id TEXT NOT NULL REFERENCES prompts(id),text TEXT NOT NULL,origin TEXT NOT NULL,label TEXT,UNIQUE(prompt_id,text));
CREATE TABLE categories(code TEXT PRIMARY KEY,name TEXT NOT NULL);
CREATE TABLE response_categories(response_id TEXT NOT NULL REFERENCES responses(id),category_code TEXT NOT NULL REFERENCES categories(code),PRIMARY KEY(response_id,category_code));
CREATE TABLE category_statistics(category_code TEXT PRIMARY KEY REFERENCES categories(code),category_name TEXT NOT NULL,positive INTEGER NOT NULL,negative INTEGER NOT NULL,negative_safe INTEGER NOT NULL,negative_other_unsafe INTEGER NOT NULL);
CREATE TABLE export_metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
''')
c.executemany('INSERT INTO categories VALUES(?,?)',LABELS.items());total=0; counts={'safe':0,'unsafe':0};category_counts={x:0 for x in LABELS};ts=time.time()
with src.open(encoding='utf8') as f:
 for line in f:
  if not line.strip():continue
  r=json.loads(line);pid=r['prompt_id'];prompt=r['prompt'];text=r['response'];safety=r['label'];cats=r['categories']
  if safety not in counts or not isinstance(cats,list) or len(cats)!=len(set(cats)) or not set(cats)<=set(LABELS) or (safety=='safe' and cats) or (safety=='unsafe' and not cats):raise ValueError(f'invalid label {pid}')
  payload=json.dumps({'prompt':prompt,'context':[],'images':[]},ensure_ascii=False)
  before=c.total_changes;c.execute('INSERT OR IGNORE INTO prompts(id,payload,updated) VALUES(?,?,?)',(pid,payload,ts))
  if c.total_changes==before:
   old=c.execute('SELECT payload FROM prompts WHERE id=?',(pid,)).fetchone()
   if old[0]!=payload:raise ValueError('prompt_id collision: '+pid)
  rid=hashlib.sha256(json.dumps([pid,text],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
  origin=json.dumps({'source':r['source'],'round':r['round'],'kind':'generated'},ensure_ascii=False)
  label=json.dumps({'task':'response','safety':safety,'categories':cats},ensure_ascii=False)
  c.execute('INSERT INTO responses VALUES(?,?,?,?,?)',(rid,pid,text,origin,label))
  c.executemany('INSERT INTO response_categories VALUES(?,?)',[(rid,x) for x in cats])
  for x in cats:category_counts[x]+=1
  counts[safety]+=1;total+=1
  if total%10000==0:c.commit()
if total!=293603 or counts!={'safe':216073,'unsafe':77530}:raise ValueError((total,counts))
stats=[]
for code,name in LABELS.items():
 n=category_counts[code];stats.append((code,name,n,total-n,counts['safe'],counts['unsafe']-n))
c.executemany('INSERT INTO category_statistics VALUES(?,?,?,?,?,?)',stats)
c.executemany('INSERT INTO export_metadata VALUES(?,?)',[(k,json.dumps(v,ensure_ascii=False)) for k,v in {'source':str(src),'responses':total,'safe':counts['safe'],'unsafe':counts['unsafe'],'definition':'One-vs-rest: positive=category present; negative=category absent; labels are multi-label and negatives include unsafe samples of other categories.','warning':'Response labels are inherited judge results; prompt labels remain unset.'}.items()])
c.execute('CREATE INDEX responses_prompt_id ON responses(prompt_id)');c.execute('CREATE INDEX response_categories_code ON response_categories(category_code)');c.commit()
print('integrity',c.execute('pragma integrity_check').fetchone()[0]);print('foreign_key_violations',len(c.execute('pragma foreign_key_check').fetchall()));print('prompts',c.execute('select count(*) from prompts').fetchone()[0],'responses',total,'counts',counts)
with (P/'category-statistics-gty-1125.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['category','name','positive','negative','negative_safe','negative_other_unsafe']);w.writerows(stats)
for row in stats:print('\t'.join(map(str,row)))
c.close()
