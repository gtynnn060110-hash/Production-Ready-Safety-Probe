"""Copy existing prompt judgments into the generated-response database."""
import sys as _prep_sys; from pathlib import Path as _PrepPath; _prep_sys.path.insert(0, str(_PrepPath(__file__).resolve().parents[2])); from scripts._bootstrap import setup as _prep_setup; _prep_setup()
import collections,json,sqlite3
from pathlib import Path
TARGET=Path('/workspace/models/gty/probe-generated-gty-1125.sqlite')
UPSTREAM=Path('data/total-text-prompts-v1/index.sqlite').resolve()
CATS={f'R{i:02}' for i in range(1,20)}

def main():
 c=sqlite3.connect('file:'+str(TARGET)+'?mode=rw',uri=True,timeout=60);c.execute('pragma foreign_keys=on');c.execute('attach database ? as upstream',('file:'+str(UPSTREAM)+'?mode=ro',));c.execute('begin immediate')
 try:
  if c.execute('select count(*) from prompts where label is not null').fetchone()[0]!=0:raise ValueError('Prompt labels already exist; refusing overwrite')
  c.execute('create table if not exists prompt_judgment_provenance (prompt_id text not null references prompts(id), origin text not null, result text not null, primary key(prompt_id,origin))')
  records=collections.defaultdict(list)
  for pid,origin,raw in c.execute('select p.id,j.origin,j.result from prompts p join upstream.prompt_judgments j on j.pid=p.id'):
   r=json.loads(raw);s=r.get('safety');cats=r.get('categories')
   if s not in ('safe','unsafe') or not isinstance(cats,list) or len(cats)!=len(set(cats)) or not set(cats)<=CATS or (s=='safe' and cats) or (s=='unsafe' and not cats):raise ValueError('Invalid judgment for '+pid)
   records[pid].append((origin,raw,s,tuple(sorted(cats))))
  if len(records)!=31803:raise ValueError(f'Unexpected matching prompt count: {len(records)}')
  counts=collections.Counter();provenance=0
  for pid,items in records.items():
   values={(x[2],x[3]) for x in items}
   if len(values)!=1:raise ValueError('Conflicting judgments: '+pid)
   safety,cats=next(iter(values));label=json.dumps({'task':'input','safety':safety,'categories':list(cats)},ensure_ascii=False)
   if c.execute('update prompts set label=? where id=? and label is null',(label,pid)).rowcount!=1:raise ValueError('Update failed: '+pid)
   for origin,raw,_,_ in items:c.execute('insert into prompt_judgment_provenance values(?,?,?)',(pid,origin,raw));provenance+=1
   counts[safety]+=1
  info={'prompt_labels_source':str(UPSTREAM),'prompt_labels_matched':len(records),'prompt_labels_safe':counts['safe'],'prompt_labels_unsafe':counts['unsafe'],'prompt_labels_unmatched':'left null; no inference from response labels','prompt_label_provenance_table':'prompt_judgment_provenance'}
  for k,v in info.items():c.execute('insert or replace into export_metadata(key,value) values (?,?)',(k,json.dumps(v,ensure_ascii=False)))
  c.execute('insert or replace into export_metadata(key,value) values (?,?)',('warning',json.dumps('Response labels are inherited judge results; prompt labels are partially populated from upstream judgments.',ensure_ascii=False)))
  c.commit();print('merged',dict(counts),'provenance_rows',provenance,'unlabeled',c.execute('select count(*) from prompts where label is null').fetchone()[0]);print('responses',c.execute('select count(*),sum(label is not null) from responses').fetchone());print('quick_check',c.execute('pragma quick_check').fetchone()[0]);print('foreign_key_violations',len(c.execute('pragma foreign_key_check').fetchall()))
 except BaseException:
  c.rollback();raise
 finally:c.close()
if __name__=='__main__':main()
