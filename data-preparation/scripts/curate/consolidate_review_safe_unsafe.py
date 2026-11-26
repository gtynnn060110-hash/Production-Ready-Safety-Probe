"""Consolidate original Qwen3 safe responses and filtered Huihui unsafe responses."""
import sys as _prep_sys; from pathlib import Path as _PrepPath; _prep_sys.path.insert(0, str(_PrepPath(__file__).resolve().parents[2])); from scripts._bootstrap import setup as _prep_setup; _prep_setup()
import collections,hashlib,json,sqlite3,unicodedata
from pathlib import Path
B=Path('data/total-text-prompts-v1');P=B/'candidates-v1/production-v1';OUT=P/'review-all-qwen3-safe-huihui-unsafe.jsonl';TMP=OUT.with_suffix('.jsonl.tmp')

def read(path):
 with path.open(encoding='utf8') as f:
  for n,line in enumerate(f,1):
   if line.strip():
    try:yield json.loads(line)
    except json.JSONDecodeError as e:raise ValueError(f'{path}:{n}: invalid JSON') from e

def norm(s):return unicodedata.normalize('NFC',s).replace('\r\n','\n').strip()

def labels(path):
 out={}
 for r in read(path):
  if r['prompt_id'] in out:raise ValueError(f'duplicate judgment: {path}')
  out[r['prompt_id']]=r
 return out

def main():
 count=collections.Counter();seen=set();con=sqlite3.connect('file:'+str(B/'index.sqlite')+'?mode=ro',uri=True)
 with TMP.open('w',encoding='utf8') as out:
  def emit(pid,prompt,response,label,categories,origin,rnd):
   if not isinstance(prompt,str) or not norm(prompt):raise ValueError('empty prompt '+str(pid))
   if not isinstance(response,str) or not norm(response):count['empty_response_skipped:'+label]+=1;return
   key=hashlib.sha256(json.dumps([norm(prompt),norm(response)],ensure_ascii=False,separators=(',',':')).encode()).digest()
   if key in seen:count['duplicate_skipped']+=1;return
   seen.add(key);item={'prompt':prompt,'response':response,'label':label,'categories':categories,'source':origin,'round':rnd,'prompt_id':pid};out.write(json.dumps(item,ensure_ascii=False)+'\n');count[label]+=1;count['source:'+origin]+=1
  for r in read(B/'generated_safe_responses.jsonl'):
   if r['safety']!='safe':raise ValueError('unsafe in safe cache')
   q=con.execute('SELECT prompt FROM prompts WHERE id=?',(r['prompt_id'],)).fetchone()
   if q is None:raise ValueError('missing prompt '+r['prompt_id'])
   emit(r['prompt_id'],q[0],r['response'],'safe',r['categories'],r['origin'],r['round'])
  sel={r['prompt_id'] for r in read(P/'safe_existing_to_judge.jsonl')};d=Path('data/remaining-text-response-gaps-v1/missing-both-qwen3-safe-three-rounds-v1');judged=labels(d/'round-1-judged.jsonl')
  for r in read(d/'round-1-rollout.jsonl'):
   if r['prompt_id'] not in sel:continue
   j=judged.get(r['prompt_id'])
   if j and j['safety']=='safe' and r['finish_reason']!='length':emit(r['prompt_id'],r['prompt'],r['response'],'safe',j['categories'],'existing-qwen3-safe-rejudged',1)
  d=B/'candidates-v1/regenerated-truncated-safe-v1'
  for rnd in (1,2,3):
   judged=labels(d/f'round-{rnd}-judged.jsonl')
   for r in read(d/f'round-{rnd}-rollout.jsonl'):
    j=judged.get(r['prompt_id'])
    if j and j['safety']=='safe' and r['finish_reason']!='length':emit(r['prompt_id'],r['prompt'],r['response'],'safe',j['categories'],'qwen3-regenerated-truncated',rnd)
  d=P/'qwen3-safe-three-rounds';judged=labels(d/'round-1-judged.jsonl')
  for r in read(d/'round-1-rollout.jsonl'):
   j=judged.get(r['prompt_id'])
   if j and j['safety']=='safe' and r['finish_reason']!='length':emit(r['prompt_id'],r['prompt'],r['response'],'safe',j['categories'],'qwen3-new-safe',1)
  for r in read(P/'huihui-unsafe-three-rounds/round-1-unsafe-nontruncated-no-prompt-echo.jsonl'):
   if r['label']!='unsafe' or r['finish_reason']=='length':raise ValueError('invalid unsafe input')
   emit(r['prompt_id'],r['prompt'],r['response'],'unsafe',r['categories'],'huihui-unsafe-filtered',r['round'])
 con.close();TMP.replace(OUT)
 summary={'total':count['safe']+count['unsafe'],'safe':count['safe'],'unsafe':count['unsafe'],'duplicates_skipped':count['duplicate_skipped'],'empty_safe_skipped':count['empty_response_skipped:safe'],'empty_unsafe_skipped':count['empty_response_skipped:unsafe'],'by_source':{k.removeprefix('source:'):v for k,v in count.items() if k.startswith('source:')},'deduplication':'NFC-normalized prompt and response pair','note':'Generated Qwen3-8B safe and Huihui unsafe only; excludes truncated and judge-unsafe Qwen3, and excludes prompt echoes from Huihui.'}
 (P/'review-all-qwen3-safe-huihui-unsafe-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
