"""Export only completed, judge-safe generated responses for the prompt index."""
import sys as _prep_sys; from pathlib import Path as _PrepPath; _prep_sys.path.insert(0, str(_PrepPath(__file__).resolve().parents[2])); from scripts._bootstrap import setup as _prep_setup; _prep_setup()
from pathlib import Path
import json
from screen_response_gaps import digest,normalize
root=Path('data'); safe=root/'saferlhf-response-gaps-v1';other=root/'remaining-text-response-gaps-v1';out=root/'total-text-prompts-v1'/'generated_safe_responses.jsonl';tmp=out.with_suffix('.jsonl.tmp')
groups=[safe/'qwen3-original-safe-refusal-three-rounds-v1',other/'qwen3-original-safe-three-rounds-v1',other/'missing-both-qwen3-safe-three-rounds-v1',]
seen=set(); records=0
with tmp.open('x',encoding='utf8') as dest:
 for group in groups:
  for rnd in (1,2,3):
   j=group/f'round-{rnd}-judged.jsonl'; r=group/f'round-{rnd}-rollout.jsonl'
   if not j.exists() or not r.exists():continue
   labels={}
   with j.open(encoding='utf8') as f:
    for line in f:
     x=json.loads(line)
     if x['safety']=='safe':labels[x['prompt_id']]=x
   with r.open(encoding='utf8') as f:
    for line in f:
     x=json.loads(line); pid=x['prompt_id']
     if pid not in labels or x.get('finish_reason')!='stop' or not isinstance(x.get('response'),str) or not x['response'].strip():continue
     rid=digest([pid,normalize(x['response'])])
     if rid in seen:continue
     seen.add(rid);item={'prompt_id':pid,'response_id':rid,'response':x['response'],'safety':'safe','categories':labels[pid]['categories'],'origin':group.name,'round':rnd};dest.write(json.dumps(item,ensure_ascii=False)+'\n');records+=1
 print('safe records',records,flush=True)
tmp.replace(out)
