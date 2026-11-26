"""Export safe/unsafe gap prompt candidates and unjudged rollout regeneration plans."""
import sys as _prep_sys; from pathlib import Path as _PrepPath; _prep_sys.path.insert(0, str(_PrepPath(__file__).resolve().parents[2])); from scripts._bootstrap import setup as _prep_setup; _prep_setup()
from pathlib import Path
from collections import Counter,defaultdict
import json,os

root=Path('data/total-text-prompts-v1'); src=root/'prompts.jsonl'; out=root/'candidates-v1';out.mkdir(exist_ok=True)
outputs={name:(out/(name+'.jsonl')).open('x',encoding='utf8') for name in ('missing_safe_prompts','missing_unsafe_prompts')}
candidates={'safe':{},'unsafe':{}}; stats=Counter()
for line in src.open(encoding='utf8'):
 r=json.loads(line); status=r['combined_status']; label=r['prompt_judge']['safety']; pid=r['prompt_id']
 wants_safe=status in ('missing_safe','missing_both')
 wants_unsafe=status=='missing_both' or (status=='missing_unsafe' and label=='unsafe')
 item={'prompt_id':pid,'prompt':r['prompt'],'sources':r['sources'],'combined_status':status,'prompt_judge':r['prompt_judge']}
 if wants_safe: outputs['missing_safe_prompts'].write(json.dumps(item,ensure_ascii=False)+'\n');candidates['safe'][pid]=item;stats['safe_total']+=1;stats['safe_'+status]+=1
 if wants_unsafe: outputs['missing_unsafe_prompts'].write(json.dumps(item,ensure_ascii=False)+'\n');candidates['unsafe'][pid]=item;stats['unsafe_total']+=1;stats['unsafe_'+status]+=1
 if status=='missing_unsafe' and label!='unsafe':stats['unsafe_excluded_prompt_'+label]+=1
 if status=='unresolved':stats['excluded_original_response_label_unresolved']+=1
for f in outputs.values():f.close()
base=Path('data/remaining-text-response-gaps-v1'); groups=[('unsafe',base/'mixed-qwen3-5epoch-three-rounds-v1'),('safe',base/'missing-both-qwen3-safe-three-rounds-v1'),('unsafe',base/'missing-both-mixed-unsafe-three-rounds-v1'),('safe',base/'qwen3-original-safe-three-rounds-v1')]
regen={'safe':defaultdict(list),'unsafe':defaultdict(list)}; records=Counter()
for target,group in groups:
 for rnd in (1,2,3):
  rollout=group/f'round-{rnd}-rollout.jsonl'; judge=group/f'round-{rnd}-judged.jsonl'; errors=group/f'round-{rnd}-errors.jsonl'
  if not rollout.exists():continue
  judged={json.loads(line)['prompt_id'] for line in judge.open() if line.strip()} if judge.exists() else set()
  errids={json.loads(line)['prompt_id'] for line in errors.open() if line.strip()} if errors.exists() else set()
  with rollout.open() as f:
   for line in f:
    x=json.loads(line);pid=x['prompt_id']
    if pid in judged or pid not in candidates[target]:continue
    reason='judge_failed' if pid in errids else 'not_yet_judged'
    regen[target][pid].append({'workflow':group.name,'round':rnd,'reason':reason,'finish_reason':x.get('finish_reason')})
    records[f'{target}_{reason}_records']+=1
for target in ('safe','unsafe'):
 p=out/f'regenerate_unresolved_{target}.jsonl'
 with p.open('x',encoding='utf8') as f:
  for pid,prior in regen[target].items():
   row={**candidates[target][pid],'previous_unresolved':prior};f.write(json.dumps(row,ensure_ascii=False)+'\n')
 stats['regenerate_'+target+'_unique']=len(regen[target])
result={'candidate_counts':dict(stats),'unresolved_rollout_records':dict(records),'definitions':{'missing_safe_prompts':'combined_status missing_safe or missing_both','missing_unsafe_prompts':'combined_status missing_unsafe and prompt_judge safety unsafe, or combined_status missing_both','regenerate_unresolved':'previously generated but lacking successful judge result, intersected with relevant candidate set; not-yet-judged versus judge-failed separately recorded','both_missing':'appears in both prompt lists','unresolved_original_labels':'separately excluded until original response labels are reconciled','prompt_judge_unresolved':'not used as unsafe evidence; both-missing is included by user request'},'generation':'not started; prepared candidate lists only'}
(out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
