"""Keep non-truncated judge-unsafe responses, excluding high-confidence prompt echoes."""
import sys as _prep_sys; from pathlib import Path as _PrepPath; _prep_sys.path.insert(0, str(_PrepPath(__file__).resolve().parents[2])); from scripts._bootstrap import setup as _prep_setup; _prep_setup()
import ast,json,re,unicodedata,collections
from pathlib import Path
ROOT=Path('data/total-text-prompts-v1/candidates-v1/production-v1/huihui-unsafe-three-rounds')

def norm(s):
 return ''.join(c for c in unicodedata.normalize('NFKC',s).casefold() if c.isalnum())

def literal(node,variables):
 if isinstance(node,ast.Constant) and isinstance(node.value,str):return node.value
 if isinstance(node,ast.Name):return variables.get(node.id)
 if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):
  a,b=literal(node.left,variables),literal(node.right,variables)
  return a+b if isinstance(a,str) and isinstance(b,str) else None
 if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='format' and not node.keywords:
  template=literal(node.func.value,variables);args=[literal(x,variables) for x in node.args]
  if isinstance(template,str) and all(isinstance(x,str) for x in args) and len(template)<10000:
   try:return template.format(*args)
   except (ValueError,IndexError,KeyError):return None
 return None

def code_print_outputs(prompt):
 outputs=[]
 for block in re.findall(r'```(?:python|py)?\s*\n(.*?)```',prompt,flags=re.I|re.S):
  try:tree=ast.parse(block)
  except (SyntaxError,ValueError):continue
  variables={}
  for item in tree.body:
   if isinstance(item,ast.Assign) and len(item.targets)==1 and isinstance(item.targets[0],ast.Name):
    val=literal(item.value,variables)
    if val is not None:variables[item.targets[0].id]=val
   elif isinstance(item,ast.Expr) and isinstance(item.value,ast.Call) and isinstance(item.value.func,ast.Name) and item.value.func.id=='print' and len(item.value.args)==1:
    val=literal(item.value.args[0],variables)
    if val is not None:outputs.append(val)
 return outputs

def reason(prompt,response):
 a=norm(response)
 if len(a)<8:return None
 for out in code_print_outputs(prompt):
  if a==norm(out):return 'code_print_output_only'
 if len(a)>=25 and len(response)<=300 and a in norm(prompt):return 'prompt_substring_only'
 return None

def main():
 labels={}
 with (ROOT/'round-1-judged.jsonl').open(encoding='utf8') as f:
  for line in f:
   if line.strip():
    r=json.loads(line)
    if r['safety']=='unsafe':labels[r['prompt_id']]=r
 kept=ROOT/'round-1-unsafe-nontruncated-no-prompt-echo.jsonl';excluded=ROOT/'round-1-unsafe-prompt-echo-excluded.jsonl';counts=collections.Counter();seen=set()
 with (ROOT/'round-1-rollout.jsonl').open(encoding='utf8') as src,kept.open('w',encoding='utf8') as yes,excluded.open('w',encoding='utf8') as no:
  for line in src:
   r=json.loads(line);pid=r['prompt_id'];lab=labels.get(pid)
   if lab is None or r.get('finish_reason')=='length':continue
   if pid in seen:raise ValueError('duplicate prompt id')
   seen.add(pid);why=reason(r['prompt'],r['response']);item={**r,'label':lab['safety'],'categories':lab['categories']}
   if why:item['exclusion_reason']=why;no.write(json.dumps(item,ensure_ascii=False)+'\n');counts[why]+=1
   else:yes.write(json.dumps(item,ensure_ascii=False)+'\n');counts['kept']+=1
 print(json.dumps({'total':len(seen),**counts},ensure_ascii=False))
if __name__=='__main__':main()
