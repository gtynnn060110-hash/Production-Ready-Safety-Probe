#!/usr/bin/env python3
"""Joint text/image 19-category response probe training with prompt-grouped splits."""
import argparse,hashlib,json,math,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score,roc_auc_score
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor,AutoModelForMultimodalLM
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'text_eval/src'))
from probe.contracts import FeatureBatch
from probe.probes.mqa_rmattn import MQARMAttnProbe
from probe.losses.token import SingProbeWeighted
HEADS=[f'R{i:02}' for i in range(1,20)]+['overall_unsafe']

def rows(path,limit=None):
 result=[]
 with path.open(encoding='utf8') as f:
  for line in f:
   if line.strip():result.append(json.loads(line))
   if limit and len(result)>=limit:break
 return result

def atomic_save(value,path):
 tmp=path.with_name('.'+path.name+'.tmp');torch.save(value,tmp);os.replace(tmp,path)

def final_response(text):
 if not isinstance(text,str):raise ValueError("Response must be a string")
 if '</think>' in text:
  thinking,text=text.rsplit('</think>',1);thinking=thinking.removeprefix('<think>').strip();text=text.lstrip()
  # A repeated thinking block after the marker is not a final answer.
  if not text.strip() or (thinking and thinking==text.strip()):return None
 if '<think>' in text or '</think>' in text:raise ValueError("Unclosed or nested thinking marker")
 if not text.strip():return None
 return text

def inputs_for(row,processor,device):
 images=[Image.open(path).convert('RGB') for path in row['images']]
 try:
  content=[{'type':'image','image':im} for im in images]
  if row['prompt']:content.append({'type':'text','text':row['prompt']})
  msgs=list(row['context'])+[{'role':'user','content':content}]
  kwargs=dict(tokenize=True,return_dict=True,return_tensors='pt',enable_thinking=False)
  prefix=processor.apply_chat_template(msgs,add_generation_prompt=True,**kwargs)
  answer=final_response(row['response'])
  if answer is None:raise ValueError('No final response: '+row['id'])
  full=processor.apply_chat_template(msgs+[{'role':'assistant','content':[{'type':'text','text':answer}]}],add_generation_prompt=False,**kwargs)
  prefix_ids=prefix['input_ids'][0];n=prefix_ids.shape[0];ids=full['input_ids'][0]
  if not torch.equal(ids[:n],prefix_ids):raise ValueError('Assistant prefix token mismatch: '+row['id'])
  end_id=processor.tokenizer.convert_tokens_to_ids('<|im_end|>');ends=(ids==end_id).nonzero().flatten();end=int(ends[-1]) if len(ends) else len(ids)
  if end<=n:raise ValueError('Empty response token span: '+row['id'])
  return full.to(device),n,end
 finally:
  for im in images:im.close()

def extract(row,processor,backbone,device,layer,max_length,drop_long):
 if final_response(row['response']) is None:return 'no_final_response'
 inputs,start,end=inputs_for(row,processor,device);length=inputs['input_ids'].shape[1]
 if drop_long and max_length>0 and length>max_length:return None
 with torch.no_grad():out=backbone(**inputs,output_hidden_states=True,use_cache=False,return_dict=True)
 hidden=out.hidden_states[layer][0].detach().float();
 if hidden.shape[0]!=length:raise ValueError('Hidden-token alignment mismatch: '+row['id'])
 hidden=hidden*torch.rsqrt(hidden.square().mean(-1,keepdim=True)+1e-6)
 response_mask=torch.zeros(length,dtype=torch.bool,device=device);response_mask[start:end]=True
 y=torch.tensor([float(c in row['categories']) for c in HEADS[:19]]+[float(row['safety']=='unsafe')],device=device)
 return hidden,response_mask,y

def collate(examples):
 h=pad_sequence([e[0] for e in examples],batch_first=True);rm=pad_sequence([e[1] for e in examples],batch_first=True);lens=torch.tensor([len(e[0]) for e in examples],device=h.device);attention=torch.arange(h.shape[1],device=h.device)[None,:]<lens[:,None];y=torch.stack([e[2] for e in examples]);return FeatureBatch(ids=[],hidden_states=h,attention_mask=attention,response_mask=rm,labels=y,label_mask=torch.ones_like(y,dtype=torch.bool))

def evaluate(records,processor,backbone,device,probe,layer):
 ys=[];scores=[];probe.eval()
 with torch.no_grad():
  for n,row in enumerate(records,1):
   e=extract(row,processor,backbone,device,layer,0,False)
   if e=='no_final_response':continue
   batch=collate([e]);o=probe(batch);x=o.local_logits.masked_fill(~o.local_mask[:,:,None],-torch.inf).max(1).values.sigmoid()
   ys.append(e[2].cpu().numpy());scores.append(x[0].cpu().numpy())
   if n%1000==0:print(f'validation {n}/{len(records)}',flush=True)
 y=np.asarray(ys);s=np.asarray(scores);metrics={}
 for i,name in enumerate(HEADS):
  z=y[:,i];metrics[name]={'n':len(z),'positives':int(z.sum()),'auprc':float(average_precision_score(z,s[:,i])) if z.sum() else None,'auroc':float(roc_auc_score(z,s[:,i])) if len(np.unique(z))==2 else None}
 return metrics

def main():
 p=argparse.ArgumentParser();p.add_argument('--ffn',type=int,default=0);p.add_argument('--data',default='data/processed/joint_19_gty_1125');p.add_argument('--model',default='/workspace/models/Qwen3.5-9B');p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=4);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--accumulation',type=int,default=4);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--max-length',type=int,default=4096);p.add_argument('--max-train-rows',type=int);p.add_argument('--max-validation-rows',type=int);p.add_argument('--resume',action='store_true');p.add_argument('--save-every-updates',type=int,default=100);a=p.parse_args()
 if a.ffn<0 or min(a.epochs,a.batch_size,a.accumulation,a.save_every_updates)<1:raise ValueError('invalid configuration')
 root=Path(a.data);tr=rows(root/'train.jsonl',a.max_train_rows);val=rows(root/'validation.jsonl',a.max_validation_rows);output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);last=output.with_name(output.stem+'.last.pt')
 conf={'model':a.model,'data':str(root.resolve()),'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),'ffn':a.ffn,'heads':HEADS,'layer':15,'epochs':a.epochs,'batch_size':a.batch_size,'accumulation':a.accumulation,'lr':a.lr,'max_length':a.max_length,'max_train_rows':a.max_train_rows,'max_validation_rows':a.max_validation_rows,'seed':42,'loss':'singprobe_weighted','temperature':0.1,'beta_tau':0.35,'answer_only':True,'response_thinking_policy':'strip_explicit_thinking_before_input'}
 if last.exists() and not a.resume:raise SystemExit('Checkpoint exists; use --resume or new output')
 random.seed(42);np.random.seed(42);torch.manual_seed(42)
 print('config',json.dumps(conf),'train',len(tr),'validation',len(val),flush=True)
 proc=AutoProcessor.from_pretrained(a.model,local_files_only=True);backbone=AutoModelForMultimodalLM.from_pretrained(a.model,local_files_only=True,dtype=torch.bfloat16,device_map='auto');backbone.eval();backbone.requires_grad_(False);device=backbone.get_input_embeddings().weight.device
 model_kwargs=dict(input_dim=4096,num_heads=20,context_window=0,risk_window=40,num_query_heads=4,num_kv_heads=1,head_dim=64,internal_heads=10)
 torch.manual_seed(42);reference=MQARMAttnProbe(**model_kwargs,context_ffn_hidden_size=0)
 if a.ffn:
  model=MQARMAttnProbe(**model_kwargs,context_ffn_hidden_size=a.ffn);missing,unexpected=model.load_state_dict(reference.state_dict(),strict=False)
  if unexpected or not missing or any(not k.startswith(('context_ffn.','context_ffn_norm.')) for k in missing):raise ValueError('Shared baseline initialization mismatch')
  del reference
 else:model=reference
 model=model.to(device);objective=SingProbeWeighted(temperature=.1,beta_tau=.35,detach_weights=True);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
 groups=math.ceil(len(tr)/a.batch_size);upe=math.ceil(groups/a.accumulation);total=upe*a.epochs;warm=max(1,math.ceil(total*.05))
 def lrscale(step):return step/max(1,warm) if step<warm else .5*(1+math.cos(math.pi*min(1,(step-warm)/max(1,total-warm))))
 sched=torch.optim.lr_scheduler.LambdaLR(opt,lrscale);epoch0=0;batch0=0;updates=0;best=-1.;history=[];skipped=0;skipped_no_final=0
 if a.resume and last.exists():
  ck=torch.load(last,map_location='cpu',weights_only=False)
  old_conf=ck['config']
  if old_conf!=conf:
   compatible={k:v for k,v in conf.items() if k!='response_thinking_policy'}==old_conf
   marked=[i for i,r in enumerate(tr) if '<think>' in r['response'] or '</think>' in r['response']]
   if compatible and marked:
    if ck['epoch']>0:compatible=False
    else:
     order=list(range(len(tr)));random.Random(42+ck['epoch']).shuffle(order);marked_set=set(marked)
     compatible=not any(i in marked_set for i in order[:ck['next_batch']*a.batch_size])
   if not compatible:raise ValueError('resume config mismatch or previous checkpoint trained on thinking content')
   print('Resume: legacy checkpoint verified to precede all explicitly marked thinking samples',flush=True)

  model.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler']);epoch0=ck['epoch'];batch0=ck['next_batch'];updates=ck['updates'];best=ck['best'];history=ck['history'];skipped=ck['skipped'];skipped_no_final=ck.get('skipped_no_final',0);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
 def save(ep,next_batch,path):atomic_save({'architecture':'mqa_rmattn_joint19','config':conf,'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'epoch':ep,'next_batch':next_batch,'updates':updates,'best':best,'history':history,'skipped':skipped,'skipped_no_final':skipped_no_final,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},path)
 for epoch in range(epoch0,a.epochs):
  order=list(range(len(tr)));random.Random(42+epoch).shuffle(order);model.train();opt.zero_grad(set_to_none=True);start=batch0 if epoch==epoch0 else 0;pending=0;loss_sum=0.;steps=0
  for bi in range(start,groups):
   indices=order[bi*a.batch_size:(bi+1)*a.batch_size];es=[]
   for i in indices:
    e=extract(tr[i],proc,backbone,device,15,a.max_length,True)
    if e is None:skipped+=1
    elif e=='no_final_response':skipped_no_final+=1
    else:es.append(e)
   if es:
    b=collate(es);loss=objective(model(b),b.labels,b.label_mask)
    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
    (loss/a.accumulation).backward();pending+=1;loss_sum+=float(loss.detach());steps+=1
   if (bi+1)%a.accumulation==0 or bi+1==groups:
    if pending:
     if pending<a.accumulation:
      for par in model.parameters():
       if par.grad is not None:par.grad.mul_(a.accumulation/pending)
     torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();sched.step();opt.zero_grad(set_to_none=True);updates+=1;pending=0
     if updates%a.save_every_updates==0:save(epoch,bi+1,last)
   if (bi+1)%20==0 or bi+1==groups:print(f'epoch {epoch+1}/{a.epochs} batch {bi+1}/{groups} updates={updates} loss={loss_sum/max(1,steps):.5f} skipped_long={skipped} skipped_no_final={skipped_no_final}',flush=True)
  metrics=evaluate(val,proc,backbone,device,model,15);metric=metrics['overall_unsafe']['auprc'];history.append({'epoch':epoch+1,'train_loss':loss_sum/max(1,steps),'validation':metrics});improved=metric is not None and metric>best
  if improved:best=metric
  save(epoch+1,0,last)
  if improved:save(epoch+1,0,output)
  (output.with_suffix('.history.json')).write_text(json.dumps(history,ensure_ascii=False,indent=2)+'\n');print('epoch_summary',json.dumps(history[-1]),'best',best,flush=True)
 print('TRAINING COMPLETE',flush=True)
if __name__=='__main__':main()
