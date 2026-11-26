#!/usr/bin/env python3
"""Two-rank training for the joint MQA probe; global batches match the single-GPU trainer."""
import argparse, hashlib, json, math, os, random
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoProcessor, AutoModelForMultimodalLM
from train_joint_mqa_probe import HEADS, rows, atomic_save, extract, collate, MQARMAttnProbe
from probe.losses.token import inputs


def distributed_loss(output, labels, label_mask):
    z, mask, valid, bce = inputs(output, labels, label_mask)
    m = mask[:, :, None] & label_mask[:, None, :]
    p = torch.where(m, z, 0.0).sigmoid().detach()
    # Match the single-device population standard deviation over all valid tokens.
    stats = torch.stack((m.sum((0, 1)).float(), (p*m).sum((0, 1)), (p.square()*m).sum((0, 1))))
    dist.all_reduce(stats)
    count, total, squared = stats
    mean = total / count.clamp_min(1)
    beta = ((squared / count.clamp_min(1) - mean.square()).clamp_min(0).sqrt() / .35).clamp(0, 1)
    weights = (p * beta / .1).masked_fill(~m, -1e30).softmax(dim=1) * m
    local_sum = (((weights*bce).sum(1)) * valid).sum(0)
    counts = valid.sum(0).float()
    dist.all_reduce(counts)
    active = counts > 0
    # DDP averages gradients across ranks; compensate for this average.
    return ((local_sum / counts.clamp_min(1)) * active).sum() / active.sum().clamp_min(1) * dist.get_world_size()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ffn',type=int,default=0);p.add_argument('--data',default='data/processed/joint_19_gty_1125');p.add_argument('--model',default='/workspace/models/Qwen3.5-9B');p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=4);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--accumulation',type=int,default=4);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--max-length',type=int,default=4096);p.add_argument('--max-train-rows',type=int);p.add_argument('--max-validation-rows',type=int);p.add_argument('--resume',action='store_true');p.add_argument('--save-every-updates',type=int,default=100)
    a=p.parse_args(); rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK']); world=int(os.environ['WORLD_SIZE'])
    if world != 2: raise ValueError('This trainer requires exactly two ranks')
    torch.cuda.set_device(local);device=torch.device('cuda',local);dist.init_process_group('nccl',timeout=timedelta(minutes=90))
    root=Path(a.data);tr=rows(root/'train.jsonl',a.max_train_rows);val=rows(root/'validation.jsonl',a.max_validation_rows);output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);last=output.with_name(output.stem+'.last.pt')
    conf={'model':a.model,'data':str(root.resolve()),'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),'ffn':a.ffn,'heads':HEADS,'layer':15,'epochs':a.epochs,'batch_size':a.batch_size,'accumulation':a.accumulation,'lr':a.lr,'max_length':a.max_length,'max_train_rows':a.max_train_rows,'max_validation_rows':a.max_validation_rows,'seed':42,'loss':'singprobe_weighted','temperature':0.1,'beta_tau':0.35,'answer_only':True,'response_thinking_policy':'strip_explicit_thinking_before_input'}
    if last.exists() and not a.resume:raise SystemExit('Checkpoint exists; use --resume or new output')
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if rank==0:print('config',json.dumps(conf),'train',len(tr),'validation',len(val),'world_size',world,flush=True)
    proc=AutoProcessor.from_pretrained(a.model,local_files_only=True)
    backbone=AutoModelForMultimodalLM.from_pretrained(a.model,local_files_only=True,dtype=torch.bfloat16,device_map={'':local});backbone.eval();backbone.requires_grad_(False)
    kwargs=dict(input_dim=4096,num_heads=20,context_window=0,risk_window=40,num_query_heads=4,num_kv_heads=1,head_dim=64,internal_heads=10)
    torch.manual_seed(42);reference=MQARMAttnProbe(**kwargs,context_ffn_hidden_size=0)
    if a.ffn:
        model=MQARMAttnProbe(**kwargs,context_ffn_hidden_size=a.ffn);missing,unexpected=model.load_state_dict(reference.state_dict(),strict=False)
        if unexpected or not missing or any(not k.startswith(('context_ffn.','context_ffn_norm.')) for k in missing):raise ValueError('Shared baseline initialization mismatch')
        del reference
    else:model=reference
    model=model.to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    groups=math.ceil(len(tr)/a.batch_size);upe=math.ceil(groups/a.accumulation);total=upe*a.epochs;warm=max(1,math.ceil(total*.05))
    def lrscale(step):return step/max(1,warm) if step<warm else .5*(1+math.cos(math.pi*min(1,(step-warm)/max(1,total-warm))))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lrscale);epoch0=0;batch0=0;updates=0;best=-1.;history=[];skipped=0;skipped_no_final=0
    if a.resume:
        ck=torch.load(last,map_location='cpu',weights_only=False)
        if ck['config'] != conf:raise ValueError('resume config mismatch')
        model.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler']);epoch0=ck['epoch'];batch0=ck['next_batch'];updates=ck['updates'];best=ck['best'];history=ck['history'];skipped=ck['skipped'];skipped_no_final=ck.get('skipped_no_final',0);torch.set_rng_state(ck['torch_rng'])
        if rank==0:print(f'resuming epoch={epoch0+1} next_batch={batch0} updates={updates}',flush=True)
    ddp=DDP(model,device_ids=[local],broadcast_buffers=False)
    def save(ep,next_batch,path):
        if rank==0:atomic_save({'architecture':'mqa_rmattn_joint19','config':conf,'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'epoch':ep,'next_batch':next_batch,'updates':updates,'best':best,'history':history,'skipped':skipped,'skipped_no_final':skipped_no_final,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},path)
    for epoch in range(epoch0,a.epochs):
        order=list(range(len(tr)));random.Random(42+epoch).shuffle(order);ddp.train();opt.zero_grad(set_to_none=True);start=batch0 if epoch==epoch0 else 0;loss_sum=0.;steps=0
        for bi in range(start,groups):
            indices=order[bi*a.batch_size:(bi+1)*a.batch_size][rank::world];es=[];local_skips=[0,0]
            for i in indices:
                e=extract(tr[i],proc,backbone,device,15,a.max_length,True)
                if e is None:local_skips[0]+=1
                elif e=='no_final_response':local_skips[1]+=1
                else:es.append(e)
            sc=torch.tensor(local_skips,device=device);dist.all_reduce(sc);skipped+=int(sc[0]);skipped_no_final+=int(sc[1])
            # Empty partitions must still participate in every DDP collective.
            if not es:
                for fallback in order:
                    e=extract(tr[fallback],proc,backbone,device,15,a.max_length,True)
                    if e is not None and e!='no_final_response':break
                es=[e];b=collate(es);b.label_mask[:]=False
            else:b=collate(es)
            loss=distributed_loss(ddp(b),b.labels,b.label_mask)
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            window=min(a.accumulation,groups-(bi//a.accumulation)*a.accumulation)
            (loss/window).backward();loss_sum+=float(loss.detach());steps+=1
            if (bi+1)%a.accumulation==0 or bi+1==groups:
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();sched.step();opt.zero_grad(set_to_none=True);updates+=1
                if updates%a.save_every_updates==0:save(epoch,bi+1,last)
            if rank==0 and ((bi+1)%20==0 or bi+1==groups):print(f'epoch {epoch+1}/{a.epochs} batch {bi+1}/{groups} updates={updates} loss_rank0={loss_sum/max(1,steps):.5f} skipped_long={skipped} skipped_no_final={skipped_no_final}',flush=True)
        model.eval();ys=[];scores=[]
        with torch.no_grad():
            for n,row in enumerate(val[rank::world],1):
                e=extract(row,proc,backbone,device,15,0,False)
                if e=='no_final_response':continue
                b=collate([e]);o=model(b);x=o.local_logits.masked_fill(~o.local_mask[:,:,None],-torch.inf).max(1).values.sigmoid();ys.append(e[2].cpu().numpy());scores.append(x[0].cpu().numpy())
                if rank==0 and n%1000==0:print(f'validation rank0 {n}/{len(val[rank::world])}',flush=True)
        gathered=[None]*world;dist.all_gather_object(gathered,(ys,scores));metrics={}
        if rank==0:
            y=np.asarray([x for pair in gathered for x in pair[0]]);s=np.asarray([x for pair in gathered for x in pair[1]])
            for i,name in enumerate(HEADS):
                z=y[:,i];metrics[name]={'n':len(z),'positives':int(z.sum()),'auprc':float(average_precision_score(z,s[:,i])) if z.sum() else None,'auroc':float(roc_auc_score(z,s[:,i])) if len(np.unique(z))==2 else None}
            metric=metrics['overall_unsafe']['auprc'];history.append({'epoch':epoch+1,'train_loss_rank0':loss_sum/max(1,steps),'validation':metrics});improved=metric is not None and metric>best
            if improved:best=metric
            save(epoch+1,0,last)
            if improved:save(epoch+1,0,output)
            output.with_suffix('.history.json').write_text(json.dumps(history,ensure_ascii=False,indent=2)+'\n');print('epoch_summary',json.dumps(history[-1]),'best',best,flush=True)
        dist.barrier()
    if rank==0:print('TRAINING COMPLETE',flush=True)
    dist.destroy_process_group()
if __name__=='__main__':main()
