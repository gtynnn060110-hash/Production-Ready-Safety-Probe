"""Prepare a deterministic prompt-grouped split from text and multimodal SQLite datasets."""
import argparse,collections,hashlib,json,os,sqlite3,unicodedata
from pathlib import Path
CATS={f'R{i:02}' for i in range(1,20)}
def norm(x):return unicodedata.normalize('NFC',x).replace('\r\n','\n').strip()
def main():
 p=argparse.ArgumentParser();p.add_argument('--text-db',default='/workspace/models/gty/probe-generated-gty-1125.sqlite');p.add_argument('--image-db',default='/workspace/models/gty/probe_1w_image_qwen3vl8b_huihui_judge_glm-4.6v.db');p.add_argument('--out',default='data/processed/joint_19_gty_1125');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 if any((out/f'{x}.jsonl').exists() for x in ('train','validation','test')):raise SystemExit('Output exists; choose a new output directory')
 files={k:(out/(k+'.jsonl.tmp')).open('w',encoding='utf8') for k in ('train','validation','test')};c=collections.Counter();missing=[]
 try:
  for modality,path in [('text',a.text_db),('image',a.image_db)]:
   db=sqlite3.connect('file:'+path+'?mode=ro',uri=True);sql='SELECT r.id,r.prompt_id,p.payload,r.text,r.label,r.origin FROM responses r JOIN prompts p ON p.id=r.prompt_id WHERE r.label IS NOT NULL ORDER BY r.id'
   for rid,pid,payload,text,label,origin in db.execute(sql):
    payload=json.loads(payload);lab=json.loads(label);safety=lab.get('safety');cats=lab.get('categories');imgs=payload.get('images',[]);prompt=payload.get('prompt','');context=payload.get('context',[])
    if lab.get('usable',True) is False:c[modality+':unusable']+=1;continue
    if safety not in ('safe','unsafe') or not isinstance(cats,list) or len(cats)!=len(set(cats)) or not set(cats)<=CATS or (safety=='safe' and cats) or (safety=='unsafe' and not cats):c[modality+':invalid_label']+=1;continue
    if not isinstance(text,str) or not norm(text) or not isinstance(prompt,str) or not isinstance(imgs,list) or (not norm(prompt) and not imgs):c[modality+':invalid_content']+=1;continue
    if any(not isinstance(x,str) or not os.path.isfile(x) for x in imgs):
     c[modality+':missing_image']+=1
     if len(missing)<5:missing.append(pid)
     continue
    if bool(imgs)!=(modality=='image'):raise ValueError('unexpected modality '+pid)
    group=json.dumps([norm(prompt),context,imgs],ensure_ascii=False,sort_keys=True,separators=(',',':'))
    ghash=hashlib.sha256(group.encode()).hexdigest();v=int(ghash[:16],16)%10000;split='train' if v<8000 else 'validation' if v<9000 else 'test'
    row={'id':rid,'prompt_id':pid,'group_hash':ghash,'prompt':prompt,'context':context,'images':imgs,'response':text,'safety':safety,'categories':cats,'modality':modality,'origin':json.loads(origin)}
    files[split].write(json.dumps(row,ensure_ascii=False)+'\n');c[f'{split}:{modality}:{safety}']+=1
   db.close()
 finally:
  for f in files.values():f.close()
 if missing:raise ValueError(f'missing images; examples={missing}')
 for k in files:(out/(k+'.jsonl.tmp')).replace(out/(k+'.jsonl'))
 summary={'counts':dict(c),'split':'sha256 of NFC-normalized prompt, context and image paths; 80/10/10 grouped by prompt content','categories':'R01-R19 plus overall unsafe','input_databases':[a.text_db,a.image_db]}
 (out/'manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
