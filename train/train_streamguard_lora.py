#!/usr/bin/env python3
"""SFT a Qwen chat model on prompt/response JSONL with assistant-only loss."""
import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--model', default='/workspace/models/Qwen3-8B')
    p.add_argument('--max-length', type=int, default=2048)
    p.add_argument('--epochs', type=float, default=1.0)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--gradient-accumulation', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=2e-4)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows, dropped = [], 0
    with open(args.data, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if 'messages' in row:
                messages = row['messages']
                if (not isinstance(messages, list) or len(messages) != 2
                    or not all(isinstance(m, dict) for m in messages)
                    or [m.get('role') for m in messages] != ['user', 'assistant']):
                    raise ValueError('Expected exactly one user message followed by one assistant message')
                row = {'prompt': messages[0].get('content'), 'response': messages[1].get('content')}
            if not isinstance(row.get('prompt'), str) or not isinstance(row.get('response'), str) or not row['response'].strip():
                dropped += 1
                continue
            prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            full = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}, {'role': 'assistant', 'content': row['response']}], tokenize=False, enable_thinking=False)
            prefix_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
            ids = tokenizer(full, add_special_tokens=False)['input_ids']
            if ids[:len(prefix_ids)] != prefix_ids:
                raise ValueError('Chat template prompt is not a prefix of the full conversation')
            ids = ids[:args.max_length]
            if len(ids) <= len(prefix_ids):
                dropped += 1
                continue
            rows.append({'input_ids': ids, 'labels': [-100] * len(prefix_ids) + ids[len(prefix_ids):]})
    if not rows:
        raise ValueError('No usable examples')
    if int(os.environ.get('RANK', '0')) == 0:
        print(f'Examples: {len(rows)}; dropped: {dropped}', flush=True)
    def collate(features):
        n = max(len(x['input_ids']) for x in features)
        return {'input_ids': torch.tensor([x['input_ids'] + [tokenizer.pad_token_id] * (n-len(x['input_ids'])) for x in features]),
                'attention_mask': torch.tensor([[1]*len(x['input_ids']) + [0]*(n-len(x['input_ids'])) for x in features]),
                'labels': torch.tensor([x['labels'] + [-100]*(n-len(x['labels'])) for x in features])}
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation='sdpa')
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'], task_type='CAUSAL_LM'))
    training_args = TrainingArguments(output_dir=args.output_dir, num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation, learning_rate=args.learning_rate, bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False}, logging_steps=10, save_strategy='epoch', save_total_limit=2, lr_scheduler_type='cosine', warmup_ratio=0.03, optim='adamw_torch', report_to='none', ddp_find_unused_parameters=False, seed=args.seed, dataloader_num_workers=0)
    trainer = Trainer(model=model, args=training_args, train_dataset=Dataset.from_list(rows), data_collator=collate, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)
        Path(args.output_dir, 'run_config.json').write_text(json.dumps(vars(args), indent=2, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main()
