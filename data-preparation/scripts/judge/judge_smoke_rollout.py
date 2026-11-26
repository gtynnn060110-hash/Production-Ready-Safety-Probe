"""Judge the isolated base-model smoke output; resume completed records safely."""
import sys as _prep_sys; from pathlib import Path as _PrepPath; _prep_sys.path.insert(0, str(_PrepPath(__file__).resolve().parents[2])); from scripts._bootstrap import setup as _prep_setup; _prep_setup()
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

from common import judge


def record_id(row):
    content = {k: row[k] for k in ('source_file', 'source_line', 'prompt', 'target', 'response', 'model')}
    return hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default=os.environ.get('JUDGE_MODEL', 'DeepSeek-V4.1-Flash'))
    parser.add_argument('--base-url', default=os.environ.get('JUDGE_BASE_URL', 'https://llmapi.paratera.com'))
    parser.add_argument('--concurrency', type=int, default=4, help='Maximum simultaneous judge requests')
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error('--concurrency must be positive')
    # This long-running comparison was launched before the judge key rotated.
    # Its parent shell cannot inherit an updated environment from another shell.
    if Path(args.output).name == 'qwen2-vl-7b-base-text-comparison-judged.jsonl':
        key_file = Path(__file__).resolve().parents[2] / 'data/.qwen2vl-judge-key'
        if key_file.is_symlink() or not key_file.is_file():
            parser.error(f'New judge key file is required: {key_file}')
        if key_file.stat().st_mode & 0o077:
            parser.error('New judge key file must have permissions 600 or stricter')
        os.environ['JUDGE_API_KEY'] = key_file.read_text(encoding='utf-8').strip()
    if not os.environ.get('JUDGE_API_KEY'):
        parser.error('JUDGE_API_KEY is not set in this shell')
    cfg = {'policy_file': None, 'taxonomy_version': '19-v1', 'judge': {
        'base_url': args.base_url, 'model': args.model, 'api_key_env': 'JUDGE_API_KEY',
        'temperature': 0, 'max_tokens': 1024, 'timeout': 180, 'retries': 2,
        'json_mode': True, 'extra_body': {'thinking': {'type': 'disabled'}}}}
    with open(args.input, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if output.exists():
        with output.open(encoding='utf-8') as f:
            existing = [json.loads(line) for line in f if line.strip()]
    ids = {record_id(row) for row in rows}
    if any(row.get('record_id') not in ids or row.get('response_label', {}).get('judge_model') != args.model for row in existing):
        raise ValueError('Existing output does not match input or judge model; choose a new output')
    completed = {row['record_id'] for row in existing}
    prompt_labels = {(row['source_file'], row['source_line']): row['input_label'] for row in existing}
    pending = [row for row in rows if record_id(row) not in completed]
    cache_path = output.with_name(output.name + '.input-cache.jsonl')
    def prompt_id(payload):
        return hashlib.sha256(json.dumps({'model': args.model, 'prompt': payload['prompt']}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    cached_inputs = {}
    if cache_path.exists():
        with cache_path.open(encoding='utf-8') as cache:
            for line in cache:
                if line.strip():
                    entry = json.loads(line)
                    cached_inputs[entry['id']] = entry['label']
    prompt_jobs = {}
    for row in pending:
        key = (row['source_file'], row['source_line'])
        if key not in prompt_labels:
            payload = {'prompt': row['prompt'], 'context': [], 'images': []}
            if prompt_id(payload) in cached_inputs:
                prompt_labels[key] = cached_inputs[prompt_id(payload)]
            else:
                prompt_jobs[key] = payload
    failed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        print(f'Judging {len(prompt_jobs)} inputs with concurrency={args.concurrency}', flush=True)
        tasks = {pool.submit(judge, cfg, payload): key for key, payload in prompt_jobs.items()}
        with cache_path.open('a', encoding='utf-8') as cache:
            for future in as_completed(tasks):
                key = tasks[future]
                try:
                    label = future.result()
                    cache.write(json.dumps({'id': prompt_id(prompt_jobs[key]), 'label': label}, ensure_ascii=False) + '\n')
                    cache.flush()
                    prompt_labels[key] = label
                except Exception as exc:
                    print(f'Failed input {key}: {type(exc).__name__}', flush=True)
        failed += sum((row['source_file'], row['source_line']) not in prompt_labels for row in pending)
        print(f'Judging responses; already completed {len(completed)}/{len(rows)}', flush=True)
        tasks = {}
        for row in pending:
            key = (row['source_file'], row['source_line'])
            if key in prompt_labels:
                payload = {'prompt': row['prompt'], 'context': [], 'images': []}
                tasks[pool.submit(judge, cfg, payload, row['response'])] = row
        with output.open('a', encoding='utf-8') as f:
            for future in as_completed(tasks):
                row = tasks[future]
                rid = record_id(row)
                key = (row['source_file'], row['source_line'])
                try:
                    response_label = future.result()
                except Exception as exc:
                    failed += 1
                    print(f"Failed {row['language']}/{row['target']} line {row['source_line']}: {type(exc).__name__}", flush=True)
                    continue
                result = {**row, 'record_id': rid, 'input_label': prompt_labels[key], 'response_label': response_label}
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()
                existing.append(result)
                completed.add(rid)
                print(f'Judged {len(completed)}/{len(rows)}', flush=True)
    counts = Counter((r['target'], r['response_label']['safety'], r['response_label']['usable'], r['finish_reason']) for r in existing)
    pairs = defaultdict(set)
    for row in existing:
        label = row['response_label']
        if label['usable'] and row['finish_reason'] != 'length' and label['safety'] in ('safe', 'unsafe'):
            pairs[(row['source_file'], row['source_line'])].add(label['safety'])
    print('Counts (target, safety, usable, finish_reason):', dict(counts))
    print('Prompts with usable non-truncated safe and unsafe answers:', sum(v == {'safe', 'unsafe'} for v in pairs.values()))
    print('Judged:', len(completed), '/', len(rows), 'failed:', failed)
    if failed or len(completed) != len(rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
