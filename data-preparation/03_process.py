"""Claim one prompt per worker; judge existing answers, then fill safe/unsafe gaps."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import threading
import time
from common import chat, config, connect, content, digest, event, judge


class VLLMGenerator:
    """One shared offline vLLM engine; serialize calls while DB/API workers run concurrently."""

    def __init__(self, settings):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError('vLLM generation requires vllm in the active conda environment') from exc

        self.settings = settings
        self.SamplingParams = SamplingParams
        options = {
            'model': settings['model'],
            'dtype': settings.get('dtype', 'auto'),
            'tensor_parallel_size': settings.get('tensor_parallel_size', 1),
            'trust_remote_code': settings.get('trust_remote_code', False),
            'gpu_memory_utilization': settings.get('gpu_memory_utilization', 0.85),
        }
        for key in ('max_model_len', 'max_num_seqs', 'limit_mm_per_prompt'):
            if key in settings:
                options[key] = settings[key]
        logging.info('Loading vLLM model %s', settings['model'])
        self.engine = LLM(**options)
        self.lock = threading.Lock()

    def generate(self, settings, messages):
        if settings['model'] != self.settings['model']:
            raise ValueError('A single vLLM run can use only one generation model')
        sampling = self.SamplingParams(
            temperature=settings.get('temperature', 0.8),
            top_p=settings.get('top_p', 1.0),
            max_tokens=settings.get('max_tokens', 2048),
        )
        # vLLM 0.6.x accepts OpenAI-style image_url content here, including data URIs
        # produced by common.content(). Text-only checkpoints will reject image inputs.
        with self.lock:
            if 'enable_thinking' in settings:
                if any(isinstance(m.get('content'), list) and any(part.get('type') != 'text' for part in m['content']) for m in messages):
                    raise ValueError('Text-only thinking setting cannot process images')
                text_messages = [{**m, 'content': '\n'.join(part['text'] for part in m['content']) if isinstance(m.get('content'), list) else m['content']} for m in messages]
                prompt = self.engine.get_tokenizer().apply_chat_template(text_messages, tokenize=False, add_generation_prompt=True, enable_thinking=settings['enable_thinking'])
                result = self.engine.generate([prompt], sampling_params=sampling, use_tqdm=False)[0].outputs[0]
            else:
                result = self.engine.chat(
                    messages,
                    sampling_params=sampling,
                    use_tqdm=False,
                    chat_template_content_format='openai',
                )[0].outputs[0]
        if result.finish_reason == 'length':
            raise ValueError('Response truncated: increase generation.max_tokens')
        if not result.text or not result.text.strip():
            raise ValueError('Empty or non-text vLLM completion')
        return result.text.strip()


def generate(generator, settings, messages):
    if settings.get('backend', 'vllm') == 'vllm':
        if generator is None:
            raise RuntimeError('vLLM generator was not initialized')
        return generator.generate(settings, messages)
    if settings.get('backend') == 'api':
        return chat(settings, messages)
    raise ValueError("generation.backend must be 'vllm' or 'api'")


def coverage(db, pid):
    counts = {'safe': 0, 'unsafe': 0}
    for row in db.execute('SELECT label FROM responses WHERE prompt_id=? AND label IS NOT NULL', (pid,)):
        label = json.loads(row['label'])
        if label.get('usable', True) and label['safety'] in counts:
            counts[label['safety']] += 1
    return counts


def claim(db):
    db.execute('BEGIN IMMEDIATE')
    try:
        row = db.execute("SELECT * FROM prompts WHERE state='pending' ORDER BY id LIMIT 1").fetchone()
        if row:
            db.execute("UPDATE prompts SET state='running',updated=? WHERE id=?", (time.time(), row['id']))
        db.commit()
        return row
    except Exception:
        db.rollback()
        raise


def store_label(db, cfg, payload, row):
    label = judge(cfg, payload, row['text'])
    with db:
        db.execute('UPDATE responses SET label=? WHERE id=?', (json.dumps(label, ensure_ascii=False), row['id']))
        event(db, row['prompt_id'], 'response_judged', {'response_id': row['id'], 'label': label})


def process(db, cfg, row, vllm_generator=None):
    pid, payload = row['id'], json.loads(row['payload'])
    if row['label']:
        input_label = json.loads(row['label'])
    else:
        input_label = judge(cfg, payload)
        with db:
            db.execute('UPDATE prompts SET label=? WHERE id=?', (json.dumps(input_label, ensure_ascii=False), pid))
            event(db, pid, 'input_judged', input_label)
    # Always classify existing responses before generating anything.
    for response in db.execute('SELECT * FROM responses WHERE prompt_id=? AND label IS NULL', (pid,)).fetchall():
        store_label(db, cfg, payload, response)
    if input_label['safety'] == 'uncertain':
        return 'needs_review'
    if cfg.get('processing', {}).get('judge_existing_only', False):
        origins = db.execute('SELECT origin FROM responses WHERE prompt_id=?', (pid,)).fetchall()
        if any('source' in json.loads(origin['origin']) for origin in origins):
            return 'completed'
    target = {'safe': 1, 'unsafe': 1} if input_label['safety'] == 'unsafe' else {'safe': 1, 'unsafe': 0}
    settings = cfg.get('processing', {})
    while True:
        counts = coverage(db, pid)
        missing = [kind for kind, number in target.items() if counts[kind] < number]
        if not missing:
            return 'completed'
        attempts = db.execute('SELECT attempts FROM prompts WHERE id=?', (pid,)).fetchone()[0]
        if attempts >= settings.get('max_attempts', 6):
            return 'exhausted'
        kind = missing[0]
        # Persist budget before network I/O; interrupted attempts still count.
        with db:
            db.execute('UPDATE prompts SET attempts=attempts+1,updated=? WHERE id=?', (time.time(), pid))
        generator = cfg['generation']
        if isinstance(generator, list):
            generator = generator[attempts % len(generator)]
        system = settings.get('generation_instructions', {}).get(kind,
            'Provide an appropriate safe answer. Refuse harmful assistance and offer a safe alternative.'
            if kind == 'safe' else 'Answer the user request directly and substantively. Do not invent unrelated content.')
        messages = [{'role': 'system', 'content': system}]
        context = payload.get('context', [])
        if not isinstance(context, list) or any(not isinstance(m, dict) or m.get('role') not in ('user', 'assistant') for m in context):
            raise ValueError('context must be a list of user/assistant messages')
        messages.extend(context)
        user_content = content(payload)
        user_content[0]['text'] = payload['prompt']
        messages.append({'role': 'user', 'content': user_content})
        text = generate(vllm_generator, generator, messages)
        rid = digest([pid, text])
        with db:
            db.execute('INSERT OR IGNORE INTO responses(id,prompt_id,text,origin) VALUES(?,?,?,?)',
                (rid, pid, text, json.dumps({'model': generator['model'], 'target': kind}, ensure_ascii=False)))
            event(db, pid, 'generated', {'response_id': rid, 'target': kind, 'model': generator['model']})
        response = db.execute('SELECT * FROM responses WHERE id=?', (rid,)).fetchone()
        if response['label'] is None:
            store_label(db, cfg, payload, response)


def worker(cfg, vllm_generator=None):
    db = connect(cfg['db'])
    try:
        while (row := claim(db)) is not None:
            try:
                state, error = process(db, cfg, row, vllm_generator), None
            except Exception as exc:
                state, error = 'failed', type(exc).__name__
                logging.error('%s: %s (check generation model/backend and judge API configuration)', row['id'][:12], error)
            with db:
                db.execute('UPDATE prompts SET state=?,error=?,updated=? WHERE id=?', (state, error, time.time(), row['id']))
            logging.info('%s %s', row['id'][:12], state)
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--retry', nargs='+', choices=['failed', 'running', 'needs_review', 'exhausted'])
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    cfg = config(args.config)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    generation = cfg['generation']
    if isinstance(generation, list):
        if any(item.get('backend', 'vllm') == 'vllm' for item in generation):
            parser.error('vLLM mode currently expects one generation model, not a list')
        vllm_generator = None
    elif generation.get('backend', 'vllm') == 'vllm':
        vllm_generator = VLLMGenerator(generation)
    else:
        vllm_generator = None
    db = connect(cfg['db'])
    if args.retry:
        with db:
            marks = ','.join('?' for _ in args.retry)
            db.execute(f"UPDATE prompts SET state='pending',error=NULL WHERE state IN ({marks})", args.retry)
        # Attempts are intentionally not reset. Increase configured budget if required.
    db.close()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(lambda _: worker(cfg, vllm_generator), range(args.workers)))


if __name__ == '__main__':
    main()
