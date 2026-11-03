"""Claim one prompt per worker; judge existing answers, then fill safe/unsafe gaps."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import time
from common import chat, config, connect, content, digest, event, judge


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


def process(db, cfg, row):
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
        text = chat(generator, messages)
        rid = digest([pid, text])
        with db:
            db.execute('INSERT OR IGNORE INTO responses(id,prompt_id,text,origin) VALUES(?,?,?,?)',
                (rid, pid, text, json.dumps({'model': generator['model'], 'target': kind}, ensure_ascii=False)))
            event(db, pid, 'generated', {'response_id': rid, 'target': kind, 'model': generator['model']})
        response = db.execute('SELECT * FROM responses WHERE id=?', (rid,)).fetchone()
        if response['label'] is None:
            store_label(db, cfg, payload, response)


def worker(cfg):
    db = connect(cfg['db'])
    try:
        while (row := claim(db)) is not None:
            try:
                state, error = process(db, cfg, row), None
            except Exception as exc:
                state, error = 'failed', type(exc).__name__
                logging.error('%s: %s (see API configuration and endpoint logs)', row['id'][:12], error)
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
    db = connect(cfg['db'])
    if args.retry:
        with db:
            marks = ','.join('?' for _ in args.retry)
            db.execute(f"UPDATE prompts SET state='pending',error=NULL WHERE state IN ({marks})", args.retry)
        # Attempts are intentionally not reset. Increase configured budget if required.
    db.close()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(lambda _: worker(cfg), range(args.workers)))


if __name__ == '__main__':
    main()
