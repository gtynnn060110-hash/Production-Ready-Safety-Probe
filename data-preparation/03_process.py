"""Claim one prompt per worker; judge existing answers, then fill safe/unsafe gaps."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
import uuid
from common import chat, config, connect, content, digest, event, judge, normalize, validate_label


def validate_api_settings(cfg):
    """Reject missing endpoints/credentials before claiming or changing any rows."""
    endpoints = [('judge', cfg.get('judge'))]
    targets = cfg.get('generation_by_target')
    if targets is None:
        endpoints.append(('generation', cfg.get('generation')))
    else:
        if not isinstance(targets, dict):
            raise ValueError('generation_by_target must be an object')
        endpoints.extend((f'generation_by_target.{kind}', targets.get(kind))
                         for kind in ('safe', 'unsafe'))
    for name, settings in endpoints:
        options = settings if isinstance(settings, list) and name != 'judge' else [settings]
        if not options:
            raise ValueError(f'{name} generator list cannot be empty')
        for index, item in enumerate(options):
            location = f'{name}[{index}]' if isinstance(settings, list) else name
            if not isinstance(item, dict):
                raise ValueError(f'{location} must be an API configuration object')
            for field in ('base_url', 'model'):
                if not isinstance(item.get(field), str) or not item[field].strip():
                    raise ValueError(f'{location}.{field} must be a nonempty string')
            key = item.get('api_key_env')
            if key is not None and (not isinstance(key, str) or not key.strip()):
                raise ValueError(f'{location}.api_key_env must be a nonempty environment variable name')
            if key is not None and not os.environ.get(key, '').strip():
                raise ValueError(f'Missing API key environment variable: {key} ({location})')


def api_cost(settings, usage):
    """Price provider-reported tokens; unknown usage/prices remain explicitly unknown."""
    usage = usage if isinstance(usage, dict) else {}
    def tokens(value):
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
    prompt = tokens(usage.get('prompt_tokens', usage.get('input_tokens')))
    completion = tokens(usage.get('completion_tokens', usage.get('output_tokens')))
    details = usage.get('prompt_tokens_details') or {}
    cached = tokens(details.get('cached_tokens', usage.get('prompt_cache_hit_tokens', 0)))
    cached = min(cached or 0, prompt or 0)
    pricing = settings.get('pricing', {})
    rate = pricing
    if pricing.get('tiers') and prompt is not None:
        rate = next((tier for tier in pricing['tiers'] if prompt < tier['input_tokens_lt']), {})
    keys = ('input_cny_per_million', 'output_cny_per_million', 'cached_input_cny_per_million')
    rates = [rate.get(keys[0]), rate.get(keys[1]), rate.get(keys[2], rate.get(keys[0]))]
    valid = pricing.get('currency') == 'CNY' and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0 for x in rates)
    cost = None
    if valid and (prompt is not None and completion is not None):
        cost = ((prompt - cached) * rates[0] + completion * rates[1] + cached * rates[2]) / 1_000_000
    elif pricing.get('local') is True and valid and not any(rates):
        cost = 0.0
    return {'prompt_tokens': prompt, 'completion_tokens': completion, 'cached_tokens': cached,
            'usage_known': prompt is not None and completion is not None, 'cost_cny': cost,
            'pricing_source': pricing.get('source'),
            'rates_cny_per_million': dict(zip(('input', 'output', 'cached_input'), rates)) if valid else None}


class Progress:
    def __init__(self, db, interval):
        self.run_id = uuid.uuid4().hex
        self.started = time.monotonic()
        self.interval = interval
        self.lock = threading.RLock()
        self.states = Counter({r[0]: r[1] for r in db.execute('SELECT state,count(*) FROM prompts GROUP BY state')})
        self.total = self.states['pending']
        self.done = 0
        self.in_flight = 0
        self.groups = defaultdict(Counter)
        self.stop = threading.Event()

    def snapshot(self):
        with self.lock:
            groups = {k: dict(v) for k, v in self.groups.items()}
            elapsed = time.monotonic() - self.started
            return {'run_id': self.run_id, 'pending_at_start': self.total, 'processed_this_run': self.done,
                    'states': dict(self.states), 'elapsed_seconds': elapsed,
                    'eta_seconds': elapsed / self.done * (self.total - self.done) if self.done else None,
                    'api_in_flight': self.in_flight, 'api_groups': groups,
                    'estimated_cost_cny': sum(g.get('cost_cny', 0) for g in groups.values()),
                    'unpriced_calls': sum(g.get('unpriced_calls', 0) for g in groups.values()),
                    'usage_missing_calls': sum(g.get('usage_missing_calls', 0) for g in groups.values()),
                    'billing_note': 'Configured CNY tariff estimate, not a provider invoice. Missing usage may hide charges. Local GPU compute costs excluded.'}

    def show(self):
        s = self.snapshot()
        states = s['states']
        groups = s['api_groups']
        calls = sum(g.get('calls', 0) for g in groups.values())
        errors = sum(g.get('errors', 0) for g in groups.values())
        pct = 100 * s['processed_this_run'] / self.total if self.total else 100
        eta = f'{s["eta_seconds"]:.0f}s' if s['eta_seconds'] is not None else '待估'
        logging.info('进度 %d/%d (%.1f%%) | pending=%d running=%d completed=%d exhausted=%d failed=%d | '
                     'API=%d errors=%d in_flight=%d | 费用估算=¥%.6f 未定价=%d usage缺失=%d | elapsed=%.0fs ETA=%s',
                     s['processed_this_run'], self.total, pct, states.get('pending', 0), states.get('running', 0),
                     states.get('completed', 0), states.get('exhausted', 0), states.get('failed', 0),
                     calls, errors, s['api_in_flight'], s['estimated_cost_cny'], s['unpriced_calls'],
                     s['usage_missing_calls'], s['elapsed_seconds'], eta)

    def heartbeat(self):
        while not self.stop.wait(self.interval):
            self.show()

    def claimed(self):
        with self.lock:
            self.states['pending'] -= 1
            self.states['running'] += 1

    def finished(self, state):
        with self.lock:
            self.done += 1
            self.states['running'] -= 1
            self.states[state] += 1
        self.show()

    def observer(self, db, pid, role, settings, modality):
        def record(call):
            detail = {**call, 'run_id': self.run_id, 'role': role, 'modality': modality}
            if call['phase'] == 'start':
                with db:
                    event(db, pid, 'api_call_started', detail)
                with self.lock:
                    self.in_flight += 1
                logging.info('API开始 %s %s %s HTTP_attempt=%d', pid[:12], role, call['model'], call['attempt'])
                return
            detail.update(api_cost(settings, call.get('usage')))
            with db:
                event(db, pid, 'api_call', detail)
            with self.lock:
                self.in_flight -= 1
                group = self.groups[f'{role}/{modality}']
                group.update(calls=1, errors=int(call['status'] != 'ok'),
                             prompt_tokens=detail['prompt_tokens'] or 0,
                             completion_tokens=detail['completion_tokens'] or 0,
                             cached_tokens=detail['cached_tokens'], seconds=call['seconds'],
                             cost_cny=detail['cost_cny'] or 0,
                             unpriced_calls=int(detail['cost_cny'] is None),
                             usage_missing_calls=int(not detail['usage_known']))
            cost = f'¥{detail["cost_cny"]:.6f}' if detail['cost_cny'] is not None else '未知'
            logging.info('API结束 %s %s %s | input=%s cached=%s output=%s | 估算=%s | %.2fs',
                         pid[:12], role, call['status'], detail['prompt_tokens'], detail['cached_tokens'],
                         detail['completion_tokens'], cost, call['seconds'])
        return record


def coverage(db, pid):
    counts = {'safe': 0, 'unsafe': 0}
    for row in db.execute('SELECT label FROM responses WHERE prompt_id=? AND label IS NOT NULL', (pid,)):
        label = cached_label(row['label'], 'response')
        if label is not None:
            counts[label['safety']] += 1
    return counts


def cached_label(value, task):
    try:
        return validate_label(value, task)
    except (ValueError, TypeError):
        return None


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


def store_label(db, cfg, payload, row, progress=None):
    observer = (progress.observer(db, row['prompt_id'], 'judge_response', cfg['judge'],
                                'image_text' if payload.get('images') else 'text') if progress else None)
    label = judge(cfg, payload, row['text'], on_api=observer)
    with db:
        db.execute('UPDATE responses SET label=? WHERE id=?', (json.dumps(label, ensure_ascii=False), row['id']))
        event(db, row['prompt_id'], 'response_judged', {'response_id': row['id'], 'label': label})


def process(db, cfg, row, progress=None):
    pid, payload = row['id'], json.loads(row['payload'])
    modality = 'image_text' if payload.get('images') else 'text'
    input_label = cached_label(row['label'], 'input')
    if input_label is None:
        observer = progress.observer(db, pid, 'judge_input', cfg['judge'], modality) if progress else None
        input_label = judge(cfg, payload, on_api=observer)
        with db:
            db.execute('UPDATE prompts SET label=? WHERE id=?', (json.dumps(input_label, ensure_ascii=False), pid))
            event(db, pid, 'input_judged', input_label)
    # Always classify existing responses before generating anything.
    for response in db.execute('SELECT * FROM responses WHERE prompt_id=?', (pid,)).fetchall():
        if cached_label(response['label'], 'response') is None:
            store_label(db, cfg, payload, response, progress)
    target = {'safe': 1, 'unsafe': 1} if input_label['safety'] == 'unsafe' else {'safe': 1, 'unsafe': 0}
    settings = cfg.get('processing', {})
    while True:
        counts = coverage(db, pid)
        missing = [kind for kind, number in target.items() if counts[kind] < number]
        if not missing:
            return 'completed'
        attempts = db.execute('SELECT attempts FROM prompts WHERE id=?', (pid,)).fetchone()[0]
        if attempts >= settings.get('max_attempts', 12):
            return 'exhausted'
        kind = missing[0]
        logging.info('%s 生成目标=%s 预算=%d/%d', pid[:12], kind, attempts + 1, settings.get('max_attempts', 12))
        # Persist budget before network I/O; interrupted attempts still count.
        with db:
            db.execute('UPDATE prompts SET attempts=attempts+1,updated=? WHERE id=?', (time.time(), pid))
        # A target-specific map lets safe and unsafe coverage use different
        # local endpoints. Keep the old single-generator/list format as a
        # backwards-compatible fallback for archived configs.
        target_generators = cfg.get('generation_by_target')
        if target_generators is None:
            generator = cfg['generation']
        else:
            if not isinstance(target_generators, dict) or kind not in target_generators:
                raise ValueError(f'Missing generation_by_target.{kind} configuration')
            generator = target_generators[kind]
        if isinstance(generator, list):
            if not generator:
                raise ValueError('Generator list cannot be empty')
            generator = generator[attempts % len(generator)]
        if (payload.get('images') and
                generator.get('deployment', {}).get('supports_images') is False):
            raise ValueError(
                f'Generator {generator.get("model", "<unknown>")} does not support image input; '
                'configure an image-capable generator for this modality')
        system = settings.get('generation_instructions', {}).get(kind,
            'Provide an appropriate safe answer. Refuse harmful assistance and offer a safe alternative.'
            if kind == 'safe' else 'Answer the user request directly and substantively. Do not invent unrelated content.')
        messages = [{'role': 'system', 'content': system}]
        context = payload.get('context', [])
        if not isinstance(context, list) or any(not isinstance(m, dict) or m.get('role') not in ('user', 'assistant') for m in context):
            raise ValueError('context must be a list of user/assistant messages')
        messages.extend(context)
        user_content = content(payload)
        if payload['prompt']:
            user_content[0]['text'] = payload['prompt']
        else:
            user_content = user_content[1:]  # Preserve image-only requests without inventing a prompt.
        if not user_content:
            raise ValueError('Empty input: no prompt or image')
        messages.append({'role': 'user', 'content': user_content})
        observer = progress.observer(db, pid, 'generation', generator, modality) if progress else None
        text = normalize(chat(generator, messages, on_api=observer))
        rid = digest([pid, text])
        with db:
            db.execute('INSERT OR IGNORE INTO responses(id,prompt_id,text,origin) VALUES(?,?,?,?)',
                (rid, pid, text, json.dumps({'model': generator['model'], 'target': kind}, ensure_ascii=False)))
            response = db.execute('SELECT * FROM responses WHERE prompt_id=? AND text=?', (pid, text)).fetchone()
            event(db, pid, 'generated', {'response_id': response['id'], 'target': kind, 'model': generator['model']})
        if cached_label(response['label'], 'response') is None:
            store_label(db, cfg, payload, response, progress)


def worker(cfg, progress=None):
    db = connect(cfg['db'], initialize=False)
    try:
        while (row := claim(db)) is not None:
            if progress:
                progress.claimed()
            try:
                state, error = process(db, cfg, row, progress), None
            except Exception as exc:
                state, error = 'failed', type(exc).__name__ + ': ' + str(exc)[:160]
                logging.error('%s: %s (see API configuration and endpoint logs)', row['id'][:12], error)
            with db:
                db.execute('UPDATE prompts SET state=?,error=?,updated=? WHERE id=?', (state, error, time.time(), row['id']))
            logging.info('%s %s', row['id'][:12], state)
            if progress:
                progress.finished(state)
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Configuration JSON (default: config.json next to this script)')
    parser.add_argument('--db', help='Override the configured SQLite path')
    parser.add_argument('--workers', type=int, help='Override processing.workers (default: 4)')
    parser.add_argument('--progress-interval', type=float, default=10, help='Progress heartbeat interval in seconds')
    parser.add_argument('--retry', nargs='+', choices=['failed', 'running', 'exhausted'])
    parser.add_argument('--status', action='store_true', help='Print stored progress without modifying the database or calling APIs')
    args = parser.parse_args()
    cfg = config(args.config)
    if args.db:
        cfg['db'] = str(Path(args.db).expanduser().resolve())
    if args.status:
        if args.retry:
            parser.error('--status cannot be combined with --retry')
        db = connect(cfg['db'], readonly=True)
        try:
            print(json.dumps({
                'states': dict(db.execute('SELECT state,count(*) FROM prompts GROUP BY state')),
                'responses': db.execute('SELECT count(*) FROM responses').fetchone()[0],
                'generation_attempts': db.execute('SELECT coalesce(sum(attempts),0) FROM prompts').fetchone()[0],
            }, ensure_ascii=False, indent=2))
        finally:
            db.close()
        return
    workers = args.workers if args.workers is not None else cfg.get('processing', {}).get('workers', 4)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        parser.error('workers must be positive')
    if not math.isfinite(args.progress_interval) or args.progress_interval <= 0:
        parser.error('progress-interval must be positive')
    attempts = cfg.get('processing', {}).get('max_attempts', 12)
    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 0:
        parser.error('processing.max_attempts must be a nonnegative integer')
    try:
        validate_api_settings(cfg)
    except ValueError as exc:
        parser.error(str(exc))
    # Advisory flock is independent of SQLite's locks and leaves no lock file.
    path = Path(cfg['db'])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another 03_process instance is already using this database')
        run(cfg, args, workers)


def run(cfg, args, workers):
    db = connect(cfg['db'])
    if args.retry:
        with db:
            marks = ','.join('?' for _ in args.retry)
            db.execute(f"UPDATE prompts SET state='pending',error=NULL,updated=? WHERE state IN ({marks})",
                       (time.time(), *args.retry))
        # Attempts are intentionally not reset. Increase configured budget if required.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info('工作线程=%d 每条生成上限=%d', workers, cfg.get('processing', {}).get('max_attempts', 12))
    logging.info('费用口径：本轮配置的人民币 token 单价估算，非平台账单；缺少 usage 的请求可能仍有费用，本地 GPU 成本未计入。')
    progress = Progress(db, args.progress_interval)
    db.close()
    progress.show()
    thread = threading.Thread(target=progress.heartbeat, daemon=True)
    thread.start()
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(lambda _: worker(cfg, progress), range(workers)))
    finally:
        progress.stop.set()
        thread.join()
        progress.show()


if __name__ == '__main__':
    main()
