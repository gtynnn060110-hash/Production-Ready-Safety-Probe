#!/usr/bin/env python3
"""Import judged three-round rollout results into the prepared SQLite database.

By default only validate and report what would be written. Use --apply to commit.
No model generation or judge requests are performed.
"""
import sys as _prep_sys; from pathlib import Path as _PrepPath; _prep_sys.path.insert(0, str(_PrepPath(__file__).resolve().parents[2])); from scripts._bootstrap import setup as _prep_setup; _prep_setup()
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import unicodedata

from common import LABELS, digest


def normalize(text):
    return unicodedata.normalize('NFC', text).replace('\r\n', '\n').strip()


def read_jsonl(path):
    rows = []
    with path.open(encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{line_no}: invalid JSON') from exc
                if not isinstance(row, dict):
                    raise ValueError(f'{path}:{line_no}: expected an object')
                rows.append(row)
    return rows


def validate_judgment(item, round_number):
    if item.get('round') != round_number or item.get('safety') not in ('safe', 'unsafe'):
        raise ValueError(f'Round {round_number}: invalid judgment round or safety')
    cats = item.get('categories')
    if (not isinstance(cats, list) or any(not isinstance(c, str) or c not in LABELS for c in cats)
            or len(cats) != len(set(cats))):
        raise ValueError(f'Round {round_number}: invalid categories')
    if (item['safety'] == 'safe' and cats) or (item['safety'] == 'unsafe' and not cats):
        raise ValueError(f'Round {round_number}: safety/categories disagree')


def collect(directory, source_file):
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    actual_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
    if actual_hash != manifest.get('source_sha256'):
        raise ValueError('Prompt source does not match the rollout manifest')
    source = json.loads(source_file.read_text(encoding='utf-8'))
    if not isinstance(source, list):
        raise ValueError('Prompt source must be a JSON array')
    expected = {}
    for row in source:
        if not isinstance(row, dict) or not isinstance(row.get('prompt'), str):
            raise ValueError('Invalid source prompt')
        prompt = normalize(row['prompt'])
        pid = row.get('prompt_id')
        if not prompt or pid != digest(prompt) or pid in expected:
            raise ValueError('Source prompt ID mismatch or duplicate')
        if row.get('context') or row.get('images'):
            raise ValueError('This importer supports only the text-only rollout')
        expected[pid] = prompt
    if len(expected) != manifest.get('source_count') or manifest.get('max_rounds') != 3:
        raise ValueError('Unexpected rollout manifest source count or round count')
    judge_model, taxonomy = manifest.get('judge_model'), manifest.get('taxonomy')
    if not isinstance(judge_model, str) or not judge_model or not isinstance(taxonomy, str) or not taxonomy:
        raise ValueError('Judge model and taxonomy are required in the manifest')

    grouped = {}
    round_counts = {}
    for round_number in range(1, 4):
        generated = read_jsonl(directory / f'round-{round_number}-rollout.jsonl')
        judged = read_jsonl(directory / f'round-{round_number}-judged.jsonl')
        by_id = {}
        for row in generated:
            pid = row.get('prompt_id')
            if pid not in expected or pid in by_id or row.get('round') != round_number:
                raise ValueError(f'Round {round_number}: unexpected, duplicate or mismatched rollout ID')
            if not isinstance(row.get('prompt'), str) or normalize(row['prompt']) != expected[pid]:
                raise ValueError(f'Round {round_number}: rollout prompt mismatch')
            if row.get('finish_reason') != 'stop':
                raise ValueError(f'Round {round_number}: incomplete/truncated rollout; review before import')
            if not isinstance(row.get('response'), str) or not normalize(row['response']):
                raise ValueError(f'Round {round_number}: missing response text')
            by_id[pid] = row
        if set(by_id) != set(expected):
            raise ValueError(f'Round {round_number}: rollout is incomplete or has unexpected prompts')
        judgments = {}
        for row in judged:
            pid = row.get('prompt_id')
            if pid not in by_id or pid in judgments:
                raise ValueError(f'Round {round_number}: unexpected or duplicate judgment ID')
            validate_judgment(row, round_number)
            judgments[pid] = row
        if set(judgments) != set(by_id):
            raise ValueError(f'Round {round_number}: judgments are incomplete')
        round_counts[str(round_number)] = dict(Counter(r['safety'] for r in judgments.values()))
        for input_pid, row in by_id.items():
            prompt = expected[input_pid]
            db_pid = digest({'prompt': prompt, 'context': [], 'images': []})
            text = normalize(row['response'])
            rid = digest([db_pid, text])
            decision = judgments[input_pid]
            label = {'safety': decision['safety'], 'categories': decision['categories'],
                     'judge_model': judge_model, 'taxonomy_version': taxonomy}
            if rid in grouped:
                if grouped[rid]['label'] != label:
                    raise ValueError('Identical generated response has conflicting judgments across rounds')
                grouped[rid]['rounds'].append(round_number)
            else:
                grouped[rid] = {'response_id': rid, 'prompt_id': db_pid, 'input_prompt_id': input_pid,
                                'prompt': prompt, 'text': text, 'label': label, 'rounds': [round_number]}
        expected = {pid: expected[pid] for pid, decision in judgments.items() if decision['safety'] == 'safe'}
    return manifest, grouped, round_counts


def process(db_path, manifest, records, apply=False, backup_path=None,
            source_name='qwen3 8b em generate'):
    uri = 'file:' + str(db_path.resolve()) + ('?mode=rw' if apply else '?mode=ro')
    db = sqlite3.connect(uri, uri=True, timeout=60)
    db.execute('PRAGMA foreign_keys=ON')
    if backup_path:
        if not apply:
            raise ValueError('--backup requires --apply')
        backup_path = backup_path.resolve()
        if backup_path.exists():
            raise FileExistsError(f'Backup already exists: {backup_path}')
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(backup_path)) as snapshot:
            db.backup(snapshot)
        print(f'Backup created: {backup_path}', flush=True)
    if apply:
        db.execute('BEGIN IMMEDIATE')
    stats = Counter()
    try:
        prior_events = set()
        for (text,) in db.execute("SELECT detail FROM events WHERE kind='response_imported'"):
            try:
                prior_events.add(json.loads(text).get('import_key'))
            except (TypeError, json.JSONDecodeError):
                continue
        for item in records.values():
            pid, rid = item['prompt_id'], item['response_id']
            found = db.execute('SELECT payload FROM prompts WHERE id=?', (pid,)).fetchone()
            if not found:
                raise ValueError(f'Prompt missing from database: {item["input_prompt_id"]}')
            payload = json.loads(found[0])
            if payload.get('prompt') != item['prompt'] or payload.get('context') != [] or payload.get('images') != []:
                raise ValueError(f'Database prompt payload mismatch: {item["input_prompt_id"]}')
            if not db.execute("SELECT 1 FROM provenance WHERE prompt_id=? AND source IN ('saferlhf','saferlhf_qa') LIMIT 1", (pid,)).fetchone():
                raise ValueError(f'SafeRLHF provenance missing: {item["input_prompt_id"]}')
            old = db.execute('SELECT id, origin, label FROM responses WHERE prompt_id=? AND text=?', (pid, item['text'])).fetchone()
            origin = {'source': source_name, 'model': manifest['base'],
                      'adapter': manifest['adapter'], 'rounds': item['rounds'], 'finish_reason': 'stop',
                      'input_prompt_id': item['input_prompt_id'], 'run_source_sha256': manifest['source_sha256']}
            label = json.dumps(item['label'], ensure_ascii=False)
            if old is None:
                stats['inserted'] += 1
                response_id = rid
                if apply:
                    db.execute('INSERT INTO responses(id,prompt_id,text,origin,label) VALUES(?,?,?,?,?)',
                               (rid, pid, item['text'], json.dumps(origin, ensure_ascii=False), label))
            else:
                response_id = old[0]
                if old[2] is None:
                    stats['existing_label_added'] += 1
                    if apply:
                        db.execute('UPDATE responses SET label=? WHERE id=? AND label IS NULL', (label, response_id))
                else:
                    previous = json.loads(old[2])
                    if not isinstance(previous, dict) or previous.get('safety') != item['label']['safety']:
                        raise ValueError(f'Existing label disagrees for response {response_id}; no changes committed')
                    stats['existing_unchanged'] += 1
            event_key = digest([origin['source'], manifest['source_sha256'], rid])
            if event_key not in prior_events:
                stats['new_provenance_events'] += 1
                if apply:
                    detail = {'import_key': event_key, 'response_id': response_id, 'origin': origin,
                              'judgment': item['label']}
                    db.execute('INSERT INTO events(prompt_id,kind,detail,created) VALUES(?,?,?,?)',
                               (pid, 'response_imported', json.dumps(detail, ensure_ascii=False), time.time()))
        if apply:
            db.commit()
    except BaseException:
        if apply:
            db.rollback()
        raise
    finally:
        db.close()
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db', type=Path, default=Path('data/probe.sqlite'))
    p.add_argument('--dataset-dir', type=Path, default=Path('data/saferlhf-response-gaps-v1/mixed-qwen3-5epoch-three-rounds-v1'))
    p.add_argument('--source-prompts', type=Path, default=Path('data/saferlhf-response-gaps-v1/missing_unsafe_prompt_unsafe.json'))
    p.add_argument('--apply', action='store_true', help='Commit changes; default is read-only validation')
    p.add_argument('--backup', type=Path, help='Create a SQLite backup before applying')
    p.add_argument('--include-safe', action='store_true', help='Also import safe generated responses; default is unsafe only')
    args = p.parse_args(argv)
    manifest, records, rounds = collect(args.dataset_dir, args.source_prompts)
    print('Round judge counts:', json.dumps(rounds, ensure_ascii=False), flush=True)
    if not args.include_safe:
        records = {rid: row for rid, row in records.items() if row['label']['safety'] == 'unsafe'}
    print('Distinct selected prompt/response pairs:', len(records), flush=True)
    result = process(args.db, manifest, records, args.apply, args.backup)
    print(('Committed: ' if args.apply else 'Dry run; no database changes: ') + json.dumps(dict(result), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
