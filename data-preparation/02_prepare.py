"""Import configured JSON/JSONL/Parquet fields; exact normalized content dedup."""
import argparse
from collections import Counter
import glob
import csv
import gzip
import hashlib
import lzma
import io
import json
from pathlib import Path
import time
from common import config, connect, digest, normalize


class RecordIssue(ValueError):
    """A source record that must be quarantined instead of partially imported."""

    def __init__(self, reason, detail):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def get(row, key, default=None):
    if not key:
        return default
    for part in key.split('.'):
        if not isinstance(row, dict) or part not in row:
            return default
        row = row[part]
    return row


def records(path, records_key=None):
    suffixes = path.suffixes
    opener = gzip.open if path.suffix == '.gz' else lzma.open if path.suffix == '.xz' else open
    kind = suffixes[-2] if path.suffix in ('.gz', '.xz') else path.suffix
    if kind in ('.jsonl', '.csv', '.tsv', '.json'):
        with opener(path, 'rt', encoding='utf-8-sig') as f:
            if kind == '.jsonl':
                for line in f:
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError as exc:
                            yield RecordIssue('invalid_json', {'message': str(exc)})
            elif kind in ('.csv', '.tsv'):
                yield from csv.DictReader(f, delimiter='\t' if kind == '.tsv' else ',')
            else:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as exc:
                    yield RecordIssue('invalid_json', {'message': str(exc)})
                    return
                if records_key:
                    data = get(data, records_key)
                if isinstance(data, list):
                    yield from data
                elif isinstance(data, dict):
                    for key, value in data.items():
                        if not isinstance(value, dict):
                            yield RecordIssue('invalid_record', {'record_id': key, 'type': type(value).__name__})
                        else:
                            yield {'_record_key': key, **value}
                else:
                    yield RecordIssue('invalid_records', {'type': type(data).__name__,
                                                        'message': 'Expected a JSON array or ID mapping'})
    elif kind == '.parquet':
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            yield from batch.to_pylist()
    elif kind == '.arrow':
        import pyarrow as pa
        import pyarrow.ipc as ipc
        with pa.memory_map(str(path), 'r') as stream:
            for batch in ipc.open_stream(stream):
                yield from batch.to_pylist()
    else:
        raise ValueError(f'Unsupported file: {path}')


def adapt(row, source):
    """Expand exceptional formats into records; ordinary sources only use field maps."""
    if isinstance(row, RecordIssue):
        raise row
    if not isinstance(row, dict):
        raise RecordIssue('invalid_record', {'type': type(row).__name__})
    adapter = source.get('adapter')
    if adapter == 'mss_chat':
        if not isinstance(row.get('queries'), list):
            raise RecordIssue('invalid_queries', {'type': type(row.get('queries')).__name__})
        for question in row['queries']:
            for key in ('safe_image_path', 'unsafe_image_path'):
                yield {**row, '_prompt': question, '_image': row.get(key)}
    elif adapter == 'wildjailbreak':
        yield {**row, '_prompt': row.get('adversarial') or row.get('vanilla')}
    elif adapter == 'vlguard':
        if not isinstance(row.get('instr-resp'), list) or not row['instr-resp']:
            raise RecordIssue('invalid_pairs', {'type': type(row.get('instr-resp')).__name__})
        for pair_index, pair in enumerate(row['instr-resp']):
            if not isinstance(pair, dict):
                raise RecordIssue('invalid_pair', {'pair_index': pair_index, 'type': type(pair).__name__})
            question = next((pair[k] for k in ('instruction', 'safe_instruction', 'unsafe_instruction') if k in pair), None)
            if question is None:
                raise RecordIssue('missing_instruction', {'pair_index': pair_index})
            yield {**row, '_prompt': question, '_response': pair.get('response'),
                   '_response_fields': {'_response': f'instr-resp[{pair_index}].response'}}
    elif adapter == 'messages':
        messages = row.get(source.get('message_field', 'message'))
        if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
            raise RecordIssue('invalid_messages', {'type': type(messages).__name__})
        # One sample per assistant turn, preserving its preceding conversation.
        for i, message in enumerate(messages):
            if message.get('role') == 'assistant' and i and messages[i-1].get('role') == 'user':
                yield {**row, '_prompt': messages[i-1].get('content'), '_response': message.get('content'),
                       '_context': messages[:i-1]}
    elif adapter == 'aegis' and (row.get('prompt') == 'REDACTED' or row.get('reconstruction_id_if_redacted')):
        return  # Content not actually released: do not generate from the placeholder.
    else:
        yield row


def path_mappings(cfg, source, part):
    mappings = []
    for item in source.get('image_path_maps', []) + part.get('image_path_maps', []):
        prefix = item['prefix']
        if not isinstance(prefix, str) or not prefix or not prefix.endswith('/'):
            raise ValueError('Image path mapping prefixes must be nonempty directories ending in /')
        target = Path(item['root'])
        if not target.is_absolute():
            target = Path(cfg.get('raw_dir', 'data/raw')) / target
        mappings.append((prefix, target))
    return mappings


def image_candidates(reference, root, mappings):
    original = Path(reference)
    candidates = [original if original.is_absolute() else root / original]
    if not original.is_absolute():
        for prefix, mapped_root in mappings:
            if reference.startswith(prefix):
                candidates.append(mapped_root / reference[len(prefix):])
    return candidates


def file_digest(path, cache=None):
    stat = path.stat()
    cache_key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    key = cache.get(cache_key) if cache is not None else None
    if key is None:
        key = hashlib.sha256(path.read_bytes()).hexdigest()
        if cache is not None:
            cache[cache_key] = key
    return key


def verify_image(value):
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(value) as decoded:
            fmt = decoded.format.lower()
            decoded.verify()
        return fmt
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise RecordIssue('invalid_image', {'error': type(exc).__name__}) from None


def images(value, root, assets, *, required=False, path_maps=(), hash_cache=None):
    if value is None or (isinstance(value, list) and not value):
        if required:
            raise RecordIssue('missing_image_field', {'message': 'Expected a nonempty image field'})
        return [], []
    if not isinstance(value, list):
        value = [value]
    paths, hashes = [], []
    for image in value:
        raw = image.get('bytes') if isinstance(image, dict) else image
        if isinstance(image, dict) and raw is not None and not isinstance(raw, (bytes, bytearray, memoryview)):
            raise RecordIssue('invalid_image_bytes', {'type': type(raw).__name__})
        if isinstance(raw, (bytes, bytearray, memoryview)):
            raw = bytes(raw)
            key = hashlib.sha256(raw).hexdigest()
            fmt = verify_image(io.BytesIO(raw))
            assets.mkdir(parents=True, exist_ok=True)
            path = assets / f'{key}.{fmt}'
            if path.exists():
                if file_digest(path, hash_cache) != key:
                    raise RecordIssue('corrupt_extracted_image', {'path': str(path)})
            else:
                path.write_bytes(raw)
        else:
            reference = image.get('path') if isinstance(image, dict) else image
            if not isinstance(reference, (str, Path)) or not str(reference).strip():
                raise RecordIssue('invalid_image_reference', {'type': type(reference).__name__})
            reference = str(reference)
            candidates = image_candidates(reference, root, path_maps)
            available = [p for p in candidates if p.is_file()]
            if not available:
                raise RecordIssue('missing_image', {
                    'reference': reference, 'candidates': [str(p) for p in candidates]})
            path = available[0]
            key = file_digest(path, hash_cache)
            if any(file_digest(candidate, hash_cache) != key for candidate in available[1:]):
                raise RecordIssue('conflicting_image_mapping', {
                    'reference': reference, 'candidates': [str(p) for p in available]})
            if hash_cache is None or ('verified', key) not in hash_cache:
                try:
                    verify_image(path)
                except RecordIssue as exc:
                    raise RecordIssue(exc.reason, {**exc.detail, 'path': str(path)}) from None
                if hash_cache is not None:
                    hash_cache['verified', key] = True
        hashes.append(key)
        paths.append(str(path.resolve()))
    return paths, hashes


def image_settings(cfg, source, part):
    fields = {**source['fields'], **part.get('fields', {})}
    required = part.get('images_required', source.get('images_required', bool(fields.get('images'))))
    if required and not fields.get('images'):
        raise ValueError(f'{source["name"]}: images_required needs fields.images')
    return required, path_mappings(cfg, source, part)


def file_provenance(source, part, path):
    """Explicit native config/split; templates may reference relative path components."""
    def value(key):
        template = part.get(key, source.get(key))
        return template.format(path=path.parts) if template is not None else None
    split = value('source_split')
    if not split:
        raise ValueError(f'{source["name"]}/{path}: source_split must be specified')
    return {'source_dataset': source.get('repo_id', source['name']),
            'source_config': value('source_config'), 'source_split': split,
            'revision': source.get('revision'), 'file': path.as_posix()}


def import_record(db, cfg, source, part, root, assets, row, source_id, hash_cache, location):
    """Validate input before writing; the caller commits batches of complete records."""
    fields = {**source['fields'], **part.get('fields', {})}
    prompt = get(row, fields['prompt'])
    allow_image_only = source.get('allow_image_only', False)
    if not isinstance(prompt, str) or (not prompt.strip() and not allow_image_only):
        raise RecordIssue('empty_prompt' if isinstance(prompt, str) else 'invalid_prompt', {
            'field': fields['prompt'], 'type': type(prompt).__name__})
    required, mappings = image_settings(cfg, source, part)
    image_root = root / part.get('image_root', source.get('image_root', ''))
    img, image_hashes = images(get(row, fields.get('images')), image_root, assets,
                              required=required, path_maps=mappings, hash_cache=hash_cache)
    if not prompt.strip() and not img:
        raise RecordIssue('empty_input', {'message': 'Image-only inputs must contain an image'})
    context = get(row, fields.get('context'), [])
    if context is None:
        context = []
    if isinstance(context, str):
        context = [{'role': 'user', 'content': context}] if context.strip() else []
    if not isinstance(context, list) or any(
            not isinstance(message, dict) or not isinstance(message.get('role'), str)
            or not isinstance(message.get('content'), str) for message in context):
        raise RecordIssue('invalid_context', {'type': type(context).__name__})
    payload = {'prompt': normalize(prompt), 'context': context, 'images': img}
    pid = digest({**payload, 'images': image_hashes})
    original_labels = ({key: get(row, key) for key in fields['metadata']} if fields.get('metadata')
                       else get(row, fields.get('labels')))
    responses = []
    for key in fields.get('responses', []):
        text = get(row, key)
        if text is None or (isinstance(text, float) and text != text):
            continue
        if not isinstance(text, str):
            raise RecordIssue('invalid_response', {'field': key, 'type': type(text).__name__})
        if not text.strip() or text.strip().lower() in ('nan', 'none', 'null'):
            continue
        original_field = row.get('_response_fields', {}).get(key, key)
        responses.append((normalize(text), original_field))
    db.execute('INSERT OR IGNORE INTO prompts(id,payload,updated) VALUES(?,?,?)',
               (pid, json.dumps(payload, ensure_ascii=False), time.time()))
    # Update metadata on reimport, including corrections to field mappings.
    db.execute('''INSERT INTO provenance(
                    prompt_id,source,source_id,original_labels,source_dataset,source_config,
                    source_split,revision,file,raw_index,item_index,record_id)
                  VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                  ON CONFLICT(prompt_id,source,source_id)
                  DO UPDATE SET original_labels=excluded.original_labels,
                    source_dataset=excluded.source_dataset,source_config=excluded.source_config,
                    source_split=excluded.source_split,revision=excluded.revision,file=excluded.file,
                    raw_index=excluded.raw_index,item_index=excluded.item_index,record_id=excluded.record_id''',
               (pid, source['name'], source_id, json.dumps(original_labels, ensure_ascii=False),
                *(location[key] for key in ('source_dataset', 'source_config', 'source_split',
                                           'revision', 'file', 'raw_index', 'item_index', 'record_id'))))
    for text, original_field in responses:
        response_id = digest([pid, text])
        cur = db.execute('INSERT OR IGNORE INTO responses(id,prompt_id,text,origin) VALUES(?,?,?,?)',
                         (response_id, pid, text, json.dumps({'source': source['name'], 'field': original_field})))
        if not cur.rowcount:
            # Merged databases may retain a different ID for the same prompt/text pair.
            response_id = db.execute('SELECT id FROM responses WHERE prompt_id=? AND text=?',
                                     (pid, text)).fetchone()[0]
        db.execute('''INSERT OR IGNORE INTO response_provenance
                      (response_id,prompt_id,source,source_id,field) VALUES(?,?,?,?,?)''',
                   (response_id, pid, source['name'], source_id, original_field))
        if cur.rowcount:
            db.execute("UPDATE prompts SET state='pending',error=NULL,updated=? WHERE id=?", (time.time(), pid))
    db.execute('DELETE FROM import_issues WHERE source=? AND source_id=?', (source['name'], source_id))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    parser.add_argument('--db', help='Override the SQLite database path')
    parser.add_argument('--sources', nargs='+', help='Only import selected source names')
    parser.add_argument('--limit', type=int, help='Maximum source records per file for smoke testing')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error('limit must not be negative')
    cfg = config(args.config)
    unknown = set(args.sources or ()) - {source['name'] for source in cfg['sources']}
    if unknown:
        parser.error('Unknown sources: ' + ', '.join(sorted(unknown)))
    assets = Path(cfg.get('assets_dir', 'data/images'))
    assets.mkdir(parents=True, exist_ok=True)
    db = connect(args.db or cfg['db'])
    db.execute('PRAGMA cache_size=-131072')  # 128 MiB for large imports and dedup indexes.
    count, quarantined = 0, 0
    hash_cache = {}
    for source in cfg['sources']:
        if not source.get('enabled', True) or (args.sources and source['name'] not in args.sources):
            continue
        root = Path(cfg.get('raw_dir', 'data/raw')) / source['name']
        fields = source['fields']
        imported, raw_count, skipped = 0, 0, 0
        issues = Counter()
        for part in source.get('imports', []):
            image_settings(cfg, source, part)  # Invalid configuration is fatal, not a record issue.
            files = sorted(glob.glob(str(root / part['glob']), recursive=True))
            if not files:
                raise ValueError(f'No files matched: {source["name"]}/{part["glob"]}')
            for filename in files:
                path = Path(filename)
                location = file_provenance(source, part, path.relative_to(root))
                for raw_index, raw in enumerate(records(path, part.get('records_key'))):
                    if args.limit is not None and raw_index >= args.limit:
                        break
                    raw_count += 1
                    try:
                        items = list(adapt(raw, source))
                    except RecordIssue as exc:
                        items = [exc]
                    if not items:
                        skipped += 1
                    record_id = get(raw, fields.get('id'))
                    if record_id is None and isinstance(raw, dict):
                        record_id = next((raw[k] for k in ('id', 'instruction_id', 'conv_id', 'traceid', '_record_key')
                                          if raw.get(k) is not None), None)
                    for item_index, row in enumerate(items):
                        sid = (f'{source.get("revision", "")}:{location["source_config"]}:'
                               f'{location["source_split"]}:{location["file"]}:{raw_index}:{item_index}')
                        origin = {**location, 'raw_index': raw_index, 'item_index': item_index,
                                  'record_id': str(record_id) if record_id is not None else None}
                        try:
                            if isinstance(row, RecordIssue):
                                raise row
                            import_record(db, cfg, source, part, root, assets, row, sid, hash_cache, origin)
                        except RecordIssue as exc:
                            detail = {**origin, **exc.detail}
                            db.execute('''INSERT INTO import_issues VALUES(?,?,?,?,?)
                                          ON CONFLICT(source,source_id) DO UPDATE SET
                                          reason=excluded.reason,detail=excluded.detail,updated=excluded.updated''',
                                       (source['name'], sid, exc.reason, json.dumps(detail, ensure_ascii=False), time.time()))
                            issues[exc.reason] += 1
                            quarantined += 1
                            continue
                        imported += 1
                        count += 1
                    # Commit complete raw records together, including all expanded pairs and answers.
                    if (raw_index + 1) % 1000 == 0:
                        db.commit()
                db.commit()
        print(f'{source["name"]}: raw {raw_count}, imported {imported}, skipped {skipped}, '
              f'quarantined {sum(issues.values())} {dict(issues)}', flush=True)
    print(f'Imported source records: {count}; quarantined this run: {quarantined}')
    print('Unique prompts:', db.execute('SELECT count(*) FROM prompts').fetchone()[0])
    print('Unresolved import issues:', db.execute('SELECT count(*) FROM import_issues').fetchone()[0])
    db.close()


if __name__ == '__main__':
    main()
