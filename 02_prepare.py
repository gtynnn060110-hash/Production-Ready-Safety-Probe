"""Import configured JSON/JSONL/Parquet fields; exact normalized content dedup."""
import argparse
import glob
import csv
import gzip
import lzma
import io
import json
from pathlib import Path
import time
import unicodedata
from common import config, connect, digest, event


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
                        yield json.loads(line)
            elif kind in ('.csv', '.tsv'):
                yield from csv.DictReader(f, delimiter='\t' if kind == '.tsv' else ',')
            else:
                data = json.load(f)
                if records_key:
                    data = get(data, records_key)
                if isinstance(data, list):
                    yield from data
                elif isinstance(data, dict):
                    for key, value in data.items():
                        if not isinstance(value, dict):
                            raise ValueError(f'{path}: expected ID -> record dictionary')
                        yield {'_record_key': key, **value}
                else:
                    raise ValueError(f'{path}: expected JSON array or ID mapping')
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
    adapter = source.get('adapter')
    if adapter == 'mss_chat':
        for question in row['queries']:
            for key in ('safe_image_path', 'unsafe_image_path'):
                yield {**row, '_prompt': question, '_image': row[key]}
    elif adapter == 'wildjailbreak':
        yield {**row, '_prompt': row.get('adversarial') or row.get('vanilla')}
    elif adapter == 'vlguard':
        for pair in row['instr-resp']:
            question = next((pair[k] for k in ('instruction', 'safe_instruction', 'unsafe_instruction') if k in pair), None)
            if question is None:
                raise ValueError('Unknown VLGuard instruction keys')
            yield {**row, '_prompt': question, '_response': pair.get('response')}
    elif adapter == 'messages':
        messages = row[source.get('message_field', 'message')]
        # One sample per assistant turn, preserving its preceding conversation.
        for i, message in enumerate(messages):
            if message['role'] == 'assistant' and i and messages[i-1]['role'] == 'user':
                yield {**row, '_prompt': messages[i-1]['content'], '_response': message['content'],
                       '_context': messages[:i-1]}
    elif adapter == 'aegis' and (row.get('prompt') == 'REDACTED' or row.get('reconstruction_id_if_redacted')):
        return  # Content not actually released: do not generate from the placeholder.
    else:
        yield row


def normalize(text):
    return unicodedata.normalize('NFC', text).replace('\r\n', '\n').strip()


def images(value, root, assets):
    if value is None:
        return [], []
    if not isinstance(value, list):
        value = [value]
    paths, hashes = [], []
    for image in value:
        if isinstance(image, dict):
            if image.get('bytes') is not None:
                import hashlib
                raw = image['bytes']
                key = hashlib.sha256(raw).hexdigest()
                from PIL import Image
                fmt = Image.open(io.BytesIO(raw)).format.lower()
                path = assets / f'{key}.{fmt}'
                if not path.exists():
                    path.write_bytes(raw)
            else:
                path = Path(image['path'])
                path = path if path.is_absolute() else root / path
        else:
            path = Path(str(image))
            path = path if path.is_absolute() else root / path
        if not path.is_file():
            raise ValueError(f'Image missing: {path}')
        import hashlib
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        paths.append(str(path.resolve()))
    return paths, hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--sources', nargs='+', help='Only import selected source names')
    parser.add_argument('--limit', type=int, help='Maximum source records per file for smoke testing')
    args = parser.parse_args()
    cfg = config(args.config)
    assets = Path(cfg.get('assets_dir', 'data/images'))
    assets.mkdir(parents=True, exist_ok=True)
    db = connect(cfg['db'])
    count = 0
    for source in cfg['sources']:
        if not source.get('enabled', True) or (args.sources and source['name'] not in args.sources):
            continue
        root = Path(cfg.get('raw_dir', 'data/raw')) / source['name']
        fields = source['fields']
        for part in source.get('imports', []):
            files = sorted(glob.glob(str(root / part['glob']), recursive=True))
            if not files:
                raise ValueError(f'No files matched: {source["name"]}/{part["glob"]}')
            for filename in files:
                path = Path(filename)
                expanded = (item for raw in records(path, part.get('records_key')) for item in adapt(raw, source))
                for index, row in enumerate(expanded):
                    if args.limit is not None and index >= args.limit:
                        break
                    prompt = get(row, fields['prompt'])
                    if not isinstance(prompt, str) or not prompt.strip():
                        raise ValueError(f'{path}:{index}: missing text prompt; configure fields or convert first')
                    image_root = root / part.get('image_root', source.get('image_root', ''))
                    img, image_hashes = images(get(row, fields.get('images')), image_root, assets)
                    context = get(row, fields.get('context'), []) or []
                    if isinstance(context, str):
                        context = [{'role': 'user', 'content': context}] if context.strip() else []
                    payload = {'prompt': normalize(prompt), 'context': context, 'images': img}
                    pid = digest({**payload, 'images': image_hashes})
                    with db:
                        db.execute('INSERT OR IGNORE INTO prompts(id,payload,updated) VALUES(?,?,?)',
                                   (pid, json.dumps(payload, ensure_ascii=False), time.time()))
                        sid = f'{path.relative_to(root)}:{index}:' + str(get(row, fields.get('id'), ''))
                        db.execute('INSERT OR IGNORE INTO provenance VALUES(?,?,?,?)', (pid, source['name'], sid,
                            json.dumps({key: get(row, key) for key in fields.get('metadata', [])} if fields.get('metadata') else get(row, fields.get('labels')), ensure_ascii=False)))
                        for key in fields.get('responses', []):
                            text = get(row, key)
                            if not isinstance(text, str) or not text.strip() or text.strip().lower() in ('nan', 'none', 'null'):
                                continue
                            text = normalize(text)
                            cur = db.execute('INSERT OR IGNORE INTO responses(id,prompt_id,text,origin) VALUES(?,?,?,?)',
                                (digest([pid, text]), pid, text, json.dumps({'source': source['name'], 'field': key})))
                            if cur.rowcount:
                                db.execute("UPDATE prompts SET state='pending' WHERE id=?", (pid,))
                    count += 1
        print(f'{source["name"]}: cumulative {count} imported source records')
    print('Unique prompts:', db.execute('SELECT count(*) FROM prompts').fetchone()[0])
    db.close()


if __name__ == '__main__':
    main()
