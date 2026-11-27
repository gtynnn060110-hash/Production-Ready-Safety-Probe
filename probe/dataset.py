"""Read labeled SQLite pairs, group splits, and load self-contained feature shards."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import logging
import math
import os
from pathlib import Path
import random
import sqlite3
import tempfile

import torch
from torch.utils.data import Dataset, Sampler


HEADS = tuple(f'R{i:02}' for i in range(1, 20)) + ('overall_unsafe',)
SPLITS = ('train', 'val', 'test')


def response_labels(value):
    """Return targets and known-label masks; missing categories are not negatives."""
    label = json.loads(value) if isinstance(value, str) else value
    if not isinstance(label, dict) or label.get('task', 'response') != 'response':
        raise ValueError('Expected a response label object')
    safety = label.get('safety')
    if safety not in ('safe', 'unsafe'):
        raise ValueError('Missing binary response safety')
    categories = label.get('categories')
    targets = [0.0] * len(HEADS)
    known = [categories is not None] * 19 + [True]
    targets[-1] = float(safety == 'unsafe')
    if categories is not None:
        if (not isinstance(categories, list) or any(not isinstance(c, str) for c in categories)
                or len(set(categories)) != len(categories)
                or any(c not in HEADS[:19] for c in categories)):
            raise ValueError('Invalid response categories')
        if (safety == 'safe' and categories) or (safety == 'unsafe' and not categories):
            raise ValueError('Inconsistent response safety/categories')
        for category in categories:
            targets[HEADS.index(category)] = 1.0
    return targets, known


def read_connection(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA query_only=ON')
    return db


class SQLiteDataset(Dataset):
    """Keep only the pair index in memory; read original text/images on demand.

    Every valid labeled response is eligible regardless of the prompt label,
    generation state, coverage or historical usable flag. Sampling and splitting
    operate on prompt IDs, never individual answers.
    """

    def __init__(self, db, modality='all', sources=None, max_prompts=None,
                 seed=42, ratios=(0.8, 0.1, 0.1)):
        self.path = Path(db).expanduser().resolve()
        if modality not in ('all', 'image', 'text'):
            raise ValueError('modality must be all, image or text')
        if max_prompts is not None and (type(max_prompts) is not int or max_prompts < 1):
            raise ValueError('max_prompts must be a positive integer')
        if (len(ratios) != 3 or any(not math.isfinite(x) or x < 0 for x in ratios)
                or not math.isclose(sum(ratios), 1.0)):
            raise ValueError('Three nonnegative split ratios must sum to 1')
        if type(seed) is not int:
            raise ValueError('seed must be an integer')
        sources = sorted(set(sources or ()))
        clauses = ['r.label IS NOT NULL', 'length(trim(r.text)) > 0']
        if modality != 'all':
            clauses.append("coalesce(json_array_length(p.payload,'$.images'),0) "
                           + ('> 0' if modality == 'image' else '= 0'))
        if sources:
            placeholders = ','.join('?' for _ in sources)
            clauses.append('EXISTS (SELECT 1 FROM provenance v WHERE v.prompt_id=p.id '
                           f'AND v.source IN ({placeholders}))')
        query = '''SELECT r.id,r.prompt_id,r.label,
                   coalesce(json_array_length(p.payload,'$.images'),0) AS image_count
                   FROM responses r JOIN prompts p ON p.id=r.prompt_id
                   WHERE ''' + ' AND '.join(clauses) + ' ORDER BY p.id,r.id'
        candidates, invalid = [], Counter()
        db_conn = read_connection(self.path)
        try:
            for row in db_conn.execute(query, sources):
                try:
                    labels, mask = response_labels(row['label'])
                except (ValueError, TypeError) as exc:
                    invalid[str(exc)] += 1
                    continue
                candidates.append({'id': row['id'], 'prompt_id': row['prompt_id'],
                                   'labels': labels, 'label_mask': mask,
                                   'modality': 'image' if row['image_count'] else 'text'})
        finally:
            db_conn.close()
        prompts = sorted({r['prompt_id'] for r in candidates})
        random.Random(seed).shuffle(prompts)
        if max_prompts is not None:
            prompts = prompts[:max_prompts]
        first = int(len(prompts) * ratios[0])
        second = first + int(len(prompts) * ratios[1])
        split = {pid: 'train' if i < first else 'val' if i < second else 'test'
                 for i, pid in enumerate(prompts)}
        self.rows = [{**row, 'split': split[row['prompt_id']]} for row in candidates
                     if row['prompt_id'] in split]
        self.selection = {'db': str(self.path), 'modality': modality, 'sources': sources,
                          'max_prompts': max_prompts, 'seed': seed, 'ratios': list(ratios)}
        self.invalid_labels = dict(invalid)
        self.summary = {'prompts': len(prompts), 'responses': len(self.rows),
                        'prompt_splits': dict(Counter(split.values())),
                        'response_splits': dict(Counter(r['split'] for r in self.rows)),
                        'modalities': dict(Counter(r['modality'] for r in self.rows)),
                        'invalid_labels': dict(invalid)}
        self._db = None
        self._pid = None
        if invalid:
            logging.warning('Skipped %d invalid response labels: %s', sum(invalid.values()), dict(invalid))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self._db is None or self._pid != os.getpid():
            if self._db is not None:
                self._db.close()
            self._db, self._pid = read_connection(self.path), os.getpid()
        meta = self.rows[index]
        row = self._db.execute('''SELECT p.payload,r.text,r.label FROM responses r
                                JOIN prompts p ON p.id=r.prompt_id
                                WHERE r.id=? AND r.prompt_id=?''',
                               (meta['id'], meta['prompt_id'])).fetchone()
        if row is None:
            raise ValueError(f'Database pair disappeared after selection: {meta["id"]}')
        labels, mask = response_labels(row['label'])
        if labels != meta['labels'] or mask != meta['label_mask']:
            raise ValueError(f'Response label changed after selection: {meta["id"]}')
        payload = json.loads(row['payload'])
        prompt, context, images = payload.get('prompt'), payload.get('context', []), payload.get('images', [])
        if (not isinstance(prompt, str) or not isinstance(context, list)
                or not isinstance(images, list) or any(not isinstance(p, str) for p in images)
                or (not prompt.strip() and not images) or not row['text'].strip()):
            raise ValueError(f'Invalid database input: {meta["id"]}')
        actual_modality = 'image' if images else 'text'
        if actual_modality != meta['modality']:
            raise ValueError(f'Database input modality changed: {meta["id"]}')
        return {**meta, 'prompt': prompt, 'context': context, 'images': images, 'response': row['text']}

    def __getstate__(self):
        return {**self.__dict__, '_db': None, '_pid': None}

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name,
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(value, stream)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def feature_protocol(protocol):
    """Compare the feature definition, independently of cache precision/splits."""
    required = ('model', 'block_index', 'hidden_size', 'tap', 'normalization', 'feature_template_options')
    missing = [key for key in required if key not in protocol]
    if missing:
        raise ValueError(f'Feature protocol is missing {missing}')
    if protocol['normalization'] != 'raw':
        raise ValueError('Expected raw features; normalization belongs to the probe')
    result = {key: protocol[key] for key in required}
    result['model'] = str(Path(result['model']).expanduser().resolve())
    result.update({key: protocol.get(key) for key in ('model_config', 'chat_template', 'compute_dtype')})
    return result


def _cache_records(value):
    if not isinstance(value, dict) or not isinstance(value.get('protocol'), dict):
        raise ValueError('Feature file must contain a protocol')
    if 'records' in value:
        if value.get('format') != 1 or not isinstance(value['records'], list):
            raise ValueError('Invalid feature shard format')
        return value['records']
    return [value] if value.get('stage') == 'complete' else []


def _record_meta(record):
    """Adapt standalone OOD samples and main-library records to one contract."""
    if not isinstance(record, dict):
        raise ValueError('Feature record must be an object')
    identity = record.get('id')
    if not isinstance(identity, str) or not identity:
        raise ValueError('Feature record needs a stable ID')
    original = record.get('input') or {}
    modality = record.get('modality') or ('image' if original.get('image') else 'text')
    split = record.get('split', 'ood')
    if modality not in ('text', 'image') or split not in (*SPLITS, 'ood'):
        raise ValueError(f'Invalid cached modality/split: {identity}')
    if 'labels' in record:
        # Detach small metadata from the potentially very large mmap storage.
        labels = torch.as_tensor(record['labels'], dtype=torch.float32).clone()
        mask = torch.as_tensor(record['label_mask'], dtype=torch.bool).clone()
    else:
        label = record.get('label')
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f'Invalid OOD binary label: {identity}')
        labels = torch.zeros(len(HEADS), dtype=torch.float32)
        mask = torch.zeros(len(HEADS), dtype=torch.bool)
        labels[-1], mask[-1] = label, True
    if (labels.shape != (len(HEADS),) or mask.shape != labels.shape
            or not mask[-1] or not torch.isfinite(labels[mask]).all()
            or not ((labels[mask] == 0) | (labels[mask] == 1)).all()):
        raise ValueError(f'Invalid cached targets: {identity}')
    return {'id': identity, 'prompt_id': record.get('prompt_id', identity), 'split': split,
            'modality': modality, 'labels': labels, 'label_mask': mask}


class CacheDataset(Dataset):
    """Index shard metadata via mmap; load only the requested shard's tensors."""

    def __init__(self, root, split=None, modality='all'):
        self.root = Path(root).expanduser().resolve()
        if split not in (None, *SPLITS, 'ood') or modality not in ('all', 'text', 'image'):
            raise ValueError('Invalid cache split/modality')
        files = [self.root] if self.root.is_file() else sorted(self.root.rglob('*.pt'))
        if not files:
            raise ValueError(f'No feature .pt files in {self.root}')
        self.protocol = None
        self.rows, self._index = [], []
        self.shard_groups = []
        self.failures, self.incomplete = 0, 0
        failed_ids, filtered_ids, failure_events = set(), set(), 0
        identities, prompt_splits = set(), {}
        for path in files:
            value = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            records = _cache_records(value)
            protocol = value['protocol']
            if self.protocol is None:
                self.protocol = protocol
            elif feature_protocol(protocol) != feature_protocol(self.protocol):
                raise ValueError(f'Incompatible feature protocols: {path}')
            feature_protocol(protocol)
            # A main cache must not mix different selections/partitions.
            if protocol.get('selection') != self.protocol.get('selection'):
                raise ValueError(f'Mixed database selection/splits in cache: {path}')
            for failure in value.get('failures', []):
                target = filtered_ids if failure.get('status') == 'filtered' else failed_ids
                target.add(failure['id'])
                failure_events += 1
            self.incomplete += int('records' not in value and value.get('stage') != 'complete')
            group = []
            for position, record in enumerate(records):
                meta = _record_meta(record)
                if meta['id'] in identities:
                    raise ValueError(f'Duplicate cached response ID: {meta["id"]}')
                identities.add(meta['id'])
                old_split = prompt_splits.setdefault(meta['prompt_id'], meta['split'])
                if old_split != meta['split']:
                    raise ValueError(f'Prompt crosses cached splits: {meta["prompt_id"]}')
                if ((split is not None and meta['split'] != split)
                        or (modality != 'all' and meta['modality'] != modality)):
                    continue
                group.append(len(self._index))
                self._index.append((path, position))
                self.rows.append(meta)
            if group:
                self.shard_groups.append(group)
            del value, records
        self._loaded_path, self._loaded_records = None, None
        # A later successful resume supersedes a historical failure record.
        self.failures = len(failed_ids - identities)
        self.summary = {'responses': len(self.rows), 'shards': len(self.shard_groups),
                        'splits': dict(Counter(r['split'] for r in self.rows)),
                        'modalities': dict(Counter(r['modality'] for r in self.rows)),
                        'failed_records': self.failures, 'filtered_records': len(filtered_ids - identities),
                        'incomplete_records': self.incomplete, 'failure_events': failure_events}

    def __len__(self):
        return len(self._index)

    def __getitem__(self, index):
        path, position = self._index[index]
        if path != self._loaded_path:
            value = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            if (feature_protocol(value['protocol']) != feature_protocol(self.protocol)
                    or value['protocol'].get('selection') != self.protocol.get('selection')):
                raise ValueError(f'Feature shard protocol changed after indexing: {path}')
            self._loaded_records, self._loaded_path = _cache_records(value), path
        if position >= len(self._loaded_records):
            raise ValueError(f'Feature shard records changed after indexing: {path}')
        record = self._loaded_records[position]
        actual, expected = _record_meta(record), self.rows[index]
        if (any(actual[key] != expected[key] for key in ('id', 'prompt_id', 'split', 'modality'))
                or not torch.equal(actual['label_mask'], expected['label_mask'])
                or not torch.equal(actual['labels'].masked_fill(~actual['label_mask'], 0),
                                   expected['labels'].masked_fill(~expected['label_mask'], 0))):
            raise ValueError(f'Feature record identity/labels changed after indexing: {path}')
        hidden, attention, response = (record.get(k) for k in
                                        ('hidden_states', 'attention_mask', 'response_mask'))
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 2
                or hidden.dtype not in (torch.bfloat16, torch.float32)
                or hidden.shape[1] != self.protocol['hidden_size']
                or not isinstance(attention, torch.Tensor) or attention.dtype != torch.bool
                or not isinstance(response, torch.Tensor) or response.dtype != torch.bool
                or attention.shape != hidden.shape[:1] or response.shape != attention.shape
                or not response.any() or (response & ~attention).any()
                or not torch.isfinite(hidden).all()):
            raise ValueError(f'Invalid feature tensor/masks: {self.rows[index]["id"]}')
        return {**self.rows[index], 'hidden_states': hidden,
                'attention_mask': attention, 'response_mask': response}

    def __getstate__(self):
        return {**self.__dict__, '_loaded_path': None, '_loaded_records': None}


class ShardSampler(Sampler):
    """Shuffle shards and rows each epoch, retaining disk locality within a shard."""

    def __init__(self, dataset, seed=42):
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        groups = list(range(len(self.dataset.shard_groups)))
        rng.shuffle(groups)
        for number in groups:
            indices = list(self.dataset.shard_groups[number])
            rng.shuffle(indices)
            yield from indices


def collate(samples):
    if not samples:
        raise ValueError('Cannot collate an empty batch')
    from torch.nn.utils.rnn import pad_sequence

    dtype = samples[0]['hidden_states'].dtype
    for sample in samples[1:]:
        dtype = torch.promote_types(dtype, sample['hidden_states'].dtype)
    return {
        'ids': [r['id'] for r in samples],
        'prompt_ids': [r['prompt_id'] for r in samples],
        'modalities': [r['modality'] for r in samples],
        'splits': [r['split'] for r in samples],
        'hidden_states': pad_sequence([r['hidden_states'].to(dtype) for r in samples], batch_first=True),
        'attention_mask': pad_sequence([r['attention_mask'] for r in samples], batch_first=True),
        'response_mask': pad_sequence([r['response_mask'] for r in samples], batch_first=True),
        'labels': torch.stack([r['labels'] for r in samples]),
        'label_mask': torch.stack([r['label_mask'] for r in samples]),
    }
