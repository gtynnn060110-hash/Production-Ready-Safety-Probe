"""Cache labeled prompt/response activations, with one full backbone per GPU."""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import gc
import hashlib
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import time


class ContextOverflow(ValueError):
    pass


def protocol(model, block_index, cache_dtype, device):
    model = Path(model).expanduser().resolve()
    configuration = json.loads((model / 'config.json').read_text(encoding='utf-8'))
    text = configuration.get('text_config', configuration)
    if not 0 <= block_index < text['num_hidden_layers']:
        raise ValueError(f'Invalid zero-based block index: {block_index}')
    return {
        'model': str(model), 'block_index': block_index, 'hidden_size': text['hidden_size'],
        'cache_dtype': cache_dtype, 'tap': 'block_output_pre_final_norm', 'normalization': 'raw',
        'feature_template_options': {'enable_thinking': False, 'reasoning_content': ''},
        'compute_dtype': 'float32' if str(device).startswith('cpu') else 'bfloat16',
        'model_config': configuration,
        'chat_template': (model / 'chat_template.jinja').read_text(encoding='utf-8'),
    }


class Backbone:
    def __init__(self, settings, device):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.device = torch.device(device)
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        self.processor = AutoProcessor.from_pretrained(settings['model'], local_files_only=True)
        model = AutoModelForMultimodalLM.from_pretrained(
            settings['model'], local_files_only=True,
            dtype=torch.bfloat16 if self.device.type == 'cuda' else torch.float32,
            device_map={'': str(self.device)},
        ).eval()
        model.requires_grad_(False)
        self.model = getattr(model, 'model', model)
        language = getattr(self.model, 'language_model', self.model)
        self.block = language.layers[settings['block_index']]
        text = getattr(model.config, 'text_config', model.config)
        self.context_length = text.max_position_embeddings
        self.cache_dtype = getattr(torch, settings['cache_dtype'])
        self.observed_dtype = None

    @staticmethod
    def messages(row, response):
        from PIL import Image

        result = []
        context = row.get('context', [])
        if not isinstance(context, list):
            raise ValueError('context must be a list')
        for turn in context:
            if not isinstance(turn, dict) or turn.get('role') not in ('user', 'assistant'):
                raise ValueError('context must contain user/assistant messages')
            item = {'role': turn['role'], 'content': turn['content']}
            if item['role'] == 'assistant':
                item['reasoning_content'] = ''
            result.append(item)
        content = []
        for path in row.get('images', []):
            with Image.open(path) as image:
                content.append({'type': 'image', 'image': image.convert('RGB')})
        if row['prompt']:
            content.append({'type': 'text', 'text': row['prompt']})
        if not content:
            raise ValueError('No prompt text or images')
        result.append({'role': 'user', 'content': content})
        result.append({'role': 'assistant', 'content': response, 'reasoning_content': ''})
        return result

    def encode(self, rows):
        import torch

        marker = '__PROBE_RESPONSE_BODY_52f2bdbeb5874a2a__'
        options = {'add_generation_prompt': False, 'enable_thinking': False}
        conversations, suffixes, bodies = [], [], []
        for row in rows:
            body = row['response']
            if not isinstance(body, str) or not body.strip():
                raise ValueError(f'Empty response: {row["id"]}')
            body = body.strip()
            if marker in json.dumps([row['prompt'], row.get('context', []), body], ensure_ascii=False):
                raise ValueError('Response boundary marker collision')
            messages = self.messages(row, body)
            marked_messages = [*messages[:-1], {**messages[-1], 'content': marker}]
            marked = self.processor.apply_chat_template(marked_messages, tokenize=False, **options)
            if marked.count(marker) != 1:
                raise ValueError('Chat template did not preserve the answer marker')
            prefix, suffix = marked.split(marker)
            rendered = self.processor.apply_chat_template(messages, tokenize=False, **options)
            if rendered != prefix + body + suffix:
                raise ValueError('Chat template changed the answer body')
            conversations.append(messages)
            suffixes.append(suffix)
            bodies.append(body)
        tokenizer = self.processor.tokenizer
        previous_padding = tokenizer.padding_side
        tokenizer.padding_side = 'right'
        try:
            inputs = self.processor.apply_chat_template(
                conversations, tokenize=True, return_dict=True, return_tensors='pt',
                processor_kwargs={'padding': True, 'truncation': False, 'return_offsets_mapping': True},
                **options,
            )
        finally:
            tokenizer.padding_side = previous_padding
        ids, attention = inputs.get('input_ids'), inputs.get('attention_mask')
        offsets = inputs.pop('offset_mapping', None)
        if (ids is None or attention is None or ids.ndim != 2 or attention.shape != ids.shape
                or offsets is None or offsets.shape != (*ids.shape, 2)):
            raise ValueError('Processor input IDs, attention mask, and token offsets do not align')
        if ids.shape[1] > self.context_length:
            raise ContextOverflow(f'Encoded length {ids.shape[1]} exceeds model context '
                                  f'{self.context_length}; no input truncation was applied')
        masks = []
        for index, (body, suffix) in enumerate(zip(bodies, suffixes)):
            end = int(offsets[index, :, 1].max()) - len(suffix)
            start = end - len(body)
            mask = (offsets[index, :, 1] > start) & (offsets[index, :, 0] < end)
            decoded = tokenizer.decode(ids[index, mask], skip_special_tokens=False)
            if start < 0 or not mask.any() or decoded.strip() != body or not torch.all(attention[index, mask] == 1):
                raise ValueError(f'Response token alignment failed: {rows[index]["id"]}')
            masks.append(mask)
        return inputs.to(self.device), torch.stack(masks).to(self.device)

    def features(self, rows):
        import torch

        inputs, response = self.encode(rows)
        captured = []

        def capture(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            actual = str(value.dtype).removeprefix('torch.')
            if self.observed_dtype is None:
                logging.info('Raw block output dtype=%s; cache dtype=%s', actual, self.cache_dtype)
                if value.dtype == torch.float32 and self.cache_dtype == torch.bfloat16:
                    logging.warning('FP32 block output will be rounded to BF16 for storage')
                self.observed_dtype = actual
            captured.append(value.detach().to(device='cpu', dtype=self.cache_dtype))

        handle = self.block.register_forward_hook(capture)
        try:
            with torch.inference_mode():
                # Skip the vocabulary projection and do not retain every layer.
                self.model(**inputs, use_cache=False, output_hidden_states=False, return_dict=True)
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].ndim != 3 or captured[0].shape[:2] != response.shape:
            raise ValueError('Language block output does not align with encoded tokens')
        if not torch.isfinite(captured[0]).all():
            raise ValueError('Non-finite block activations')
        attention = inputs['attention_mask'].to(device='cpu', dtype=torch.bool)
        response = response.to(device='cpu', dtype=torch.bool)
        results = []
        for index in range(len(rows)):
            valid = attention[index]
            selected = response[index, valid]
            if not selected.any() or (response[index] & ~valid).any():
                raise ValueError('Invalid response/context mask')
            # Remove padding before storage, while preserving every context token.
            results.append({
                'hidden_states': captured[0][index, valid].contiguous(),
                'attention_mask': valid[valid].clone(), 'response_mask': selected.clone(),
                'activation_dtype': self.observed_dtype,
            })
        return results


def input_digest(row, image_digests):
    images = []
    for value in row.get('images', []):
        path = Path(value).expanduser().resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in image_digests:
            image_digests[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        images.append([str(path), image_digests[key]])
    value = {key: row[key] for key in ('id', 'prompt_id', 'prompt', 'response', 'split', 'modality', 'labels', 'label_mask')}
    value.update(context=row.get('context', []), images=images)
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def cached_records(output, expected, selection, resume):
    import torch

    completed = {}
    files = sorted(Path(output).glob('*.pt'))
    if files and not resume:
        raise FileExistsError('Output already contains .pt files; choose an empty output or use --resume')
    for path in files:
        shard = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if shard.get('format') != 1 or shard.get('protocol') != {**expected, 'selection': selection}:
            raise ValueError(f'Cannot resume changed model/feature/dataset settings: {path}')
        for row in shard['records']:
            if row['id'] in completed:
                raise ValueError(f'Duplicate completed response ID in existing shards: {row["id"]}')
            hidden, attention, response = row['hidden_states'], row['attention_mask'], row['response_mask']
            if (hidden.ndim != 2 or hidden.shape[1] != expected['hidden_size']
                    or hidden.dtype != getattr(torch, expected['cache_dtype'])
                    or attention.shape != hidden.shape[:1] or response.shape != attention.shape
                    or attention.dtype != torch.bool or response.dtype != torch.bool
                    or not isinstance(row.get('input_digest'), str)):
                raise ValueError(f'Invalid cached features or input identity: {path}')
            completed[row['id']] = row['input_digest']
    return completed


class ShardWriter:
    def __init__(self, output, rank, expected, selection, token_limit):
        self.output, self.rank = Path(output), rank
        self.protocol = {**expected, 'selection': selection}
        self.token_limit = token_limit
        previous = list(self.output.glob(f'part-{rank:03d}-*.pt'))
        self.index = 1 + max((int(p.stem.rsplit('-', 1)[1]) for p in previous), default=-1)
        self.records, self.failures, self.tokens = [], [], 0

    def add(self, record):
        self.records.append(record)
        self.tokens += record['hidden_states'].shape[0]
        if self.tokens >= self.token_limit:
            self.flush()

    def failed(self, row, status, error):
        self.failures.append({'id': row['id'], 'prompt_id': row['prompt_id'], 'split': row['split'],
                              'modality': row['modality'], 'status': status, 'error': str(error)})
        if len(self.failures) >= 100:
            self.flush()

    def flush(self):
        if not self.records and not self.failures:
            return
        from dataset import atomic_save

        path = self.output / f'part-{self.rank:03d}-{self.index:06d}.pt'
        atomic_save({'format': 1, 'protocol': self.protocol,
                     'records': self.records, 'failures': self.failures}, path)
        self.index += 1
        self.records, self.failures, self.tokens = [], [], 0


def extract_batch(rows, extractor, writer, counts):
    """Retry smaller batches; every single-row failure remains visible in the cache."""
    import torch

    try:
        features = extractor.features(rows)
        if len(features) != len(rows):
            raise ValueError('Feature batch returned the wrong number of samples')
    except Exception as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            logging.warning('CUDA OOM for batch size %d; retrying smaller batches', len(rows))
            counts['batch_size_limit'] = min(counts.get('batch_size_limit', len(rows)), max(1, len(rows) // 2))
        if len(rows) > 1:
            # Do not retain traceback references to failed GPU tensors during retry.
            exc.__traceback__ = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            midpoint = len(rows) // 2
            extract_batch(rows[:midpoint], extractor, writer, counts)
            extract_batch(rows[midpoint:], extractor, writer, counts)
        else:
            status = 'filtered' if isinstance(exc, ContextOverflow) else 'failed'
            counts[status] += 1
            writer.failed(rows[0], status, f'{type(exc).__name__}: {exc}')
            logging.error('%s %s: %s', rows[0]['id'], status, exc)
        return
    for row, values in zip(rows, features):
        record = {key: row[key] for key in
                  ('id', 'prompt_id', 'split', 'modality', 'labels', 'label_mask', 'input_digest')}
        record.update(values)
        writer.add(record)
        counts['completed'] += 1
        counts['tokens'] += values['hidden_states'].shape[0]


def run_rank(rank, world_size, device, args, expected, completed, dataset=None, extractor=None):
    from dataset import SQLiteDataset

    logging.basicConfig(level=logging.INFO, format=f'%(asctime)s rank={rank} %(levelname)s %(message)s')
    dataset = dataset if dataset is not None else SQLiteDataset(
        args.db, modality=args.modality, sources=args.sources, max_prompts=args.max_prompts,
        seed=args.seed, ratios=tuple(args.ratios),
    )
    writer = ShardWriter(args.output, rank, expected, dataset.selection, args.shard_tokens)
    counts, images, pending = Counter(), {}, []
    started = time.monotonic()
    try:
        for index in range(rank, len(dataset), world_size):
            metadata = dataset.rows[index]
            try:
                row = dict(dataset[index])
                row['input_digest'] = input_digest(row, images)
            except Exception as exc:
                if metadata['id'] in completed:
                    raise ValueError(f'Previously cached input is no longer readable: {metadata["id"]}') from exc
                writer.failed(metadata, 'failed', f'{type(exc).__name__}: {exc}')
                counts['failed'] += 1
                logging.error('%s input failed: %s', metadata['id'], exc)
                continue
            if row['id'] in completed:
                if completed[row['id']] != row['input_digest']:
                    raise ValueError(f'Previously cached input/label changed: {row["id"]}')
                counts['resumed'] += 1
                continue
            pending.append(row)
            if len(pending) >= min(args.batch_size, counts.get('batch_size_limit', args.batch_size)):
                if extractor is None:
                    extractor = Backbone(expected, device)
                extract_batch(pending, extractor, writer, counts)
                pending = []
                elapsed = time.monotonic() - started
                logging.info('completed=%d resumed=%d filtered=%d failed=%d tokens=%d elapsed=%.1fs',
                             counts['completed'], counts['resumed'], counts['filtered'], counts['failed'],
                             counts['tokens'], elapsed)
        if pending:
            if extractor is None:
                extractor = Backbone(expected, device)
            extract_batch(pending, extractor, writer, counts)
    finally:
        writer.flush()
        dataset.close()
    logging.info('Finished: %s; elapsed=%.1fs', dict(counts), time.monotonic() - started)
    if counts['failed']:
        raise RuntimeError(f'{counts["failed"]} samples failed; details are stored in the shards')
    return counts


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument('--db', type=Path, default=root.parent / 'data-preparation/data/database/probe.sqlite')
    parser.add_argument('--output', type=Path, default=root / 'activations/id')
    parser.add_argument('--model', type=Path, default=Path('/workspace/models/Qwen3.5-9B'))
    parser.add_argument('--block-index', type=int, default=14,
                        help='Zero-based decoder block index; default 14 matches hidden_states[15]')
    parser.add_argument('--cache-dtype', choices=('bfloat16', 'float32'), default='bfloat16')
    parser.add_argument('--device', default='cuda:0', help='Single device, or cpu; --gpus overrides this')
    parser.add_argument('--gpus', help='Comma-separated visible GPU indices; one complete model per GPU')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--shard-tokens', type=int, default=32768)
    parser.add_argument('--modality', choices=('all', 'image', 'text'), default='all')
    parser.add_argument('--sources', nargs='+')
    parser.add_argument('--max-prompts', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ratios', nargs=3, type=float, default=(0.8, 0.1, 0.1), metavar=('TRAIN', 'VAL', 'TEST'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.shard_tokens <= 0:
        parser.error('batch-size and shard-tokens must be positive')
    if args.max_prompts is not None and args.max_prompts <= 0:
        parser.error('max-prompts must be positive')
    for key in ('db', 'model', 'output'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    return args


def main(argv=None):
    args = arguments(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        from dataset import SQLiteDataset

        if args.gpus is not None:
            gpu_ids = [int(value) for value in args.gpus.split(',')]
            if not gpu_ids or min(gpu_ids) < 0 or len(set(gpu_ids)) != len(gpu_ids):
                raise ValueError('--gpus must contain distinct nonnegative GPU indices')
            devices = [f'cuda:{value}' for value in gpu_ids]
        else:
            devices = [args.device]
        expected = protocol(args.model, args.block_index, args.cache_dtype, devices[0])
        dataset = SQLiteDataset(args.db, modality=args.modality, sources=args.sources,
                                max_prompts=args.max_prompts, seed=args.seed, ratios=tuple(args.ratios))
        if not len(dataset):
            raise ValueError('No labeled responses match the requested filters')
        args.output.mkdir(parents=True, exist_ok=True)
        # Lock the directory itself, without introducing a lock/manifest file.
        lock = os.open(args.output, os.O_RDONLY)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            completed = cached_records(args.output, expected, dataset.selection, args.resume)
            eligible = {row['id'] for row in dataset.rows}
            if not completed.keys() <= eligible:
                raise ValueError('Cached response IDs are absent from the current dataset selection')
            logging.info('Selected %d responses; cached=%d; devices=%s', len(dataset), len(completed), devices)
            if len(devices) == 1:
                run_rank(0, 1, devices[0], args, expected, completed, dataset=dataset)
            else:
                processes = []
                context = mp.get_context('spawn')
                try:
                    for rank, device in enumerate(devices):
                        process = context.Process(target=run_rank,
                            args=(rank, len(devices), device, args, expected, completed, dataset))
                        process.start()
                        processes.append(process)
                    for process in processes:
                        process.join()
                except BaseException:
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                    for process in processes:
                        process.join()
                    raise
                if any(process.exitcode != 0 for process in processes):
                    raise RuntimeError('One or more GPU workers failed; completed shards were preserved')
        finally:
            os.close(lock)
            dataset.close()
        return 0
    except Exception as exc:
        logging.error('%s: %s', type(exc).__name__, exc)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
