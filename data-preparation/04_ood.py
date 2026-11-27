#!/usr/bin/env python3
"""Build independent OOD activations: official WildGuardTest and MM-SafetyBench.

Each .pt file contains one sample, its label, and full-context block activations.
Only stage='complete' files are evaluation inputs. MM answers are saved before
judging, so an explicit --resume can recover API failures without regenerating.
The main SQLite database and previous experiment caches are never read or written.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
from pathlib import Path
import sys
import tempfile

from common import chat, config


SCENARIOS = (
    '01-Illegal_Activitiy', '02-HateSpeech', '03-Malware_Generation',
    '04-Physical_Harm', '05-EconomicHarm', '06-Fraud', '07-Sex',
    '08-Political_Lobbying', '09-Privacy_Violence', '10-Legal_Opinion',
    '11-Financial_Advice', '12-Health_Consultation', '13-Gov_Decision',
)
SPECIAL_TEMPLATES = {
    '08-Political_Lobbying': 'political_lobbing', '10-Legal_Opinion': 'legal',
    '11-Financial_Advice': 'financial', '12-Health_Consultation': 'health',
    '13-Gov_Decision': 'gov_decision',
}


def required_text(value, description):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'Empty or invalid {description}')
    return value.strip()


def read_wildguard(source):
    import pyarrow.parquet as pq

    files = sorted(source.glob('*.parquet')) if source.is_dir() else [source]
    if not files:
        raise FileNotFoundError(f'No WildGuardTest parquet files in {source}')
    rows, missing = [], 0
    for path in files:
        for index, value in enumerate(pq.read_table(path).to_pylist()):
            label = value.get('response_harm_label')
            if label is None:
                missing += 1
                continue
            if label not in ('harmful', 'unharmful'):
                raise ValueError(f'Unknown response_harm_label {label!r}: {path.name}:{index}')
            rows.append({
                'id': f'{path.stem}-{index:06d}', 'dataset': 'wildguardtest',
                'prompt': required_text(value.get('prompt'), f'prompt {path.name}:{index}'),
                'response': required_text(value.get('response'), f'response {path.name}:{index}'),
                'image': None, 'label': int(label == 'harmful'),
                'label_source': 'official:response_harm_label',
            })
    logging.info('WildGuardTest: %d labeled rows; %d missing official response labels skipped',
                 len(rows), missing)
    if not rows:
        raise ValueError('WildGuardTest contains no labeled responses')
    return rows


def read_mmsafetybench(source):
    rows = []
    for scenario in SCENARIOS:
        path = source / 'data/processed_questions' / f'{scenario}.json'
        values = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(values, dict) or not values:
            raise ValueError(f'Invalid MM-SafetyBench questions: {path}')
        for question_id, value in values.items():
            if not question_id.isdecimal() or not isinstance(value, dict):
                raise ValueError(f'Invalid question ID or record in {path}')
            for variant in ('SD', 'SD_TYPO', 'TYPO'):
                field = 'Rephrased Question(SD)' if variant == 'SD' else 'Rephrased Question'
                prompt = required_text(value.get(field), f'{scenario}/{question_id}/{field}')
                image = source / 'data/imgs/MM-SafetyBench(imgs)' / scenario / variant / f'{question_id}.jpg'
                if not image.is_file() or image.stat().st_size == 0:
                    raise FileNotFoundError(f'Missing MM-SafetyBench image: {image}')
                rows.append({
                    'id': f'{scenario}-{variant}-{question_id}', 'dataset': 'mmsafetybench',
                    'scenario': scenario, 'variant': variant, 'question_id': question_id,
                    'prompt': prompt, 'image': str(image.resolve()),
                    'label_source': 'MM-SafetyBench official evaluation.py via configured judge',
                })
    if len(rows) != 5040:
        raise ValueError(f'Expected all 5,040 MM-SafetyBench pairs, found {len(rows)}')
    logging.info('MM-SafetyBench: %d pairs, all 13 scenarios and 3 image variants', len(rows))
    return rows


def official_templates(source):
    """Read constant strings without executing the benchmark's evaluation script."""
    prefix = 'gpt_query_template_safety'
    names = {prefix, *(prefix + '_' + suffix for suffix in SPECIAL_TEMPLATES.values())}
    found = {}
    tree = ast.parse((source / 'evaluation.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, str) or '{prediction}' not in value:
                        raise ValueError(f'Invalid official template {target.id}')
                    found[target.id] = value
    if set(found) != names:
        raise ValueError(f'Missing official MM-SafetyBench templates: {sorted(names - found.keys())}')
    return {
        scenario: found[prefix + ('_' + SPECIAL_TEMPLATES[scenario]
                                 if scenario in SPECIAL_TEMPLATES else '')]
        for scenario in SCENARIOS
    }


def judge_answer(row, response, settings, templates):
    # The official evaluator supplies only the answer, using scenario-specific
    # rules. In particular, legal/financial/health disclaimers are intentional.
    prompt = templates[row['scenario']].format(prediction=response)
    result = chat(settings, [{'role': 'user', 'content': prompt}], json_output=False)
    if not isinstance(result, str) or result.strip().lower() not in ('safe', 'unsafe'):
        raise ValueError('Official MM judge must return exactly safe or unsafe')
    return {'label': int(result.strip().lower() == 'unsafe'), 'judge_output': result,
            'judge_model': settings['model']}


class Backbone:
    """One local model; batch generation followed by memory-bounded feature passes."""

    def __init__(self, model_path, device, block_index, cache_dtype='bfloat16'):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.device = torch.device(device)
        self.cache_dtype = getattr(torch, cache_dtype)
        self.activation_dtype = None
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        self.processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            str(model_path), local_files_only=True,
            dtype=torch.bfloat16 if self.device.type == 'cuda' else torch.float32,
            device_map={'': str(self.device)},
        ).eval()
        self.base = getattr(self.model, 'model', self.model)
        language = getattr(self.base, 'language_model', self.base)
        if not hasattr(language, 'layers') or not 0 <= block_index < len(language.layers):
            raise ValueError(f'Language block {block_index} does not exist in this model')
        self.block = language.layers[block_index]
        text_config = getattr(self.model.config, 'text_config', self.model.config)
        self.context_length = getattr(text_config, 'max_position_embeddings', None)
        if not isinstance(self.context_length, int) or self.context_length <= 0:
            raise ValueError('Model does not declare its supported context length')

    @staticmethod
    def messages(row, response=None):
        from PIL import Image

        content = []
        if row.get('image'):
            with Image.open(row['image']) as image:
                content.append({'type': 'image', 'image': image.convert('RGB')})
        content.append({'type': 'text', 'text': row['prompt']})
        result = [{'role': 'user', 'content': content}]
        if response is not None:
            result.append({'role': 'assistant', 'content': response, 'reasoning_content': ''})
        return result

    def check_inputs(self, inputs, output_budget=0):
        ids, mask = inputs.get('input_ids'), inputs.get('attention_mask')
        if ids is None or mask is None or ids.ndim != 2 or mask.shape != ids.shape:
            raise ValueError('Processor input IDs and attention mask do not align')
        if ids.shape[1] + output_budget > self.context_length:
            raise ValueError(f'Input length {ids.shape[1]} + output budget {output_budget} '
                             f'exceeds model context {self.context_length}; input was not truncated')

    def generate(self, rows, max_new_tokens):
        import torch

        tokenizer = self.processor.tokenizer
        padding_side = tokenizer.padding_side
        tokenizer.padding_side = 'left'
        try:
            inputs = self.processor.apply_chat_template(
                [self.messages(row) for row in rows], tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors='pt',
                processor_kwargs={'padding': True, 'truncation': False},
            ).to(self.device)
        finally:
            tokenizer.padding_side = padding_side
        self.check_inputs(inputs, max_new_tokens)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        tokens = generated[:, inputs['input_ids'].shape[1]:]
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = tokenizer.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        answers = []
        for values in tokens.tolist():
            stop = next((i for i, token in enumerate(values) if token in eos), len(values))
            response = tokenizer.decode(values[:stop], skip_special_tokens=True).strip()
            answers.append({'response': response, 'generation': {
                'tokens': stop, 'finish_reason': 'stop' if stop < len(values) else 'length',
            }})
        return answers

    def response_inputs(self, row, response):
        """Align the answer body against offsets after visual token expansion."""
        import torch

        body = required_text(response, f'response {row["id"]}')
        marker = '__OOD_RESPONSE_BODY_52f2bdbeb5874a2a__'
        if marker in body or marker in row['prompt']:
            raise ValueError('Response boundary marker collision')
        options = {'add_generation_prompt': False, 'enable_thinking': False}
        marked = self.processor.apply_chat_template(self.messages(row, marker), tokenize=False, **options)
        if marked.count(marker) != 1:
            raise ValueError('Chat template did not preserve the answer marker')
        prefix, suffix = marked.split(marker)
        rendered = self.processor.apply_chat_template(self.messages(row, body), tokenize=False, **options)
        if rendered != prefix + body + suffix:
            raise ValueError('Chat template changed the answer body')
        inputs = self.processor.apply_chat_template(
            self.messages(row, body), tokenize=True, return_dict=True, return_tensors='pt',
            processor_kwargs={'return_offsets_mapping': True, 'truncation': False}, **options,
        )
        self.check_inputs(inputs)
        offsets = inputs.pop('offset_mapping', None)
        if offsets is None or offsets.shape != (*inputs['input_ids'].shape, 2):
            raise ValueError('Processor did not provide token offsets for response alignment')
        end = int(offsets[0, :, 1].max()) - len(suffix)
        start = end - len(body)
        response_mask = (offsets[:, :, 1] > start) & (offsets[:, :, 0] < end)
        ids = inputs['input_ids'][0, response_mask[0]]
        decoded = self.processor.tokenizer.decode(ids, skip_special_tokens=False)
        if (start < 0 or not response_mask.any() or decoded.strip() != body
                or not torch.all(inputs['attention_mask'][response_mask] == 1)):
            raise ValueError(f'Response token alignment failed for {row["id"]}')
        return inputs.to(self.device), response_mask.to(self.device)

    def features(self, row, response):
        import torch

        inputs, response_mask = self.response_inputs(row, response)
        captured = []

        def capture(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            if self.activation_dtype is None:
                self.activation_dtype = str(value.dtype).removeprefix('torch.')
                logging.info('Raw block output dtype=%s; cache dtype=%s', self.activation_dtype, self.cache_dtype)
                if value.dtype == torch.float32 and self.cache_dtype == torch.bfloat16:
                    logging.warning('FP32 block output will be rounded to BF16 for storage')
            captured.append(value.detach().to(device='cpu', dtype=self.cache_dtype))

        handle = self.block.register_forward_hook(capture)
        try:
            with torch.inference_mode():
                # Calling the multimodal base omits the very large vocabulary
                # projection. Teacher forcing includes the final response token.
                self.base(**inputs, output_hidden_states=False, use_cache=False, return_dict=True)
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].ndim != 3 or captured[0].shape[:2] != response_mask.shape:
            raise ValueError('Language block output does not match the encoded tokens')
        if not torch.isfinite(captured[0]).all():
            raise ValueError('Language block produced non-finite activations')
        attention = inputs['attention_mask'].to(device='cpu', dtype=torch.bool)[0]
        response_mask = response_mask.to(device='cpu', dtype=torch.bool)[0]
        if not response_mask.any() or (response_mask & ~attention).any():
            raise ValueError('Invalid context/response masks')
        return {'hidden_states': captured[0][0], 'attention_mask': attention,
                'response_mask': response_mask, 'activation_dtype': self.activation_dtype}


def save_sample(path, state):
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name, delete=False) as output:
            temporary = Path(output.name)
            torch.save(state, output)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sample_input(row):
    value = dict(row)
    if row.get('image'):
        value['image_sha256'] = hashlib.sha256(Path(row['image']).read_bytes()).hexdigest()
    return value


def load_sample(path, row, protocol, resume):
    import torch

    source = sample_input(row)
    if not path.exists():
        state = {'id': row['id'], 'dataset': row['dataset'], 'stage': 'pending',
                 'input': source, 'protocol': protocol}
        if row['dataset'] == 'wildguardtest':
            state.update(response=row['response'], label=row['label'], stage='judged')
        return state
    if not resume:
        raise FileExistsError(f'{path} exists; choose an empty output or explicitly use --resume')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('input') != source or state.get('protocol') != protocol:
        raise ValueError(f'Cannot resume changed input/model/judge/feature protocol: {path}')
    if state.get('id') != row['id'] or state.get('stage') not in ('pending', 'generated', 'judged', 'complete'):
        raise ValueError(f'Invalid cached sample identity/stage: {path}')
    if state['stage'] != 'pending':
        required_text(state.get('response'), f'cached response {path}')
    if state['stage'] in ('judged', 'complete') and (type(state.get('label')) is not int or state['label'] not in (0, 1)):
        raise ValueError(f'Invalid cached label: {path}')
    if row['dataset'] == 'wildguardtest' and (state.get('response') != row['response'] or state.get('label') != row['label']):
        raise ValueError(f'Cached WildGuard response/label differs from the official source: {path}')
    if state['stage'] == 'complete':
        hidden, attention, response = (state.get(key) for key in ('hidden_states', 'attention_mask', 'response_mask'))
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 2
                or hidden.dtype != getattr(torch, protocol['cache_dtype'])
                or not isinstance(attention, torch.Tensor) or attention.dtype != torch.bool
                or not isinstance(response, torch.Tensor) or response.dtype != torch.bool
                or attention.shape != hidden.shape[:1] or response.shape != attention.shape
                or not response.any() or (response & ~attention).any()):
            raise ValueError(f'Invalid cached activations/masks: {path}')
    return state


def protocol_for(settings, dataset, templates):
    model = Path(settings['model'])
    model_config = json.loads((model / 'config.json').read_text(encoding='utf-8'))
    text_config = model_config.get('text_config', model_config)
    protocol = {
        'format': 1, 'model': str(model), 'block_index': settings['block_index'],
        'cache_dtype': settings.get('cache_dtype', 'bfloat16'),
        'compute_dtype': 'float32' if str(settings.get('device', 'cuda:0')).startswith('cpu') else 'bfloat16',
        'hidden_size': text_config['hidden_size'], 'hidden_layout': 'tokens,hidden',
        'tap': 'block_output_pre_final_norm', 'normalization': 'raw',
        'feature_mode': 'full-context teacher-forced raw decoder block output; response-body mask',
        'feature_template_options': {'enable_thinking': False, 'reasoning_content': ''},
        'model_config': model_config,
        'chat_template': (model / 'chat_template.jinja').read_text(encoding='utf-8'),
    }
    if dataset == 'mmsafetybench':
        protocol['generation'] = {'do_sample': False, 'max_new_tokens': settings['max_new_tokens'],
                                  'template_options': 'native defaults'}
        protocol['judge'] = {key: settings['judge'].get(key) for key in
                             ('base_url', 'model', 'temperature', 'max_tokens', 'extra_body')}
        protocol['judge']['official_templates'] = templates
    else:
        protocol['label'] = 'WildGuardTest official response_harm_label: unharmful=0,harmful=1'
    return protocol


def run_dataset(rows, settings, templates, resume, backbone=None):
    dataset = rows[0]['dataset']
    protocol = protocol_for(settings, dataset, templates)
    root = Path(settings['output']) / dataset
    if not resume and root.exists() and any(root.glob('*.pt')):
        raise FileExistsError(f'{root} is not empty; use an empty output or explicitly use --resume')
    done, failed = 0, 0
    with ThreadPoolExecutor(max_workers=settings['judge_workers']) as judges:
        for offset in range(0, len(rows), settings['batch_size']):
            batch = rows[offset:offset + settings['batch_size']]
            entries = []
            for row in batch:
                path = root / f'{row["id"]}.pt'
                state = load_sample(path, row, protocol, resume)
                if state['stage'] == 'complete':
                    done += 1
                else:
                    entries.append((row, path, state))
            if entries and backbone is None:
                backbone = Backbone(settings['model'], settings['device'], settings['block_index'],
                                    settings.get('cache_dtype', 'bfloat16'))
            generation = [entry for entry in entries if entry[2]['stage'] == 'pending']
            if generation:
                try:
                    answers = backbone.generate([entry[0] for entry in generation], settings['max_new_tokens'])
                    if len(answers) != len(generation):
                        raise ValueError('Generation batch returned the wrong number of answers')
                    for (row, path, state), answer in zip(generation, answers):
                        state.update(answer, stage='generated')
                        if not isinstance(state.get('response'), str) or not state['response'].strip():
                            state.update(stage='pending', error='Empty generated response')
                            logging.error('%s: %s', row['id'], state['error'])
                        save_sample(path, state)
                except Exception as exc:
                    for row, path, state in generation:
                        state['error'] = f'{type(exc).__name__}: {exc}'
                        save_sample(path, state)
                        logging.error('%s generation failed: %s', row['id'], state['error'])
            futures = {}
            for row, path, state in entries:
                if state['stage'] == 'generated':
                    future = judges.submit(judge_answer, row, state['response'], settings['judge'], templates)
                    futures[future] = (row, path, state)
            for future in as_completed(futures):
                row, path, state = futures[future]
                try:
                    state.update(future.result(), stage='judged')
                    state.pop('error', None)
                    save_sample(path, state)
                except Exception as exc:
                    state['error'] = f'{type(exc).__name__}: {exc}'
                    save_sample(path, state)
                    logging.error('%s judge failed: %s', row['id'], state['error'])
            for row, path, state in entries:
                if state['stage'] == 'judged':
                    try:
                        state.update(backbone.features(row, state['response']), stage='complete')
                        state.pop('error', None)
                        save_sample(path, state)
                    except Exception as exc:
                        state['error'] = f'{type(exc).__name__}: {exc}'
                        save_sample(path, state)
                        logging.error('%s features failed: %s', row['id'], state['error'])
                done += int(state['stage'] == 'complete')
                failed += int(state['stage'] != 'complete')
            logging.info('%s %d/%d visited; complete=%d failed=%d',
                         dataset, min(offset + len(batch), len(rows)), len(rows), done, failed)
    return failed, backbone


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--dataset', choices=('wildguardtest', 'mmsafetybench', 'all'), default='all')
    parser.add_argument('--source', type=Path, help='Source override; requires one specific --dataset')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--device', help='Single device for the independent OOD model, e.g. cuda:0')
    parser.add_argument('--block-index', type=int,
                        help='Zero-based decoder block index; configured default 14 matches hidden_states[15]')
    parser.add_argument('--cache-dtype', choices=('bfloat16', 'float32'))
    parser.add_argument('--batch-size', type=int, help='Generation batch size; feature passes use one sample')
    parser.add_argument('--max-new-tokens', type=int, help='Generation output limit, never an input truncation limit')
    parser.add_argument('--limit', type=int, help='Explicit small test: first N rows of each selected dataset')
    parser.add_argument('--resume', action='store_true', help='Resume only matching samples from this script')
    parser.add_argument('--validate-source', action='store_true', help='Check sources/images/templates without loading torch or a model')
    args = parser.parse_args(argv)
    if args.source is not None and args.dataset == 'all':
        parser.error('--source requires --dataset wildguardtest or mmsafetybench')
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    return args


def main(argv=None):
    args = arguments(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        settings = config(args.config).get('ood', {})
        if not settings:
            raise ValueError('Missing config.ood settings')
        base = args.config.expanduser().resolve().parent
        for key in ('output', 'model'):
            value = getattr(args, key)
            settings[key] = str((value if value is not None else base / settings[key]).expanduser().resolve())
        for key in ('device', 'block_index', 'batch_size', 'max_new_tokens', 'cache_dtype'):
            if getattr(args, key) is not None:
                settings[key] = getattr(args, key)
        for key in ('batch_size', 'max_new_tokens', 'judge_workers'):
            if type(settings.get(key)) is not int or settings[key] <= 0:
                raise ValueError(f'ood.{key} must be a positive integer')
        if type(settings.get('block_index')) is not int or settings['block_index'] < 0:
            raise ValueError('ood.block_index must be a nonnegative integer')
        if settings.setdefault('cache_dtype', 'bfloat16') not in ('bfloat16', 'float32'):
            raise ValueError('ood.cache_dtype must be bfloat16 or float32')
        datasets = ('wildguardtest', 'mmsafetybench') if args.dataset == 'all' else (args.dataset,)
        prepared = []
        for dataset in datasets:
            source = (args.source if args.source is not None else base / settings['sources'][dataset]).expanduser().resolve()
            rows = read_wildguard(source) if dataset == 'wildguardtest' else read_mmsafetybench(source)
            if len({row['id'] for row in rows}) != len(rows):
                raise ValueError(f'Duplicate sample IDs in {dataset}')
            templates = official_templates(source) if dataset == 'mmsafetybench' else {}
            if args.validate_source:
                from PIL import Image
                for row in rows:
                    if row.get('image'):
                        with Image.open(row['image']) as image:
                            image.verify()
                logging.info('%s source validation passed: %d eligible samples', dataset, len(rows))
            prepared.append((rows[:args.limit] if args.limit else rows, templates))
        if args.validate_source:
            return 0
        failed, backbone = 0, None
        for rows, templates in prepared:
            count, backbone = run_dataset(rows, settings, templates, args.resume, backbone)
            failed += count
        return int(failed > 0)
    except (ValueError, KeyError, FileNotFoundError, FileExistsError, OSError, RuntimeError) as exc:
        logging.error('%s: %s', type(exc).__name__, exc)
        return 1


if __name__ == '__main__':
    sys.exit(main())
