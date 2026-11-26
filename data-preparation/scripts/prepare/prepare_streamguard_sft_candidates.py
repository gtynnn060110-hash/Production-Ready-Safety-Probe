"""Quarantine bio-related keywords and export provisional SFT candidates.

This is a high-recall lexical screen, not a biosafety classifier. Its outputs
must be reviewed before use as targets for a generative model.
"""
import sys as _prep_sys; from pathlib import Path as _PrepPath; _prep_sys.path.insert(0, str(_PrepPath(__file__).resolve().parents[2])); from scripts._bootstrap import setup as _prep_setup; _prep_setup()
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import unicodedata

import pyarrow as pa
import pyarrow.ipc as ipc

SOURCE_ROOT = Path('data/raw/streamguard')
SOURCES = ('wildguard', 's_eval')
REVISION = '7fa1203b2d9be5671d3b40f2c1df0035deeb098c'
# Deliberately overinclusive, including medicine and computer-virus references.
BIO = re.compile(
    r'bio|virol|virus|viral|virion|bacter|microb|pathogen|germ|infect|contag|'
    r'epidem|pandem|outbreak|disease|illness|plague|vacci|immun|medic|clinical|'
    r'anthrax|smallpox|orthopox|poxvirus|botulin|ricin|toxin|poison|venom|'
    r'genet|genom|\bdna\b|\brna\b|gene[ -]?edit|cell[ -]?cult|cultivat|'
    r'ferment|fung|spore|laborator|aerosol|contaminat|synthesi[sz]|sequen[cs]|'
    r'epidemi|enfermedad|infecci|bact[eé]ri|maladie|toxine|laboratoire|'
    r'生物|生化|病毒|细菌|細菌|病原|病菌|疾病|疫苗|疫病|疫情|传染|傳染|感染|'
    r'流感|瘟疫|炭疽|天花|毒素|毒液|中毒|投毒|基因|遗传|遺傳|核酸|实验室|實驗室|'
    r'微生物|培养|培養|发酵|發酵|气溶胶|氣溶膠|医药|醫藥|药物|藥物|医疗|醫療',
    re.IGNORECASE,
)


def norm(text):
    return unicodedata.normalize('NFKC', text).strip()


def pid(text):
    return hashlib.sha256(norm(text).encode('utf-8')).hexdigest()


def read_source(source):
    directory = SOURCE_ROOT / source / 'qwen3_8b' / 'train'
    paths = sorted(directory.glob('*.arrow'))
    if not paths:
        raise FileNotFoundError(f'No Arrow files in {directory}')
    for path in paths:
        row_number = 0
        with pa.memory_map(str(path), 'r') as stream:
            for batch in ipc.open_stream(stream):
                names = batch.schema.names
                fields = ['prompt', 'response', 'label'] + (['reason'] if 'reason' in names else [])
                for row in batch.select(fields).to_pylist():
                    row_number += 1
                    yield path, row_number, row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        p.error(f'output directory already exists: {output}')
    quarantine_prompts = set()
    raw = Counter()
    hits = Counter()
    for source in SOURCES:
        for path, row_number, row in read_source(source):
            prompt, response, label = row.get('prompt'), row.get('response'), row.get('label')
            if not isinstance(prompt, str) or not prompt.strip() or not isinstance(response, str) or not response.strip() or label not in (0, 1):
                raise ValueError(f'Invalid row: {path}:{row_number}')
            raw[(source, label)] += 1
            text = '\n'.join(norm(str(row.get(k) or '')) for k in ('prompt', 'response', 'reason'))
            if BIO.search(text):
                quarantine_prompts.add(pid(prompt))
                hits[(source, label)] += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=output.name + '.tmp-', dir=output.parent))
    os.chmod(temp, 0o700)
    kept, excluded = Counter(), Counter()
    with (temp / 'all_candidates.jsonl').open('x', encoding='utf-8') as all_out, \
         (temp / 'unsafe_candidates.jsonl').open('x', encoding='utf-8') as unsafe_out:
        for source in SOURCES:
            for path, row_number, row in read_source(source):
                label = row['label']
                prompt_id = pid(row['prompt'])
                if prompt_id in quarantine_prompts:
                    excluded[(source, label)] += 1
                    continue
                record = {'prompt': row['prompt'], 'response': row['response'], 'label': label,
                          'source': source, 'source_path': str(path), 'source_row': row_number,
                          'prompt_id': prompt_id, 'generator': 'Qwen3-8B'}
                line = json.dumps(record, ensure_ascii=False) + '\n'
                all_out.write(line)
                if label == 1:
                    unsafe_out.write(line)
                kept[(source, label)] += 1
    for name in ('all_candidates.jsonl', 'unsafe_candidates.jsonl'):
        os.chmod(temp / name, 0o600)
    def show(counter):
        return {f'{source}:{label}': count for (source, label), count in sorted(counter.items())}
    metadata = {
        'status': 'provisional_keyword_screen_only_requires_review',
        'dataset': 'Alibaba-AAIG/StreamGuardBench', 'revision': REVISION,
        'source_splits': list(SOURCES), 'generator': 'Qwen3-8B',
        'label_meaning': {'0': 'safe', '1': 'unsafe'},
        'filter_fields': ['prompt', 'response', 'reason'],
        'filter': 'conservative keyword screen; all rows sharing a matched normalized prompt are excluded',
        'filter_pattern_sha256': hashlib.sha256(BIO.pattern.encode('utf-8')).hexdigest(),
        'raw_counts': show(raw), 'direct_keyword_matches': show(hits),
        'excluded_counts_including_prompt_group': show(excluded), 'candidate_counts': show(kept),
        'deduplication': 'none; original source rows preserved',
        'warning': 'Keyword matching does not guarantee absence of biosecurity risk; human review is required before generative training.',
    }
    (temp / 'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.chmod(temp / 'manifest.json', 0o600)
    os.rename(temp, output)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print('Candidates:', output)


if __name__ == '__main__':
    main()
