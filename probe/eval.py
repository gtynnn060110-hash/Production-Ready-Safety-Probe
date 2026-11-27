"""Evaluate cached ID/OOD responses with thresholds fixed on training validation data."""
import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from dataset import CacheDataset
from model import Probe
from train import check_protocol, load_checkpoint, metrics, predict


def print_table(name, rows):
    def percentage(value):
        return '--' if value is None else f'{100 * value:.2f}'
    print(f'\n{name}', flush=True)
    print(f'{"head":<16} {"n":>7} {"positive":>8} {"negative":>8} '
          f'{"Recall%":>8} {"FPR%":>8} {"AP%":>8} {"F1%":>8} {"AUROC%":>8}')
    for head, row in rows.items():
        values = ' '.join(f'{percentage(row[key]):>8}' for key in ('recall', 'fpr', 'ap', 'f1', 'auroc'))
        print(f'{head:<16} {row["n"]:>7} {row["positives"]:>8} {row["negatives"]:>8} {values}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path(__file__).parent / 'activations/id')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--ood', type=Path, nargs='*', default=[], help='One activation cache directory per OOD dataset')
    parser.add_argument('--output', type=Path, required=True, help='Metrics JSON path')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error('batch-size must be positive and workers must be nonnegative')
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    thresholds, calibration = checkpoint.get('thresholds'), checkpoint.get('calibration')
    if thresholds is None or not calibration or calibration.get('split') != 'val':
        raise ValueError('Checkpoint has no validation calibration; use a best or completed-epoch checkpoint')
    device = torch.device(args.device)
    probe = Probe(**checkpoint['model_config']).to(device)
    probe.load_state_dict(checkpoint['state_dict'])
    test = CacheDataset(args.cache, split='test')
    if not len(test):
        raise ValueError('ID cache contains no complete test samples')
    check_protocol(checkpoint['protocol'], test.protocol)
    selection = checkpoint['protocol'].get('selection')
    if selection is None or selection != test.protocol.get('selection'):
        raise ValueError('ID cache database selection/prompt split differs from the training cache')
    print(f'ID cache: {test.summary}', flush=True)
    predictions = predict(probe, test, device, args.batch_size, args.workers)
    result = {'checkpoint': str(checkpoint_path), 'cache': str(args.cache.expanduser().resolve()),
              'protocol': checkpoint['protocol'], 'calibration': calibration, 'thresholds': thresholds,
              'cache_summary': test.summary,
              'id': {}, 'ood': {}}
    modalities = np.asarray(predictions['modalities'])
    for modality in ('all', 'text', 'image'):
        select = np.ones(len(modalities), dtype=bool) if modality == 'all' else modalities == modality
        rows = metrics(predictions, thresholds, select=select)
        result['id'][modality] = {'samples': int(select.sum()), 'heads': rows}
        print_table(f'ID {modality}', rows)
    for path in args.ood:
        path = path.expanduser().resolve()
        name = path.name
        if name in result['ood']:
            raise ValueError(f'OOD cache names must be unique: {name}')
        samples = CacheDataset(path)
        if not len(samples):
            raise ValueError(f'OOD cache contains no complete samples: {path}')
        check_protocol(checkpoint['protocol'], samples.protocol)
        print(f'OOD {name} cache: {samples.summary}', flush=True)
        scores = predict(probe, samples, device, args.batch_size, args.workers)
        rows = metrics(scores, thresholds, heads={'overall_unsafe'})
        result['ood'][name] = {'cache': str(path), 'samples': len(samples),
                               'cache_summary': samples.summary, 'heads': rows}
        print_table(f'OOD {name}', rows)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output.parent,
                                         prefix='.' + output.name, suffix='.tmp', delete=False) as stream:
            temp = Path(stream.name)
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temp, output)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    print(f'\nSaved metrics: {output}', flush=True)


if __name__ == '__main__':
    main()
