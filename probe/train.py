"""Train a response safety probe from cached, prompt-grouped hidden states."""
import argparse
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

from dataset import HEADS, CacheDataset, ShardSampler, atomic_save, collate, feature_protocol
from model import Probe


def weighted_loss(logits, local_mask, labels, label_mask, temperature=0.1, beta_tau=0.35):
    """SingProbeWeighted: FP32 BCE, detached confidence weights, equal valid-head weight."""
    z = logits.float()
    valid = label_mask.bool() & local_mask.any(1, keepdim=True)
    mask = local_mask[:, :, None] & label_mask[:, None, :]
    if not valid.any():
        raise ValueError('A training batch has no labeled response positions')
    z = torch.where(mask, z, 0.0)
    probability = z.sigmoid().detach()
    count = mask.sum((0, 1)).clamp_min(1)
    mean = (probability * mask).sum((0, 1)) / count
    variance = ((probability - mean).square() * mask).sum((0, 1)) / count
    std = torch.where(variance > 0, variance.clamp_min(1e-20).sqrt(), variance * 0)
    beta = (std / beta_tau).clamp(0, 1)
    weights = (probability * beta / temperature).masked_fill(~mask, -1e30).softmax(1) * mask
    target = torch.where(label_mask, labels.float(), 0.0)[:, None, :].expand_as(z)
    bce = F.binary_cross_entropy_with_logits(z, target, reduction='none')
    per_example = (weights * bce).sum(1)
    head_counts = valid.sum(0)
    per_head = (per_example * valid).sum(0) / head_counts.clamp_min(1)
    return (per_head * (head_counts > 0)).sum() / (head_counts > 0).sum()


def move_batch(batch, device):
    result = {key: batch[key].to(device) for key in
              ('hidden_states', 'attention_mask', 'response_mask', 'labels', 'label_mask')}
    labels, mask = result['labels'], result['label_mask']
    if labels.shape != mask.shape or labels.shape[-1] != len(HEADS):
        raise ValueError('Cached labels do not match the probe heads')
    known = labels[mask]
    if not torch.isfinite(known).all() or not ((known == 0) | (known == 1)).all():
        raise ValueError('Known labels must be binary and finite')
    return result


def predict(probe, samples, device, batch_size=8, workers=0):
    """One maximum response-position score per sample/head; retain unknown-label masks."""
    outputs = {key: [] for key in ('labels', 'label_mask', 'scores', 'ids', 'prompt_ids', 'modalities')}
    loader = DataLoader(samples, batch_size=batch_size, num_workers=workers, collate_fn=collate,
                        generator=torch.Generator().manual_seed(0))
    probe.eval()
    with torch.inference_mode():
        for batch in loader:
            data = move_batch(batch, device)
            logits, local_mask = probe(data['hidden_states'], data['attention_mask'], data['response_mask'])
            if not local_mask.any(1).all():
                raise ValueError('A cached sample has no scored response positions')
            scores = logits.float().masked_fill(~local_mask[:, :, None], -torch.inf).amax(1).sigmoid()
            if not torch.isfinite(scores).all():
                raise ValueError('Probe produced nonfinite scores')
            outputs['scores'].append(scores.cpu().numpy())
            outputs['labels'].append(data['labels'].float().cpu().numpy())
            outputs['label_mask'].append(data['label_mask'].cpu().numpy())
            for key in ('ids', 'prompt_ids', 'modalities'):
                outputs[key].extend(batch[key])
    for key in ('scores', 'labels', 'label_mask'):
        dtype = bool if key == 'label_mask' else np.float64
        outputs[key] = (np.concatenate(outputs[key]).astype(dtype)
                        if outputs[key] else np.empty((0, len(HEADS)), dtype=dtype))
    return outputs


def calibrate(predictions, target_fpr=0.05):
    """Use only validation negatives; ties are handled conservatively with >= readout."""
    thresholds = {}
    for index, head in enumerate(HEADS):
        known = predictions['label_mask'][:, index]
        negative = known & (predictions['labels'][:, index] == 0)
        scores = np.sort(predictions['scores'][negative, index])[::-1]
        if not len(scores):
            thresholds[head] = None
            continue
        allowed = int(math.floor(target_fpr * len(scores)))
        thresholds[head] = float(np.nextafter(scores[allowed], np.inf))
    return thresholds


def metrics(predictions, thresholds=None, select=None, heads=None):
    """Report only observed labels; an uncalibrated head has no threshold-based metrics."""
    if select is None:
        select = np.ones(len(predictions['labels']), dtype=bool)
    result = {}
    for index, head in enumerate(HEADS):
        if heads is not None and head not in heads:
            continue
        known = select & predictions['label_mask'][:, index]
        labels = predictions['labels'][known, index].astype(bool)
        scores = predictions['scores'][known, index]
        positives, negatives = int(labels.sum()), int((~labels).sum())
        threshold = None if thresholds is None else thresholds[head]
        row = {'n': len(labels), 'positives': positives, 'negatives': negatives,
               'threshold': threshold, 'recall': None, 'fpr': None, 'f1': None,
               'ap': float(average_precision_score(labels, scores)) if positives else None,
               'auroc': float(roc_auc_score(labels, scores)) if positives and negatives else None}
        if threshold is not None and len(labels):
            alarms = scores >= threshold
            tp, fp = int((alarms & labels).sum()), int((alarms & ~labels).sum())
            row.update(recall=tp / positives if positives else None,
                       fpr=fp / negatives if negatives else None,
                       f1=2 * tp / (int(alarms.sum()) + positives) if alarms.any() or positives else 0.0)
        result[head] = row
    return result


def check_protocol(expected, actual):
    expected, actual = feature_protocol(expected), feature_protocol(actual)
    different = [key for key in expected.keys() | actual.keys() if expected.get(key) != actual.get(key)]
    if different:
        raise ValueError('Hidden-state feature protocol mismatch: ' + ', '.join(sorted(different)))


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('heads') != list(HEADS):
        raise ValueError('Checkpoint heads do not match dataset.HEADS')
    if checkpoint.get('architecture') != 'response_probe':
        raise ValueError('Unsupported checkpoint architecture')
    return checkpoint


def initialize_probe(input_dim, ffn, context_window, risk_window, seed):
    """FFN comparisons start from exactly the same context/risk parameters."""
    torch.manual_seed(seed)
    settings = dict(input_dim=input_dim, num_heads=len(HEADS),
                    context_window=context_window, risk_window=risk_window)
    baseline = Probe(**settings, ffn=0)
    if not ffn:
        return baseline
    probe = Probe(**settings, ffn=ffn)
    missing, unexpected = probe.load_state_dict(baseline.state_dict(), strict=False)
    if unexpected or not missing or any(not key.startswith(('context_ffn.', 'context_ffn_norm.')) for key in missing):
        raise ValueError('FFN probe does not preserve baseline parameter names/shapes')
    return probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path(__file__).parent / 'activations/id')
    parser.add_argument('--output', type=Path, required=True, help='Best checkpoint; also saves <stem>.last.pt')
    parser.add_argument('--ffn', type=int, default=0, help='Context FFN width: 0 disables it; use 1024 to enable')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--context-window', type=int, default=0)
    parser.add_argument('--risk-window', type=int, default=40)
    parser.add_argument('--modality', choices=('all', 'text', 'image'), default='all')
    parser.add_argument('--target-fpr', type=float, default=0.05)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--save-every-updates', type=int, default=100)
    parser.add_argument('--resume', action='store_true', help='Continue the corresponding .last.pt checkpoint')
    args = parser.parse_args()
    if (min(args.epochs, args.batch_size, args.accumulation, args.save_every_updates, args.risk_window) < 1
            or min(args.ffn, args.context_window, args.workers) < 0 or not math.isfinite(args.lr)
            or args.lr <= 0 or not 0 < args.target_fpr < 1):
        parser.error('Invalid training parameters')
    cache, output = args.cache.expanduser().resolve(), args.output.expanduser().resolve()
    last = output.with_name(output.stem + '.last' + output.suffix)
    if not args.resume and (output.exists() or last.exists()):
        parser.error('Checkpoint exists; use --resume or choose another --output')
    if args.resume and not last.is_file():
        parser.error(f'No resumable checkpoint: {last}')
    train = CacheDataset(cache, split='train', modality=args.modality)
    validation = CacheDataset(cache, split='val', modality=args.modality)
    if not len(train) or not len(validation):
        raise ValueError('Training and validation caches must both contain complete samples')
    check_protocol(train.protocol, validation.protocol)
    validation_labels = [int(row['labels'][-1]) for row in validation.rows if row['label_mask'][-1]]
    if not validation_labels or set(validation_labels) != {0, 1}:
        raise ValueError('Validation overall_unsafe needs both positive and negative labeled examples')
    feature = feature_protocol(train.protocol)
    settings = {'cache': str(cache), 'protocol': train.protocol, 'modality': args.modality,
                'train_samples': len(train), 'validation_samples': len(validation),
                'ffn': args.ffn, 'epochs': args.epochs, 'batch_size': args.batch_size,
                'accumulation': args.accumulation, 'lr': args.lr, 'seed': args.seed,
                'context_window': args.context_window, 'risk_window': args.risk_window,
                'target_fpr': args.target_fpr, 'loss': 'SingProbeWeighted',
                'temperature': 0.1, 'beta_tau': 0.35, 'detach_weights': True,
                'weight_decay': 0.01, 'gradient_clip': 1.0, 'sampler': 'shard_shuffle'}
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    probe = initialize_probe(feature['hidden_size'], args.ffn, args.context_window, args.risk_window, args.seed).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=0.01)
    batches = math.ceil(len(train) / args.batch_size)
    total_updates = math.ceil(batches / args.accumulation) * args.epochs
    warmup = max(1, math.ceil(total_updates * 0.05))

    def lr_scale(step):
        if step < warmup:
            return step / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, total_updates - warmup))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    progress = {'epoch': 0, 'next_batch': 0, 'updates': 0, 'loss_sum': 0.0, 'loss_steps': 0}
    history, best_ap = [], None
    if args.resume:
        checkpoint = load_checkpoint(last)
        check_protocol(checkpoint['protocol'], train.protocol)
        if checkpoint['training'] != settings or checkpoint['model_config'] != probe.get_config():
            raise ValueError('Resume configuration/cache differs from the saved training run')
        probe.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        progress, history, best_ap = checkpoint['progress'], checkpoint['history'], checkpoint['best_ap']
        random.setstate(checkpoint['rng']['python'])
        np.random.set_state(checkpoint['rng']['numpy'])
        torch.set_rng_state(checkpoint['rng']['torch'])
        if device.type == 'cuda' and checkpoint['rng']['cuda'] is not None:
            torch.cuda.set_rng_state(checkpoint['rng']['cuda'], device)

    def save(path, thresholds=None, validation_metrics=None):
        rng = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
               'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}
        atomic_save({'architecture': 'response_probe', 'heads': list(HEADS),
                     'model_config': probe.get_config(), 'protocol': train.protocol,
                     'training': settings, 'state_dict': probe.state_dict(),
                     'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                     'progress': dict(progress), 'history': history, 'best_ap': best_ap, 'rng': rng,
                     'thresholds': thresholds, 'validation': validation_metrics,
                     'calibration': {'split': 'val', 'target_fpr': args.target_fpr,
                                     'comparison': '>=', 'samples': len(validation)} if thresholds is not None else None}, path)

    print(f'train={len(train)} validation={len(validation)} device={device} ffn={args.ffn}', flush=True)
    started = time.monotonic()
    sampler = ShardSampler(train, seed=args.seed)
    for epoch in range(progress['epoch'], args.epochs):
        sampler.set_epoch(epoch)
        order = list(sampler)
        batch_order = [order[start:start + args.batch_size] for start in range(0, len(order), args.batch_size)]
        start_batch = progress['next_batch'] if epoch == progress['epoch'] else 0
        loader = DataLoader(train, batch_sampler=batch_order[start_batch:], num_workers=args.workers,
                            collate_fn=collate, generator=torch.Generator().manual_seed(args.seed + epoch))
        probe.train()
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        for batch_index, batch in enumerate(loader, start_batch):
            data = move_batch(batch, device)
            logits, local_mask = probe(data['hidden_states'], data['attention_mask'], data['response_mask'])
            loss = weighted_loss(logits, local_mask, data['labels'], data['label_mask'])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            (loss / args.accumulation).backward()
            pending += 1
            progress['loss_sum'] += float(loss.detach())
            progress['loss_steps'] += 1
            if (batch_index + 1) % args.accumulation == 0 or batch_index + 1 == batches:
                if pending < args.accumulation:
                    for parameter in probe.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(args.accumulation / pending)
                torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                progress.update(epoch=epoch, next_batch=batch_index + 1, updates=progress['updates'] + 1)
                if progress['updates'] % args.save_every_updates == 0:
                    save(last)
            if (batch_index + 1) % 20 == 0 or batch_index + 1 == batches:
                mean_loss = progress['loss_sum'] / max(1, progress['loss_steps'])
                print(f'epoch={epoch + 1}/{args.epochs} batch={batch_index + 1}/{batches} '
                      f'updates={progress["updates"]} loss={mean_loss:.5f}', flush=True)
        predictions = predict(probe, validation, device, args.batch_size, args.workers)
        thresholds = calibrate(predictions, args.target_fpr)
        validation_metrics = metrics(predictions, thresholds)
        overall = validation_metrics['overall_unsafe']
        if overall['ap'] is None or overall['threshold'] is None:
            raise ValueError('Validation overall_unsafe needs both positive and negative labeled examples')
        history.append({'epoch': epoch + 1, 'train_loss': progress['loss_sum'] / max(1, progress['loss_steps']),
                        'validation': validation_metrics})
        improved = best_ap is None or overall['ap'] > best_ap
        if improved:
            best_ap = overall['ap']
        progress.update(epoch=epoch + 1, next_batch=0, loss_sum=0.0, loss_steps=0)
        save(last, thresholds, validation_metrics)
        if improved:
            save(output, thresholds, validation_metrics)
        print(f'validation epoch={epoch + 1} overall AP={overall["ap"]:.4f} '
              f'recall={overall["recall"]:.4f} FPR={overall["fpr"]:.4f} best_AP={best_ap:.4f}', flush=True)
    print(f'Training complete in {(time.monotonic() - started) / 60:.1f} min; best={output}; last={last}', flush=True)


if __name__ == '__main__':
    main()
