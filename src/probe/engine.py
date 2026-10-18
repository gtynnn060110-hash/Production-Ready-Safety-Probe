"""Shared prediction and checkpoint helpers, not a training framework."""

import os
import random
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .contracts import HEADS
from .features import FeatureDataset, collate_features, move_batch
from .io import checked_manifest, file_hash
from .metrics import readout


def setup(seed, device):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This runner is single-process; DDP beta aggregation is not implemented")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; set device=cpu for small checks")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.enable_cudnn_sdp(False)


def autocast(config):
    if config.training.dtype == "float32":
        return nullcontext()
    return torch.autocast(torch.device(config.training.device).type, dtype=torch.bfloat16)


def feature_manifest(config, paths):
    prepared = checked_manifest(paths.prepared / "manifest.json")
    manifest = checked_manifest(paths.features / "manifest.json")
    signature = manifest["signature"]
    if signature["features"] != asdict(config.features):
        raise ValueError("Feature manifest does not match requested extraction settings")
    if signature["prepared_hash"] != file_hash(paths.prepared / "manifest.json"):
        raise ValueError("Prepared data changed since extraction")
    if prepared["signature"]["data"] != asdict(config.data):
        raise ValueError("Prepared data does not match requested data configuration")
    for info in manifest["splits"].values():
        for shard in info["shards"]:
            marker = paths.features / shard["tensor"].replace(".safetensors", ".manifest.json")
            checked_manifest(marker, signature)
    return manifest


def run_signature(config, paths):
    return {
        "feature_manifest_hash": file_hash(paths.features / "manifest.json"),
        "probe": asdict(config.probe),
        "loss": asdict(config.loss),
        "training": asdict(config.training) | {"resume": True},
        "seed": config.seed,
        "heads": list(HEADS),
        "implementation": 1,
    }


def loader(config, paths, manifest, split, train=False, epoch=0):
    dataset = FeatureDataset(paths.features, manifest, split)
    batch_size = config.training.micro_batch_size if train else config.evaluation.batch_size
    if train and len(dataset) < batch_size:
        raise ValueError("Training split is smaller than the fixed microbatch")
    generator = torch.Generator().manual_seed(config.seed + epoch)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        collate_fn=collate_features,
        num_workers=config.training.num_workers,
        generator=generator,
    )


def predict(model, batches, config, manifest, readouts):
    model.eval()
    ids, labels, masks, lengths = [], [], [], []
    scores = {name: [] for name in readouts}
    with torch.inference_mode():
        for batch in batches:
            batch = move_batch(
                batch, config.training.device, config.features, manifest["hidden_size"]
            )
            with autocast(config):
                output = model(batch)
            ids.extend(batch.ids)
            labels.append(batch.labels.cpu().numpy())
            masks.append(batch.label_mask.cpu().numpy())
            lengths.extend(batch.response_mask.sum(1).cpu().tolist())
            for name in readouts:
                score = readout(output, name)
                if not torch.isfinite(score).all():
                    raise ValueError("Non-finite probe logits during prediction")
                scores[name].append(score.cpu().numpy())
    return {
        "ids": ids,
        "labels": np.concatenate(labels),
        "label_mask": np.concatenate(masks),
        "response_lengths": lengths,
        "scores": {k: np.concatenate(v).astype(np.float64) for k, v in scores.items()},
    }


def save_checkpoint(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".pt.tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def load_checkpoint(path, signature):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value["signature"] != signature:
        raise ValueError(f"Checkpoint configuration/source mismatch: {path}")
    return value
