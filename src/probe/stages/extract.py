"""Frozen block extraction with disk budget checks and resumable raw shards."""

import logging
import os
import shutil
from dataclasses import asdict

import torch
from safetensors.torch import save_file
from tqdm.auto import tqdm

from ..artifacts import Artifacts
from ..config import Config
from ..features import BlockTap, encode_sample, get_text_model
from ..io import (
    SCHEMA,
    checked_manifest,
    file_hash,
    fingerprint,
    read_json,
    read_jsonl,
    stage_lock,
    write_json,
    write_jsonl,
)


def tokenize_splits(config, paths, prepared, tokenizer):
    """Establish real lengths/disk demand before loading weights."""
    stats, all_tokens = {}, 0
    selected = config.features.splits if config.features.splits is not None else prepared["splits"]
    missing = set(selected) - prepared["splits"].keys()
    if missing:
        raise ValueError(f"Requested feature splits are not prepared: {sorted(missing)}")
    for split in selected:
        rows, dropped, tokens, maximum = [], 0, 0, 0
        for sample in tqdm(
            read_jsonl(paths.prepared / f"{split}.jsonl"),
            total=prepared["splits"][split]["samples"],
            desc=f"tokenize {split}",
        ):
            encoded = encode_sample(tokenizer, sample["prompt"], sample["response"])
            length = len(encoded["input_ids"])
            if split == "train" and length > config.features.max_length:
                dropped += 1
                continue
            rows.append(
                {k: sample[k] for k in ("id", "labels", "label_mask")}
                | {
                    "input_ids": encoded["input_ids"],
                    "response_mask": encoded["response_mask"],
                    "body_char_span": encoded["body_char_span"],
                }
            )
            tokens += length
            maximum = max(maximum, length)
        if not rows:
            raise ValueError(f"No samples left after tokenization in {split}")
        write_jsonl(paths.features / "tokens" / f"{split}.jsonl", rows)
        stats[split] = {
            "samples": len(rows),
            "tokens": tokens,
            "max_length": maximum,
            "dropped_overlength": dropped,
        }
        all_tokens += tokens
    return stats, all_tokens


def extract_split(config, paths, split, model, tap, tokenizer, signature):
    """A crash only repeats the last uncommitted shard."""
    directory = paths.features / split
    directory.mkdir(parents=True, exist_ok=True)
    shards, processed = [], 0
    while (marker := directory / f"shard-{len(shards):05d}.manifest.json").exists():
        item = checked_manifest(marker, signature)
        if item["source_start"] != processed:
            raise ValueError(f"Non-contiguous extraction progress: {marker}")
        shards.append(item)
        processed = item["source_stop"]
    rows = list(read_jsonl(paths.features / "tokens" / f"{split}.jsonl"))
    pending, hidden, response_masks, token_ids = [], [], [], []
    cursor, shard_start = 0, processed

    def flush(stop):
        nonlocal pending, hidden, response_masks, token_ids, cursor, shard_start
        if not pending:
            return
        name = f"shard-{len(shards):05d}"
        tensor_path = directory / f"{name}.safetensors"
        index_path = directory / f"{name}.jsonl"
        temp = tensor_path.with_suffix(".safetensors.tmp")
        save_file(
            {
                "hidden_states": torch.cat(hidden).contiguous(),
                "response_mask": torch.cat(response_masks),
                "input_ids": torch.cat(token_ids),
            },
            str(temp),
        )
        os.replace(temp, tensor_path)
        write_jsonl(index_path, pending)
        item = {
            "schema": SCHEMA,
            "signature": signature,
            "index": str(index_path.relative_to(paths.features)),
            "tensor": str(tensor_path.relative_to(paths.features)),
            "samples": len(pending),
            "tokens": cursor,
            "source_start": shard_start,
            "source_stop": stop,
            "files": {
                tensor_path.name: file_hash(tensor_path),
                index_path.name: file_hash(index_path),
            },
        }
        write_json(directory / f"{name}.manifest.json", item)
        shards.append(item)
        pending, hidden, response_masks, token_ids = [], [], [], []
        cursor, shard_start = 0, stop

    batch_size = config.features.batch_size
    for start in tqdm(range(processed, len(rows), batch_size), desc=f"extract {split}"):
        batch = rows[start : start + batch_size]
        longest = max(len(r["input_ids"]) for r in batch)
        ids = torch.full(
            (len(batch), longest),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=config.features.device,
        )
        attention = torch.zeros_like(ids, dtype=torch.bool)
        for i, row in enumerate(batch):
            n = len(row["input_ids"])
            ids[i, :n] = torch.tensor(row["input_ids"], device=ids.device)
            attention[i, :n] = True
        with torch.inference_mode():
            model(input_ids=ids, attention_mask=attention, use_cache=False)
            features = tap.collect().to(device="cpu", dtype=getattr(torch, config.features.dtype))
        for i, row in enumerate(batch):
            n = len(row["input_ids"])
            pending.append(
                {
                    "id": row["id"],
                    "labels": row["labels"],
                    "label_mask": row["label_mask"],
                    "shard": f"{split}/shard-{len(shards):05d}.safetensors",
                    "start": cursor,
                    "stop": cursor + n,
                }
            )
            hidden.append(features[i, :n].clone())
            response_masks.append(torch.tensor(row["response_mask"], dtype=torch.bool))
            token_ids.append(torch.tensor(row["input_ids"], dtype=torch.int32))
            cursor += n
        if cursor >= config.features.shard_tokens:
            flush(start + len(batch))
    flush(len(rows))
    return {"samples": len(rows), "tokens": sum(s["tokens"] for s in shards), "shards": shards}


def run(config: Config, paths: Artifacts) -> None:
    from huggingface_hub import HfApi
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

    prepared = checked_manifest(paths.prepared / "manifest.json")
    with stage_lock(paths.features):
        revision = config.features.revision or HfApi().model_info(config.features.model).sha
        tokenizer = AutoTokenizer.from_pretrained(config.features.model, revision=revision)
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must provide a padding token")
        signature = {
            "features": asdict(config.features),
            "revision": revision,
            "prepared_hash": file_hash(paths.prepared / "manifest.json"),
            "template": fingerprint(tokenizer.chat_template),
            "tokenizer": fingerprint(tokenizer.backend_tokenizer.to_str()),
            "implementation": 1,
        }
        target = paths.features / "manifest.json"
        if target.exists():
            completed = checked_manifest(target, signature)
            for info in completed["splits"].values():
                for shard in info["shards"]:
                    marker = paths.features / shard["tensor"].replace(
                        ".safetensors", ".manifest.json"
                    )
                    checked_manifest(marker, signature)
            logging.info("Reusing complete features: %s", paths.features)
            return
        intent = paths.features / "intent.json"
        if intent.exists() and read_json(intent) != signature:
            raise ValueError("Partial feature signature mismatch; use a new output_dir")
        write_json(intent, signature)
        stats, tokens = tokenize_splits(config, paths, prepared, tokenizer)
        model_config = AutoConfig.from_pretrained(config.features.model, revision=revision)
        text_config = getattr(model_config, "text_config", model_config)
        hidden_size = text_config.hidden_size
        if max(config.features.block_indices) >= text_config.num_hidden_layers:
            raise ValueError("Requested block index is outside this model")
        bytes_per_token = (
            hidden_size
            * len(config.features.block_indices)
            * (2 if config.features.dtype == "bfloat16" else 4)
            + 5
        )
        estimated = tokens * bytes_per_token
        logging.info(
            "Raw cache estimate: %.2f GB for %s tokens; lengths: %s", estimated / 1e9, tokens, stats
        )
        if estimated > config.features.max_cache_gb * 1e9:
            raise ValueError(
                "Feature estimate exceeds features.max_cache_gb; use a pilot subset "
                "or explicitly increase the disk budget"
            )
        existing_bytes = sum(p.stat().st_size for p in paths.features.rglob("*.safetensors"))
        if max(0, estimated - existing_bytes) * 1.1 > shutil.disk_usage(paths.features).free:
            raise ValueError("Insufficient free disk for cache plus safety margin")
        if max(s["max_length"] for s in stats.values()) > text_config.max_position_embeddings:
            raise ValueError("A complete evaluation sequence exceeds model context capacity")
        model = (
            AutoModelForImageTextToText.from_pretrained(
                config.features.model,
                revision=revision,
                dtype=getattr(torch, config.features.dtype),
                device_map={"": config.features.device},
                attn_implementation="sdpa",
            )
            .eval()
            .requires_grad_(False)
        )
        text_model = get_text_model(model)
        tap = BlockTap(text_model, config.features.block_indices)
        try:
            splits = {
                name: extract_split(config, paths, name, text_model, tap, tokenizer, signature)
                | {"length_stats": stats[name]}
                for name in stats
            }
        finally:
            tap.close()
        write_json(
            target,
            {
                "schema": SCHEMA,
                "signature": signature,
                "hidden_size": hidden_size,
                "input_dim": hidden_size * len(config.features.block_indices),
                "cache_is_raw": True,
                "splits": splits,
                "files": {
                    f"tokens/{name}.jsonl": file_hash(paths.features / "tokens" / f"{name}.jsonl")
                    for name in splits
                },
            },
        )
        logging.info("Feature extraction complete: %s", target)
