"""Chat-body alignment, selected raw block taps, and variable-length batches."""

from dataclasses import fields
from pathlib import Path

import torch
from safetensors import safe_open
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .contracts import FeatureBatch
from .io import read_jsonl


def encode_sample(tokenizer, prompt: str, response: str) -> dict:
    """Render boundaries, then jointly tokenize; never concatenate separately encoded text."""
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with offset_mapping is required")
    body = response.strip()
    if not body:
        raise ValueError("Response body is empty")
    marker = "__PROBE_BODY_72862c705b374f89__"
    if marker in prompt or marker in body:
        raise ValueError("Response boundary marker collision")

    def render(content):
        # Do not let the template parse </think> out of original response text.
        return tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": content, "reasoning_content": ""},
            ],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

    marked = render(marker)
    if marked.count(marker) != 1:
        raise ValueError("Chat template does not preserve the response marker")
    prefix, suffix = marked.split(marker)
    rendered = render(body)
    if rendered != prefix + body + suffix:
        raise ValueError("Chat template rewrote response; inspect alignment")
    start, end = len(prefix), len(prefix) + len(body)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    mask = []
    for token, (left, right) in zip(encoded["input_ids"], encoded["offset_mapping"], strict=True):
        overlap = right > start and left < end and right > left
        if overlap:
            outside = rendered[left : min(right, start)] + rendered[max(left, end) : right]
            if outside.strip() or token in tokenizer.all_special_ids:
                raise ValueError("Response token overlaps template/special text; inspect alignment")
        mask.append(overlap)
    if not any(mask):
        raise ValueError("No response tokens after alignment")
    return {
        "input_ids": encoded["input_ids"],
        "response_mask": mask,
        "body_char_span": [start, end],
        "rendered": rendered,
    }


class BlockTap:
    def __init__(self, text_model, indices):
        if max(indices) >= len(text_model.layers):
            raise ValueError("Requested block index is outside this model")
        self.values, self.handles = {}, []
        for index in indices:

            def hook(module, args, output, index=index):
                self.values[index] = (output[0] if isinstance(output, tuple) else output).detach()

            self.handles.append(text_model.layers[index].register_forward_hook(hook))
        self.indices = indices

    def collect(self):
        result = torch.cat([self.values[i] for i in self.indices], dim=-1)
        self.values.clear()
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.values.clear()


def get_text_model(model):
    base = getattr(model, "model", model)
    text = getattr(base, "language_model", base)
    if not hasattr(text, "layers"):
        raise ValueError("Text backbone must expose .layers; add an explicit adapter")
    return text


class FeatureDataset(Dataset):
    """Metadata in RAM, tensor slices mmap'd on demand."""

    def __init__(self, root: Path, manifest: dict, split: str):
        if split not in manifest["splits"]:
            raise ValueError(f"Feature split is unavailable: {split}")
        self.root, self.rows = root, []
        for shard in manifest["splits"][split]["shards"]:
            self.rows.extend(read_jsonl(root / shard["index"]))
        if not self.rows:
            raise ValueError(f"Empty feature split: {split}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with safe_open(str(self.root / row["shard"]), framework="pt", device="cpu") as stream:
            start, stop = row["start"], row["stop"]
            hidden = stream.get_slice("hidden_states")[start:stop]
            mask = stream.get_slice("response_mask")[start:stop].bool()
        return {
            "id": row["id"],
            "hidden_states": hidden,
            "response_mask": mask,
            "labels": torch.tensor(row["labels"], dtype=torch.float32),
            "label_mask": torch.tensor(row["label_mask"], dtype=torch.bool),
        }


def collate_features(rows) -> FeatureBatch:
    hidden = pad_sequence([r["hidden_states"] for r in rows], batch_first=True)
    lengths = torch.tensor([r["hidden_states"].shape[0] for r in rows])
    return FeatureBatch(
        ids=[r["id"] for r in rows],
        hidden_states=hidden,
        attention_mask=torch.arange(hidden.shape[1])[None] < lengths[:, None],
        response_mask=pad_sequence([r["response_mask"] for r in rows], batch_first=True),
        labels=torch.stack([r["labels"] for r in rows]),
        label_mask=torch.stack([r["label_mask"] for r in rows]),
    )


def move_batch(batch: FeatureBatch, device, feature_config, hidden_size: int) -> FeatureBatch:
    values = {f.name: getattr(batch, f.name) for f in fields(batch)}
    values = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in values.items()}
    x = values["hidden_states"].float()
    if feature_config.normalize == "rms_no_affine":
        shape = x.shape
        x = x.reshape(*shape[:-1], -1, hidden_size)
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + feature_config.norm_eps)
        x = x.reshape(shape)
    values["hidden_states"] = x
    return FeatureBatch(**values)
