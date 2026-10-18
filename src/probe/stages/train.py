"""Train only probe parameters; select checkpoints on validation Overall AUPRC."""

import logging
import math
from dataclasses import asdict

import torch
from sklearn.metrics import average_precision_score

from ..artifacts import Artifacts
from ..config import Config
from ..contracts import HEADS
from ..engine import (
    autocast,
    feature_manifest,
    load_checkpoint,
    loader,
    predict,
    run_signature,
    save_checkpoint,
    setup,
)
from ..features import move_batch
from ..io import SCHEMA, checked_manifest, file_hash, stage_lock, write_json
from ..losses import build_loss
from ..probes import build_probe


def train_epochs(config, paths, manifest, model, objective, signature):
    batches_per_epoch = len(loader(config, paths, manifest, "train", train=True))
    accum = config.training.gradient_accumulation_steps
    updates_per_epoch = math.ceil(batches_per_epoch / accum)
    total_updates = config.training.epochs * updates_per_epoch
    warmup = int(total_updates * config.training.warmup_ratio)

    def schedule(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total_updates - warmup)))
        return 0.5 * (1 + math.cos(math.pi * progress))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=tuple(config.training.betas),
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start_epoch, best, history = 0, -1.0, []
    latest = paths.run / "checkpoints/last.pt"
    if latest.exists():
        if not config.training.resume:
            raise ValueError(
                "Existing checkpoint; resume is disabled. Choose a new experiment name"
            )
        state = load_checkpoint(latest, signature)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        start_epoch, best, history = state["epoch"] + 1, state["best_score"], state["history"]
        logging.info("Resuming from completed epoch %s", start_epoch)
    for epoch in range(start_epoch, config.training.epochs):
        batches = loader(config, paths, manifest, "train", train=True, epoch=epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for step, batch in enumerate(batches):
            batch = move_batch(
                batch, config.training.device, config.features, manifest["hidden_size"]
            )
            group_start = (step // accum) * accum
            group_size = min(accum, len(batches) - group_start)
            with autocast(config):
                output = model(batch)
            loss = objective(output, batch.labels, batch.label_mask)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss at epoch {epoch + 1}, microbatch {step + 1}")
            (loss / group_size).backward()
            loss_sum += float(loss.detach())
            if (step + 1) % accum == 0 or step + 1 == len(batches):
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.max_grad_norm, error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (step + 1) % config.training.log_every == 0:
                logging.info(
                    "epoch=%s microbatch=%s/%s loss=%.5f lr=%.3g diagnostics=%s",
                    epoch + 1,
                    step + 1,
                    len(batches),
                    float(loss.detach()),
                    scheduler.get_last_lr()[0],
                    getattr(objective, "diagnostics", {}),
                )
        val = predict(
            model,
            loader(config, paths, manifest, "validation"),
            config,
            manifest,
            [config.training.selection_readout],
        )
        valid = val["label_mask"][:, -1]
        y = val["labels"][valid, -1]
        if len(set(y.tolist())) != 2:
            raise ValueError("Validation Overall needs both classes for checkpoint selection")
        score = float(
            average_precision_score(y, val["scores"][config.training.selection_readout][valid, -1])
        )
        improved = score > best
        best = max(best, score)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": loss_sum / len(batches),
                "validation_overall_auprc": score,
                "selection_readout": config.training.selection_readout,
                "microbatches": len(batches),
                "optimizer_updates": updates_per_epoch,
                "dropped_last_samples": len(batches.dataset) % config.training.micro_batch_size,
                "last_grad_norm": float(norm),
            }
        )
        state = {
            "signature": signature,
            "config": asdict(config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_score": best,
            "history": history,
            "torch_rng": torch.get_rng_state(),
        }
        if improved:
            save_checkpoint(paths.run / "checkpoints/best.pt", state)
        save_checkpoint(latest, state)
        write_json(paths.run / "history.json", history)
        logging.info("epoch=%s val Overall AUPRC=%.5f best=%.5f", epoch + 1, score, best)


def run(config: Config, paths: Artifacts) -> None:
    setup(config.seed, config.training.device)
    manifest = feature_manifest(config, paths)
    signature = run_signature(config, paths)
    model = build_probe(config.probe, input_dim=manifest["input_dim"], num_heads=len(HEADS))
    model = model.to(config.training.device)
    objective = build_loss(config.loss)
    with stage_lock(paths.run):
        target = paths.run / "manifest.json"
        if target.exists():
            checked_manifest(target, signature)
            logging.info("Training already complete: %s", paths.run)
            return
        # JSON is valid YAML; atomic save, no second serialization implementation.
        write_json(paths.run / "config.yaml", asdict(config))
        logging.info(
            "Training %s + %s (%s parameters), fixed microbatch=%s, accumulation=%s",
            config.probe.name,
            config.loss.name,
            sum(p.numel() for p in model.parameters()),
            config.training.micro_batch_size,
            config.training.gradient_accumulation_steps,
        )
        train_epochs(config, paths, manifest, model, objective, signature)
        write_json(
            target,
            {
                "schema": SCHEMA,
                "signature": signature,
                "files": {
                    name: file_hash(paths.run / name)
                    for name in ["checkpoints/best.pt", "checkpoints/last.pt", "history.json"]
                },
            },
        )
