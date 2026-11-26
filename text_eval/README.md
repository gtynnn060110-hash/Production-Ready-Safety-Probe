# YHT text-probe workflow

This isolated subproject contains the YHT data preparation, frozen-backbone
feature extraction, probe training, and evaluation stages. Run commands from
this directory; all prepared data, cached hidden states, checkpoints, and
reports are written under `text_eval/artifacts/` and are Git-ignored.

## Setup

Use Python 3.11 and install the pinned workflow dependencies in an isolated
environment:

```bash
cd /home/gty/Production-Ready-Safety-Probe/text_eval
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The Hugging Face datasets are BeaverTails (`330k_train` and `330k_test`) and
WildGuardMix's `wildguardtest`. The pipeline groups BeaverTails by prompt,
creates train/validation/ID-test splits, and maps labels to four risk heads plus
overall safety. Feature extraction downloads Qwen3.5-9B at the pinned revision,
then caches block-15 BF16 activations with a maximum sequence length of 4096.
Ensure Hugging Face access is configured if your environment requires it, and
allow substantial free disk space (the configured cache budget is 500 GB).

## Full-data run

Prepare and extract once. The two probe runs share the same dataset and feature
configuration, so the second training run reuses those artifacts:

```bash
probe prepare \
  --config configs/base.yaml \
  --config configs/runs/mqa_rmattn_full_fpr5.yaml

probe extract \
  --config configs/base.yaml \
  --config configs/runs/mqa_rmattn_full_fpr5.yaml

probe train \
  --config configs/base.yaml \
  --config configs/runs/mqa_rmattn_full_fpr5.yaml

probe train \
  --config configs/base.yaml \
  --config configs/runs/singprobe_full_fpr5.yaml
```

Evaluate each trained probe. Thresholds are calibrated on BeaverTails
validation at FPR 5%, then applied to BeaverTails ID test and WildGuardTest:

```bash
probe evaluate \
  --config configs/base.yaml \
  --config configs/runs/mqa_rmattn_full_fpr5.yaml

probe evaluate \
  --config configs/base.yaml \
  --config configs/runs/singprobe_full_fpr5.yaml
```

## Train the causal attention + SWiM probe

This adaptation keeps YHT's five labels (privacy, misinformation, harmful
language, malicious uses, and overall safety) while using the two-layer MLP,
eight per-task causal attention subheads, 16-token SWiM loss, positive-class
weights capped at 50, and EMA-based head-loss weighting. It uses Qwen3.5-9B
block 15, the same feature layer as YHT's text runs, so the prepared data and
prepared-data manifest can be shared. The local model path creates a separate
feature-cache signature, so activations are extracted from the local checkpoint.
Inference adds the same `alpha=0.5` token
probability EMA and max readout; raw max/mean/last are also reported for
comparison. Training follows the multimodal probe's one-epoch, batch-one,
`1e-4` learning-rate, and `1e-3` weight-decay settings.
The Gty config loads the backbone from `/ssd/models/Qwen3.5-9B` in local-only
mode; its feature cache has a distinct manifest from the Hugging Face model-ID
cache because the configured model path differs.

```bash
probe prepare \
  --config configs/base.yaml \
  --config configs/runs/causal_attention_swim_layer15_fpr5_gty.yaml

probe extract \
  --config configs/base.yaml \
  --config configs/runs/causal_attention_swim_layer15_fpr5_gty.yaml

probe train \
  --config configs/base.yaml \
  --config configs/runs/causal_attention_swim_layer15_fpr5_gty.yaml

probe evaluate \
  --config configs/base.yaml \
  --config configs/runs/causal_attention_swim_layer15_fpr5_gty.yaml
```

`prepare` will reuse its completed cache if the data manifest already exists.

To check artifact paths and configuration without starting a stage, append
`--dry-run` to any command. Extraction is resumable at completed shard
boundaries; do not change the feature config between stages if you want to
reuse the same cache.
