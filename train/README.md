# Training

`train_rolling_probe.py` freezes Qwen3.5-9B, extracts one multimodal hidden-state sequence per BeaverTails-V record, and fits four Max-of-Rolling-Means heads: Physical Harm, Hate Speech, Privacy, and overall response safety. The probe scans the complete image + question + candidate-response sequence; this is an offline reproduction baseline, not a prompt-only pre-generation detector.

The default rolling window is 40 tokens. Layer and threshold are stored in the checkpoint. The threshold is selected on BeaverTails-V validation only; MM-SafetyBench is never used to fit or tune the probe.

This is a dataset-adapted reproduction of the probe architecture, not an exact reproduction of the paper's experiment: the paper detects high-stakes user requests using pre-layer-norm residual activations from Llama-3.3-70B and cached activations, while this project predicts BeaverTails-V candidate-response safety using Qwen3.5-9B hidden states. Because response text is part of the training sequence, this version monitors completed answers; it is not a pre-generation prompt-risk detector.

For a quick end-to-end smoke test, cap both splits:

```bash
conda activate safety-compare
cd /home/gty/agent-probe
python train/train_rolling_probe.py --max-train-rows 8 --max-validation-rows 8 --epochs 1 --output outputs/rolling_probe_smoke.pt
```

Train on all BeaverTails-V categories so the three domain heads see both in-domain examples and explicit out-of-domain negatives. BeaverTails-V has no exact hate-speech label, so its proxy combines discriminatory depictions with insulting/harassing behavior:

```bash
python train/train_rolling_probe.py \
  --output outputs/rolling_probe_4head.pt
```

Each row trains `overall_safety` from the response label. A domain head is positive only for an unsafe response in its mapped category; it is explicitly negative for other categories, which penalizes cross-domain false alarms. This assumes the BeaverTails-V topic label is exclusive; genuinely overlapping or unlabeled topics would need multi-label annotations or an ignore mask. The loss is BCE-with-logits with per-head positive weighting (capped at 50); each head's threshold is tuned separately on validation. Hate Speech remains an approximate mapping and the discriminatory category has relatively few unsafe examples. MM-SafetyBench calls its privacy slice `09-Privacy_Violence`; its prompts are predominantly privacy-invasive, with some doxing/targeted-harassment overlap.

The selected risk dimensions are semantically transferable to text-only settings, but this dataset pipeline still consumes images. The current experiment's labels are response safety after generation, not prompt-only risk.

Remove the `--max-*-rows` options for the full BeaverTails-V training run. It extracts model activations online (no large activation cache), so the full run will take substantially longer than the smoke test.

## Causal attention + SWiM comparison probe

The first alternative probe is implemented in `train_attention_swim_probe.py`. It keeps the same task labels, train/validation manifests, hidden-state extraction, and shared CLI options as the rolling-max trainer, so the two checkpoints can be compared on the same rows. It uses the final hidden layer, a two-layer width-256 MLP, 8 task-specific causal-attention subheads per output, SWiM window 16, token-softmax temperature 1, task-loss EMA `rho=0.9`, and one epoch by default. The first 3,000 optimizer steps use equal four-head weights while collecting the EMA loss statistics; dynamic weights activate afterward. Inference thresholds are tuned on BeaverTails-V validation from max EMA-smoothed token probabilities (`alpha=0.5`).

```bash
python train/train_attention_swim_probe.py \
  --output outputs/attention_swim_probe_4head.pt
```

Use the same evaluator and options as for the baseline; the evaluator detects the checkpoint architecture automatically:

```bash
python eval/eval_rolling_probe.py \
  --checkpoint outputs/attention_swim_probe_4head.pt \
  --dataset beavertails --split official_evaluation
```

The alternative trainer adds `--attention-heads`, `--mlp-hidden-size`, `--dropout`, `--temperature`, `--ema-alpha`, `--loss-ema-rho`, and head-weight clamp options. To match the baseline's scan scope, both training and evaluation currently score the complete image + question + response sequence, not only assistant answer tokens. MM-SafetyBench generation, caching/resume, judge options, and CSV fields remain in the existing evaluator.
