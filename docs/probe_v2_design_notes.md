# Probe v2 design notes

Status: first implementation landed; settings below define the initial comparison run, not a tuned result.

## Fixed comparison setup

- Keep the task, labels, model responses, dataset splits, and evaluation protocol the same when comparing probe designs.
- Target four response-risk outputs: `physical_harm`, `hate_speech`, `privacy`, and `overall_safety`.
- Preserve explicit negatives for the non-matching risk heads so a false alarm on another category is penalized.
- Retain the existing Max-of-Rolling-Means probe as the baseline.

## Training loss: SWiM + softmax-weighted BCE

Inspired by *Constitutional Classifiers++: Efficient Production-Grade Defenses against Universal Jailbreaks* (arXiv:2601.04603), §3.2.

Let `a[i,t,k]` be the token-level logit for example `i`, answer position `t`, and head `k`. The probe architecture that produces these logits is still undecided.

1. During training, smooth logits with a sliding-window mean of width `M`:

   `zbar[i,t,k] = mean(a[i,t-M+1,k], ..., a[i,t,k])` for full windows.

2. For each example and head, assign a softmax weight across valid smoothed token positions:

   `w[i,t,k] = softmax_t(zbar[i,t,k] / tau)`.

3. Compute sequence-label BCE at every valid position and weight it by `w`:

   `L[i] = sum_k sum_t w[i,t,k] * BCE(y[i,k], sigmoid(zbar[i,t,k]))`.

This is a four-head adaptation of the paper's binary objective, implemented in `train/train_attention_swim_probe.py`. The softmax is per example and per head. It emphasizes positions where that head has the strongest harmful logit; on a negative label, a highly confident false alarm receives more emphasis. This is model-induced token weighting, not human-provided token/span attribution labels.

For sequences shorter than `M`, average the available positions and make one prediction, following the paper's stated short-sequence handling. The intended token scope is assistant-response tokens; whether to include additional positions remains to be confirmed against the chosen architecture.

## Dynamic weighting across the four heads

This is a separate, outer level of weighting inspired by SingProbe (arXiv:2608.30703): keep the SWiM + token-softmax BCE above as each head's loss, then dynamically weight the four head losses according to their recent EMA loss.

Let `L[k,s]` be head `k`'s SWiM/token-weighted loss at optimizer step `s`. Maintain:

`m[k,s] = rho * m[k,s-1] + (1-rho) * L[k,s]`

and set:

`lambda[k,s] = lambda_base[k] * m[k,s] / mean_j(m[j,s])`

`L_total[s] = sum_k lambda[k,s] * L[k,s]`.

Initial implementation uses equal base weights (`lambda_base[k] = 1`) and `rho=0.9`. The per-head EMA is updated after each training example; relative weights are normalized by their mean and clipped to `[0.5, 2]`.

Important caveat: BeaverTails-V head prevalence differs, and any per-head positive-class weighting changes loss scale. A large EMA loss may therefore reflect class weighting, label noise, or an intrinsically difficult head—not only a head that deserves more importance. Track per-head raw losses, EMA losses, and effective weights; consider warm-up and a weight clamp, and compare dynamic weighting with equal head weights if time permits.

## Architecture: MLP + causal attention pooling

Inspired by the Attention Probe in *Building Production-Ready Probes for Gemini* (arXiv:2601.11516).

- First apply a shared two-layer MLP to each selected token activation: `y[t] = MLP_2(h[t])`, with LayerNorm, ReLU and dropout between layers.
- Then apply **global causal attention pooling** at each sequence position: at position `t`, pool over all valid representations `1..t` (no local attention-window mask), producing a token/prefix logit rather than only one final sequence score. Attention is task-specific and has 8 learned subheads per output; the subhead scalar values are summed.
- Adapt the pool to four target labels: each risk output needs its own scoring path (or explicitly tested shared attention parameters). Keep the four risk outputs distinct from the paper's internal attention subheads `H`.
- Feed the resulting per-prefix logits into the already selected SWiM + softmax-weighted BCE objective, then the candidate EMA-based dynamic weighting across the four task losses.

The paper's displayed attention equation pools all token activations into a sequence-level score. Our global causal per-prefix version applies that pool to each prefix, so it can feed per-position logits into SWiM; SWiM's fixed local window is over these prefix logits, not over the attention's input tokens. Computation is chunked over query positions to limit attention-matrix memory. The first run uses the final hidden layer and MLP width 256.

## Inference aggregation

- Replace the training-time sliding-window buffer with an EMA over sigmoid token scores at inference, coefficient `alpha=0.5`, initialized from the first valid token. The sequence score is the maximum EMA probability across valid positions.
- This is an efficient inference aggregation, not mathematically identical to the training-time SWiM objective; compare it against validation metrics and the rolling-max baseline.

## Follow-up choices

- Whether to add any consistency term between the three fine-grained heads and `overall_safety`.
- Whether to restrict loss/score aggregation to assistant response tokens. For a fair first comparison the current implementation follows the baseline and scores the complete image + question + response sequence.
- Whether to add a warm-up or change the dynamic loss weighting after inspecting per-head loss/weight traces.

## Initial parameter ranges (engineering starting points)

These are starting ranges for the frozen Qwen3.5-9B + four-head setup, not values established by the cited papers on this dataset.

| Parameter | Suggested range | Initial value / note |
|---|---:|---|
| Backbone hidden width | fixed by model: 4096 | Qwen3.5-9B config |
| Source layers | 1-4 layers | For a multi-layer first pass, try 1-based layers `[8, 16, 24, 32]`; use only layer 32 if keeping it simple. Avoid concatenating all 32 initially. |
| MLP | 2 layers; hidden width 128-512; output width 64-256 | Start `4096 (or 4*4096) -> 256 -> 128`; LayerNorm before MLP, ReLU to match the Gemini paper or GELU; dropout 0.05-0.15 (start 0.1). |
| Attention subheads per risk output | 4-10 | Start 4 or 8. This is separate from the four risk outputs; the Gemini probe used 10 attention heads by default. |
| SWiM window `M` | 8-32 tokens | Start 16. CC++ used `M=16` and reports best results at a moderate window near 16 in its ablation. |
| Token-loss softmax temperature `tau` | 0.5-2 | Start 1, the CC++ setting; lower values concentrate loss more strongly on the highest-risk prefixes. |
| Inter-head loss EMA `rho` | 0.9-0.99 | Start 0.9 from SingProbe; use the same four base head weights (=1) initially. |
| Dynamic head-weight safeguards | warm-up 50-200 steps; clamp normalized weights around 0.5-2 | Engineering safeguard, not a paper default; log raw/EMA losses and applied weights. |
| Per-head positive weight | `min(n_negative/n_positive, cap)`, cap 10-50 | Initially preserve the existing training code's cap of 50 for comparability; reduce if rare heads dominate despite dynamic weighting. |
| Inference EMA coefficient `alpha_score` | 0.3-0.7 | Start 0.5; this is distinct from `rho` for training-time head-loss EMA. Take max EMA over answer positions for a sequence-level score if following the cited EMA aggregation. |
| Optimizer | AdamW; LR 1e-4-3e-4; weight decay 1e-4-3e-3 | Start LR 1e-4 for the multi-layer MLP-attention probe, weight decay 1e-3, gradient clip 1.0. Gemini's reported general settings include LR 1e-4 and weight decay 3e-3. |
| Epochs / stopping | 2-5 head-only epochs if activations are cached; otherwise start with 1-2 full backbone passes | Early-stop on validation objective/metric, patience 2; each online epoch re-runs the frozen Qwen backbone, so measure cost before increasing. Do not select on official evaluation. |
| Batch | effective batch 16-32 | If extracting activations live on the 80GB A100, start microbatch 1-4 and increase to 4-8 after a memory smoke test; use accumulation to reach the effective batch. |
| Decision threshold | calibrated per output head | If applying Huawei's FPR target, select the highest-recall threshold subject to validation FPR <= 0.5%; also report F1-optimal thresholds for comparison. |

Reference values: the Gemini probes paper reports a two-layer width-100 MLP, 10 attention heads by default, and a rolling attention window of 10; CC++ reports `M=16`, `tau=1`; SingProbe uses `rho=0.9` for EMA task-loss weighting. These are reference points only because model, data, and task differ.

## First-run configuration selected

- Source activation: final Qwen layer only.
- Attention pooling: 8 internal attention subheads per risk output.
- Training loss: SWiM window `M=16`; token softmax temperature `tau=1`.
- Inference smoothing: score EMA coefficient `alpha_score=0.5`.
- Training duration: 1 epoch, as an initial comparison against the existing rolling probe.
- Other starting settings remain as above unless changed: two-layer MLP, AdamW, `lr=1e-4`, `weight_decay=1e-3`, dropout `0.1`, task-loss EMA `rho=0.9`, equal base head weights, 3,000-step equal-weight warm-up, normalized dynamic head-weight clamp `[0.5, 2]`, and existing per-head `pos_weight` policy (negative/positive ratio capped at 50). The loss EMA is still updated during warm-up.

## Commands and API compatibility

- Train with `python train/train_attention_swim_probe.py`; shared CLI options match `train/train_rolling_probe.py` (`--model`, `--train`, `--validation`, `--output`, `--layer`, `--window-size`, `--epochs`, `--learning-rate`, `--weight-decay`, row caps, and `--seed`). New architecture/loss parameters are additional flags, including `--head-weight-warmup-steps` (default 3000).
- Evaluate using the same `python eval/eval_rolling_probe.py` interface as the baseline. It detects the architecture from checkpoint metadata, and retains the same dataset, scenario, generation batching/resume, judge, and output options.
- MM-SafetyBench scoring reuses generated answer hidden states as before; it does not regenerate differently for the new probe. Thresholds are selected on BeaverTails-V validation only.
