# Evaluation

`eval_rolling_probe.py` supports two checks:

```bash
conda activate safety-compare
cd /home/gty/agent-probe

# Labeled held-out BeaverTails-V responses (no generation or external API)
python eval/eval_rolling_probe.py --checkpoint outputs/rolling_probe_smoke.pt --dataset beavertails --split official_evaluation --max-rows 8

# Stage 1: generate all responses and score the probe. No --judge means no
# judge API calls; every completed row is flushed to CSV immediately.
python eval/eval_rolling_probe.py --checkpoint outputs/rolling_probe_4head.pt --dataset mm_safetybench --scenarios 02-HateSpeech --max-new-tokens 512 --batch-size 4 --output outputs/mm_hatespeech_generated.csv

# If generation is interrupted, rerun the same command and add --resume.
python eval/eval_rolling_probe.py --checkpoint outputs/rolling_probe_4head.pt --dataset mm_safetybench --scenarios 02-HateSpeech --max-new-tokens 512 --batch-size 4 --output outputs/mm_hatespeech_generated.csv --resume

# Stage 2: judge cached responses with images; this does not load Qwen or regenerate.
export OPENAI_API_KEY="<your-key>"
export OPENAI_MODEL="GLM-4.6V"
export OPENAI_BASE_URL="https://llmapi.paratera.com/v1"
python eval/judge_cached_responses.py --input outputs/mm_hatespeech_generated.csv --judge-with-image --workers 8

# If judging is interrupted, resume from the partial *_judged.csv output.
python eval/judge_cached_responses.py --input outputs/mm_hatespeech_generated.csv --judge-with-image --workers 8 --resume
```

The MM-SafetyBench generation stage uses a left-padded micro-batch (default 4) for image/question generation and captures the probe layer's hidden states during generation, avoiding a second full-sequence model pass. It streams each completed row to CSV and supports `--resume`; the adjacent manifest prevents resuming with mismatched settings. Lower `--batch-size` if GPU memory is insufficient. The separate `judge_cached_responses.py` stage reads those saved responses and sends the question, response, and optionally the image to an OpenAI-compatible judge, with bounded parallelism (`--workers`, default 8) and a progress bar. It saves each successful judgment immediately and supports `--resume`, without loading Qwen or repeating generation. The judged CSV contains the original probe predictions plus the four labels; LLM labels are exploratory pseudo-ground truth, not human-verified annotations.

To compare a different probe on the exact same generated answers, pass the previous generation or judged CSV through `--input-csv`. The evaluator reuses each response and image/question record, runs a teacher-forced forward pass to extract the new checkpoint's hidden states, and does not call `generate()`. Existing `judge_label_*` columns are copied into the new results; if the input CSV has no judge labels, run `judge_cached_responses.py` on the newly scored output afterward. For example:

```bash
python eval/eval_rolling_probe.py \
  --checkpoint outputs/attention_swim_probe_4head.pt \
  --dataset mm_safetybench \
  --input-csv outputs/mm_safetybench_3scenarios_4096_b8_generated.csv \
  --scenarios 02-HateSpeech 04-Physical_Harm 09-Privacy_Violence \
  --output outputs/mm_safetybench_attention_swim_reused.csv
```

This reuses generated text, not cached activations: the model still needs a forward pass over each saved image/question/response to extract the selected layer. It is substantially cheaper than autoregressive answer generation, but not zero-cost.

The selected slice is `02-HateSpeech`, `04-Physical_Harm`, and `09-Privacy_Violence`. Physical Harm is a direct match; Hate Speech is an approximate training match because BeaverTails-V lacks an exact hate-speech category; and the MM-SafetyBench `Privacy_Violence` prompts are predominantly privacy-invasive but include a small amount of doxing/targeted-harassment content. Use `--scenarios` to select a slice. Each scenario includes the three official image variants (`SD`, `SD_TYPO`, `TYPO`), reported separately.

MM-SafetyBench is test-only. Do not use it for fitting or threshold selection. Without a judge pass, it reports probe scores only; its built-in prompt label is not an answer-safety label. LLM-judge labels are noisy; preserve responses and labels for review. Full 5,040-row generation + image judging can be slow and incur API cost, so smoke-test each stage with `--max-rows` before a full run. The stages are separate so the slow local generation need not be repeated after an API issue.

## Streaming first-alarm evaluation

`eval_streaming_probe.py` compares the four-head rolling-max checkpoint with the two-epoch causal-attention+SWiM checkpoint. It calibrates one threshold per head and probe using BeaverTails-V training negatives (default: up to 1,000 negatives per head) to an empirical 5% FPR. At evaluation, it keeps every example positive in any labeled head and samples up to 500 examples negative in all four heads; set `--max-safe-eval 0` to use every labeled MM-SafetyBench case. Alarm means the first assistant-answer token whose score is strictly above threshold. The image/question/prompt remain causal context, but cannot themselves trigger an alarm. The script replays saved responses teacher-forced rather than calling generation; this gives a causal prefix trajectory for response tokens, not wall-clock streaming latency. The attention probe's prefix pooling is computed with an equivalent online softmax recurrence to avoid quadratic attention materialization.

```bash
CUDA_VISIBLE_DEVICES=0 python eval/eval_streaming_probe.py \
  --calibration-negatives-per-head 1000 \
  --max-safe-eval 500 \
  --target-fpr 0.05 \
  --output-prefix outputs/streaming_probe_eval_fpr05
```

The run writes per-head thresholds, empirical calibration FPR, recall, test FPR, detected-positive mean first-alarm position (tokens and fraction of response length), plus per-example CSV rows.

## Full-response FPR calibration

`score_fpr_calibration_validation.py` scores the BeaverTails-V validation split for rolling max, one-epoch dynamic-weight attention, two-epoch dynamic-weight attention, and one-epoch equal/stable-weight attention. It runs the Qwen backbone once per validation example and reuses that hidden state across all four probes. Then `report_full_response_fpr.py` calibrates each probe/head threshold to validation FPR <= 5% and applies it to the existing BeaverTails official (ID) and MM-SafetyBench (OOD) score CSVs.

```bash
CUDA_VISIBLE_DEVICES=0 python eval/score_fpr_calibration_validation.py
python eval/report_full_response_fpr.py --target-fpr 0.05
```

The report is written to `outputs/full_response_fpr05_report.json`. The one-epoch dynamic-attention MM scores are joined by ID to the judged baseline CSV; the script verifies cached response equality before using those labels. No test-set scoring or generation is repeated.
