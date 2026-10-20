# BeaverTails-V safety probe on Qwen3.5-9B

This project prepares BeaverTails-V for probe training and MM-SafetyBench for held-out multimodal evaluation. The local model is `/ssd/models/Qwen3.5-9B`.

## Dataset and split policy

- BeaverTails-V: train and validation are derived only from its official `train` split. The official `evaluation` split is kept separate. Each source example is expanded into one record per candidate response, with `label=1` meaning unsafe and `label=0` meaning safe. The two responses from a source example always stay in the same split.
- MM-SafetyBench: evaluation only. Its scenarios, question text, image attack variants, and paths are preserved. Do not use it to fit probe parameters or choose hyperparameters.
- Images are materialized once under `data/raw/` and referenced by relative path. Processed manifests are JSONL so image-text records can be streamed.

Both upstream datasets are CC BY-NC 4.0; use is restricted to non-commercial research. BeaverTails-V contains image/question/response safety preferences and severity annotations. Its response-safety target is not identical to MM-SafetyBench jailbreak success, so report transfer evaluation separately from in-domain validation.

## Layout

```text
Production-Ready-Safety-Probe/
  data/raw/                 # downloads and image files (ignored by git)
  data/processed/           # JSONL manifests (ignored by git)
  data/activations/         # future hidden-state cache (ignored by git)
  scripts/                  # dataset download, normalization, integrity checks
  train/                    # probe training and activation extraction
  eval/                     # MM-SafetyBench evaluation
  outputs/                  # checkpoints and metrics (ignored by git)
```

## Prepare datasets

Run from this directory with the `safety-compare` conda environment:

```bash
conda activate safety-compare
python scripts/download_beavertails_v.py
python scripts/prepare_beavertails_v.py
```

For MM-SafetyBench, fetch the official repository and image archive using the links in the upstream README. Pass the repository path after the images have been unpacked in `data/imgs`:

```bash
python scripts/fetch_mm_safetybench.py
```

The question JSON files will be in `data/raw/mm_safetybench/source`. Unpack the official `MM-SafetyBench(imgs).zip` into that repository's `data/imgs`, then create the test manifest:

```bash
python -m pip install gdown
gdown "1xjW9k-aGkmwycqGCXbru70FaSKhSDcR_" -O data/raw/mm_safetybench/MM-SafetyBench-imgs.zip
mkdir -p data/raw/mm_safetybench/source/data/imgs
unzip -q data/raw/mm_safetybench/MM-SafetyBench-imgs.zip -d data/raw/mm_safetybench/source/data/imgs
python scripts/prepare_mm_safetybench.py --source data/raw/mm_safetybench/source
```

If BeaverTails-V download stops, rerun `HF_HUB_DISABLE_XET=1 python scripts/download_beavertails_v.py --workers 4`; completed configs are cached and reused. Then rerun `python scripts/prepare_beavertails_v.py` to rebuild the full manifests.

The helper writes the official question/image variants to `data/processed/mm_safetybench.jsonl` and reports missing images. Its official image archive is hosted externally (Google Drive/Baidu Netdisk), so it is not fetched automatically by this project.

Check prepared manifests and image references with:

```bash
python scripts/check_data.py
```

## Probe design notes

The reference baseline to reproduce is Max of Rolling Means: learned linear scores at each sequence position, rolling mean over a fixed token window, then maximum across windows. The paper's Appendix A.1 uses a 40-token window. The current training target is four logits over the completed multimodal exchange: Physical Harm, Hate Speech (approximate BeaverTails-V mapping), Privacy, and overall response safety. Overall safety is supervised on every response; the three category heads use one-vs-rest category labels.

Qwen3.5-9B is a vision-language model and accepts images. The installed server environment has PyTorch 2.5.1+cu124, Transformers 5.12.1, and Datasets 5.0.0; the model's local config exposes a vision encoder and image token IDs.

## Upstream sources

- [BeaverTails-V](https://huggingface.co/datasets/saferlhf-v/BeaverTails-V)
- [MM-SafetyBench](https://github.com/isXinLiu/MM-SafetyBench)
- [Detecting High-Stakes Interactions using Activation Probes](https://arxiv.org/abs/2506.10805)
- [Reference implementation](https://github.com/Arrrlex/models-under-pressure)
