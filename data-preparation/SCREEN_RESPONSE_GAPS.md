# Response gap screening

`screen_response_gaps.py` merges original response-level labels, asks the judge
to resolve conflicting labels for the same prompt and response, and separately
judges the safety of prompts that lack an unsafe response. It does not generate
answers. Judge requests include the prompt, associated context/images, and, for
response adjudication, the response; they are sent to the configured service.

## Configure datasets

Copy `screen_response_gaps.example.json` and edit its `datasets` list. File
patterns and relative `image_root` values are relative to the configuration
file, not the shell's current directory. Each dataset has a unique `name`, a
nonempty `files` list of glob patterns, and a `fields` mapping. A given file
must not be listed by more than one dataset.

`fields.prompt` gives the text field. `fields.responses` is a list of mappings:
each provides the response `text` field, the corresponding `label` field,
and explicit `safe_values` and `unsafe_values`. Field names support dotted
object paths such as `conversation.question`. A dataset with prompt-only
records uses `"responses": []`. For responses without original labels use,
for example, `{"text": "model_output"}`.

Label values are compared with their JSON types intact: `true`, `1`, and
`"1"` are distinct. CSV and TSV field values are strings, so a CSV label needs
string values in the mapping. Unknown values are never assumed safe or unsafe.
Overlapping safe/unsafe values are rejected.

Optional `fields.context` names a list of text messages, each with `role` and
`content`. Optional `fields.images` names a local image path or list of local
paths; set `image_root` on the dataset when paths are relative. Context and image
contents participate in prompt identity, so equal text with different context or
images remains separate. Image files must already exist. URLs, embedded image
bytes, arrays of nested conversation pairs and dataset-specific adapters are not
supported by this generic reader; convert such data to the documented schema
first. A row with an empty or non-text prompt is skipped and counted; a file
with no valid prompts fails instead of silently accepting a wrong mapping.

Supported files: JSONL, CSV, TSV, JSON arrays, Parquet and Arrow IPC streams.
JSONL/CSV/TSV/JSON may additionally be gzip- or xz-compressed. Parquet and
Arrow require `pyarrow`.

## Run

Inspect scope without making API calls:

```bash
python screen_response_gaps.py \
  --dataset-config screen_response_gaps.example.json \
  --output-dir data/gap-screen-example --dry-run
```

Run in a shell with `JUDGE_API_KEY` set. `JUDGE_MODEL` and `JUDGE_BASE_URL`
can override the default judge; `--concurrency` limits parallel requests.

```bash
python -u screen_response_gaps.py \
  --dataset-config screen_response_gaps.example.json \
  --output-dir data/gap-screen-example --concurrency 4
```

The default judges conflicting original response labels only. Add
`--judge-unlabeled-responses` to also judge answers with no recognized label;
otherwise those answers stay unresolved. This can substantially increase API
requests. Prompt-only records are exported separately and are not yet judged.
Completed judge decisions are cached in `judge_cache.sqlite` and reused when
rerunning the same inputs with the same judge configuration. The current
labels-only judge uses a new cache signature; older decisions that requested
explanations are retained but are not reused. Use a separate output directory
for each concurrent run.

The legacy `--inputs ...` interface remains available for raw SafeRLHF and
SafeRLHF-QA JSONL files with their original fields.

## Output

All prompt lists are JSON arrays. `missing_unsafe.json` contains prompts with
known safe answers and no known unsafe answer; `missing_safe.json` is the
reverse. `missing_both.json` contains prompts without answers, and
`unresolved.json` contains prompts for which an unknown or conflicted answer
prevents a reliable gap determination. A prompt with known answers of both
classes is counted in the summary but omitted from the gap lists.

The `missing_unsafe` list is additionally split into
`missing_unsafe_prompt_safe.json`, `missing_unsafe_prompt_unsafe.json`, and
`missing_unsafe_prompt_unresolved.json` using an independent input-only judge
call. Judge decisions contain only `safety` and `categories`, without evidence
or a separate usability assessment. A prompt
judged safe is not a calibrated guarantee that it is exceptionally safe, and
the absence of a safe answer does not imply that a refusal is appropriate.
Judge errors and uncertain judgments remain unresolved and are retried
on the next run; inspect `summary.json` for counts and failures.
