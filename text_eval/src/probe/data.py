"""BeaverTails annotation voting, post-vote filtering, and prompt-grouped splits."""

import unicodedata
from collections import Counter

from .contracts import HEADS
from .io import fingerprint

# Names, not dictionary order or numeric positions, define the mapping.
CATEGORY_HEAD = {
    "privacy_violation": "privacy",  # 10
    "misinformation_regarding_ethics,laws_and_safety": "misinformation",  # 8
    "discrimination,stereotype,injustice": "harmful_language",  # 4
    "hate_speech,offensive_language": "harmful_language",  # 7
    "sexually_explicit,adult_content": "harmful_language",  # 12
    "drug_abuse,weapons,banned_substance": "malicious_uses",  # 5
    "financial_crime,property_crime,theft": "malicious_uses",  # 6
    "self_harm": "malicious_uses",  # 11
    "violence,aiding_and_abetting,incitement": "malicious_uses",  # 14
}
EXCLUDED = {
    "animal_abuse",
    "child_abuse",
    "controversial_topics,politics",
    "non_violent_unethical_behavior",
    "terrorism,organized_crime",
}


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def beavertails_annotation(row: dict, source: str) -> tuple[dict | None, str]:
    """Validate one vote; do not filter excluded categories before aggregation."""
    prompt, response = row.get("prompt"), row.get("response")
    if not isinstance(prompt, str) or not isinstance(response, str) or not response.strip():
        return None, "missing_text"
    category = row.get("category", {})
    if not isinstance(category, dict) or set(category) != set(CATEGORY_HEAD) | EXCLUDED:
        raise ValueError("BeaverTails category schema changed; inspect the source mapping")
    if type(row.get("is_safe")) is not bool or any(type(v) is not bool for v in category.values()):
        return None, "invalid_labels"
    active = {name for name, value in category.items() if value}
    if row["is_safe"] != (not active):
        return None, "inconsistent_safe_label"
    active_heads = {CATEGORY_HEAD[c] for c in active if c in CATEGORY_HEAD}
    labels = [float(h in active_heads) for h in HEADS[:-1]]
    labels.append(float(not row["is_safe"]))
    pair = [normalize(prompt), normalize(response)]
    return {
        "id": fingerprint(pair),
        "prompt": prompt,
        "response": response,
        "labels": labels,
        "label_mask": [True] * len(HEADS),
        "sources": [source],
        "excluded_votes": {name: int(name in active) for name in sorted(EXCLUDED)},
    }, "valid_annotations"


def aggregate_annotations(sources):
    """One vote per valid source row, strict per-head majority, ties are safe."""
    pairs, counts = {}, Counter()
    for split, rows in sources:
        for i, row in enumerate(rows):
            vote, reason = beavertails_annotation(row, f"{split}:{i}")
            counts[reason] += 1
            if vote is None:
                continue
            key = vote["id"]
            if key not in pairs:
                pairs[key] = {
                    "sample": {k: vote[k] for k in ("id", "prompt", "response", "label_mask")},
                    "sources": [],
                    "positive_votes": [0] * len(HEADS),
                    "excluded_positive_votes": dict.fromkeys(sorted(EXCLUDED), 0),
                }
            else:
                counts["duplicate_pairs"] += 1
            group = pairs[key]
            sample = group["sample"]
            # Keep an actual source text, deterministically, even for normalized duplicates.
            if (vote["prompt"], vote["response"]) < (sample["prompt"], sample["response"]):
                sample.update(prompt=vote["prompt"], response=vote["response"])
            group["sources"].extend(vote["sources"])
            for c, value in enumerate(vote["labels"]):
                group["positive_votes"][c] += int(value)
            for name, value in vote["excluded_votes"].items():
                group["excluded_positive_votes"][name] += value
    samples, audit = [], []
    for key, group in sorted(pairs.items()):
        n = len(group["sources"])
        positive = group["positive_votes"]
        excluded = group["excluded_positive_votes"]
        counts["disagreement_pairs"] += int(any(0 < v < n for v in [*positive, *excluded.values()]))
        counts["tied_head_votes"] += sum(2 * v == n for v in positive)
        counts["tied_excluded_votes"] += sum(2 * v == n for v in excluded.values())
        sample = group["sample"]
        sample["labels"] = [float(2 * value > n) for value in positive]
        filtered = any(2 * value > n for value in excluded.values())
        counts["excluded_category_pairs"] += int(filtered)
        audit.append(
            {
                "id": key,
                "annotation_count": n,
                "positive_votes": dict(zip(HEADS, positive, strict=True)),
                "excluded_positive_votes": excluded,
                "sources": sorted(group["sources"]),
                "excluded": filtered,
            }
        )
        if not filtered:
            samples.append(sample)
    counts["unique_pairs"] = len(pairs)
    counts["retained_pairs"] = len(samples)
    return samples, audit, counts


def wildguard_sample(row: dict, source: str) -> tuple[dict | None, str]:
    label = row.get("response_harm_label")
    if label not in {"harmful", "unharmful"}:
        return None, "missing_response_label"
    if not isinstance(row.get("prompt"), str) or not isinstance(row.get("response"), str):
        return None, "missing_text"
    if not row["response"].strip():
        return None, "empty_response"
    return {
        "id": "wildguard-" + source,
        "prompt": row["prompt"],
        "response": row["response"],
        "labels": [0.0] * 4 + [float(label == "harmful")],
        "label_mask": [False] * 4 + [True],
        "sources": [source],
    }, "retained"


def prepare_records(sources, benchmark, config):
    """Aggregate all annotations, filter final labels, split, then select subsets."""
    samples, audit, counts = aggregate_annotations(sources)
    external = []
    for i, row in enumerate(benchmark):
        sample, reason = wildguard_sample(row, str(i))
        counts["benchmark_" + reason] += 1
        if sample is not None:
            external.append(sample)
    external_prompts = {normalize(s["prompt"]) for s in external}
    groups = {}
    for sample in samples:
        prompt = normalize(sample["prompt"])
        if prompt in external_prompts:
            counts["benchmark_prompt_overlap_removed"] += 1
            continue
        groups.setdefault(prompt, []).append(sample)
    ordered = sorted(groups, key=lambda p: fingerprint([config.split_seed, p]))
    n = len(ordered)
    first = int(n * config.split_ratios[0])
    second = first + int(n * config.split_ratios[1])
    result = {}
    for name, prompts in zip(
        ["train", "validation", "beavertails_test"],
        [ordered[:first], ordered[first:second], ordered[second:]],
        strict=True,
    ):
        result[name] = sorted((s for p in prompts for s in groups[p]), key=lambda s: s["id"])
    if config.benchmark_enabled:
        result["wildguardtest"] = external
    full_splits = {
        name: label_stats(rows) | {"membership_sha256": fingerprint(sorted(s["id"] for s in rows))}
        for name, rows in result.items()
    }
    for name, rows in result.items():
        limit = config.max_samples
        if isinstance(limit, dict):
            limit = limit.get(name)
        if limit is not None and len(rows) > limit:
            # Split membership is already frozen; selecting individual pairs here is safe.
            # Hash order gives nested, label-independent subsets regardless of source order.
            selected = sorted(rows, key=lambda s: fingerprint([config.split_seed, name, s["id"]]))
            result[name] = sorted(selected[:limit], key=lambda s: s["id"])
    counts["full_prompt_groups"] = n
    return result, dict(counts) | {"full_splits": full_splits, "annotation_audit": audit}


def label_stats(rows):
    return {
        "samples": len(rows),
        "heads": {
            h: {
                "valid": sum(s["label_mask"][c] for s in rows),
                "positive": sum(int(s["labels"][c]) for s in rows if s["label_mask"][c]),
            }
            for c, h in enumerate(HEADS)
        },
    }
