"""BeaverTails response labels, conservative filtering, and prompt-grouped splits."""

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


def beavertails_sample(row: dict, source: str) -> tuple[dict | None, str]:
    prompt, response = row.get("prompt"), row.get("response")
    if not isinstance(prompt, str) or not isinstance(response, str) or not response.strip():
        return None, "missing_text"
    category = row.get("category", {})
    if set(category) != set(CATEGORY_HEAD) | EXCLUDED:
        raise ValueError("BeaverTails category schema changed; inspect the source mapping")
    if type(row.get("is_safe")) is not bool or any(type(v) is not bool for v in category.values()):
        return None, "invalid_labels"
    active = {name for name, value in category.items() if value}
    if row["is_safe"] != (not active):
        return None, "inconsistent_safe_label"
    if active & EXCLUDED:
        return None, "excluded_category"
    labels = [float(any(CATEGORY_HEAD[c] == h for c in active)) for h in HEADS[:-1]]
    labels.append(float(not row["is_safe"]))
    pair = [normalize(prompt), normalize(response)]
    return {
        "id": fingerprint(pair),
        "prompt": prompt,
        "response": response,
        "labels": labels,
        "label_mask": [True] * len(HEADS),
        "sources": [source],
    }, "retained"


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
    """Pure adapter; all duplicate/conflict decisions precede grouped splitting."""
    counts = Counter()
    external = []
    for i, row in enumerate(benchmark):
        sample, reason = wildguard_sample(row, str(i))
        counts["benchmark_" + reason] += 1
        if sample is not None:
            external.append(sample)
    external_prompts = {normalize(s["prompt"]) for s in external}
    pairs, conflicts = {}, set()
    for split, rows in sources:
        for i, row in enumerate(rows):
            sample, reason = beavertails_sample(row, f"{split}:{i}")
            counts[reason] += 1
            if sample is None:
                continue
            key = sample["id"]
            if key in pairs:
                if pairs[key]["labels"] != sample["labels"]:
                    conflicts.add(key)
                pairs[key]["sources"].extend(sample["sources"])
                counts["duplicate_pairs"] += 1
            else:
                pairs[key] = sample
    groups = {}
    for key, sample in pairs.items():
        if key in conflicts:
            counts["conflicting_pair_removed"] += 1
            continue
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
    return result, dict(counts) | {"full_splits": full_splits}


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
