from copy import deepcopy

import numpy as np
import pytest

from probe.config import DataConfig
from probe.data import CATEGORY_HEAD, EXCLUDED, beavertails_sample, normalize, prepare_records
from probe.metrics import (
    best_f1_threshold,
    binary_metrics,
    calibrate,
    evaluate_scores,
    fpr_threshold,
)


def raw_row(prompt="question", response="answer", active=()):
    return {
        "prompt": prompt,
        "response": response,
        "category": {c: c in active for c in set(CATEGORY_HEAD) | EXCLUDED},
        "is_safe": not bool(active),
    }


@pytest.mark.parametrize("category,head", list(CATEGORY_HEAD.items()))
def test_each_retained_category(category, head):
    sample, reason = beavertails_sample(raw_row(active=[category]), "train:0")
    assert reason == "retained"
    assert sample["labels"][-1] == 1
    assert sum(sample["labels"][:-1]) == 1
    from probe.contracts import HEADS

    assert sample["labels"][HEADS.index(head)] == 1


@pytest.mark.parametrize("category", sorted(EXCLUDED))
def test_excluded_category_filters_the_entire_unsafe_pair(category):
    sample, reason = beavertails_sample(raw_row(active=[category, "privacy_violation"]), "0")
    assert sample is None
    assert reason == "excluded_category"


def test_safe_consistency_and_multilabel():
    row = raw_row(active=["privacy_violation", "hate_speech,offensive_language"])
    sample, _ = beavertails_sample(row, "0")
    assert sample["labels"] == [1, 0, 1, 0, 1]
    row["is_safe"] = True
    assert beavertails_sample(row, "0")[1] == "inconsistent_safe_label"


def test_grouped_dedup_conflicts_and_benchmark_overlap():
    rows = [raw_row(f"Question {i}", f"answer {j}") for i in range(40) for j in range(2)]
    duplicate = deepcopy(rows[0])
    duplicate["prompt"] = " QUESTION  0 "
    duplicate["response"] = "ANSWER  0 "
    conflict = raw_row("question 1", "answer 0", ["privacy_violation"])
    external = [
        {
            "prompt": "question 2",
            "response": "benchmark answer",
            "response_harm_label": "unharmful",
        },
        {"prompt": "other", "response": "unknown", "response_harm_label": None},
    ]
    splits, counts = prepare_records(
        [("330k_train", [*rows, duplicate, conflict])], external, DataConfig()
    )
    assert counts["duplicate_pairs"] == 2
    assert counts["conflicting_pair_removed"] == 1
    assert counts["benchmark_prompt_overlap_removed"] == 2
    prompt_sets = [
        {normalize(r["prompt"]) for r in splits[s]}
        for s in ["train", "validation", "beavertails_test"]
    ]
    assert not prompt_sets[0] & prompt_sets[1]
    assert not prompt_sets[0] & prompt_sets[2]
    assert not prompt_sets[1] & prompt_sets[2]
    assert all("question 2" not in s for s in prompt_sets)
    assert splits["wildguardtest"][0]["label_mask"] == [False] * 4 + [True]
    repeat, _ = prepare_records(
        [("330k_train", list(reversed([*rows, duplicate, conflict])))], external, DataConfig()
    )
    assert {s: [r["id"] for r in v] for s, v in splits.items()} == {
        s: [r["id"] for r in v] for s, v in repeat.items()
    }


@pytest.mark.parametrize("limit", [None, 1, 7, 10000, {}, {"train": 7, "validation": 3}])
def test_limits_are_hard_caps_within_the_original_splits(limit):
    rows = [raw_row(f"question {i}", f"answer {j}") for i in range(60) for j in range(5)]
    benchmark = [
        {"prompt": f"external {i}", "response": f"answer {i}", "response_harm_label": "unharmful"}
        for i in range(12)
    ]
    source = [("330k_train", rows)]
    full, full_counts = prepare_records(source, benchmark, DataConfig())
    selected, counts = prepare_records(source, benchmark, DataConfig(max_samples=limit))
    assert counts["full_splits"] == full_counts["full_splits"]
    for split, samples in selected.items():
        cap = limit.get(split) if isinstance(limit, dict) else limit
        expected_size = len(full[split]) if cap is None else min(cap, len(full[split]))
        assert len(samples) == expected_size
        assert {s["id"] for s in samples} <= {s["id"] for s in full[split]}
        assert counts["full_splits"][split]["samples"] == len(full[split])
    # A cap below a single prompt group's size is still respected exactly.
    if limit == 1:
        assert all(len(samples) == 1 for samples in selected.values())


def test_subset_caps_are_independent_nested_and_source_order_invariant():
    rows = [raw_row(f"question {i}", f"answer {j}") for i in range(100) for j in range(3)]

    def select(caps, source=rows):
        return prepare_records([("330k_train", source)], [], DataConfig(max_samples=caps))[0]

    small = select({"train": 5, "validation": 4})
    large = select({"train": 13, "validation": 4})
    repeat = select({"train": 5, "validation": 4}, list(reversed(rows)))
    assert {name: [s["id"] for s in rows] for name, rows in small.items()} == {
        name: [s["id"] for s in rows] for name, rows in repeat.items()
    }
    assert {s["id"] for s in small["train"]} < {s["id"] for s in large["train"]}
    assert small["validation"] == large["validation"]
    assert small["beavertails_test"] == large["beavertails_test"]


def test_thresholds_and_known_metrics():
    y, z = np.array([0, 0, 1, 1]), np.array([-2.0, -1.0, 1.0, 2.0])
    threshold = best_f1_threshold(y, z)
    assert threshold == 1
    metrics = binary_metrics(y, z, threshold)
    assert metrics["auroc"] == metrics["auprc"] == metrics["acc"] == metrics["f1"] == 1
    assert metrics["fpr"] == 0


def test_fpr_calibration_is_conservative_with_ties():
    y, z = np.array([0] * 100 + [1]), np.array([2.0] * 3 + [0.0] * 97 + [3.0])
    for f in [0.01, 0.05]:
        threshold = fpr_threshold(y, z, f)
        assert binary_metrics(y, z, threshold)["fpr"] <= f


def test_unknown_heads_and_single_class_are_explicitly_unavailable():
    labels = np.zeros((3, 5))
    mask = np.zeros((3, 5), dtype=bool)
    mask[:, -1] = True
    scores = np.ones((3, 5))
    thresholds = calibrate(labels, mask, scores, [0.01])
    metrics = evaluate_scores(labels, mask, scores, thresholds)
    assert metrics["privacy"]["n"] == 0
    assert metrics["privacy"]["f1"] is None
    assert metrics["overall"]["auroc"] is None
    assert metrics["overall"]["auprc"] is None
    assert metrics["overall"]["fpr"] == 0
