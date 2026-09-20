"""Corpus-level measures for closed-set clip classification: top-1 and top-3 accuracy, macro recall and macro F1
over the label set, the per-label confusion, and two non-adapted baselines (chance, majority-training-label).

Every rate is computed over the supplied records with the supplied closed label set; nothing is calibrated and no
dispersion is estimated (one seeded split of one sample gives one number)."""
# ruff: noqa: E501  -- adaptation-contract lines are kept at the fleet width

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .pipeline import format_labels

TOP_K = 3

METRIC_DEFINITIONS = {
    "top1_accuracy": "fraction of clips whose highest-scoring label is the reference label",
    "top3_accuracy": f"fraction of clips whose reference label is among the {TOP_K} highest-scoring labels (1.0 by construction when the label set has {TOP_K} or fewer labels)",
    "macro_recall": "mean over the label set of the per-label recall (correct clips of that label / clips of that label); labels absent from the records are skipped",
    "macro_f1": "mean over the label set of the per-label F1 between precision and recall of the top-1 decision; labels absent from both references and predictions are skipped",
    "confusion": "reference label -> predicted top-1 label -> count",
    "baselines": "chance = 1 / |labels| top-1 by construction; majority = the most frequent training label predicted for every clip",
}


def _rank(ranking: Sequence[str], reference: str) -> int:
    for index, label in enumerate(ranking):
        if label == reference:
            return index + 1
    return len(ranking) + 1


def classification_metrics(
    rankings: Sequence[Sequence[str]], records: Sequence[Mapping[str, Any]], labels: Sequence[str]
) -> dict[str, Any]:
    """Score one ranking of `labels` (best first) per record against the record's `label`."""
    label_set = list(format_labels(labels))
    if len(rankings) != len(records):
        raise ValueError(f"{len(rankings)} rankings for {len(records)} records")
    if not records:
        raise ValueError("no records to score")
    refs = [format_labels([r["label"], "__other__"])[0] for r in records]
    for ref in refs:
        if ref not in label_set:
            raise ValueError(f"record label {ref!r} is not in the label set")
    ranks = [_rank(list(ranking), ref) for ranking, ref in zip(rankings, refs, strict=True)]
    tops = [list(ranking)[0] if ranking else "" for ranking in rankings]
    confusion: dict[str, dict[str, int]] = {label: {} for label in label_set}
    for ref, top in zip(refs, tops, strict=True):
        confusion[ref][top] = confusion[ref].get(top, 0) + 1
    support = Counter(refs)
    predicted = Counter(tops)
    recalls, f1s = [], []
    per_label = {}
    for label in label_set:
        tp = confusion[label].get(label, 0)
        n_ref, n_pred = support.get(label, 0), predicted.get(label, 0)
        recall = tp / n_ref if n_ref else None
        precision = tp / n_pred if n_pred else None
        f1 = None
        if n_ref or n_pred:
            f1 = (2 * tp / (n_ref + n_pred)) if (n_ref + n_pred) else None
        per_label[label] = {"support": n_ref, "predicted": n_pred, "correct": tp, "recall": recall, "precision": precision, "f1": f1}
        if recall is not None:
            recalls.append(recall)
        if f1 is not None:
            f1s.append(f1)
    k = min(TOP_K, len(label_set))
    return {
        "n": len(records),
        "n_labels": len(label_set),
        "top1_accuracy": sum(r == 1 for r in ranks) / len(ranks),
        "top3_accuracy": sum(r <= k for r in ranks) / len(ranks),
        "macro_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "mean_rank": sum(ranks) / len(ranks),
        "per_label": per_label,
        "confusion": confusion,
        "definitions": dict(METRIC_DEFINITIONS),
    }


def chance_baseline(records: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> dict[str, Any]:
    """Uniform guessing over the label set: top-1 = 1 / |labels|, top-3 = min(3, |labels|) / |labels|."""
    label_set = list(format_labels(labels))
    n = len(label_set)
    return {
        "n": len(records),
        "n_labels": n,
        "top1_accuracy": 1.0 / n,
        "top3_accuracy": min(TOP_K, n) / n,
        "macro_recall": 1.0 / n,
        "macro_f1": 1.0 / n,
        "kind": "chance",
    }


def majority_label(train: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> str:
    """The most frequent training label (ties broken by label-set order)."""
    label_set = list(format_labels(labels))
    counts = Counter(format_labels([r["label"], "__other__"])[0] for r in train)
    return max(label_set, key=lambda label: (counts.get(label, 0), -label_set.index(label)))


def majority_baseline(
    train: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]], labels: Sequence[str]
) -> dict[str, Any]:
    """Predict the most frequent training label for every clip, ranking the rest in label-set order."""
    label_set = list(format_labels(labels))
    top = majority_label(train, label_set)
    ranking = [top] + [label for label in label_set if label != top]
    out = classification_metrics([ranking] * len(records), records, label_set)
    out["kind"] = "majority"
    out["label"] = top
    return out
