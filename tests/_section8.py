"""Stand-ins for executing the notebook's Section 7/8/10 cells without a model (plumbing evidence, not model evidence)."""
# ruff: noqa: E501

from __future__ import annotations

import json
import time
from pathlib import Path

from xclip_video_classification_pipeline import metrics as mt
from xclip_video_classification_pipeline import protocol as pr
from xclip_video_classification_pipeline import samples as sm

METRICS = ("top1_accuracy", "top3_accuracy", "macro_recall", "macro_f1")
LABELS = ["a", "b"]


def metrics_for(top1: float, n: int = 4, predictions: list | None = None) -> dict:
    return {**{k: top1 for k in METRICS}, "n": n, "n_labels": 2, "label": "a", "mean_rank": 1.5, "verdict": "measured-small-sample", "definitions": {},
            "per_label": {"a": {"recall": top1, "support": n // 2}, "b": {"recall": top1, "support": n - n // 2}},
            "rankings": [["a", "b"]] * len(predictions or []), "top1_probability": [0.5] * len(predictions or []), "predictions": list(predictions or [])}


FRESH = [{"id": f"hmdb51-train-0-{i}", "frames": [i], "label": "a" if i % 2 else "b", "film": f"film{i // 2}"} for i in range(8)]


def adapter(**changes) -> dict:
    base = {"epochs": 8, "batch_size": 16, "seed": 0, "lr": 1e-5, "grad_clip": 1.0, "min_improvement": 2 / 53, "n_val": 53, "n_train": 181,
            "best_epoch": 2, "history": [{"epoch": 0, "val": {"top1_accuracy": 0.8, "n": 53}}, {"epoch": 1, "val": {"top1_accuracy": 0.9, "n": 53}}, {"epoch": 2, "val": {"top1_accuracy": 0.95, "n": 53}}],
            "trainable_names": [], "head_sha256": "h"}
    base.update(changes)
    return base


def fingerprint(**changes) -> dict:
    fp = {"data_source": pr.SAMPLE_DATA_SOURCE, "split_seed": sm.SAMPLE_SEED, "train_digest": "t", "validation_digest": "v",
          "label_text_sha256": pr.label_text_digest(), "epochs": 8, "batch_size": 16, "adapt_seed": 0, "min_improvement_clips": 2}
    fp.update(changes)
    return pr.run_fingerprint(**fp)


def grid(lrs=pr.GRID_LRS, **fp_changes) -> dict:
    runs = {lr: adapter(lr=lr) for lr in lrs}
    correct = {lr: pr.kept_correct(r) for lr, r in runs.items()}
    if set(lrs) == set(pr.GRID_LRS):
        selection = pr.select_lr(correct, fingerprint=fingerprint(**fp_changes), n_val=53)
    else:
        selection = {"selected_lr": 1e-5, "correct": {f"{lr:g}": c for lr, c in correct.items()}, "fingerprint": fingerprint(**fp_changes)}
    return {"runs": runs, "selection": selection}


class StandInPipe:
    def __init__(self, adapter_record: dict, head: str = "h"):
        self.adapter = adapter_record
        self.head = head
        self.fresh_calls = 0

    def head_digest(self):
        return self.head

    def evaluate(self, records, labels, batch_size=8):
        if records is FRESH or (records and str(records[0].get("id", "")).startswith("hmdb51-train-")):
            self.fresh_calls += 1
            return metrics_for(0.75, n=len(records), predictions=[r["label"] for r in records[:6]] + ["a"] * (len(records) - 6))
        return metrics_for(0.9)

    def evaluate_base(self, records, labels, batch_size=8):
        self.fresh_calls += 1
        return metrics_for(0.5, n=len(records), predictions=["a"] * len(records))


def namespace(pipe: StandInPipe, grid_record, fetch_log: list, *, split_seed: int = sm.SAMPLE_SEED, data_source: str = pr.SAMPLE_DATA_SOURCE) -> dict:
    ns = {}
    for module in (mt, sm, pr):
        ns.update({k: getattr(module, k) for k in dir(module) if not k.startswith("__")})

    def fetch_fresh_test(**kwargs):
        fetch_log.append(kwargs)
        return list(FRESH)

    ns.update({
        "pipe": pipe, "grid_record": grid_record, "fetch_fresh_test": fetch_fresh_test, "clip_digest": lambda frames: f"clip-{frames}",
        "train_records": [], "val_records": [], "test_records": [], "splits": {"train": [], "validation": [], "test": []},
        "time": time, "json": json, "Path": Path, "LABELS": LABELS, "EVAL_BATCH_SIZE": 8, "METRICS": METRICS,
        "baseline_chance": metrics_for(0.5), "baseline_majority": metrics_for(0.25), "frozen_test": metrics_for(0.75), "frozen_verdict": "pre-trained model above both baselines",
        "MODEL_ID": "stand-in", "MODEL_REVISION": "0" * 40, "MODEL_KEY": "stand-in", "data_source": "stand-in", "run_data_source": data_source, "SPLIT_SEED": split_seed,
        "dataset_manifests": {"train": {"digest": "t"}, "validation": {"digest": "v"}, "test": {"digest": "d"}}, "disjoint": {}, "films_shared": {},
        "adapt_result": dict(pipe.adapter or {}), "adapt_seconds": 0.0,
    })
    return ns
