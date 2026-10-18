"""The tutorial's pre-registered evaluation protocol (docs/xclip-preregistration.md, 2026-10-18).

A three-value learning-rate grid is compared on validation only; a challenger replaces the default 1e-5 only when its
kept-epoch validation top-1 beats the default's by at least two clips, and an adapted epoch replaces the frozen model
only when it beats it by two clips (`adapt(min_improvement=...)`). The fresh test (the HMDB51 split-1 training clips of
the ten classes, `samples.fetch_fresh_test`; scored by the 2026-10-18 evaluation of record) is scored only when every pre-registered setting holds,
the grid ran complete on this run's data, and the model in memory is the selected system (SCORE_ONCE_RULE). Any other
setting is exploratory: it reads validation and the labelled exposed development test (the 66 clips of the default
split, which a 2026-09-20 sweep read for five learning rates) and writes separate `*_exploratory_<n>` files.
"""
# ruff: noqa: E501  -- written at the fleet's protocol-module width

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .pipeline import GRAD_CLIP
from .samples import FRESH_TEST_LIST_SHA256, SAMPLE_CLASS_TEXT, SAMPLE_SEED

PREREGISTRATION = "docs/xclip-preregistration.md"
AMENDMENTS = ("docs/xclip-preregistration-amendment-1.md",)
FETCH_CHECK_ENV = "DIMER_FRESH_FETCH_CHECK"
# Amendment 1, A4: 80 % power minimum detectable paired top-1 difference on 669 clips (assumed d 0.10-0.20, ICC 0-0.3).
MDE_RANGE = (0.034, 0.067)
DEFAULT_LR = 1e-5
GRID_LRS = (1e-5, 5e-6, 2e-5)  # C0 (default), C1, C2
GRID_MARGIN_CLIPS = 2  # validation clips; a practical tolerance, not a significance threshold
KEEP_FROZEN_MARGIN_CLIPS = 2
SAMPLE_DATA_SOURCE = "HMDB51 sample"
N_BOOT = 2000
BOOT_SEED = 113
MASK_ENV = "DIMER_MASK_FRESH_TEST"


def label_text_digest(class_text: Mapping[str, str] | None = None) -> str:
    """SHA-256 of the class -> label phrasing the text tower ranks (a phrasing edit is exploratory)."""
    text = json.dumps(dict(class_text if class_text is not None else SAMPLE_CLASS_TEXT), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Every setting the pre-registration fixes. `lr` must also equal the grid's selection in the same run.
PREREGISTERED_SETTINGS: dict[str, Any] = {
    "data_source": SAMPLE_DATA_SOURCE,
    "split_seed": SAMPLE_SEED,
    "label_text_sha256": label_text_digest(),
    "fresh_test_list_sha256": FRESH_TEST_LIST_SHA256,
    "epochs": 8,
    "batch_size": 16,
    "adapt_seed": 0,
    "grad_clip": GRAD_CLIP,
    "min_improvement_clips": KEEP_FROZEN_MARGIN_CLIPS,
    "grid_lrs": list(GRID_LRS),
    "grid_run": True,
}
# SCORE_ONCE_RULE item 8: a runtime cannot know about other runtimes, so the labels say "(this runtime)"; which run is
# the evaluation of record (the first hosted pre-registered run) is stated only in docs/release-verification.md.
TEST_LOOKS = (
    "pre-registered evaluation (this runtime)",
    "reproduction",
    "not scored (exploratory)",
    "not scored (fresh test masked for a development check)",
)
# The evaluation of record: the first hosted T4 run with the pre-registered settings (docs/release-verification.md,
# "Fresh-test evaluation of record"). Values as printed by Section 8 of that run (3 decimals). Once it is set, every
# later scoring with identical settings, including the first one in a new runtime, is a "reproduction": it writes
# `*_reproduction_<n>` files, never the main ones, and prints whether it matches this record. Nothing here changes a
# setting.
RECORDED_EVALUATION: dict[str, Any] | None = {
    "run": "Colab CLI sequential execution (colab exec -f) on a fresh Colab T4 runtime, 2026-10-18",
    "commit": "1bf9b91",
    "notebook_blob": "0c9b1f0f",
    "evidence": "docs/execution-evidence/2026-10-18-1bf9b91/",
    "selected_system": "adapted",
    "lr": 1e-05,
    "n": 669,
    "frozen_top1": 0.786,
    "selected_top1": 0.895,
    "delta_top1": 0.109,
    "ci95": [0.066, 0.151],
    "frozen_only_correct": 17,
    "selected_only_correct": 90,
    "verdict": "adapted helps",
}
# The fields a reproduction is compared on, at the record's printed precision.
RECORD_MATCH_FIELDS = ("selected_system", "lr", "n", "frozen_top1", "selected_top1", "delta_top1", "ci95",
                       "frozen_only_correct", "selected_only_correct", "verdict")
_MODULE_RECORD = object()
MODEL_STATE_MISMATCH = "model state is not the selected system"
_ADAPTER_KEYS = ("lr", "epochs", "batch_size", "seed", "best_epoch", "n_train", "min_improvement")


def fresh_test_masked() -> bool:
    return os.environ.get(MASK_ENV, "").strip() not in ("", "0")


def fresh_fetch_check_requested() -> bool:
    """Masked runs only: also fetch, verify and decode the fresh test, without a model (amendment-1 round)."""
    return fresh_test_masked() and os.environ.get(FETCH_CHECK_ENV, "").strip() not in ("", "0")


def run_fingerprint(*, data_source: str, split_seed: int, train_digest: str, validation_digest: str, label_text_sha256: str,
                    epochs: int, batch_size: int, adapt_seed: int, min_improvement_clips: int) -> dict[str, Any]:
    """The inputs a grid selection is valid for: a selection made on other data or settings does not carry over."""
    return {"data_source": data_source, "split_seed": split_seed, "train_digest": train_digest,
            "validation_digest": validation_digest, "label_text_sha256": label_text_sha256, "epochs": epochs,
            "batch_size": batch_size, "adapt_seed": adapt_seed, "min_improvement_clips": min_improvement_clips}


def kept_correct(adapt_result: Mapping[str, Any]) -> int:
    """Validation clips the kept epoch ranks first (top-1 x n, rounded)."""
    val = adapt_result["history"][adapt_result["best_epoch"]]["val"]
    return int(round(val["top1_accuracy"] * val["n"]))


def select_lr(correct: Mapping[float, int], *, margin: int = GRID_MARGIN_CLIPS, fingerprint: Mapping[str, Any] | None = None,
              n_val: int | None = None) -> dict[str, Any]:
    """Apply the pre-registered rule to the kept-epoch validation top-1 (in clips) of each grid learning rate.

    1e-5 stays unless a challenger beats it by >= `margin` clips (ties and smaller wins go to the default). When both
    challengers clear it, the one with more correct clips wins; equal counts go to the smaller learning rate."""
    keys = {float(k) for k in correct}
    if keys != set(GRID_LRS):
        raise ValueError(f"grid scores must cover exactly {list(GRID_LRS)}, got {sorted(keys)}")
    by_lr = {float(k): int(v) for k, v in correct.items()}
    base = by_lr[DEFAULT_LR]
    clear = [lr for lr in GRID_LRS if lr != DEFAULT_LR and by_lr[lr] >= base + margin]
    if not clear:
        selected, reason = DEFAULT_LR, f"no challenger beat {DEFAULT_LR:g} by {margin} validation clips"
    elif len(clear) == 1:
        selected, reason = clear[0], f"{clear[0]:g} beat {DEFAULT_LR:g} by >= {margin} validation clips"
    else:
        selected = max(clear, key=lambda lr: (by_lr[lr], -lr))
        reason = f"both challengers beat {DEFAULT_LR:g} by >= {margin}; {selected:g} has more correct clips (equal: smaller lr)"
    return {"selected_lr": selected, "reason": reason, "correct": {f"{lr:g}": by_lr[lr] for lr in GRID_LRS},
            "n_val": n_val, "margin_clips": margin, "metric": "kept-epoch validation top-1 (clips)",
            "fingerprint": dict(fingerprint) if fingerprint is not None else None}


def grid_problems(grid: Mapping[str, Any] | None, fingerprint: Mapping[str, Any]) -> list[str]:
    """Why the grid record does not support a pre-registered scoring (empty = it does)."""
    if not grid:
        return ["grid (no grid selection in this run; RUN_GRID must be True)"]
    out = []
    runs = grid.get("runs") or {}
    if {float(k) for k in runs} != set(GRID_LRS) or set((grid.get("selection") or {}).get("correct", {})) != {f"{lr:g}" for lr in GRID_LRS}:
        out.append(f"grid (incomplete: ran {sorted(float(k) for k in runs)}, needs {list(GRID_LRS)})")
    selection = grid.get("selection") or {}
    if selection.get("fingerprint") != dict(fingerprint):
        out.append("grid (it ran on other data or settings than this run; re-run Section 7)")
    return out


def adapter_settings(adapter: Mapping[str, Any] | None) -> dict[str, Any]:
    """The training settings of the model in memory, read from its `adapter` record (not from form globals)."""
    a = adapter or {}
    n_val = a.get("n_val") or 0
    clips = a.get("min_improvement")
    return {"epochs": a.get("epochs"), "batch_size": a.get("batch_size"), "adapt_seed": a.get("seed"), "lr": a.get("lr"),
            "grad_clip": a.get("grad_clip"),
            "min_improvement_clips": None if clips is None or not n_val else int(round(clips * n_val))}


def model_state_problems(adapter: Mapping[str, Any] | None, selected_run: Mapping[str, Any] | None) -> list[str]:
    """Why the model in memory is not the grid's selected run, re-trained in Section 7 (empty = it is)."""
    if adapter is None or selected_run is None:
        return [MODEL_STATE_MISMATCH + " (no adapter on the pipeline, or no selected run)"]
    bad = [k for k in _ADAPTER_KEYS if adapter.get(k) != selected_run.get(k)]
    hist = lambda h: [(e["epoch"], (e.get("val") or {}).get("top1_accuracy")) for e in h or []]  # noqa: E731
    if hist(adapter.get("history")) != hist(selected_run.get("history")):
        bad.append("validation history")
    return [f"{MODEL_STATE_MISMATCH} (differs in {bad})"] if bad else []


def settings_differing(settings: Mapping[str, Any], grid: Mapping[str, Any] | None, fingerprint: Mapping[str, Any],
                       model_state: Sequence[str] = ()) -> list[str]:
    """The settings that differ from the pre-registration (empty = pre-registered): every fixed setting, the grid's
    completeness and the inputs it ran on, the lr against the grid's selection, and the model state."""
    differing = [k for k, v in PREREGISTERED_SETTINGS.items() if settings.get(k) != v]
    problems = grid_problems(grid, fingerprint)
    differing += problems
    if not problems and settings.get("lr") != grid["selection"]["selected_lr"]:
        differing.append(f"lr (used {settings.get('lr')}, grid selected {grid['selection']['selected_lr']})")
    differing += list(model_state)
    return differing


def read_looks(log_path: str | Path) -> list[dict[str, Any]]:
    """The scored looks recorded in this runtime (append-only JSON lines; unreadable lines are skipped, never fatal)."""
    path = Path(log_path)
    out = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    except (OSError, UnicodeDecodeError):
        return [{"test_look": TEST_LOOKS[1], "note": "unreadable look log"}]
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            entry = None
        # Fail safe, like the unreadable-file branch: a line that cannot be read counts as a prior look.
        out.append(entry if isinstance(entry, dict) else {"test_look": TEST_LOOKS[1], "note": "unparsable ledger line"})
    return out


def is_look(entry: Mapping[str, Any]) -> bool:
    """A ledger entry that counts as a look at the fresh test: an `attempt` (a model was about to see the fresh clips),
    a `scored` entry, or any entry labelled with a scored test_look."""
    return entry.get("event") in ("attempt", "scored") or entry.get("test_look") in TEST_LOOKS[:2]


def look_at_test(differing: Sequence[str], log_path: str | Path, canonical: str | Path, *, masked: bool = False,
                 recorded: Any = _MODULE_RECORD) -> str:
    """`not scored (exploratory)` when any setting differs; the masked label for a development check; `reproduction`
    when an evaluation of record is set (`recorded`, default `RECORDED_EVALUATION`; this includes the first run in a new
    runtime), when this runtime already scored the fresh test (look log) or when a canonical report exists, readable or
    not; else the pre-registered evaluation (this runtime)."""
    if differing:
        return TEST_LOOKS[2]
    if masked:
        return TEST_LOOKS[3]
    record = RECORDED_EVALUATION if recorded is _MODULE_RECORD else recorded
    if record is not None or any(is_look(e) for e in read_looks(log_path)) or Path(canonical).exists():
        return TEST_LOOKS[1]
    return TEST_LOOKS[0]


def compare_with_record(reading: Mapping[str, Any], lr: float, record: Mapping[str, Any] | None) -> dict[str, Any]:
    """Display only: does this run's fresh-test reading equal the evaluation of record at its printed precision (3
    decimals; counts, lr, system and verdict exactly)? Returns {'matches_record': bool | None, 'differs': {...}}."""
    if record is None:
        return {"matches_record": None, "differs": {}, "note": "no evaluation of record is set"}
    now = {"selected_system": reading["selected_system"], "lr": float(lr), "n": reading["n"],
           "frozen_top1": round(reading["frozen_top1"], 3), "selected_top1": round(reading["selected_top1"], 3),
           "delta_top1": round(reading["delta_top1"], 3), "ci95": [round(x, 3) for x in reading["ci95"]],
           "frozen_only_correct": reading["mcnemar"]["frozen_only_correct"],
           "selected_only_correct": reading["mcnemar"]["selected_only_correct"], "verdict": reading["verdict"]}
    differs = {k: {"record": record.get(k), "this_run": now[k]} for k in RECORD_MATCH_FIELDS if record.get(k) != now[k]}
    return {"matches_record": not differs, "differs": differs, "record": f"{record.get('commit')} / {record.get('notebook_blob')} ({record.get('run')})"}


def fresh_test_scored(test_look: str) -> bool:
    return test_look in TEST_LOOKS[:2]


def output_suffix(path: str | Path, test_look: str) -> str:
    """One suffix per run for every file it writes: '' for the first pre-registered scoring in this runtime,
    `_reproduction_<n>` or `_exploratory_<n>` (first free n beside `path`) otherwise, so no run overwrites another."""
    if test_look == TEST_LOOKS[0]:
        return ""
    kind = "reproduction" if test_look == TEST_LOOKS[1] else "exploratory"
    path = Path(path)
    n = 1
    while path.with_name(f"{path.stem}_{kind}_{n}{path.suffix}").exists():
        n += 1
    return f"_{kind}_{n}"


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    """Write JSON to a temporary file beside `path`, then rename it over `path`: an exception mid-dump leaves any
    previous file byte-identical."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def append_look(log_path: str | Path, entry: Mapping[str, Any]) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(entry), sort_keys=True, default=str) + "\n")


def _binom_two_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_reading(frozen_correct: Sequence[bool], selected_correct: Sequence[bool], films: Sequence[str], *,
                   n_boot: int = N_BOOT, seed: int = BOOT_SEED, selected_system: str = "adapted") -> dict[str, Any]:
    """The pre-registered reading rule on the fresh clips: the paired selected-minus-frozen top-1 difference, a 95%
    percentile bootstrap interval resampling films (clusters) with replacement, and the McNemar discordant counts with
    the exact two-sided p. "adapted helps" only when the interval excludes 0 on the positive side."""
    if not (len(frozen_correct) == len(selected_correct) == len(films)) or not films:
        raise ValueError("frozen, selected and films must be equal-length, non-empty sequences")
    f = [bool(x) for x in frozen_correct]
    s = [bool(x) for x in selected_correct]
    n = len(f)
    delta = (sum(s) - sum(f)) / n
    by_film: dict[str, list[int]] = {}
    for i, film in enumerate(films):
        by_film.setdefault(str(film), []).append(i)
    clusters = [by_film[k] for k in sorted(by_film)]
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        idx = [i for _c in range(len(clusters)) for i in clusters[rng.randrange(len(clusters))]]
        draws.append((sum(s[i] for i in idx) - sum(f[i] for i in idx)) / len(idx))
    draws.sort()
    lo, hi = draws[int(0.025 * n_boot)], draws[min(n_boot - 1, int(math.ceil(0.975 * n_boot)) - 1)]
    b = sum(1 for x, y in zip(f, s, strict=True) if x and not y)
    c = sum(1 for x, y in zip(f, s, strict=True) if y and not x)
    # Amendment 1, A5: cluster-adjusted McNemar (Durkalski et al. 2003), films as clusters; descriptive only.
    per_film = [sum(int(s[i]) - int(f[i]) for i in idx) for idx in clusters]
    denom = sum(d * d for d in per_film)
    x2 = (sum(per_film) ** 2 / denom) if denom else 0.0
    if selected_system == "frozen":
        verdict = "no demonstrated gain (the frozen model was kept on validation)"
    elif lo > 0:
        verdict = "adapted helps"
    elif hi < 0:
        verdict = "adapted is worse"
    else:
        verdict = f"no demonstrated gain (not detectable at this size: a true gain below about {MDE_RANGE[0]:.2f}-{MDE_RANGE[1]:.2f} often reads this way)"
    return {"n": n, "films": len(clusters), "frozen_top1": sum(f) / n, "selected_top1": sum(s) / n, "delta_top1": delta,
            "ci95": [lo, hi], "n_boot": n_boot, "seed": seed, "resampled": "films (strict film key)",
            "mcnemar": {"frozen_only_correct": b, "selected_only_correct": c, "exact_p_two_sided": _binom_two_sided(b, c),
                        "clustered_chi2": x2, "clustered_p": math.erfc(math.sqrt(x2 / 2)),
                        "note": "descriptive only: the exact p assumes independent clips; the clustered p (Durkalski 2003) treats films as clusters; the verdict is the bootstrap interval"},
            "mde_80pct_power": list(MDE_RANGE),
            "selected_system": selected_system, "verdict": verdict}
