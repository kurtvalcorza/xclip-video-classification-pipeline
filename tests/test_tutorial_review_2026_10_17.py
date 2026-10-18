"""Regression tests for the 2026-10-17 B1 review (R113-M1, M2, m1-m8, S1, S2) and the owner rulings R113-D1/D2.

Only CI's dependencies; no model. Notebook cells run with stand-ins (tests/_section8.py): plumbing evidence only.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from pathlib import Path

import pytest

from _section8 import FRESH, StandInPipe, adapter, fingerprint, grid, namespace
from xclip_video_classification_pipeline import metrics as mt
from xclip_video_classification_pipeline import pipeline as pl
from xclip_video_classification_pipeline import protocol as pr
from xclip_video_classification_pipeline import samples as sm

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "xclip_video_classification_colab.ipynb"
STEM = "xclip_video_classification"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cell(notebook: dict, marker: str) -> str:
    found = [c["source"] for c in notebook["cells"] if c["cell_type"] == "code" and marker in c["source"]]
    assert len(found) == 1, marker
    return found[0]


def _markdown(notebook: dict) -> str:
    return "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")


def _section8(notebook):
    return _cell(notebook, "settings_used = {")


def _run8(notebook, tmp_path, monkeypatch, pipe=None, grid_record="default", **kw):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir(exist_ok=True)
    monkeypatch.delenv(pr.MASK_ENV, raising=False)
    pipe = pipe or StandInPipe(adapter())
    fetched: list = []
    ns = namespace(pipe, grid() if grid_record == "default" else grid_record, fetched, **kw)
    exec(_section8(notebook), ns)
    return ns, fetched, pipe


def _report(tmp_path, suffix=""):
    return tmp_path / "outputs" / f"{STEM}_evaluation_report{suffix}.json"


# --- R113-M2: score-once wiring ------------------------------------------------------------------------------


def test_m2_default_run_scores_the_fresh_test_once_as_the_preregistered_evaluation(notebook, tmp_path, monkeypatch):
    ns, fetched, pipe = _run8(notebook, tmp_path, monkeypatch)
    report = json.loads(_report(tmp_path).read_text(encoding="utf-8"))
    assert report["test_look"] == pr.TEST_LOOKS[0] and report["preregistered"] is True and report["settings_differing"] == []
    assert len(fetched) == 1 and pipe.fresh_calls == 2
    reading = report["fresh_test"]["reading_rule"]
    assert reading["n"] == len(FRESH) and reading["n_boot"] == pr.N_BOOT and reading["seed"] == pr.BOOT_SEED
    assert set(reading["mcnemar"]) == {"frozen_only_correct", "selected_only_correct", "exact_p_two_sided"}
    assert report["recorded_evaluation"] is None


def test_m2_rerunning_section_8_is_a_reproduction_and_never_overwrites(notebook, tmp_path, monkeypatch):
    _run8(notebook, tmp_path, monkeypatch)
    first = _report(tmp_path).read_bytes()
    _run8(notebook, tmp_path, monkeypatch)
    _run8(notebook, tmp_path, monkeypatch)
    assert _report(tmp_path).read_bytes() == first
    for n in (1, 2):
        assert json.loads(_report(tmp_path, f"_reproduction_{n}").read_text(encoding="utf-8"))["test_look"] == pr.TEST_LOOKS[1]


@pytest.mark.parametrize(
    "change",
    [
        {"pipe": StandInPipe(adapter(epochs=3))},
        {"pipe": StandInPipe(adapter(batch_size=8))},
        {"pipe": StandInPipe(adapter(seed=1))},
        {"pipe": StandInPipe(adapter(lr=1e-4))},
        {"pipe": StandInPipe(adapter(min_improvement=0.0))},
        {"split_seed": 7},
        {"data_source": "BYOD"},
        {"pipe": StandInPipe(adapter(), head="other")},
    ],
    ids=["epochs", "batch", "seed", "lr", "keep-frozen-margin", "split-seed", "data-source", "model-state"],
)
def test_m2_a_changed_setting_never_scores_the_fresh_test(notebook, tmp_path, monkeypatch, change):
    ns, fetched, pipe = _run8(notebook, tmp_path, monkeypatch, **change)
    assert fetched == [] and pipe.fresh_calls == 0
    assert not _report(tmp_path).exists()
    report = json.loads(_report(tmp_path, "_exploratory_1").read_text(encoding="utf-8"))
    assert report["test_look"] == pr.TEST_LOOKS[2] and report["preregistered"] is False and report["settings_differing"]
    assert report["fresh_test"] is None


@pytest.mark.parametrize("grid_record", [None, grid(lrs=(1e-5, 5e-6)), grid(epochs=3)], ids=["missing", "partial", "other-settings"])
def test_m2_a_missing_partial_or_stale_grid_never_scores_the_fresh_test(notebook, tmp_path, monkeypatch, grid_record):
    ns, fetched, pipe = _run8(notebook, tmp_path, monkeypatch, grid_record=grid_record)
    assert fetched == [] and pipe.fresh_calls == 0 and not _report(tmp_path).exists()
    assert any("grid" in d for d in json.loads(_report(tmp_path, "_exploratory_1").read_text(encoding="utf-8"))["settings_differing"])


def test_m2_masked_development_check_does_not_fetch_the_fresh_test(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    monkeypatch.setenv(pr.MASK_ENV, "1")
    fetched: list = []
    pipe = StandInPipe(adapter())
    exec(_section8(notebook), namespace(pipe, grid(), fetched))
    assert fetched == [] and pipe.fresh_calls == 0
    assert json.loads(_report(tmp_path, "_exploratory_1").read_text(encoding="utf-8"))["test_look"] == pr.TEST_LOOKS[3]


def test_m2_section_8_checks_everything_before_touching_the_fresh_test(notebook):
    source = _section8(notebook)
    fetch = source.index("fetch_fresh_test(")
    for check in ("settings_differing(", "model_state_problems(", "head_digest()", "look_at_test("):
        assert source.index(check) < fetch, check


def test_m2_section_10_is_exploratory_and_a_run_after_it_is_a_reproduction(notebook, tmp_path, monkeypatch):
    section10 = _cell(notebook, "RUN_EXPERIMENT = False")
    assert "fetch_fresh_test" not in section10 and "fresh" not in section10.replace("fresh test is never read", "").replace("never the fresh test", "")
    assert "TEST_LOOKS[2]" in section10
    _run8(notebook, tmp_path, monkeypatch)
    ns, fetched, pipe = _run8(notebook, tmp_path, monkeypatch)  # Section 8 again after Section 10
    assert ns["test_look"] == pr.TEST_LOOKS[1] and _report(tmp_path, "_reproduction_1").is_file()


def test_m2_grid_is_on_by_default_and_matches_the_preregistration(notebook):
    source = _cell(notebook, "RUN_GRID = True")
    assert "LEARNING_RATE = 0.0  # @param" in source and "EPOCHS = 8  # @param" in source and "BATCH_SIZE = 16  # @param" in source
    assert "pipe.load_head_state(" in source and "min_improvement=min_improvement" in source
    assert pr.GRID_LRS == (1e-5, 5e-6, 2e-5) and pr.DEFAULT_LR == 1e-5 and pr.GRID_MARGIN_CLIPS == 2 == pr.KEEP_FROZEN_MARGIN_CLIPS
    prereg = (ROOT / pr.PREREGISTRATION).read_text(encoding="utf-8")
    for needle in ("1e-5", "5e-6", "2e-5", "8 epochs", "two clips", sm.FRESH_TEST_LIST_SHA256, str(pr.N_BOOT), str(pr.BOOT_SEED)):
        assert needle in prereg, needle


def test_d2_selection_rule_margin_and_ties():
    fp = fingerprint()
    assert pr.select_lr({1e-5: 50, 5e-6: 51, 2e-5: 51}, fingerprint=fp)["selected_lr"] == 1e-5  # +1 clip is not enough
    assert pr.select_lr({1e-5: 50, 5e-6: 52, 2e-5: 49}, fingerprint=fp)["selected_lr"] == 5e-6
    assert pr.select_lr({1e-5: 48, 5e-6: 50, 2e-5: 51}, fingerprint=fp)["selected_lr"] == 2e-5
    assert pr.select_lr({1e-5: 48, 5e-6: 50, 2e-5: 50}, fingerprint=fp)["selected_lr"] == 5e-6  # equal challengers: smaller lr
    with pytest.raises(ValueError, match="cover exactly"):
        pr.select_lr({1e-5: 50, 5e-6: 52})


def test_d2_adapt_has_the_keep_frozen_margin():
    sig = inspect.signature(pl.XClipVideoClassificationPipeline.adapt)
    assert sig.parameters["min_improvement"].default == 0.0
    source = inspect.getsource(pl.XClipVideoClassificationPipeline.adapt)
    assert "best_acc - frozen_acc >= min_improvement" in source and '"min_improvement": min_improvement' in source


# --- R113-m6: atomic report writes ---------------------------------------------------------------------------


def test_m6_an_exception_mid_dump_leaves_the_previous_report_byte_identical(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    pr.write_json_atomic(path, {"a": 1})
    before = path.read_bytes()

    def boom(*args, **kwargs):
        raise RuntimeError("mid-dump")

    monkeypatch.setattr(pr.json, "dump", boom)
    with pytest.raises(RuntimeError):
        pr.write_json_atomic(path, {"a": 2})
    assert path.read_bytes() == before and sorted(p.name for p in tmp_path.iterdir()) == ["report.json"]


@pytest.mark.parametrize("prior", [b"", b"{not json", b"\xff\xfe"], ids=["empty", "corrupt", "binary"])
def test_m6_a_corrupt_or_empty_prior_report_does_not_crash(notebook, tmp_path, monkeypatch, prior):
    (tmp_path / "outputs").mkdir()
    _report(tmp_path).write_bytes(prior)
    (tmp_path / "outputs" / f"{STEM}_test_looks.jsonl").write_bytes(prior)
    ns, fetched, pipe = _run8(notebook, tmp_path, monkeypatch)
    assert _report(tmp_path).read_bytes() == prior  # never overwritten
    assert ns["test_look"] == pr.TEST_LOOKS[1]


def test_m6_no_cell_writes_json_with_open_w():
    template = (ROOT / "tools" / "notebook_template.py").read_text(encoding="utf-8")
    assert "json.dump(" not in template and "write_json_atomic(" in template


# --- R113-m1: the reading rule ---------------------------------------------------------------------------------


def test_m1_reading_rule_bootstraps_films_and_reads_only_an_interval_excluding_zero():
    films = [f"f{i // 3}" for i in range(30)]
    helps = pr.paired_reading([False] * 30, [True] * 30, films)
    assert helps["verdict"] == "adapted helps" and helps["ci95"][0] > 0 and helps["mcnemar"]["selected_only_correct"] == 30
    mixed = pr.paired_reading([i % 2 == 0 for i in range(30)], [i % 3 == 0 for i in range(30)], films)
    assert mixed["verdict"] in ("no demonstrated gain", "adapted is worse") and mixed["films"] == 10
    assert pr.paired_reading([True] * 30, [True] * 30, films, selected_system="frozen")["verdict"].startswith("no demonstrated gain")
    assert pr.paired_reading([True, False], [False, True], ["x", "y"]) == pr.paired_reading([True, False], [False, True], ["x", "y"])  # fixed seed


# --- R113-D1: the fresh test --------------------------------------------------------------------------------------


def test_d1_fresh_test_is_pinned_and_disjoint_by_construction():
    assert sm.FRESH_FILE == "default/train/0000.parquet" and sorted(sm.FRESH_ROW_GROUP_PINS) == [0, 1, 2]
    assert sm.FRESH_URL.startswith(f"https://huggingface.co/datasets/mteb/HMDB51/resolve/{sm.CORPUS_REVISION}/")
    assert len(sm.FRESH_EXCLUDED_FILMS) == 8 and 682 - len(sm.FRESH_SCREEN_EXCLUDED) == sm.FRESH_TEST_CLIPS
    assert sm.film_of("Finding_Forrester_3_catch_f_cm_np1_fr_med_0.avi", "catch") == "finding_forrester"
    assert sm.film_of("TheBoondockSaints_drink_u_cm_np1_ba_goo_2.avi", "drink") == "theboondocksaints"


def test_d1_fresh_clip_list_refuses_anything_but_the_pinned_list():
    rows = {0: [{"video": b"x" * 10, "path": "Some_Film_catch_f_cm_np1_fr_med_0.avi", "label": sm.HMDB51_CLASSES.index("catch")}]}
    with pytest.raises(ValueError, match="fresh test list"):
        sm.fresh_test_clips(rows)
    line = sm.fresh_clip_list_digest([{"row_group": 0, "index": 0, "path": "p.avi", "size": 3, "sha256": "ab"}])
    assert line == hashlib.sha256(b"0\t0\tp.avi\t3\tab\n").hexdigest()


def test_m2r_film_overlap_reports_films_shared_across_splits():
    splits = {"train": [{"film": "theboondocksaints"}, {"film": "a"}], "validation": [{"film": "theboondocksaints"}], "test": [{"film": "b"}, {}]}
    out = sm.film_overlap(splits)
    assert out["films_shared"] == {"theboondocksaints": ["train", "validation"]} and out["clips_in_shared_films"] == 2


# --- R113-S1: majority baseline tie handling ----------------------------------------------------------------------


def test_s1_majority_baseline_is_order_independent_under_a_tie():
    labels = ["w", "x", "y", "z"]
    train = [{"label": lab} for lab in ["x"] * 3 + ["y"] * 3 + ["z"]]
    test = [{"label": lab} for lab in ["x", "y", "y", "w", "z"]]
    a = mt.majority_baseline(train, test, labels)
    b = mt.majority_baseline(train, test, list(reversed(labels)))
    for key in ("top1_accuracy", "top3_accuracy", "macro_recall", "macro_f1"):
        assert a[key] == pytest.approx(b[key]), key
    assert a["top1_accuracy"] == pytest.approx((1 / 5 + 2 / 5) / 2) and a["tied_labels"] == ["x", "y"]
    assert a["top3_accuracy"] == pytest.approx((1 + 1 + 1 + 0 + 1) / 5)


# --- R113-M1 / m3 / m4 / m7: prose, labels, spec ----------------------------------------------------------------


def test_m1_no_unqualified_never_used_for_selection_claim_and_the_sweep_is_named(notebook):
    text = _markdown(notebook)
    assert "test clips were never used for training or epoch selection" not in text
    assert "exposed development test" in text and "read the 66 test clips' scores for five learning rates" in text


def test_m3_every_quoted_run_is_the_current_blobs_colab_run(notebook):
    text = json.dumps(notebook)
    assert "Kaggle T4 release run" not in text and "release run" not in text.replace("release runtime", "")
    assert "the Colab T4 run of blob 41caa483 (commit adcb4e8, 2026-10-09)" in text


def test_m4_spec_25_kind_and_use_notice(notebook):
    meta = notebook["metadata"]["dimer"]
    assert meta["notebook_spec"] == "2.5" and meta["notebook_kind"] == "TUTORIAL"
    header = notebook["cells"][0]["source"]
    assert "**Kind:** `TUTORIAL`" in header and "For learning, research and evaluation only" in header
    assert header.index("Notice.") < header.index("**Profile:**")


def test_m7_no_exactly_claim_for_the_head_logits(notebook):
    assert "the logits equal the full model's exactly" not in json.dumps(notebook)
    assert "full model's exactly" not in inspect.getsource(pl.XClipVideoClassificationPipeline.adapt)


# --- R113-m8: BYOD refusal ------------------------------------------------------------------------------------------


def test_m8_a_refused_byod_run_leaves_no_report_under_the_canonical_names(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "outputs"
    out.mkdir()
    (out / f"{STEM}_evaluation_report.json").write_text("{}", encoding="utf-8")
    (out / f"{STEM}_result.json").write_text("{}", encoding="utf-8")
    source = _cell(notebook, "USE_BYOD = False").replace("USE_BYOD = False  # @param", "USE_BYOD = True  # @param", 1).replace("BYOD_PATH = ''  # @param", "BYOD_PATH = 'missing.zip'  # @param", 1)
    ns = {"os": __import__("os"), "Path": Path, "__name__": "__main__", "pipe": type("P", (), {"adapter": None, "restore_base": lambda self: []})()}
    for module in (pl, mt, sm):
        ns.update({k: getattr(module, k) for k in dir(module) if not k.startswith("__")})
    with pytest.raises(FileNotFoundError):
        exec(source, ns)
    assert not list(out.glob(f"{STEM}_*")) and (out / "previous_run_1" / f"{STEM}_evaluation_report.json").is_file()


# --- R113-m5 / S2: provenance and local paths -------------------------------------------------------------------------


def test_m5_stamped_revision_carries_the_embedded_modules(notebook):
    meta = notebook["metadata"]["dimer"]["generated_from"]
    try:
        for rel in meta["modules"]:
            blob = subprocess.run(["git", "-C", str(ROOT), "show", f"{meta['revision']}:{rel}"], capture_output=True, check=True).stdout
            assert blob.replace(b"\r\n", b"\n") == (ROOT / rel).read_bytes().replace(b"\r\n", b"\n"), rel
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("stamped revision not in this checkout (shallow clone)")


def test_s2_no_local_absolute_paths_in_committed_evidence():
    for path in (ROOT / "docs").rglob("*"):
        if path.is_file() and path.suffix in (".log", ".json", ".md", ".ipynb"):
            text = path.read_text(encoding="utf-8", errors="replace")
            for needle in ("/home/kurt", "Users\\Kurt", "Users/Kurt", "KURTVA~1"):
                assert needle not in text, f"{path.name}: {needle}"


def test_m2r_validation_keeps_the_film_key_so_overlap_checks_see_it():
    from PIL import Image

    frames = [Image.new("RGB", (32, 24), (i, 0, 0)) for i in range(8)]
    records = [{"id": f"c{i}", "frames": frames, "label": "brushing hair" if i % 2 else "chewing", "source": f"s{i}", "film": f"f{i}"} for i in range(16)]
    checked = sm.validate_dataset(records, ["brushing hair", "chewing"], min_per_class=1)["records"]
    assert [r["film"] for r in checked] == [f"f{i}" for i in range(16)]
