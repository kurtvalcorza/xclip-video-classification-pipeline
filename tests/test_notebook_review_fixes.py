"""Regression tests for the 2026-10-05 notebook review findings (XCL-M1..M3, XCL-m1..m3).

Every test needs only CI's dependencies and no model: the notebook's own cell sources are executed with stand-ins
where a model would be needed, and restore_base() is exercised on a stand-in model, not the checkpoint. Stand-in evidence is plumbing evidence, not model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

from xclip_video_classification_pipeline import samples as sm

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "xclip_video_classification_colab.ipynb"
LOCK = ROOT / "tutorials" / "requirements-colab.lock.txt"
PIPELINE = ROOT / "src" / "xclip_video_classification_pipeline" / "pipeline.py"
STEM = "xclip_video_classification"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(notebook: dict, marker: str) -> str:
    found = [c["source"] for c in _code_cells(notebook) if marker in c["source"]]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _markdown(notebook: dict) -> str:
    return "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")


# --- XCL-M1: no in-kernel install, no restart, idempotent Section 1 ------------------------------------------


def test_xcl_m1_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(notebook):
    code = "\n".join(c["source"] for c in _code_cells(notebook))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "Restart the runtime" not in json.dumps(notebook)
    kernel = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in c["source"]]
    assert len(kernel) == 1, "exactly one cell may run in the kernel"
    source = kernel[0]["source"]
    for needed in ("'--require-hashes', '--only-binary', ':all:'", "'--managed-python'", "UV_SHA256", "LOCK_SHA256", 'MPLBACKEND="Agg"', '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"'):
        assert needed in source


def test_xcl_m1_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(notebook):
    source = _cell(notebook, "# dimer: kernel cell")
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    spec = importlib.util.spec_from_file_location("_review_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    build.check_lock(build._pins(ROOT), lock_text)


def test_xcl_m1_section_1_is_idempotent_and_keeps_the_live_worker(notebook, tmp_path, monkeypatch, capsys):
    """The real Section 1 cell, run twice with a stand-in interpreter: the matching environment is reused (no
    download) and the live worker — with every variable later cells created — is kept."""
    source = _cell(notebook, "# dimer: kernel cell")
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = types.SimpleNamespace(input_transformers_cleanup=[])
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    def no_download(*args, **kwargs):
        raise AssertionError("a matching environment must be reused, not downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(source, "<section 1>", "exec"), namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert "'reused': True" in capsys.readouterr().out
        runtime.run("learner_value = 41 + 1\n")
        exec(compile(source, "<section 1>", "exec"), namespace)  # the learner re-runs Section 1 on its own
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("print('value', learner_value)\n")
        assert "value 42" in capsys.readouterr().out
        assert namespace["_route_to_isolated_runtime"](["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([source]) == [source]
    finally:
        runtime.close()


# --- XCL-M2: every adaptation and every frozen evaluation starts from the pinned base -------------------------


def test_xcl_m2_adapt_and_load_artifact_restore_the_base_first():
    """Torch-backed, so the order inside adapt is checked statically: pre-call state kept, base restored, then epoch 0."""
    text = PIPELINE.read_text(encoding="utf-8")
    adapt = text[text.index("    def adapt(") : text.index("    def save_artifact(")]
    order = [adapt.index(m) for m in ("backup = {name: param.detach().clone()", "restored = self.restore_base()", "self._remember_base(head_names)", '"note": "frozen model"', "for epoch in range(1, epochs + 1):")]
    assert order == sorted(order)
    failure = adapt[adapt.index("except BaseException:") :]
    assert failure.index("param.copy_(backup[name])") < failure.index("self.adapter = previous_adapter") < failure.index("raise")
    assert '"started_from": "pinned base"' in adapt
    load = text[text.index("    def load_artifact(") : text.index("    def from_artifact(")]
    assert load.index("self.restore_base()") < load.index("self._remember_base(sorted(tensors))") < load.index("model.load_state_dict(")


class _Tensor:
    def __init__(self, value):
        self.value = np.array(value, dtype=float)

    def detach(self):
        return self

    def clone(self):
        return _Tensor(self.value.copy())


class _Model:
    def __init__(self):
        self.state = {"mit.w": _Tensor([1.0, 2.0]), "prompts_generator.w": _Tensor([3.0]), "vision_model.w": _Tensor([4.0])}

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, values, strict=True):
        assert strict is False
        for name, tensor in values.items():
            self.state[name] = _Tensor(tensor.value.copy())

    def eval(self):
        return self


def test_xcl_m2_restore_base_undoes_every_earlier_change_stand_in():
    """restore_base() and _remember_base() on a stand-in model with the state_dict interface (numpy, not torch)."""
    from xclip_video_classification_pipeline import XClipVideoClassificationPipeline

    model = _Model()
    pipe = XClipVideoClassificationPipeline(_runner=lambda *a: None, device="cpu", _model=model, _processor=object())
    assert pipe.restore_base() == []
    pipe._remember_base(["mit.w", "prompts_generator.w"])
    model.state["mit.w"] = _Tensor([9.0, 9.0])
    model.state["prompts_generator.w"] = _Tensor([9.0])
    pipe._remember_base(["mit.w"])  # a second run keeps the first (base) value
    pipe.adapter = {"best_epoch": 2}
    assert pipe.restore_base() == ["mit.w", "prompts_generator.w"]
    assert model.state["mit.w"].value.tolist() == [1.0, 2.0] and model.state["prompts_generator.w"].value.tolist() == [3.0]
    assert model.state["vision_model.w"].value.tolist() == [4.0] and pipe.adapter is None


def test_xcl_m2_rerun_from_section_4_restores_the_base_and_the_experiment_has_its_own_pipeline(notebook):
    section_4 = _cell(notebook, "USE_BYOD = False")
    assert section_4.index("restored_tensors = pipe.restore_base()") < section_4.index("if USE_BYOD:")
    assert section_4.index("restored_tensors = pipe.restore_base()") < section_4.index("splits = build_sample_dataset(corpus, seed=SPLIT_SEED)")
    experiment = _cell(notebook, "RUN_EXPERIMENT = False")
    assert "experiment_pipe = XClipVideoClassificationPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=pipe.device)" in experiment
    assert f"Path('outputs/{STEM}_experiment')" in experiment
    assert "raise RuntimeError(f'the experiment changed a default export: {unchanged}')" in experiment
    assert not re.search(r"(?<!experiment_)pipe\.adapt\(", experiment)
    assert "**Predict → Change one thing → Run → Observe → Explain**" in _markdown(notebook)
    assert "they do not affect the default path" not in _markdown(notebook)


# --- XCL-M3: guided layer and infrastructure labelling ----------------------------------------------------------


def test_xcl_m3_guided_layer_is_present(notebook):
    markdown = _markdown(notebook)
    for heading in ("**Who this notebook is for.**", "**Input → Model → Output.**", "**How to use this notebook.**", "**Roadmap:**", "## Troubleshooting", "## Glossary", "## Conclusion (your notes)", "## 10. Change one thing", "**Learner:**"):
        assert heading in markdown, heading
    assert markdown.count("**Predict") >= 7
    assert markdown.count("<details><summary>Check your reasoning</summary>") >= 7
    assert markdown.count("**What to notice:**") >= 6
    assert "restart the runtime and rerun from the top" not in markdown


def test_xcl_m3_infrastructure_cells_are_labelled_and_collapsed(notebook):
    infra = [c for c in _code_cells(notebook) if c["metadata"].get("cellView") == "form"]
    assert len([c for c in infra if c["metadata"].get("dimer", {}).get("embedded_module")]) == 3
    titled = [c["source"].splitlines()[0] for c in infra if not c["metadata"].get("dimer")]
    assert len(titled) == 3 and all(t.startswith("# @title Infrastructure:") for t in titled), titled


def test_xcl_m3_no_template_placeholders_leak(notebook):
    learner = "\n".join(c["source"] for c in notebook["cells"] if not c.get("metadata", {}).get("dimer", {}).get("embedded_module"))
    for leftover in ("{{", "{MODEL_ID}", "{stem}", "@P:"):
        assert leftover not in learner, leftover
    assert "}}" not in _markdown(notebook)


# --- XCL-m1: quality outcomes are reported verdicts -------------------------------------------------------------


def test_xcl_m1_no_quality_assert_remains(notebook):
    code = "\n".join(c["source"] for c in _code_cells(notebook) if not c["metadata"].get("dimer", {}).get("embedded_module"))
    asserts = re.findall(r"(?m)^\s*assert .*$", code)
    assert len(asserts) == 1 and asserts[0].startswith("assert parity['identical_rankings'] == parity['of']")


METRICS = ("top1_accuracy", "top3_accuracy", "macro_recall", "macro_f1")


def _m(top1):
    return {**{k: top1 for k in METRICS}, "n": 4, "n_labels": 2, "label": "a", "mean_rank": 1.5, "verdict": "measured-small-sample", "definitions": {}, "per_label": {"a": {"recall": top1, "support": 2}, "b": {"recall": top1, "support": 2}}, "rankings": [], "top1_probability": [], "predictions": []}


def test_xcl_m1_a_worse_test_score_is_recorded_and_does_not_stop_the_notebook(notebook, tmp_path, monkeypatch):
    """Sections 6 and 8 with stand-ins where the adapted head loses test accuracy (a later epoch won on validation):
    both cells complete and record `worse` (stand-in evidence, no model)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    scores = iter([_m(0.75), _m(0.5), _m(0.9)])

    class StandIn:
        def evaluate(self, records, labels, batch_size=8):
            return next(scores)

    ns = {
        "pipe": StandIn(), "train_records": [], "val_records": [], "test_records": [], "time": __import__("time"), "json": json, "LABELS": ["a", "b"], "EVAL_BATCH_SIZE": 8,
        "chance_baseline": lambda *a: _m(0.5), "majority_baseline": lambda *a: _m(0.25),
        "MODEL_ID": "stand-in", "MODEL_REVISION": "0" * 40, "MODEL_KEY": "stand-in", "data_source": "stand-in", "dataset_manifests": {"test": {"digest": "d"}},
        "disjoint": {}, "adapt_result": {"history": [], "trainable_names": []}, "adapt_seconds": 0.0,
    }
    exec(_cell(notebook, "baseline_chance = chance_baseline("), ns)
    assert ns["frozen_verdict"] == "pre-trained model above both baselines"
    exec(_cell(notebook, "adapted_test = pipe.evaluate(test_records, LABELS"), ns)
    verdicts = json.loads((tmp_path / "outputs" / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))["comparison"]["verdicts"]
    assert verdicts["adapted_vs_frozen_top1_accuracy"] == "worse" and verdicts["adapted_above_both_baselines"] is False


# --- XCL-m2: Kinetics-400 overlap and the split sentence ------------------------------------------------------


def test_xcl_m2_kinetics_overlap_is_stated_and_the_leakage_sentence_matches_the_split(notebook):
    markdown = _markdown(notebook)
    assert "the pre-trained model scored on our label phrasing" in markdown
    assert "only `chewing` and `climbing stairs` have none" in markdown and "video level cannot be ruled out" in markdown
    assert "no clip and no source video appears in two splits" not in markdown
    assert "*for the same action*" in markdown


# --- XCL-m3: BYOD contract --------------------------------------------------------------------------------------


def _gif(i: int) -> bytes:
    from PIL import Image

    rng = np.random.default_rng(i)
    frames = [Image.fromarray(rng.integers(0, 255, (32, 40, 3), dtype=np.uint8)) for _ in range(8)]
    buffer = io.BytesIO()
    frames[0].save(buffer, "GIF", save_all=True, append_images=frames[1:], duration=40, loop=0)
    return buffer.getvalue()


def _zip(path: Path, per_label: int, labels=("waving", "jumping"), *, drop: str | None = None, shared_sources: bool = False, extra: dict[str, bytes] | None = None) -> Path:
    names = [(f"{label}{i}.gif", label, f"film{i}" if shared_sources else f"{label}-src{i}") for label in labels for i in range(per_label)]
    rows = "file,label,source\n" + "".join(f"{name},{label},{source}\n" for name, label, source in names)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("labels.csv", rows)
        for k, (name, _label, _source) in enumerate(names):
            if name != drop:
                archive.writestr(name, _gif(k))
        for name, data in (extra or {}).items():
            archive.writestr(name, data)
    return path


def test_xcl_m3_stated_minimum_is_what_the_split_accepts(tmp_path, notebook):
    assert sm.min_byod_records(2)["total"] == 24 and sm.min_byod_records(3)["total"] == 27
    split = sm.split_dataset(sm.load_byod_dataset(_zip(tmp_path / "ok.zip", 12)), seed=42)
    assert {k: len(v) for k, v in split.items()} == {"test": 4, "validation": 4, "train": 16}
    with pytest.raises(ValueError, match=r"the train split holds 14 clips; at least 16 are required — with 2 labels .* at least 24 distinct clips"):
        sm.split_dataset(sm.load_byod_dataset(_zip(tmp_path / "small.zip", 11)), seed=42)
    assert "**24 clips for two labels**" in _markdown(notebook)


def test_xcl_m3_a_source_with_two_labels_never_straddles_splits(tmp_path):
    records = sm.load_byod_dataset(_zip(tmp_path / "shared.zip", 15, shared_sources=True))
    for seed in range(6):
        splits = sm.split_dataset(records, seed=seed)
        assert sm.check_split_disjoint(splits)
        homes = {}
        for name, part in splits.items():
            for record in part:
                assert homes.setdefault(record["source"], name) == name


def test_xcl_m3_refusals_name_the_line_and_litter_is_skipped(tmp_path):
    with pytest.raises(ValueError, match=r"labels.csv line 5 \(file 'waving3.gif'\): names a missing clip"):
        sm.load_byod_dataset(_zip(tmp_path / "missing.zip", 12, drop="waving3.gif"))
    assert len(sm.load_byod_dataset(_zip(tmp_path / "mac.zip", 12, extra={"__MACOSX/._waving0.gif": b"\0"}))) == 24


def _section_4(notebook: dict, path: str) -> str:
    source = _cell(notebook, "USE_BYOD = False")
    source = source.replace("USE_BYOD = False  # @param", "USE_BYOD = True  # @param", 1)
    return source.replace("BYOD_PATH = ''  # @param", f"BYOD_PATH = {path!r}  # @param", 1)


def _section_4_namespace(restored: list) -> dict:
    from xclip_video_classification_pipeline import metrics as mt
    from xclip_video_classification_pipeline import pipeline as pl

    ns = {}
    for module in (pl, mt, sm):
        ns.update({k: getattr(module, k) for k in dir(module) if not k.startswith("__")})
    pipe = types.SimpleNamespace(adapter={"best_epoch": 2}, restore_base=lambda: restored.append(True) or ["a"])
    ns.update({"os": __import__("os"), "Path": Path, "pipe": pipe, "__name__": "__main__"})
    return ns


def test_xcl_m3_byod_path_runs_section_4_outside_colab_from_the_base(notebook, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _zip(tmp_path / "mine.zip", 13)
    restored: list = []
    ns = _section_4_namespace(restored)
    exec(_section_4(notebook, "mine.zip"), ns)
    out = capsys.readouterr().out
    assert restored == [True], "a BYOD re-run must put the model back to the pinned base first"
    assert ns["raw_rows"] == {"byod": 26, "labels": 2, "effective_minimum": 24}
    assert "fewer than 5 per label" in out
    assert (tmp_path / "outputs" / f"{STEM}_train.csv").is_file()


def test_xcl_m3_upload_outside_colab_cancelled_and_bad_path_are_explained(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google", None)
    with pytest.raises(RuntimeError, match="upload dialog exists only in Google Colab"):
        exec(_section_4(notebook, ""), _section_4_namespace([]))
    with pytest.raises(FileNotFoundError, match="BYOD_PATH 'nowhere.zip' does not exist"):
        exec(_section_4(notebook, "nowhere.zip"), _section_4_namespace([]))
    for uploaded, message in (({}, "received 0"), ({"a.zip": b"", "b.zip": b""}, "received 2")):
        google, colab, files = (types.ModuleType(n) for n in ("google", "google.colab", "google.colab.files"))
        files.upload = lambda uploaded=uploaded: uploaded
        colab.files, google.colab = files, colab
        for name, module in (("google", google), ("google.colab", colab), ("google.colab.files", files)):
            monkeypatch.setitem(sys.modules, name, module)
        with pytest.raises(ValueError, match=message):
            exec(_section_4(notebook, ""), _section_4_namespace([]))
