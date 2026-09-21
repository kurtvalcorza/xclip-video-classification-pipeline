"""Model-backed checks that run only where the pinned snapshot is staged: the measured model facts, batched
classification equal to single-clip classification, the fusion-head logits on cached tower features equal to the
full forward, corpus evaluation on synthetic moving-shape clips, a short adaptation of the fusion head, the artifact
round trip with reload parity, the loader's scope check, the transactional guarantee and — where CUDA is visible —
the same path on the accelerator. Skipped when the weights are absent."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pytest
from PIL import Image, ImageDraw

from xclip_video_classification_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    HEAD_PARAMETERS,
    NUM_FRAMES,
    PARAMETER_COUNT,
    WEIGHTS_FILE,
    XClipVideoClassificationPipeline,
)
from xclip_video_classification_pipeline.pipeline import _TRAINABLE_PREFIXES

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
if not (DEFAULT_WEIGHTS_DIR / WEIGHTS_FILE).is_file():
    pytest.skip("snapshot not staged", allow_module_level=True)

LABELS = ["a red square", "a blue circle", "a green triangle"]


def _clip(i, label_index, size=(256, 192)):
    """A shape of the class colour drifting across `NUM_FRAMES` white frames; clip `i` fixes its start and speed."""
    frames = []
    rng = np.random.default_rng(i)
    x0, y0 = int(rng.integers(8, 80)), int(rng.integers(8, 60))
    dx, dy = int(rng.integers(4, 14)), int(rng.integers(-4, 8))
    for k in range(NUM_FRAMES):
        image = Image.new("RGB", size, (255, 255, 255))
        draw = ImageDraw.Draw(image)
        x, y = x0 + k * dx, y0 + k * dy
        if label_index == 0:
            draw.rectangle((x, y, x + 72, y + 72), fill=(220, 30, 30))
        elif label_index == 1:
            draw.ellipse((x, y, x + 72, y + 72), fill=(30, 60, 220))
        else:
            draw.polygon([(x, y + 72), (x + 36, y), (x + 72, y + 72)], fill=(30, 160, 60))
        image.putpixel((i % size[0], 0), (i % 256, 0, 0))
        frames.append(image)
    return frames


def _record(i):
    label_index = i % len(LABELS)
    return {"id": f"clip{i:02d}", "frames": _clip(i, label_index), "label": LABELS[label_index], "source": f"src{i}"}


@pytest.fixture(autouse=True)
def _release_memory():
    yield
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def records():
    return [_record(i) for i in range(24)]


@pytest.fixture(scope="module")
def pipe():
    return XClipVideoClassificationPipeline.from_pretrained(weights_dir=DEFAULT_WEIGHTS_DIR)


def test_model_facts_batched_classification_head_parity_and_frozen_evaluation(pipe, records):
    assert sum(p.numel() for p in pipe._model.parameters()) == PARAMETER_COUNT
    assert sum(p.numel() for n, p in pipe._model.named_parameters() if n.startswith(_TRAINABLE_PREFIXES)) == HEAD_PARAMETERS
    assert pipe.weight_sha256 is not None and len(pipe.weight_sha256) == 64
    single = [pipe.classify(r["frames"], LABELS) for r in records[:4]]
    batched = pipe.classify_batch([r["frames"] for r in records[:4]], LABELS, batch_size=4)
    for s, b in zip(single, batched, strict=True):
        assert s["top1"] == b["top1"] and abs(s["predictions"][0]["probability"] - b["predictions"][0]["probability"]) < 1e-4
    # the fusion head on cached tower features equals the full forward
    cls_features, patch_features = pipe._encode_clips(records[:3], 8)
    text_features = pipe._encode_labels(LABELS)
    with torch.no_grad():
        head = pipe._head_logits(cls_features, patch_features, text_features).float().cpu().numpy()
    full = np.asarray(pipe._batch_runner([r["frames"] for r in records[:3]], LABELS))
    assert np.abs(head - full).max() < 1e-3
    metrics = pipe.evaluate(records[:12], LABELS)
    assert metrics["n"] == 12 and metrics["n_labels"] == 3 and metrics["adapted"] is False and metrics["verdict"] == "measured-small-sample"
    assert metrics["top1_accuracy"] > 1 / 3 and len(metrics["rankings"]) == 12 and set(metrics["confusion"]) == set(LABELS)


def test_short_adaptation_and_artifact_round_trip(pipe, records, tmp_path):
    result = pipe.adapt(records[:18], records[18:], LABELS, epochs=2, lr=2e-5, batch_size=6)
    assert result["n_trainable"] == HEAD_PARAMETERS and result["n_total"] == PARAMETER_COUNT and result["labels"] == LABELS
    assert result["history"][0]["note"] == "frozen model" and result["history"][1]["train_loss"] > 0.0
    assert set(result["history"][1]["val"]) == {"top1_accuracy", "top3_accuracy", "macro_recall", "macro_f1", "n"} and result["best_epoch"] in (0, 1, 2)
    assert all(n.startswith(_TRAINABLE_PREFIXES) for n in result["trainable_names"])
    assert not any(n.startswith(("vision_model", "text_model", "text_projection", "logit_scale")) for n in result["trainable_names"])
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["tensors"]) == len(result["trainable_names"]) and manifest["base"]["weight_sha256"] == pipe.weight_sha256
    assert manifest["metadata"] == {"note": "test"} and manifest["adapter"]["selection"] == "highest validation top-1 accuracy" and manifest["adapter"]["labels"] == LABELS
    reloaded = XClipVideoClassificationPipeline.from_artifact(artifact, weights_dir=DEFAULT_WEIGHTS_DIR)
    a = pipe.classify_batch([r["frames"] for r in records[:4]], LABELS)
    b = reloaded.classify_batch([r["frames"] for r in records[:4]], LABELS)
    assert [x["top1"] for x in a] == [x["top1"] for x in b]
    assert all(abs(x["predictions"][0]["probability"] - y["predictions"][0]["probability"]) < 1e-5 for x, y in zip(a, b, strict=True))
    assert reloaded.adapter["best_epoch"] == result["best_epoch"] and reloaded.evaluate(records[:12], LABELS)["adapted"] is True
    assert not any(p.requires_grad for p in pipe._model.parameters())


def test_no_validation_keeps_the_final_epoch_and_reloads_it(pipe, records, tmp_path):
    result = pipe.adapt(records[:18], None, LABELS, epochs=2, batch_size=6)
    assert result["best_epoch"] == 2 == result["epochs"] and result["selection"].startswith("final epoch")
    assert all(entry["val"] is None for entry in result["history"]) and len(result["history"]) == 3
    artifact = pipe.save_artifact(tmp_path / "final")
    reloaded = XClipVideoClassificationPipeline.from_artifact(artifact, weights_dir=DEFAULT_WEIGHTS_DIR)
    state, other = pipe._model.state_dict(), reloaded._model.state_dict()
    assert all(torch.equal(state[name], other[name]) for name in result["trainable_names"])


def test_adapt_refuses_bad_hyperparameters_and_datasets(pipe, records):
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(records[:18], None, LABELS, epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(records[:18], None, LABELS, epochs=1, lr=0.5)
    with pytest.raises(ValueError, match="batch_size"):
        pipe.adapt(records[:18], None, LABELS, epochs=1, batch_size=0)
    with pytest.raises(ValueError, match="16..5000"):
        pipe.adapt(records[:6], None, LABELS, epochs=1)
    with pytest.raises(ValueError, match="not in the label set"):
        pipe.adapt(records[:18], None, LABELS[:2], epochs=1)
    assert not any(p.requires_grad for p in pipe._model.parameters())


def test_load_artifact_refuses_a_tensor_set_that_differs_from_the_recorded_configuration(pipe, records, tmp_path):
    from safetensors.torch import load_file, save_file

    pipe.adapt(records[:18], None, LABELS, epochs=1, batch_size=6)
    artifact = pipe.save_artifact(tmp_path / "ok")
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    fewer = tmp_path / "fewer"
    shutil.copytree(artifact, fewer)
    (fewer / "manifest.json").write_text(json.dumps({**manifest, "tensors": manifest["tensors"][:-1]}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        XClipVideoClassificationPipeline.from_artifact(fewer, weights_dir=DEFAULT_WEIGHTS_DIR)
    extra = tmp_path / "extra"
    shutil.copytree(artifact, extra)
    tensors = load_file(str(extra / "adapter.safetensors"))
    tensors["mit.zz_extra"] = torch.zeros(1)
    save_file(tensors, str(extra / "adapter.safetensors"), metadata={"format": "pt"})
    digest = hashlib.sha256((extra / "adapter.safetensors").read_bytes()).hexdigest()
    files = [{**manifest["files"][0], "bytes": (extra / "adapter.safetensors").stat().st_size, "sha256": digest}]
    (extra / "manifest.json").write_text(json.dumps({**manifest, "files": files}))
    with pytest.raises(ValueError, match="tensor names differ"):
        XClipVideoClassificationPipeline.from_artifact(extra, weights_dir=DEFAULT_WEIGHTS_DIR)
    tower = tmp_path / "tower"
    shutil.copytree(artifact, tower)
    (tower / "manifest.json").write_text(json.dumps({**manifest, "tensors": [*manifest["tensors"], "vision_model.encoder.layers.0.x"]}))
    with pytest.raises(ValueError, match="fusion head"):
        XClipVideoClassificationPipeline.from_artifact(tower, weights_dir=DEFAULT_WEIGHTS_DIR)


def test_adapt_is_transactional_when_the_progress_callback_raises(pipe, records):
    before = {k: v.clone() for k, v in pipe._model.state_dict().items()}
    adapter_before = pipe.adapter

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        pipe.adapt(records[:18], None, LABELS, epochs=2, batch_size=6, progress=boom)
    after = pipe._model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert pipe.adapter is adapter_before
    assert not any(p.requires_grad for p in pipe._model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not visible")
def test_the_default_device_is_cuda_when_visible(pipe):
    assert pipe.device == "cuda:0" and next(pipe._model.parameters()).device.type == "cuda"
