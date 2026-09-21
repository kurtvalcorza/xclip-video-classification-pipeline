"""Offline checks of the adaptation contract: the labelled-clip record contract and its refusals, the pinned-corpus
refusals and the source-grouped draw, splitting, the BYOD loader, the metrics and baselines, `classify_batch` /
`evaluate` with an injected runner, the artifact-manifest checks, and the model-free refusals of `adapt` / artifacts."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import io
import json
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from xclip_video_classification_pipeline import (
    ARTIFACT_FORMAT,
    HMDB51_CLASSES,
    MODEL_ID,
    MODEL_REVISION,
    NUM_FRAMES,
    SAMPLE_CLASS_TEXT,
    SAMPLE_LABELS,
    XClipVideoClassificationPipeline,
    build_sample_dataset,
    chance_baseline,
    check_split_disjoint,
    classification_metrics,
    clip_digest,
    dataset_digest,
    fetch_corpus,
    load_byod_dataset,
    majority_baseline,
    majority_label,
    read_corpus,
    source_of,
    split_dataset,
    validate_dataset,
    write_dataset_csv,
)
from xclip_video_classification_pipeline import pipeline as pl
from xclip_video_classification_pipeline import samples as sm

LABELS = ["brushing hair", "doing a cartwheel", "catching a ball", "chewing"]


def _frames(i, size=(64, 48), n=NUM_FRAMES):
    frames = []
    for k in range(n):
        image = Image.new("RGB", size, ((i * 37) % 256, (k * 29) % 256, 90))
        image.putpixel((i % size[0], k % size[1]), (255, 0, 0))
        frames.append(image)
    return frames


def _record(i, label=None, **extra):
    return {"id": f"c{i:03d}", "frames": _frames(i), "label": LABELS[i % len(LABELS)] if label is None else label, **extra}


def _records(n=16, sources=False):
    out = []
    for i in range(n):
        extra = {"source": f"video{i // 2}/{LABELS[i % len(LABELS)]}"} if sources else {}
        out.append(_record(i, **extra))
    return out


# --- record contract -------------------------------------------------------------------------------------------


def test_validate_dataset_accepts_records_and_reports_counts_and_digest():
    info = validate_dataset(_records(), LABELS)
    assert info["n_records"] == 16 and info["labels"] == LABELS and info["per_label"] == {label: 4 for label in LABELS}
    assert info["frame_width"] == {"min": 64, "max": 64} and info["frames_per_clip"] == NUM_FRAMES
    assert info["digest"] == dataset_digest(_records()) and info["model_id"] == MODEL_ID
    assert info["records"][0]["label"] == "brushing hair"
    # without a label set the labels are the ones the records carry
    assert validate_dataset(_records())["labels"] == sorted(LABELS)
    # labels are normalised like the candidate names
    assert validate_dataset([_record(0, label="  Brushing   Hair. ")] + _records()[1:], LABELS)["records"][0]["label"] == "brushing hair"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.pop("frames"), "missing 'frames'"),
        (lambda r: r.update(id="bad id!"), "id must match"),
        (lambda r: r.update(frames=r["frames"][:3]), "NUM_FRAMES"),
        (lambda r: r.update(frames=[Image.new("RGB", (8, 8))] * NUM_FRAMES), "MIN_IMAGE_SIDE"),
        (lambda r: r.update(frames=Image.new("RGB", (64, 48))), "not a single image"),
        (lambda r: r.update(label=""), "label must be a non-empty str"),
        (lambda r: r.update(label="juggling"), "not in the label set"),
    ],
)
def test_validate_dataset_refuses_malformed_records(mutate, message):
    records = _records()
    mutate(records[3])
    with pytest.raises(ValueError, match=message):
        validate_dataset(records, LABELS)


def test_validate_dataset_enforces_bounds_unique_ids_and_class_support():
    with pytest.raises(ValueError, match="16..5000 are required"):
        validate_dataset(_records(8), LABELS)
    with pytest.raises(ValueError, match="records must be a list"):
        validate_dataset({"id": "x"}, LABELS)
    dup = _records()
    dup[1]["id"] = dup[0]["id"]
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset(dup, LABELS)
    thin = _records()
    for r in thin:
        r["label"] = "brushing hair" if r["label"] == "chewing" else r["label"]
    thin[0]["label"] = "chewing"
    with pytest.raises(ValueError, match="at least 2 clips"):
        validate_dataset(thin, LABELS)
    with pytest.raises(ValueError, match="at least two distinct labels"):
        validate_dataset([_record(i, label="chewing") for i in range(16)])


def test_validate_dataset_refuses_before_importing_model_libraries(forbid_model_imports):
    with pytest.raises(ValueError, match="NUM_FRAMES"):
        validate_dataset([_record(0, frames=_frames(0, n=2))] + _records()[1:], LABELS)


def test_digests_split_disjointness_and_source_keys():
    a, b = _record(0), _record(0)
    assert clip_digest(a["frames"]) == clip_digest(b["frames"])
    assert clip_digest(_record(1)["frames"]) != clip_digest(a["frames"])
    assert dataset_digest([a, _record(1)]) == dataset_digest([_record(1), a])
    splits = {"train": _records()[:12], "test": _records()[12:]}
    assert check_split_disjoint(splits) == {"train": 12, "test": 4}
    leaky = {"train": _records()[:12], "test": [_record(0)]}
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint(leaky)
    shared = {"train": [_record(0, source="film/chewing")], "test": [_record(1, source="film/chewing")]}
    with pytest.raises(ValueError, match="source video"):
        check_split_disjoint(shared)
    assert source_of("Aussie_Brunette_Brushing_Long_Hair_brush_hair_u_nm_np1_ba_med_3.avi", "brush_hair") == "Aussie_Brunette_Brushing_Long_Hair/brush_hair"
    assert source_of("odd-name.avi", "chew") == "odd-name/chew"


def test_split_dataset_is_stratified_seeded_and_source_grouped():
    records = _records(32, sources=True)
    splits = split_dataset(records, val_fraction=0.25, test_fraction=0.25, seed=1)
    assert check_split_disjoint(splits) == {name: len(part) for name, part in splits.items()}
    assert sum(len(part) for part in splits.values()) == 32
    assert splits == split_dataset(records, val_fraction=0.25, test_fraction=0.25, seed=1)
    assert splits != split_dataset(records, val_fraction=0.25, test_fraction=0.25, seed=2)
    for part in splits.values():
        assert len({r["label"] for r in part}) == len(LABELS)
    dup = _records(32) + [dict(_records(32)[0], id="dup")]
    assert sum(len(part) for part in split_dataset(dup).values()) == 32
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.5, test_fraction=0.6)


# --- pinned corpus ---------------------------------------------------------------------------------------------


def test_pins_sample_classes_and_draw_targets():
    assert sm.CORPUS_REVISION == "50bb2abb741074ef0868e59ba51af710c825cd1f" and sm.CORPUS_ROW_GROUPS == 1
    assert set(sm.ROW_GROUP_PINS) == {0} and sm.ROW_GROUP_PINS[0][1] == 112_858_960
    assert len(HMDB51_CLASSES) == 51 and HMDB51_CLASSES[:2] == ("brush_hair", "cartwheel")
    assert list(SAMPLE_CLASS_TEXT) == list(HMDB51_CLASSES[:10]) and tuple(SAMPLE_CLASS_TEXT.values()) == SAMPLE_LABELS
    assert sm.SAMPLE_SPLIT == {"test": 6, "validation": 4}
    assert len(sm.SAMPLE_DIGEST) == 64


def _fake_rows(n=24, label=0):
    return [{"video": {"bytes": bytes([i % 256]) * 40, "path": f"src{i // 3}_{HMDB51_CLASSES[label]}_u_cm_np1_fr_med_{i}.avi"}, "label": label} for i in range(n)]


def test_fetch_corpus_refuses_a_row_group_that_does_not_match_its_pin(tmp_path, monkeypatch):
    rows = _fake_rows()
    table = pa.Table.from_pylist([{"video": r["video"], "label": r["label"]} for r in rows])
    shard = tmp_path / "shard.parquet"
    pq.write_table(table, shard)
    monkeypatch.setattr(sm, "CORPUS_ROWS", len(rows))
    with pytest.raises(ValueError, match="sha256"):
        fetch_corpus(cache_dir=tmp_path / "cache", opener=lambda url: str(shard))
    assert not (tmp_path / "cache" / "test-rg0.parquet").exists()
    with pytest.raises(ValueError, match="no pin"):
        fetch_corpus(cache_dir=tmp_path / "cache", groups=[7], opener=lambda url: str(shard))
    # a stale cache file is refetched and refused the same way
    pq.write_table(table, tmp_path / "cache" / "test-rg0.parquet")
    with pytest.raises(ValueError, match="sha256"):
        fetch_corpus(cache_dir=tmp_path / "cache", opener=lambda url: str(shard))


def test_read_corpus_keeps_the_ten_sample_classes_and_decodes_clips(monkeypatch):
    calls = []

    def fake_decode(data, n=NUM_FRAMES):
        calls.append(len(data))
        return _frames(len(calls))

    monkeypatch.setattr(sm, "decode_clip", fake_decode)
    rows = [{"video": r["video"]["bytes"], "path": r["video"]["path"], "label": r["label"]} for r in _fake_rows(6, label=0) + _fake_rows(3, label=10) + _fake_rows(4, label=9)]
    records = read_corpus({0: rows})
    assert len(records) == 10 and len(calls) == 10  # the label-10 rows (eleventh class) are skipped, undecoded
    assert records[0] == {"id": "hmdb51-test-0-0", "frames": records[0]["frames"], "label": "brushing hair", "class": "brush_hair", "source": "src0/brush_hair", "source_row_group": 0}
    assert records[-1]["label"] == "dribbling a basketball" and records[-1]["source"] == "src1/dribble"


def test_build_sample_dataset_takes_whole_sources_smallest_first():
    records = []
    for label in LABELS:
        for src in range(6):
            for _k in range(src + 1):  # source sizes 1..6 per label
                i = len(records)
                records.append({"id": f"r{i}", "frames": _frames(i), "label": label, "source": f"s{src}/{label}"})
    splits = build_sample_dataset(records, seed=3, sizes={"test": 6, "validation": 4})
    assert check_split_disjoint(splits)
    for label in LABELS:
        test = [r for r in splits["test"] if r["label"] == label]
        val = [r for r in splits["validation"] if r["label"] == label]
        train = [r for r in splits["train"] if r["label"] == label]
        assert len(test) == 6 and len(val) == 4 and len(train) == 11  # sources 1+2+3 test, 4 val, 5+6 train
    assert splits == build_sample_dataset(records, seed=3, sizes={"test": 6, "validation": 4})
    with pytest.raises(ValueError, match="too few sources"):
        build_sample_dataset(records[:4] + records[21:25], sizes={"test": 6, "validation": 4})


@pytest.mark.skipif(not (sm.DEFAULT_CACHE_DIR / "test-rg0.parquet").is_file(), reason="pinned row group not cached")
def test_default_draw_matches_the_pinned_digest_when_the_row_group_is_cached():
    splits = build_sample_dataset(read_corpus(fetch_corpus()))
    assert check_split_disjoint(splits) == {"train": 181, "validation": 53, "test": 66}
    assert dataset_digest(splits["train"] + splits["validation"] + splits["test"]) == sm.SAMPLE_DIGEST


# --- BYOD --------------------------------------------------------------------------------------------------------


def _gif_bytes(i, n=10):
    frames = _frames(i, n=n)
    buffer = io.BytesIO()
    frames[0].save(buffer, format="GIF", save_all=True, append_images=frames[1:], duration=50, loop=0)
    return buffer.getvalue()


def test_load_byod_dataset_reads_clips_with_labels_from_a_zip_or_directory(tmp_path):
    root = tmp_path / "clips"
    root.mkdir()
    for i in range(4):
        (root / f"clip{i}.gif").write_bytes(_gif_bytes(i))
    (root / "labels.csv").write_text("file,label,source\nclip0.gif,brushing hair,filmA\nclip1.gif,chewing,\nclip2.gif,brushing hair,filmA\nclip3.gif,chewing,filmB\n", encoding="utf-8")
    records = load_byod_dataset(root)
    assert [r["id"] for r in records] == ["clip0", "clip1", "clip2", "clip3"]
    assert len(records[0]["frames"]) == NUM_FRAMES and records[0]["label"] == "brushing hair" and records[0]["source"] == "filmA"
    assert "source" not in records[1]
    archive = tmp_path / "clips.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for file in root.iterdir():
            z.write(file, f"nested/{file.name}")
    assert [r["id"] for r in load_byod_dataset(archive)] == ["clip0", "clip1", "clip2", "clip3"]
    (root / "extra.gif").write_bytes(_gif_bytes(9))
    with pytest.raises(ValueError, match="no labels.csv row"):
        load_byod_dataset(root)
    (root / "extra.gif").unlink()
    (root / "labels.csv").write_text("file,label\nmissing.gif,chewing\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing clip"):
        load_byod_dataset(root)
    (root / "labels.csv").write_text("file,text\nclip0.gif,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns file and label"):
        load_byod_dataset(root)
    (root / "short.gif").write_bytes(_gif_bytes(5, n=3))
    (root / "labels.csv").write_text("file,label\nshort.gif,chewing\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a decodable clip"):
        load_byod_dataset(root)
    with pytest.raises(ValueError, match="neither a directory nor a zip"):
        load_byod_dataset(tmp_path / "nope.txt")
    out = write_dataset_csv(records, tmp_path / "out" / "train.csv")
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "id,file,width,height,frames,label,class,source,source_row_group" and len(lines) == 5


# --- metrics and baselines --------------------------------------------------------------------------------------


def test_classification_metrics_and_baselines():
    records = _records()
    perfect = [[r["label"]] + [label for label in LABELS if label != r["label"]] for r in records]
    m = classification_metrics(perfect, records, LABELS)
    assert m["top1_accuracy"] == 1.0 and m["top3_accuracy"] == 1.0 and m["macro_f1"] == 1.0 and m["mean_rank"] == 1.0
    always_first = [list(LABELS)] * len(records)
    m = classification_metrics(always_first, records, LABELS)
    assert m["top1_accuracy"] == 0.25 and m["top3_accuracy"] == 0.75 and m["macro_recall"] == 0.25
    assert m["per_label"]["brushing hair"] == {"support": 4, "predicted": 16, "correct": 4, "recall": 1.0, "precision": 0.25, "f1": pytest.approx(0.4)}
    assert m["per_label"]["chewing"]["recall"] == 0.0 and m["confusion"]["chewing"] == {"brushing hair": 4}
    with pytest.raises(ValueError, match="rankings for"):
        classification_metrics(perfect[:3], records, LABELS)
    with pytest.raises(ValueError, match="not in the label set"):
        classification_metrics(perfect, records, LABELS[:3] + ["juggling"])
    chance = chance_baseline(records, LABELS)
    assert chance["top1_accuracy"] == 0.25 and chance["top3_accuracy"] == 0.75
    train = _records()[:8] + [_record(99, label="chewing")]
    assert majority_label(train, LABELS) == "chewing"
    majority = majority_baseline(train, records, LABELS)
    assert majority["label"] == "chewing" and majority["top1_accuracy"] == 0.25 and majority["kind"] == "majority"


# --- classify_batch / evaluate with an injected runner ---------------------------------------------------------


def _scoring_runner(frames, labels):
    # the red pixel's row index encodes the record index; score its label highest
    i = next(x for x in range(frames[0].width) if frames[0].getpixel((x, 0)) == (255, 0, 0))
    scores = np.zeros(len(labels))
    scores[i % len(labels)] = 5.0
    return scores


def test_classify_batch_and_evaluate_score_the_runner_and_flag_small_samples():
    pipe = XClipVideoClassificationPipeline(_scoring_runner, "cpu")
    records = _records()
    results = pipe.classify_batch([r["frames"] for r in records], LABELS, batch_size=5)
    assert len(results) == 16 and results[1]["top1"] == "doing a cartwheel" and results[1]["n_frames"] == NUM_FRAMES
    report = pipe.evaluate(records, LABELS)
    assert report["top1_accuracy"] == 1.0 and report["verdict"] == "measured-small-sample" and report["adapted"] is False
    assert report["predictions"][2] == "catching a ball" and report["rankings"][2][0] == "catching a ball" and report["labels"] == LABELS
    big = [_record(i) for i in range(32)]
    assert pipe.evaluate(big, LABELS)["verdict"] == "measured"
    with pytest.raises(ValueError, match="batch_size"):
        pipe.classify_batch([records[0]["frames"]], LABELS, batch_size=0)
    with pytest.raises(ValueError, match="not in the label set"):
        pipe.evaluate(records, LABELS[:3] + ["juggling"])
    batched = XClipVideoClassificationPipeline(_scoring_runner, "cpu", lambda clips, labels: np.stack([_scoring_runner(c, labels) for c in clips]))
    assert [r["top1"] for r in batched.classify_batch([r["frames"] for r in records], LABELS)] == [r["top1"] for r in results]
    wrong = XClipVideoClassificationPipeline(_scoring_runner, "cpu", lambda clips, labels: np.zeros((1, 2)))
    with pytest.raises(RuntimeError, match="logits of shape"):
        wrong.classify_batch([records[0]["frames"], records[1]["frames"]], LABELS)


def test_adapt_and_artifacts_require_a_loaded_model(forbid_model_imports):
    pipe = XClipVideoClassificationPipeline(_scoring_runner, "cpu")
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.adapt(_records(), None, LABELS)
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(_records(), None, LABELS, epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(_records(), None, LABELS, lr=0.5)
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.save_artifact("x")
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.load_artifact("x")


# --- artifact manifest checks --------------------------------------------------------------------------------------


def _manifest(tmp_path, **overrides):
    weights = tmp_path / "adapter.safetensors"
    weights.write_bytes(b"tensor-bytes")
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": "base-digest"},
        "tensors": ["mit.encoder.layers.0.self_attn.q_proj.weight", "prompts_visual_projection", "visual_projection.weight"],
        "adapter": {"best_epoch": 1, "labels": list(LABELS)},
        "files": [{"path": "adapter.safetensors", "bytes": weights.stat().st_size, "sha256": hashlib.sha256(b"tensor-bytes").hexdigest()}],
    }
    manifest.update(overrides)
    return manifest


def test_check_artifact_manifest_accepts_a_consistent_manifest_and_refuses_each_deviation(tmp_path):
    pl._check_artifact_manifest(_manifest(tmp_path), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="format"):
        pl._check_artifact_manifest(_manifest(tmp_path, format="other"), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="trained on"):
        pl._check_artifact_manifest(_manifest(tmp_path, base={"model_id": "x", "revision": MODEL_REVISION, "weight_sha256": "base-digest"}), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="base weight digest"):
        pl._check_artifact_manifest(_manifest(tmp_path), tmp_path, "another-digest")
    bad = _manifest(tmp_path)
    bad["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256"):
        pl._check_artifact_manifest(bad, tmp_path, "base-digest")
    with pytest.raises(ValueError, match="exactly adapter.safetensors"):
        pl._check_artifact_manifest(_manifest(tmp_path, files=[]), tmp_path, "base-digest")
    for name in ("vision_model.encoder.layers.11.mlp.fc1.weight", "text_model.encoder.layers.0.x", "text_projection.weight", "logit_scale", "vision_model.embeddings.patch_embedding.weight"):
        with pytest.raises(ValueError, match="fusion head"):
            pl._check_artifact_manifest(_manifest(tmp_path, tensors=[name]), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="adapter.labels"):
        pl._check_artifact_manifest(_manifest(tmp_path, adapter={"best_epoch": 1}), tmp_path, "base-digest")


def test_trainable_names_selects_the_fusion_head():
    class _Param:
        def numel(self):
            return 1

    class _Model:
        def named_parameters(self):
            names = ["vision_model.encoder.layers.0.x", "vision_model.post_layernorm.weight", "visual_projection.weight", "mit.encoder.layers.0.self_attn.q_proj.weight", "mit.position_embedding", "prompts_visual_layernorm.weight", "prompts_visual_projection", "prompts_generator.layers.0.norm1.weight", "text_model.encoder.layers.0.x", "text_projection.weight", "logit_scale"]
            return [(n, _Param()) for n in names]

    assert pl._trainable_names(_Model()) == ["visual_projection.weight", "mit.encoder.layers.0.self_attn.q_proj.weight", "mit.position_embedding", "prompts_visual_layernorm.weight", "prompts_visual_projection", "prompts_generator.layers.0.norm1.weight"]
    assert len(pl._TRAINABLE_PREFIXES) == 5 and pl.HEAD_PARAMETERS == 10_247_680 and pl.PARAMETER_COUNT == 196_585_729


def test_manifest_json_round_trip(tmp_path):
    payload = {"epoch": 1, "train_loss": 0.3, "val": {"top1_accuracy": 0.9, "top3_accuracy": 1.0, "macro_recall": 0.9, "macro_f1": 0.88, "n": 53}}
    (tmp_path / "h.json").write_text(json.dumps([payload]), encoding="utf-8")
    assert json.loads((tmp_path / "h.json").read_text(encoding="utf-8"))[0]["val"]["top1_accuracy"] == 0.9
