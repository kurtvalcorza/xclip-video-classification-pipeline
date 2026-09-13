import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from xclip_video_classification_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    FRAME_SIZE,
    MAX_IMAGE_SIDE,
    MAX_LABEL_CHARS,
    MAX_LABELS,
    MAX_TEXT_TOKENS,
    MIN_IMAGE_SIDE,
    MIN_LABELS,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    NUM_FRAMES,
    XClipVideoClassificationPipeline,
    format_labels,
    frames_from_animation,
    sample_frames,
    stage_missing_files,
    validate_clip,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPO = Path(__file__).resolve().parents[1]
LABELS = ["a ball rolling", "a ball bouncing", "a square growing"]


def _clip(n: int = NUM_FRAMES, size: tuple[int, int] = (64, 48), mode: str = "RGB") -> list[Image.Image]:
    return [Image.new(mode, size, (i * 10 % 256,) * (3 if mode == "RGB" else 1)) for i in range(n)]


def test_identity_constants():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "microsoft/xclip-base-patch32"
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    assert NUM_FRAMES == 8 and FRAME_SIZE == 224
    assert 2 <= MIN_LABELS < MAX_LABELS == 32 and MAX_LABEL_CHARS == 64 and MAX_TEXT_TOKENS == 77
    manifest = REPO / "weights" / MODEL_KEY / "dimer-base-manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["modelId"] == MODEL_ID
        assert data["revision"] == MODEL_REVISION
        paths = [entry["path"] for entry in data["files"]]
        assert "model.safetensors" in paths and "pytorch_model.bin" not in paths


def _write_snapshot(root: Path, content: bytes, sha: str | None = None, size: int | None = None) -> None:
    (root / "config.json").write_bytes(content)
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": "config.json",
                "bytes": len(content) if size is None else size,
                "sha256": hashlib.sha256(content).hexdigest() if sha is None else sha,
            }
        ],
        "totalBytes": len(content),
    }
    (root / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_snapshot_accepts_matching_manifest(tmp_path):
    _write_snapshot(tmp_path, b'{"model_type": "xclip"}')
    info = verify_snapshot(tmp_path)
    assert info["revision"] == MODEL_REVISION and info["files"] == 1


def test_verify_snapshot_rejects_tampered_digest(tmp_path):
    content = b'{"model_type": "xclip"}'
    good = hashlib.sha256(content).hexdigest()
    flipped = ("0" if good[0] != "0" else "1") + good[1:]
    _write_snapshot(tmp_path, content, sha=flipped)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_size_missing_file_and_revision(tmp_path):
    _write_snapshot(tmp_path, b"abc", size=99)
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    manifest = json.loads((tmp_path / "dimer-base-manifest.json").read_text())
    manifest["revision"] = "0" * 40
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    (tmp_path / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    listed = verify_snapshot(tmp_path)["files"]
    assert (listed if isinstance(listed, int) else len(listed)) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def test_format_labels_lowercases_and_normalises():
    assert format_labels(["Playing  Soccer", "juggling balls."]) == ["playing soccer", "juggling balls"]
    with pytest.raises(ValueError, match="distinct"):
        format_labels(["a cat", "A cat."])
    with pytest.raises(TypeError):
        format_labels("a cat")
    with pytest.raises(TypeError):
        format_labels(["a cat", 3])
    with pytest.raises(ValueError, match="empty"):
        format_labels(["a cat", " . "])
    with pytest.raises(ValueError, match="MIN_LABELS"):
        format_labels(["only one"])
    with pytest.raises(ValueError, match="MAX_LABELS"):
        format_labels([f"label {i}" for i in range(MAX_LABELS + 1)])
    with pytest.raises(ValueError, match="MAX_LABEL_CHARS"):
        format_labels(["a" * (MAX_LABEL_CHARS + 1), "b"])


def test_validate_clip_sample_frames_and_animation():
    rgb = validate_clip(_clip(mode="L"))
    assert len(rgb) == NUM_FRAMES and all(frame.mode == "RGB" for frame in rgb)
    with pytest.raises(TypeError, match="not a single image"):
        validate_clip(Image.new("RGB", (64, 48)))
    with pytest.raises(ValueError, match="NUM_FRAMES"):
        validate_clip(_clip(n=NUM_FRAMES - 1))
    with pytest.raises(ValueError, match="same size"):
        validate_clip(_clip(n=NUM_FRAMES - 1) + [Image.new("RGB", (32, 32))])
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        validate_clip(_clip(size=(MIN_IMAGE_SIDE - 1, 64)))
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        validate_clip(_clip(size=(MAX_IMAGE_SIDE + 1, 64)))
    long = _clip(n=30)
    picked = sample_frames(long)
    assert len(picked) == NUM_FRAMES and picked[0] is long[0] and picked[-1] is long[-1]
    assert sample_frames(long, n=3) == [long[0], long[14], long[29]]  # 14.5 rounds to even
    with pytest.raises(ValueError, match="at least"):
        sample_frames(_clip(n=3))
    buffer = io.BytesIO()
    frames = _clip(n=5)
    frames[0].save(buffer, format="GIF", save_all=True, append_images=frames[1:], duration=50, loop=0)
    buffer.seek(0)
    decoded = frames_from_animation(Image.open(buffer))
    assert len(decoded) == 5 and all(frame.mode == "RGB" for frame in decoded)
    with pytest.raises(TypeError):
        frames_from_animation("clip.gif")


def _fake_pipeline(calls: list | None = None) -> XClipVideoClassificationPipeline:
    def runner(frames, labels):
        if calls is not None:
            calls.append(([frame.mode for frame in frames], list(labels)))
        return np.array([1.0, 3.0, 2.0][: len(labels)], dtype=np.float32)

    return XClipVideoClassificationPipeline(runner, "cpu")


def test_classify_output_fields_and_softmax():
    calls: list = []
    pipe = _fake_pipeline(calls)
    result = pipe.classify(_clip(mode="L"), ["A ball rolling", "a ball bouncing.", "a square growing"])
    assert calls == [(["RGB"] * NUM_FRAMES, LABELS)]
    assert result["labels"] == LABELS
    assert result["top1"] == "a ball bouncing"
    assert [entry["label"] for entry in result["predictions"]] == [
        "a ball bouncing",
        "a square growing",
        "a ball rolling",
    ]
    probs = [entry["probability"] for entry in result["predictions"]]
    assert probs == sorted(probs, reverse=True) and sum(probs) == pytest.approx(1.0)
    expected = np.exp([3.0, 2.0, 1.0]) / np.exp([3.0, 2.0, 1.0]).sum()
    assert probs == pytest.approx(expected.tolist())
    assert [entry["logit"] for entry in result["predictions"]] == [3.0, 2.0, 1.0]
    assert result["n_frames"] == NUM_FRAMES and result["frame_size"] == [64, 48]
    assert (result["model_id"], result["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_classify_rejects_bad_inputs():
    pipe = _fake_pipeline()
    with pytest.raises(TypeError, match="not a single image"):
        pipe.classify(Image.new("RGB", (64, 48)), LABELS)
    with pytest.raises(ValueError, match="NUM_FRAMES"):
        pipe.classify(_clip(n=2), LABELS)
    with pytest.raises(TypeError, match="not a single string"):
        pipe.classify(_clip(), "a ball rolling")
    with pytest.raises(ValueError, match="MIN_LABELS"):
        pipe.classify(_clip(), ["one"])


def test_classify_rejects_malformed_backend_output():
    wrong_shape = XClipVideoClassificationPipeline(lambda frames, labels: np.zeros(2, np.float32), "cpu")
    with pytest.raises(RuntimeError, match="logits of shape"):
        wrong_shape.classify(_clip(), LABELS)
    non_finite = XClipVideoClassificationPipeline(lambda frames, labels: np.array([np.nan, 1.0, 2.0]), "cpu")
    with pytest.raises(RuntimeError, match="logits"):
        non_finite.classify(_clip(), LABELS)
