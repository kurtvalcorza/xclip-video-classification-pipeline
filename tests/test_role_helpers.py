"""Role-helper contract: validate_inputs (validation stage) and evaluation_report (evaluation stage)."""

from __future__ import annotations

import pytest
from PIL import Image

from xclip_video_classification_pipeline import (
    INPUT_SCHEMA,
    MAX_IMAGE_SIDE,
    MAX_LABELS,
    MIN_IMAGE_SIDE,
    MIN_LABELS,
    MODEL_ID,
    MODEL_REVISION,
    NUM_FRAMES,
    evaluation_report,
    validate_inputs,
)

LABELS = ["a ball rolling", "a ball bouncing", "a square growing"]


def _clip(n: int = NUM_FRAMES, size: tuple[int, int] = (64, 48)) -> list[Image.Image]:
    return [Image.new("RGB", size, "white") for _ in range(n)]


def _result(order: list[str], clip: str = "clip-0") -> dict:
    probs = [0.6, 0.3, 0.1][: len(order)]
    return {
        "predictions": [
            {"label": label, "probability": prob, "logit": prob * 10}
            for label, prob in zip(order, probs, strict=True)
        ],
        "top1": order[0],
        "labels": sorted(order),
        "n_frames": NUM_FRAMES,
        "frame_size": [64, 48],
        "clip": clip,
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(
        [_clip(), _clip(size=(32, 32))],
        ["A ball rolling", "a ball bouncing.", "a square growing"],
        names=["roll.gif", "bounce.gif"],
    )
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["frame_side_px"] == [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE]
    assert manifest["schema"]["frames_per_clip"] == NUM_FRAMES
    assert manifest["schema"]["labels"] == [MIN_LABELS, MAX_LABELS]
    assert manifest["inputs"] == [
        {"id": "roll.gif", "n_frames": NUM_FRAMES, "frame_mode": "RGB", "frame_size": [64, 48]},
        {"id": "bounce.gif", "n_frames": NUM_FRAMES, "frame_mode": "RGB", "frame_size": [32, 32]},
    ]
    assert manifest["labels"] == LABELS
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_ids() -> None:
    manifest = validate_inputs([_clip()], LABELS)
    assert [entry["id"] for entry in manifest["inputs"]] == ["clip-0"]


def test_validate_inputs_rejects_like_classify() -> None:
    with pytest.raises(TypeError, match="non-empty sequence"):
        validate_inputs([], LABELS)
    with pytest.raises(TypeError, match="not one clip"):
        validate_inputs(_clip(), LABELS)
    with pytest.raises(ValueError, match="NUM_FRAMES"):
        validate_inputs([_clip(n=3)], LABELS)
    with pytest.raises(ValueError, match="MIN_LABELS"):
        validate_inputs([_clip()], ["one"])
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        validate_inputs([_clip(size=(8, 8))], LABELS)
    with pytest.raises(ValueError, match="names has"):
        validate_inputs([_clip()], LABELS, names=["a", "b"])


def test_evaluation_report_not_measurable_without_labels() -> None:
    report = evaluation_report(
        [_result(["a ball rolling", "a ball bouncing", "a square growing"])], sample_kind="BYOD"
    )
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == [] and report["baselines"] == []
    assert report["n_clips"] == 1 and report["n_labels"] == [3]
    assert "top-1" in report["needs"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert "not a calibrated probability" in report["score_semantics"]


def test_evaluation_report_sample_sanity_with_labels() -> None:
    results = [
        _result(["a ball rolling", "a ball bouncing", "a square growing"], "roll"),
        _result(["a ball rolling", "a ball bouncing", "a square growing"], "bounce"),
        _result(["a square growing", "a ball rolling", "a ball bouncing"], "grow"),
    ]
    report = evaluation_report(results, ["A ball rolling", "a ball bouncing.", "a square growing"])
    assert report["verdict"] == "sample-sanity"
    by_id = {metric["id"]: metric for metric in report["metrics"]}
    assert by_id["top1_accuracy"]["value"] == pytest.approx(2 / 3)
    assert report["baselines"][0] == {
        "id": "chance",
        "value": pytest.approx(1 / 3),
        "note": "mean of 1/n_labels over the clips",
    }
    assert [entry["correct"] for entry in report["per_clip"]] == [True, False, True]
    assert [entry["rank_of_correct"] for entry in report["per_clip"]] == [1, 2, 1]
    assert report["per_clip"][1]["top1_probability"] == 0.6


def test_evaluation_report_rejects_mismatched_or_unknown_labels() -> None:
    result = _result(["a ball rolling", "a ball bouncing", "a square growing"])
    with pytest.raises(ValueError, match="correct_labels has"):
        evaluation_report([result], ["a ball rolling", "a ball bouncing"])
    with pytest.raises(ValueError, match="not among the result's labels"):
        evaluation_report([result], ["a cat"])
    with pytest.raises(ValueError, match="results"):
        evaluation_report([], None)
