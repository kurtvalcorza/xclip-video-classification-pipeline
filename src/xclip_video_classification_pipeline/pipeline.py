"""Zero-shot video classification with the pinned ``microsoft/xclip-base-patch32`` checkpoint (X-CLIP).

The class loads the processor and model only from a digest-verified local snapshot (``weights/<key>/``)
or, when explicitly allowed, from the Hugging Face Hub at the pinned revision — always with
``trust_remote_code=False``: the X-CLIP architecture comes from the pinned ``transformers`` release, the
weights are SafeTensors, and no model-repository code is executed. A clip is a sequence of exactly
NUM_FRAMES PIL frames; the caller names the candidate classes as free text and receives a softmax over
those names — a relative ranking, not a calibrated probability.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageSequence

MODEL_ID = "microsoft/xclip-base-patch32"
MODEL_REVISION = "a2e27a78a2b5d802e894b8a1ef14f3a8ce490963"
MODEL_LICENSE = "mit"
MODEL_KEY = "xclip-base-patch32"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# The checkpoint was trained on 8 frames per clip (config.json vision_config.num_frames); the temporal
# modules expect exactly that many, so a clip is exactly NUM_FRAMES frames and longer sequences are
# subsampled uniformly by sample_frames.
NUM_FRAMES = 8
# Frame preprocessing: shorter side resized to 224, centre crop 224x224, ImageNet mean/std
# (preprocessor_config.json), 32x32 patches -> 49 tokens per frame.
FRAME_SIZE = 224
# Input ceilings. Each label is one CLIP text query (77-token context); the label set is the caller's
# closed vocabulary for this request.
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 16
MIN_LABELS = 2
MAX_LABELS = 32
MAX_LABEL_CHARS = 64
MAX_TEXT_TOKENS = 77


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def format_labels(labels: Sequence[str]) -> list[str]:
    """Validate the candidate class names and normalise them: stripped, whitespace-collapsed,
    lower-cased, trailing full stop removed, distinct. Names are passed to the CLIP text tower as-is
    otherwise (the upstream zero-shot recipe uses plain class names such as "playing soccer")."""
    if isinstance(labels, str) or not isinstance(labels, Sequence):
        raise TypeError("labels must be a list of class names, not a single string")
    if not MIN_LABELS <= len(labels) <= MAX_LABELS:
        raise ValueError(
            f"label count {len(labels)} outside MIN_LABELS {MIN_LABELS}..MAX_LABELS {MAX_LABELS}"
        )
    cleaned: list[str] = []
    for name in labels:
        if not isinstance(name, str):
            raise TypeError(f"label must be str, got {type(name).__name__}")
        text = " ".join(name.split()).strip().rstrip(".").strip().lower()
        if not text:
            raise ValueError("labels must not be empty")
        if len(text) > MAX_LABEL_CHARS:
            raise ValueError(
                f"label {text[:12]!r}... is {len(text)} chars > MAX_LABEL_CHARS {MAX_LABEL_CHARS}"
            )
        cleaned.append(text)
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("labels must be distinct after normalisation")
    return cleaned


def validate_frame(frame: Any) -> Image.Image:
    if not isinstance(frame, Image.Image):
        raise TypeError(f"frame must be a PIL.Image.Image, got {type(frame).__name__}")
    width, height = frame.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"frame side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"frame side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    return frame.convert("RGB")


def validate_clip(frames: Any) -> list[Image.Image]:
    """Exactly NUM_FRAMES PIL frames of one common size, each within the side ceilings."""
    if isinstance(frames, Image.Image) or not isinstance(frames, Sequence):
        raise TypeError("frames must be a sequence of PIL.Image.Image, not a single image")
    if len(frames) != NUM_FRAMES:
        raise ValueError(
            f"a clip is exactly NUM_FRAMES={NUM_FRAMES} frames, got {len(frames)}; "
            "use sample_frames to subsample a longer sequence"
        )
    rgb = [validate_frame(frame) for frame in frames]
    if len({frame.size for frame in rgb}) != 1:
        raise ValueError("all frames of a clip must have the same size")
    return rgb


def sample_frames(frames: Sequence[Image.Image], n: int = NUM_FRAMES) -> list[Image.Image]:
    """Pick ``n`` frames at evenly spaced indices (first and last included) from a longer sequence."""
    if isinstance(frames, Image.Image) or not isinstance(frames, Sequence):
        raise TypeError("frames must be a sequence of PIL.Image.Image")
    if len(frames) < n:
        raise ValueError(f"need at least {n} frames to sample {n}, got {len(frames)}")
    indices = np.linspace(0, len(frames) - 1, num=n).round().astype(int)
    return [frames[int(index)] for index in indices]


def frames_from_animation(image: Image.Image) -> list[Image.Image]:
    """Decode every frame of an animated image (GIF, WebP, APNG) that Pillow can open into RGB copies."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(image)]
    if not frames:
        raise ValueError("the image holds no frames")
    return frames


INPUT_SCHEMA: dict[str, Any] = {
    "input": (
        f"one clip of exactly NUM_FRAMES={NUM_FRAMES} PIL.Image.Image frames of one size (any mode, "
        "converted to RGB) plus MIN_LABELS..MAX_LABELS free-text class names"
    ),
    "frame_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "frames_per_clip": NUM_FRAMES,
    "labels": [MIN_LABELS, MAX_LABELS],
    "label_chars": [1, MAX_LABEL_CHARS],
    "label_tokens": [1, MAX_TEXT_TOKENS],
    "preprocessing": (
        f"each frame: shorter side resized to {FRAME_SIZE}, centre crop {FRAME_SIZE}x{FRAME_SIZE}, ImageNet "
        "mean/std, 32x32 patches; labels normalised into one CLIP text query each (format_labels); the "
        "video embedding (frame features fused by the multi-frame integration transformer) is scored "
        "against each label embedding and the scores are softmaxed over the supplied labels"
    ),
    "output": (
        "one probability per supplied label (softmax over the label set: a relative ranking that sums to "
        "1 and is not calibrated), the raw logits, and the top-1 label"
    ),
}


def _check_inputs(frames: Any, labels: Any) -> tuple[list[Image.Image], list[str]]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the checked request.

    ``classify`` and ``validate_inputs`` both route through this function so their acceptance
    criteria cannot diverge.
    """
    return validate_clip(frames), format_labels(labels)


def validate_inputs(
    clips: Sequence[Sequence[Image.Image]],
    labels: Sequence[str],
    *,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict).

    Every clip is checked exactly as ``classify`` would check it; rejection is reported by raising,
    and a caller that wants the finding recorded catches the exception and stores ``str(exc)`` under
    ``findings``.
    """
    if isinstance(clips, Image.Image) or not isinstance(clips, Sequence) or not clips:
        raise TypeError("clips must be a non-empty sequence of frame sequences")
    if clips and isinstance(clips[0], Image.Image):
        raise TypeError("clips must be a sequence of clips (each a sequence of frames), not one clip")
    if names is not None and len(names) != len(clips):
        raise ValueError(f"names has {len(names)} entries for {len(clips)} clips")
    checked_labels = format_labels(labels)
    observed = []
    for index, clip in enumerate(clips):
        rgb, _ = _check_inputs(clip, labels)
        observed.append(
            {
                "id": names[index] if names else f"clip-{index}",
                "n_frames": len(rgb),
                "frame_mode": clip[0].mode,
                "frame_size": list(rgb[0].size),
            }
        )
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": observed,
        "labels": checked_labels,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    results: Sequence[Mapping[str, Any]],
    correct_labels: Sequence[str] | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``correct_labels`` (one class name per result, in order; normalised like the labels) the report
    carries ``top1_accuracy`` over the clips, the chance baseline (mean of 1/n_labels) and one
    per-clip entry, verdict ``sample-sanity``; without them it is ``not-measurable`` and says what
    labelled data would make the task measurable.
    """
    if not results:
        raise ValueError("results must contain at least one classification result")
    base = {
        "task": "zero-shot video classification over a caller-supplied label set (top-1 over the labels)",
        "score_semantics": (
            "probabilities are a softmax over the supplied labels only: a relative ranking that sums to 1, "
            "not a calibrated probability, and a label set without the true class still yields a confident "
            "top-1"
        ),
        "sample_kind": sample_kind,
        "n_clips": len(results),
        "n_labels": [len(result["labels"]) for result in results],
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if correct_labels is None:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no correct labels were supplied for the classified clips",
            "needs": (
                "labelled clips from the deployment domain with a class vocabulary matching the labels "
                "(Kinetics-style annotations) scored with top-1/top-5 accuracy over many clips; no such "
                "labelled set ships with this repository"
            ),
        }
    if len(correct_labels) != len(results):
        raise ValueError(f"correct_labels has {len(correct_labels)} entries for {len(results)} results")
    per_clip = []
    for result, correct in zip(results, correct_labels, strict=True):
        key = format_labels([correct, "__second__"])[0]
        if key not in result["labels"]:
            raise ValueError(f"correct label {correct!r} is not among the result's labels")
        per_clip.append(
            {
                "clip": result.get("clip"),
                "top1": result["top1"],
                "top1_probability": result["predictions"][0]["probability"],
                "correct_label": key,
                "correct": result["top1"] == key,
                "rank_of_correct": next(
                    index for index, entry in enumerate(result["predictions"]) if entry["label"] == key
                )
                + 1,
            }
        )
    chance = sum(1.0 / n for n in base["n_labels"]) / len(results)
    return {
        **base,
        "metrics": [
            {
                "id": "top1_accuracy",
                "value": sum(entry["correct"] for entry in per_clip) / len(per_clip),
                "estimation": f"{len(per_clip)} clip(s), no dispersion estimate",
            }
        ],
        "baselines": [{"id": "chance", "value": chance, "note": "mean of 1/n_labels over the clips"}],
        "per_clip": per_clip,
        "verdict": "sample-sanity",
        "reason": (
            f"{len(per_clip)} clip(s) with caller-known classes; plumbing evidence, not a video "
            "classification benchmark"
        ),
        "needs": (
            "labelled clips from the deployment domain with a matching class vocabulary for any "
            "top-1/top-5 accuracy claim; Kinetics-400 and UCF101 are not bundled"
        ),
    }


@dataclass
class XClipVideoClassificationPipeline:
    """Zero-shot video classification over the pinned X-CLIP base/32 checkpoint."""

    _runner: Callable[[list[Image.Image], list[str]], np.ndarray]
    device: str

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> XClipVideoClassificationPipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            source, kwargs = str(root), {"local_files_only": True}
        elif allow_download:
            source, kwargs = MODEL_ID, {}
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage {MODEL_ID}@{MODEL_REVISION} under weights/{MODEL_KEY}"
            )
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import XCLIPModel, XCLIPProcessor

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        processor = XCLIPProcessor.from_pretrained(
            source, revision=MODEL_REVISION, trust_remote_code=False, **kwargs
        )
        model = XCLIPModel.from_pretrained(
            source, revision=MODEL_REVISION, trust_remote_code=False, dtype=torch.float32, **kwargs
        )
        model = model.to(resolved_device).eval()

        def runner(frames: list[Image.Image], labels: list[str]) -> np.ndarray:
            # The frame list is handed to the snapshot's VideoMAE-style image processor directly: in
            # the pinned transformers release XCLIPProcessor's `videos=` keyword yields no
            # pixel_values (only `images=` does), and calling the two sub-processors makes the
            # contract explicit. One clip -> pixel_values (1, NUM_FRAMES, 3, 224, 224).
            pixel_values = processor.image_processor([frames], return_tensors="pt")["pixel_values"]
            text = processor.tokenizer(labels, padding=True, return_tensors="pt")
            with torch.inference_mode():
                outputs = model(
                    pixel_values=pixel_values.to(resolved_device),
                    input_ids=text["input_ids"].to(resolved_device),
                    attention_mask=text["attention_mask"].to(resolved_device),
                )
            return outputs.logits_per_video[0].float().cpu().numpy()

        return cls(runner, resolved_device)

    def classify(self, frames: Sequence[Image.Image], labels: Sequence[str]) -> dict[str, Any]:
        """Rank ``labels`` for one clip of exactly NUM_FRAMES frames; probabilities are a softmax over them.

        The softmax is a relative ranking over the supplied labels, not a calibrated probability.
        """
        rgb, names = _check_inputs(frames, labels)
        logits = np.asarray(self._runner(rgb, names), dtype=np.float64).reshape(-1)
        if logits.shape != (len(names),) or not np.all(np.isfinite(logits)):
            raise RuntimeError(f"backend returned logits of shape {logits.shape} for {len(names)} labels")
        shifted = np.exp(logits - logits.max())
        probs = shifted / shifted.sum()
        order = np.argsort(-probs)
        predictions = [
            {"label": names[index], "probability": float(probs[index]), "logit": float(logits[index])}
            for index in order
        ]
        return {
            "predictions": predictions,
            "top1": predictions[0]["label"],
            "labels": names,
            "n_frames": len(rgb),
            "frame_size": list(rgb[0].size),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
