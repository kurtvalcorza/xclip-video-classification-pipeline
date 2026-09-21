"""Zero-shot video classification with the pinned ``microsoft/xclip-base-patch32`` checkpoint (X-CLIP), plus the
adaptation contract for a closed label set: corpus evaluation of labelled clips, bounded fine-tuning of the fusion
head (the frame-integration transformer, the two visual projections and the video-specific prompt generator) on
cached tower features, and a verified adapter artifact.

The class loads the processor and model only from a digest-verified local snapshot (``weights/<key>/``)
or, when explicitly allowed, from the Hugging Face Hub at the pinned revision — always with
``trust_remote_code=False``: the X-CLIP architecture comes from the pinned ``transformers`` release, the
weights are SafeTensors, and no model-repository code is executed. A clip is a sequence of exactly
NUM_FRAMES PIL frames; the caller names the candidate classes as free text and receives a softmax over
those names — a relative ranking, not a calibrated probability.
"""
# ruff: noqa: E501  -- adaptation-contract lines are kept at the fleet width

from __future__ import annotations

import hashlib
import json
import random
import time
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
WEIGHTS_FILE = "model.safetensors"

# Adaptation contract: the fusion head — the multi-frame integration transformer (`mit`), the two visual projections
# and the video-specific prompt generator — is the adapter; the vision tower (12 ViT-B/32 layers with cross-frame
# attention), the text tower, the text projection and the logit scale stay frozen, so their outputs are computed
# once per clip and label set and cached.
PARAMETER_COUNT = 196_585_729
HEAD_PARAMETERS = 10_247_680
FRAME_TOKENS = 50  # 1 CLS + 49 patch tokens per 224-px frame
VISION_WIDTH = 768
_TRAINABLE_PREFIXES = ("visual_projection.", "mit.", "prompts_visual_layernorm.", "prompts_visual_projection", "prompts_generator.")
ARTIFACT_FORMAT = f"org.valcorza.{MODEL_KEY}.adapter.v1"
ARTIFACT_VERSION = 1
ADAPTER_WEIGHTS = "adapter.safetensors"
ADAPTER_MANIFEST = "manifest.json"
MIN_SCORED_RECORDS = 30  # below this a scored set is labelled a small sample
MAX_EVAL_RECORDS = 5_000
EVAL_BATCH_SIZE = 8  # clips per vision-tower forward (8 clips x NUM_FRAMES frames)
GRAD_CLIP = 1.0


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


def _weight_digest(root: Path) -> str | None:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    with open(manifest_path, encoding="utf-8") as handle:
        entries = json.load(handle).get("files", [])
    return next((e["sha256"] for e in entries if e["path"] == WEIGHTS_FILE), None)


def _trainable_names(model: Any) -> list[str]:
    """The fusion head's tensors; the vision tower, the text tower, the text projection and the logit scale stay
    frozen."""
    return [name for name, _ in model.named_parameters() if name.startswith(_TRAINABLE_PREFIXES)]


def _check_artifact_manifest(manifest: Mapping[str, Any], artifact_dir: Path, base_sha256: str) -> None:
    """Refuse an adapter that names another base, another format or a file that does not match its digest."""
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
    base = manifest.get("base", {})
    if base.get("model_id") != MODEL_ID or base.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"artifact was trained on {base.get('model_id')}@{base.get('revision')}, not {MODEL_ID}@{MODEL_REVISION}"
        )
    if base.get("weight_sha256") != base_sha256:
        raise ValueError("artifact base weight digest does not match the verified snapshot")
    files = manifest.get("files") or []
    if len(files) != 1 or files[0].get("path") != ADAPTER_WEIGHTS:
        raise ValueError(f"artifact manifest must list exactly {ADAPTER_WEIGHTS}")
    weights = artifact_dir / ADAPTER_WEIGHTS
    if not weights.is_file():
        raise FileNotFoundError(f"artifact weights missing: {weights}")
    size = weights.stat().st_size
    if size != files[0].get("bytes"):
        raise ValueError(f"{ADAPTER_WEIGHTS}: size {size} != manifest {files[0].get('bytes')}")
    digest = _sha256(weights)
    if digest != files[0].get("sha256"):
        raise ValueError(f"{ADAPTER_WEIGHTS}: sha256 {digest} != manifest {files[0].get('sha256')}")
    names = manifest.get("tensors") or []
    if not names or any(not str(n).startswith(_TRAINABLE_PREFIXES) for n in names):
        raise ValueError("artifact tensors must all belong to the fusion head (mit, visual projections, prompt generator)")
    adapter = manifest.get("adapter") or {}
    labels = adapter.get("labels")
    if not isinstance(labels, list) or len(labels) < MIN_LABELS:
        raise ValueError("artifact manifest must record adapter.labels, the closed label set the head was trained on")


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
    _batch_runner: Callable[[list[list[Image.Image]], list[str]], np.ndarray] | None = None
    _model: Any = None
    _processor: Any = None
    weight_sha256: str | None = None
    adapter: dict[str, Any] | None = None

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
        weight_sha256 = _weight_digest(root) if (root / MANIFEST_NAME).is_file() else None
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

        def batch_runner(clips: list[list[Image.Image]], labels: list[str]) -> np.ndarray:
            pixel_values = processor.image_processor(clips, return_tensors="pt")["pixel_values"]
            text = processor.tokenizer(labels, padding=True, return_tensors="pt")
            with torch.inference_mode():
                outputs = model(
                    pixel_values=pixel_values.to(resolved_device),
                    input_ids=text["input_ids"].to(resolved_device),
                    attention_mask=text["attention_mask"].to(resolved_device),
                )
            return outputs.logits_per_video.float().cpu().numpy()

        return cls(runner, resolved_device, batch_runner, model, processor, weight_sha256, None)

    def _require_model(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            raise RuntimeError("this pipeline has no loaded model (injected runner); use from_pretrained")
        return self._model, self._processor

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

    # ------------------------------------------------------------------------------------------------------
    # Adaptation contract: batched classification, corpus evaluation, bounded fine-tuning, artifacts
    # ------------------------------------------------------------------------------------------------------

    @staticmethod
    def _result(logits: np.ndarray, names: list[str]) -> dict[str, Any]:
        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        if logits.shape != (len(names),) or not np.all(np.isfinite(logits)):
            raise RuntimeError(f"backend returned logits of shape {logits.shape} for {len(names)} labels")
        shifted = np.exp(logits - logits.max())
        probs = shifted / shifted.sum()
        order = np.argsort(-probs, kind="stable")
        predictions = [
            {"label": names[index], "probability": float(probs[index]), "logit": float(logits[index])}
            for index in order
        ]
        return {"predictions": predictions, "top1": predictions[0]["label"], "labels": names}

    def classify_batch(
        self,
        clips: Sequence[Sequence[Image.Image]],
        labels: Sequence[str],
        *,
        batch_size: int = EVAL_BATCH_SIZE,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Rank `labels` for many clips, `batch_size` clips per forward; one result (as `classify` returns it, minus
        the frame fields) per clip, in order. With an injected runner and no batch runner the clips are ranked one by
        one through the runner."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 64:
            raise ValueError("batch_size must be an int in 1..64")
        names = format_labels(labels)
        checked = [validate_clip(clip) for clip in clips]
        out: list[dict[str, Any]] = []
        for start in range(0, len(checked), batch_size):
            batch = checked[start : start + batch_size]
            if self._batch_runner is not None:
                logits = np.asarray(self._batch_runner(batch, names), dtype=np.float64)
                if logits.shape != (len(batch), len(names)):
                    raise RuntimeError(f"backend returned logits of shape {logits.shape} for {len(batch)} clips x {len(names)} labels")
                rows = list(logits)
            else:
                rows = [np.asarray(self._runner(clip, names), dtype=np.float64) for clip in batch]
            for clip, row in zip(batch, rows, strict=True):
                out.append({**self._result(row, names), "n_frames": len(clip), "frame_size": list(clip[0].size)})
            if progress is not None:
                progress(min(start + batch_size, len(checked)), len(checked))
        return out

    def evaluate(
        self,
        records: Sequence[Mapping[str, Any]],
        labels: Sequence[str],
        *,
        batch_size: int = EVAL_BATCH_SIZE,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Rank the closed label set for every validated record and score the rankings against the record labels
        with ``metrics.classification_metrics`` (top-1 / top-3 accuracy, macro recall and F1, confusion). Works with
        an injected runner too."""
        from .metrics import classification_metrics
        from .samples import validate_dataset

        checked = validate_dataset(records, labels, min_records=1, max_records=MAX_EVAL_RECORDS, min_per_class=1)["records"]
        names = format_labels(labels)
        started = time.perf_counter()
        results = self.classify_batch([r["frames"] for r in checked], names, batch_size=batch_size, progress=progress)
        rankings = [[p["label"] for p in r["predictions"]] for r in results]
        metrics = classification_metrics(rankings, checked, names)
        return {
            **metrics,
            "labels": names,
            "predictions": [r["top1"] for r in results],
            "top1_probability": [r["predictions"][0]["probability"] for r in results],
            "rankings": rankings,
            "verdict": "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample",
            "adapted": self.adapter is not None,
            "seconds": round(time.perf_counter() - started, 3),
        }

    def _encode_clips(self, records: Sequence[Mapping[str, Any]], batch_size: int, progress: Callable[[int, int], None] | None = None) -> tuple[Any, Any]:
        """Run the frozen vision tower once per clip: pooled CLS features (n, NUM_FRAMES, VISION_WIDTH) and patch
        features (n, NUM_FRAMES, FRAME_TOKENS - 1, VISION_WIDTH), kept on the model's device in float32."""
        model, processor = self._require_model()
        import torch

        cls_out, patch_out = [], []
        with torch.no_grad():  # not inference_mode: the cached tensors feed autograd during adapt
            for start in range(0, len(records), batch_size):
                batch = [r["frames"] for r in records[start : start + batch_size]]
                pixel_values = processor.image_processor(batch, return_tensors="pt")["pixel_values"].to(self.device)
                vision = model.vision_model(pixel_values=pixel_values.flatten(0, 1))
                cls_out.append(vision[1].reshape(len(batch), NUM_FRAMES, -1).clone())
                patch_out.append(vision[0][:, 1:, :].reshape(len(batch), NUM_FRAMES, FRAME_TOKENS - 1, -1).clone())
                if progress is not None:
                    progress(min(start + batch_size, len(records)), len(records))
        return torch.cat(cls_out), torch.cat(patch_out)

    def _encode_labels(self, names: Sequence[str]) -> Any:
        model, processor = self._require_model()
        import torch

        text = processor.tokenizer(list(names), padding=True, return_tensors="pt")
        with torch.no_grad():
            return model.get_text_features(input_ids=text["input_ids"].to(self.device), attention_mask=text["attention_mask"].to(self.device)).clone()

    def _head_logits(self, cls_features: Any, patch_features: Any, text_features: Any) -> Any:
        """The fusion head on cached tower features: exactly what `XCLIPModel.forward` computes after the towers
        (parity with the full forward is asserted by the model-backed tests)."""
        model, _ = self._require_model()
        import torch

        batch = cls_features.shape[0]
        video = model.visual_projection(cls_features.reshape(batch * NUM_FRAMES, -1)).view(batch, NUM_FRAMES, -1)
        video = model.mit(video)[1]
        img = model.prompts_visual_layernorm(patch_features.reshape(batch * NUM_FRAMES, FRAME_TOKENS - 1, -1))
        img = (img @ model.prompts_visual_projection).view(batch, NUM_FRAMES, -1, video.shape[-1]).mean(dim=1)
        text = text_features.unsqueeze(0).expand(batch, -1, -1)
        text = text + model.prompts_generator(text, img)
        video = video / video.norm(p=2, dim=-1, keepdim=True)
        text = text / text.norm(p=2, dim=-1, keepdim=True)
        return torch.einsum("bd,bkd->bk", video, text) * model.logit_scale.exp()

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None,
        labels: Sequence[str],
        *,
        epochs: int = 8,
        lr: float = 1e-5,
        batch_size: int = 16,
        seed: int = 0,
        progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded fine-tuning of the fusion head on labelled clips with the cross-entropy over the closed label set
        (the model's own contrastive scoring, read as a classifier over `labels`). The frozen towers — the ViT-B/32
        vision encoder with its cross-frame attention and the CLIP text encoder — are run once per clip and label
        set under no gradient and their outputs are cached, so each step runs only the head; the logits equal the
        full model's exactly. AdamW (no weight decay), gradient clipping at `GRAD_CLIP`, seeded shuffling, no
        scheduler, no augmentation. Epoch 0 records the frozen model's validation metrics; the epoch with the highest
        validation top-1 accuracy (the earliest on ties) is kept. On any exception the frozen head is restored."""
        from .metrics import classification_metrics
        from .samples import validate_dataset

        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 100:
            raise ValueError("epochs must be an int in 1..100")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 128:
            raise ValueError("batch_size must be an int in 1..128")
        if not isinstance(lr, int | float) or isinstance(lr, bool) or not 0 < lr <= 1e-2:
            raise ValueError("lr must be a number in (0, 1e-2]")
        names = format_labels(labels)
        train_checked = validate_dataset(train, names)["records"]
        val_checked = validate_dataset(val, names, min_records=1, min_per_class=1)["records"] if val is not None else None
        model, _processor = self._require_model()
        import torch

        started = time.perf_counter()
        head_names = _trainable_names(model)
        params = {name: param for name, param in model.named_parameters() if name in set(head_names)}
        n_trainable = sum(p.numel() for p in params.values())
        for name, param in model.named_parameters():
            param.requires_grad_(name in params)
        backup = {name: param.detach().clone() for name, param in params.items()}
        cudnn_flags = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        try:
            model.eval()
            text_features = self._encode_labels(names)
            targets = torch.tensor([names.index(r["label"]) for r in train_checked], device=self.device)
            cls_train, patch_train = self._encode_clips(train_checked, EVAL_BATCH_SIZE)
            cached = None
            if val_checked is not None:
                cached = self._encode_clips(val_checked, EVAL_BATCH_SIZE)
            cache_seconds = round(time.perf_counter() - started, 3)

            def score_val() -> dict[str, Any] | None:
                if val_checked is None or cached is None:
                    return None
                with torch.no_grad():
                    rows = []
                    for start in range(0, len(val_checked), EVAL_BATCH_SIZE):
                        rows.append(self._head_logits(cached[0][start : start + EVAL_BATCH_SIZE], cached[1][start : start + EVAL_BATCH_SIZE], text_features))
                    logits = torch.cat(rows).float().cpu().numpy()
                rankings = [[names[i] for i in np.argsort(-row, kind="stable")] for row in logits]
                m = classification_metrics(rankings, val_checked, names)
                return {k: m[k] for k in ("top1_accuracy", "top3_accuracy", "macro_recall", "macro_f1", "n")}

            history: list[dict[str, Any]] = [{"epoch": 0, "train_loss": None, "val": score_val(), "note": "frozen model"}]
            if progress is not None:
                progress(history[-1])
            best_epoch, best_acc = 0, (history[0]["val"] or {}).get("top1_accuracy", -1.0)
            best_state = {name: param.detach().clone() for name, param in params.items()}
            optimizer = torch.optim.AdamW(list(params.values()), lr=lr, weight_decay=0.0)
            rng = random.Random(seed)
            torch.manual_seed(seed)
            order = list(range(len(train_checked)))
            for epoch in range(1, epochs + 1):
                rng.shuffle(order)
                model.train()
                total, steps = 0.0, 0
                for start in range(0, len(order), batch_size):
                    idx = torch.tensor(order[start : start + batch_size], device=self.device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = self._head_logits(cls_train[idx], patch_train[idx], text_features)
                    loss = torch.nn.functional.cross_entropy(logits, targets[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(params.values()), GRAD_CLIP)
                    optimizer.step()
                    total += float(loss.detach())
                    steps += 1
                model.eval()
                entry = {"epoch": epoch, "train_loss": round(total / max(steps, 1), 5), "val": score_val()}
                history.append(entry)
                if progress is not None:
                    progress(entry)
                acc = (entry["val"] or {}).get("top1_accuracy")
                if val_checked is None or (acc is not None and acc > best_acc):
                    best_epoch, best_acc = epoch, acc if acc is not None else best_acc
                    best_state = {name: param.detach().clone() for name, param in params.items()}
            with torch.no_grad():
                for name, param in params.items():
                    param.copy_(best_state[name])
        except BaseException:
            with torch.no_grad():
                for name, param in params.items():
                    param.copy_(backup[name])
            model.eval()
            raise
        finally:
            for param in model.parameters():
                param.requires_grad_(False)
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn_flags
        self.adapter = {
            "labels": names,
            "trainable_names": head_names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "batch_size": batch_size,
            "best_epoch": best_epoch,
            "selection": "highest validation top-1 accuracy" if val_checked is not None else "final epoch (no validation split)",
            "lr": lr,
            "seed": seed,
            "grad_clip": GRAD_CLIP,
            "objective": "cross-entropy over the closed label set on cached tower features",
            "n_train": len(train_checked),
            "n_val": len(val_checked) if val_checked is not None else 0,
            "cache_seconds": cache_seconds,
            "history": history,
            "seconds": round(time.perf_counter() - started, 3),
        }
        return dict(self.adapter)

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the trained tensors as safetensors plus a manifest naming the base, the digests, the label set and
        the training configuration. Requires a prior `adapt`."""
        model, _processor = self._require_model()  # refuse before importing torch
        import torch
        from safetensors.torch import save_file

        if self.adapter is None:
            raise RuntimeError("nothing to save: call adapt() first")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = list(self.adapter["trainable_names"])
        state = model.state_dict()
        tensors = {name: state[name].detach().cpu().contiguous() for name in names}
        weights = out / ADAPTER_WEIGHTS
        save_file(tensors, str(weights), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "version": ARTIFACT_VERSION,
            "base": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "weight_file": WEIGHTS_FILE, "weight_sha256": self.weight_sha256},
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": names,
            "files": [{"path": ADAPTER_WEIGHTS, "bytes": weights.stat().st_size, "sha256": _sha256(weights)}],
            "torch": torch.__version__,
            "metadata": dict(metadata or {}),
        }
        with open(out / ADAPTER_MANIFEST, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        return out

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Overlay a saved adapter onto this (freshly loaded) pipeline after checking its manifest, digest and exact
        tensor set. Refuses tensors outside the fusion head."""
        model, _processor = self._require_model()  # refuse before importing safetensors
        from safetensors.torch import load_file

        artifact = Path(artifact_dir)
        manifest_path = artifact / ADAPTER_MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(f"artifact manifest missing: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        _check_artifact_manifest(manifest, artifact, self.weight_sha256 or "")
        expected = _trainable_names(model)
        if sorted(manifest["tensors"]) != sorted(expected):
            raise ValueError("artifact tensor set does not match its recorded configuration")
        tensors = load_file(str(artifact / ADAPTER_WEIGHTS))
        if sorted(tensors) != sorted(expected):
            raise ValueError("artifact tensor names differ from the manifest")
        state = model.state_dict()
        for name, tensor in tensors.items():
            if tuple(tensor.shape) != tuple(state[name].shape):
                raise ValueError(f"artifact tensor {name} has shape {tuple(tensor.shape)}, base has {tuple(state[name].shape)}")
        model.load_state_dict({k: v.to(state[k].device, state[k].dtype) for k, v in tensors.items()}, strict=False)
        model.eval()
        self.adapter = {**manifest["adapter"], "trainable_names": expected, "history": manifest.get("history", [])}
        return dict(self.adapter)

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> XClipVideoClassificationPipeline:
        """Check the adapter manifest against the base snapshot's recorded weight digest, load the verified base, then
        overlay the adapter (checked again, and the tensor set, before deserialising). A refused manifest never loads
        a model."""
        artifact = Path(artifact_dir)
        manifest_path = artifact / ADAPTER_MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(f"artifact manifest missing: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        _check_artifact_manifest(manifest, artifact, _weight_digest(root) or "")
        pipe = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipe.load_artifact(artifact_dir)
        return pipe
