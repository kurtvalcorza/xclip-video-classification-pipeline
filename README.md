# X-CLIP base/32 video classification pipeline

DIMER pipeline for **X-CLIP** (`microsoft/xclip-base-patch32`), Microsoft's ~197M-parameter video–text model (a CLIP ViT-B/32 frame encoder with cross-frame attention, a multi-frame integration transformer and a CLIP text encoder, trained on Kinetics-400) that ranks a caller-supplied set of class names for one 8-frame clip — pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot. The pipeline accepts exactly 8 PIL frames and 2–32 class names, returns a softmax over those names (a relative ranking, not a calibrated probability) with the raw logits and the top-1 name, and ships `sample_frames` / `frames_from_animation` so any longer frame sequence or Pillow-readable animation can be reduced to a clip; it never abstains. On top of inference it carries the **adaptation contract for a closed label set**: labelled `{id, frames, label}` clip records (decoded from video containers by PyAV or from Pillow animations), top-1 / top-3 accuracy with macro recall and F1 beside two non-adapted baselines, a bounded fine-tuning of the fusion head on cached tower features, and a verified safetensors adapter that reloads against the pinned base.

## Upstream alignment

- Model: `microsoft/xclip-base-patch32`
- Revision: `a2e27a78a2b5d802e894b8a1ef14f3a8ce490963`
- Upstream weight license: MIT
- Upstream task: video classification (Kinetics-400, fully supervised); used here zero-shot over a caller-supplied label set
- Repository adaptation: **bounded supervised fine-tuning of the fusion head** — the multi-frame integration transformer, the two visual projections and the video-specific prompt generator (10,247,680 of 196,585,729 parameters) on `{id, frames, label}` records with the cross-entropy over the closed label set; the ViT-B/32 vision tower, the CLIP text tower, the text projection and the logit scale stay frozen and their outputs are cached. Trained tensors are exported as a safetensors adapter with a manifest and overlaid on a freshly loaded, re-verified base. Every other label set is inference-only; the tuned head ranks them too, which the tutorial shows on five drawn clips and the card records.

## Quick start

```python
from PIL import Image
from xclip_video_classification_pipeline import (
    XClipVideoClassificationPipeline,
    frames_from_animation,
    sample_frames,
)

pipe = XClipVideoClassificationPipeline.from_pretrained()        # stages + verifies weights/xclip-base-patch32 first
frames = sample_frames(frames_from_animation(Image.open("clip.gif")))   # any Pillow-readable animation -> 8 frames
result = pipe.classify(frames, ["playing soccer", "juggling balls", "dancing"])
print(result["top1"], result["predictions"])   # softmax over YOUR label set; no calibration, no abstention

from xclip_video_classification_pipeline import SAMPLE_LABELS, fetch_sample_dataset
splits = fetch_sample_dataset()                          # 300 digest-pinned HMDB51 clips of ten actions, 181 / 53 / 66 by source video
print(pipe.evaluate(splits["test"], SAMPLE_LABELS)["top1_accuracy"])      # frozen zero-shot top-1 on the closed label set
pipe.adapt(splits["train"], splits["validation"], SAMPLE_LABELS)          # fusion head, highest-validation-top-1 epoch kept
print(pipe.evaluate(splits["test"], SAMPLE_LABELS)["top1_accuracy"])
pipe.save_artifact("outputs/adapter")
again = XClipVideoClassificationPipeline.from_artifact("outputs/adapter")   # re-verifies the base, checks the manifest, overlays
```

Install into a Python 3.12 environment that already holds the pinned dependencies with `pip install -e . --no-deps`; run `pytest -q -o addopts= tests` for the offline test suite (no weights needed; `tests/test_model_backed.py` runs only where the snapshot is staged). `av==18.1.0` (PyAV, BSD-3, bundles FFmpeg) and `pyarrow==25.0.1` are pinned for the clip decoder and the parquet sample. On a fresh clone the manifest is committed but the weights are not: `XClipVideoClassificationPipeline.from_pretrained(allow_download=True)` fetches exactly the missing manifest-listed files at the pinned revision, then verifies them.

## Weights layout

```
weights/xclip-base-patch32/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + SHA-256 (9 files)
  config.json                # XClipModel: ViT-B/32 frame encoder (8 frames), 1-layer MIT, CLIP text tower, prompt generator
  preprocessor_config.json   # VideoMAEImageProcessor: shorter side 224, centre crop 224, ImageNet mean/std
  merges.txt  vocab.json  tokenizer.json  special_tokens_map.json  tokenizer_config.json
  model.safetensors          # git-ignored, 786,414,772 bytes
  README.md
```

`pytorch_model.bin` exists upstream and is deliberately not listed (pickle; DIMER does not accept `.bin`).

## Input ceilings and request parameters

`NUM_FRAMES = 8` (exactly; `sample_frames` subsamples longer sequences), `FRAME_SIZE = 224`; `MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096` per frame; `MIN_LABELS = 2`, `MAX_LABELS = 32`, `MAX_LABEL_CHARS = 64` (distinct after normalisation; `MAX_TEXT_TOKENS = 77`). The label set is the request parameter: the softmax ranks only the names you supply. See `MODEL_CARD.md` for what the probabilities are and are not, the recorded chance-level result on drawn clips, and the measured CPU timings.

## Adaptation contract

- **Records:** `{id, frames, label}` — exactly `NUM_FRAMES` (8) PIL frames of one size within the side ceilings (`decode_clip` reads a video file into 8 uniformly spaced frames with PyAV; `sample_frames` subsamples any longer sequence), `label` one of the closed label set (normalised like `format_labels`), and an optional `source` naming the video the clip was cut from; `validate_dataset` checks the structure and the label set, `split_dataset` de-duplicates by decoded pixels and keeps every clip of one source in one split, `check_split_disjoint` asserts it. The default sample (`samples.py`) is row group 0 of the HMDB51 test shard (`mteb/HMDB51` parquet repack of the CC BY 4.0 dataset; every test clip of the first ten action classes, 300 clips) read over an HTTPS range request at an immutable Hub revision and refused on any SHA-256 or byte-total mismatch; `build_sample_dataset` draws a seeded source-grouped 181 / 53 / 66; `load_byod_dataset` reads a zip or directory of clips plus `labels.csv`.
- **Measures (`metrics.py`):** `classification_metrics` — top-1 and top-3 accuracy, macro recall and macro F1 over the label set, the per-label confusion and the mean rank of the reference; `chance_baseline` (1 / |labels| by construction) and `majority_baseline` (the most frequent training label for every clip).
- **Fine-tuning:** `adapt(train, val, labels, *, epochs=8, lr=1e-5, batch_size=16, seed=0)` runs the frozen towers once per clip and label set (the pooled CLS and 49 patch features of every frame; the label embeddings) and caches them, then trains the fusion head on those features with the cross-entropy over the closed label set, AdamW (no weight decay), gradient clipping at 1.0 and seeded shuffling; the logits equal the full model's exactly. Epoch 0 records the frozen validation rates; the epoch with the highest validation top-1 (the earliest on ties) is kept; on any exception the frozen head is restored.
- **Artifact:** `save_artifact` writes `adapter.safetensors` (about 41 MB) + `manifest.json` (`org.valcorza.xclip-base-patch32.adapter.v1`: base identity and weight digest, tensor names, file size and SHA-256, the label set, configuration, history); `from_artifact` re-verifies the base and checks the manifest, digest and exact tensor set before deserialising.
- **Build record (Tesla T4, seed 42 split):** frozen top-1 @P:FROZEN_TOP1@ / macro F1 @P:FROZEN_F1@ on the 66 held-out clips (@P:FROZEN_READ@), adapted **@P:ADAPTED_TOP1@** / **@P:ADAPTED_F1@** (epoch @P:BEST_EPOCH@ of 8), reload parity 8/8; the five drawn clips after adaptation: @P:DRAWING_AFTER@. One seeded split of one 300-clip sample; no dispersion estimate — one clip is 1.5 points.

## Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/xclip-video-classification-pipeline/blob/main/tutorials/xclip_video_classification_colab.ipynb)

`tutorials/xclip_video_classification_colab.ipynb` is declared `E2E` / `GUIDED` under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the three pipeline modules, the model identity, the manifest digests and the runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py`). Its default `Run all` path stages and verifies the pinned snapshot, fetches the pinned HMDB51 row group and decodes the 300 clips into 8-frame records split 181 / 53 / 66 by source video, ranks five drawn clips through the inference contract (the four recorded misses of the inference-only tutorial are kept — the model does not read drawn motion), measures the frozen model's top-1 / top-3 accuracy, macro recall and F1 on the held-out clips beside the chance and majority-label baselines, runs `adapt` with validation-accuracy epoch selection, scores the held-out clips again, writes six clip panels and re-ranks the drawn clips with the adapted model, and exports the adapter and reloads it with verified ranking parity. BYOD (your own clips plus `labels.csv`) is optional and gated off by default. See `tutorials/README.md` for the registry and `docs/release-verification.md` for the release gate.

## Release status

**Candidate.** Static/unit checks — including the standalone generator parity checks (`tools/build_notebook.py --check`, `tests/test_notebook_parity.py`) — do not constitute clean-runtime notebook evidence. The earlier `TASK-INFERENCE` notebook's Kaggle CPU run (2026-09-14) is retained as history and is not evidence for the `E2E` blob; the supported-runtime run of the exact release revision is recorded in `docs/release-verification.md` when it exists.

## Documentation

- `MODEL_CARD.md` — MODEL_CARD_SPEC 1.1 card, provenance digests, input/output contract, measured runtime.
- `docs/WEIGHTS.md` — weight provenance, the pinned-processor quirk, the adapter artifacts, the sample corpus and hosting notes.
- `STATUS.md` — release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). The upstream weights are MIT; the HMDB51 sample is CC BY 4.0 (Serre Lab, Brown University) and is not redistributed; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
