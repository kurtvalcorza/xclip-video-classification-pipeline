---
license: mit
model_card_spec: "1.1"
pipeline_tag: video-classification
task: "Others - Video Classification"
base_model: microsoft/xclip-base-patch32
date_published: "2022-08-25"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt` 2022-08-25T13:06:15Z, https://huggingface.co/api/models/microsoft/xclip-base-patch32 — the Transformers-format conversion); the X-CLIP paper is arXiv:2208.02816 (2022-08) and the pinned revision is the Hub's `main` as of 2026-09-14"
---

# X-CLIP base/32 — Zero-Shot Video Classification (Inference)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-microsoft%2Fxclip--base--patch32-ffcc4d?style=flat)](https://huggingface.co/microsoft/xclip-base-patch32)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-microsoft%2FVideoX-181717?style=flat&logo=github&logoColor=white)](https://github.com/microsoft/VideoX/tree/master/X-CLIP)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2208.02816-b31b1b.svg)](https://arxiv.org/abs/2208.02816)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — stage and verify the pinned upstream revision in a fresh runtime, validate an input, run the task, and inspect and export the outputs:

- **Task Inference Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/xclip-video-classification-pipeline/blob/main/tutorials/xclip_video_classification_colab.ipynb) [`xclip_video_classification_colab.ipynb`](https://github.com/kurtvalcorza/xclip-video-classification-pipeline/blob/main/tutorials/xclip_video_classification_colab.ipynb)  
  *Five 8-frame cartoon clips drawn in code ranked against their five class names with the pinned `microsoft/xclip-base-patch32` weights; `top1_accuracy` against a chance baseline as sanity evidence only (four recorded misses kept on purpose) — no Kinetics benchmark.*

---

#### Description

`microsoft/xclip-base-patch32` is the Transformers-format release of the X-CLIP *base* model with 32-pixel patches and 8 frames, from "Expanding Language-Image Pretrained Models for General Video Recognition" (Ni et al., arXiv:2208.02816), converted by the Hugging Face team and pinned here to revision `a2e27a78a2b5d802e894b8a1ef14f3a8ce490963` (the Hub's `main` on 2026-09-14). The snapshot `config.json` declares `XClipModel`: a CLIP ViT-B/32 frame encoder (12 layers, hidden size 768, 224×224 frames, 8 frames per clip, with cross-frame attention message tokens), a one-layer multi-frame integration transformer (hidden size 512) that fuses the frame embeddings into one video embedding, a CLIP text encoder (12 layers, hidden size 512, 77-token context, vocabulary 49,408), and a two-layer video-specific prompt generator that conditions the text embeddings on the video — about 197M parameters in the 786 MB float32 `model.safetensors`. The model was trained fully supervised on Kinetics-400 (the upstream card reports top-1 80.4 % and top-5 95.0 % there, upstream-reported and not reproduced here); used zero-shot, it scores a clip against any set of class names. At inference the processor (`VideoMAEImageProcessor`, `preprocessor_config.json`: shorter side to 224, centre crop 224×224, ImageNet mean/std) encodes the 8 frames, the tokenizer encodes each class name, and the model returns one logit per name; this package softmaxes those logits over the supplied names. Nothing is trained or adapted here. What this repository adds is packaging: `verify_snapshot` and `stage_missing_files` (manifest digest checking and fresh-clone staging), `XClipVideoClassificationPipeline.from_pretrained` (verified local loading with `trust_remote_code=False`, calling the two sub-processors explicitly because the pinned `XCLIPProcessor`'s `videos=` keyword yields no pixel values), `format_labels`, `validate_clip`, `sample_frames`, `frames_from_animation`, `classify` (input validation, ranking, backend output checks), and the `validate_inputs` and `evaluation_report` stage helpers.

#### Intended Use and Limitations

The uses below are the ones the package was built to support; everything else is either out of scope (§Out-of-scope use cases) or prohibited (§Use cases).

###### Primary Intended Uses

The task is closed-set zero-shot video classification: input one clip of exactly 8 frames (`PIL.Image.Image`, one common size, any mode, converted to RGB) and 2–32 free-text class names; output the names ranked by a softmax over them with the raw logits and the top-1 name. Envisioned applications are tagging short clips of everyday human actions and sports against a vocabulary the operator defines, triage of video collections by a handful of coarse categories, and prototyping of video–language systems — with the ranking checked by a person before it is stored or acted on. Within DIMER the pipeline is an inference component and a zero-configuration baseline for video classification, inference-only until a DIMER video dataset contract exists, not a certified classifier for any action vocabulary.

###### Primary Intended Users

Intended users are machine-learning engineers, computer-vision developers and researchers integrating zero-shot video classification into research prototypes, internal tooling, or the DIMER workbench. A user is expected to understand that the probabilities are a *softmax over the label set they supplied* — a relative ranking that sums to 1, not a calibrated probability, and never an abstention — that a label set missing the true class still produces a confident top-1, that the model was trained on Kinetics-400 human-action footage so drawn animations, static scenes, screen recordings, non-human subjects and non-English names are distribution shifts (the tutorial's cartoons scored at chance), that a clip is exactly 8 uniformly sampled frames and everything between them is unseen, that the forward pass is deterministic on a fixed device but GPU and CPU logits can differ slightly, and that accuracy can only be measured on labelled clips they supply. Users who need video decoding, temporal localisation, open-set recognition or batch throughput are expected to know none of that is provided here.

###### Out-of-scope use cases

1. **Capability boundary:** no video decoding (frames or a Pillow-readable animation are the input), no temporal localisation or per-frame output (one ranking per clip), no abstention or open-set rejection (the softmax always crowns one name), no calibrated confidence, no video–text retrieval over a corpus, no batching across clips, and no reading of drawn or diagrammatic motion — the smoke run ranked `a ball standing still` first for five cartoon clips of different motions.
2. **Input boundary:** `classify` rejects a single image or a clip that is not exactly `NUM_FRAMES = 8` frames of one size (`sample_frames` reduces longer sequences), frames with sides below `MIN_IMAGE_SIDE = 16` px or above `MAX_IMAGE_SIDE = 4096` px, a single string instead of a list, fewer than `MIN_LABELS = 2` or more than `MAX_LABELS = 32` names, and empty, non-string, over-long (`MAX_LABEL_CHARS = 64`) or duplicate-after-normalisation names (`ValueError`/`TypeError`). Every frame is resized on its shorter side to 224 and centre-cropped, so wide frames lose their edges and small subjects lose detail; names longer than the 77-token CLIP context are truncated by the tokenizer.
3. **Input boundary:** the training data is Kinetics-400 — 10-second YouTube clips of 400 human actions. Cartoons and synthetic renders (the tutorial's), static or near-static footage, non-human activity, surveillance or dashcam footage, screen recordings, clips much shorter or longer than the 8 sampled frames can represent, and non-English class names fall outside what the upstream authors evaluated and what this repository measured; results on them are undefined, not merely degraded. A clip of eight blank frames still produces a confident ranking (see §Risks and harms).
4. **Decision boundary:** not for autonomous decisions that act on rankings — content moderation, safety or incident detection, behavioural monitoring of people, automated tagging that is published unreviewed — without a human checking the clip, and a locally measured top-1/top-5 accuracy on the deployment's own labelled clips with the deployment's own label set.

#### Factors

###### Groups

This pipeline is human-centric by construction: Kinetics-400 is a dataset of people performing actions, most class names describe human behaviour, and any clip of people is ranked against names that may describe them (`dancing`, `fighting`, `crying`). Published audits of Kinetics document geographic and demographic skews in its YouTube sourcing and action vocabulary, and the CLIP text tower inherits the social biases of its web pretraining in how names match footage; neither the upstream authors nor this repository audited ranking accuracy by the depicted people's gender, age, skin tone, disability, dress or setting, and the tutorial contains no people at all. Non-human groups whose accuracy is unknown, not known to be equal: non-photographic clips (the tutorial's cartoons, ranked at chance), non-Western activities and settings, low-light or low-resolution footage, and names in languages other than English. An operator whose clips contain people is responsible for a fairness audit on their own clips, stratified by depicted group and action, before relying on the output — and for deciding which names about people are permitted at all (see §Use cases).

###### Instrumentation

The upstream "instrument" is YouTube footage as uploaded by its authors — consumer cameras and phones, edited, compressed and resized by the platform — temporally sampled by the Kinetics pipeline into 8 frames per clip at training time. Inference clips arrive from whatever produced them — a phone, a webcam, a GIF exported by a tool, a drawing library — and frame rate, the span the 8 frames cover, resolution, exposure, compression and aspect ratio all change the visual evidence; the fixed shorter-side-224 resize and centre crop discard edges and detail regardless of the source, and `sample_frames` keeps only 8 evenly spaced frames of whatever sequence it is given. The pipeline validates frame count, size and label shape only; it cannot detect a non-photographic clip, a span too short to show the action, or a label set that misses the truth. The synthetic tutorial clips (flat Pillow shapes moving over eight frames, no texture or camera motion) are themselves a rendering instrument unlike any Kinetics clip, which is why their chance-level result is recorded rather than tuned away.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, float32 on CPU; CUDA is used automatically when visible (float32) but was not exercised for this card. Measured on the reference machine with the GPU hidden (`CUDA_VISIBLE_DEVICES=-1`) and the Hub offline (`HF_HUB_OFFLINE=1`): `verify_snapshot` on the 9-file, 786 MB snapshot 0.41 s; load 5.18 s; one 8-frame 320×240 clip with five names 0.36 s on the first call and 0.11 s thereafter; eight 4096×4096 blank frames 1.32 s — cost is one 8-frame encoding plus one text encoding per name, roughly independent of the frame resolution after the resize. Data environment: the model assumes a real-footage clip of a human action and a label set that contains its class; the synthetic tutorial clips violate the first assumption on purpose (flat cartoons), which is where the recorded chance-level result comes from. Static footage, non-human subjects, label sets that omit the truth and non-English names violate it to degrees this repository did not measure, and the pipeline reports no signal when they do.

#### Metrics

###### Performance Measures

The pipeline reports no accuracy measure. The probabilities are a softmax over the supplied names — a relative ranking that carries no calibration and no abstention — and the logits are raw video–text similarities. The repository ships the benchmark's own measure because it is what a caller would use to evaluate: **top-1 accuracy** (the top-ranked name equals the labelled class) with a **chance baseline** (the mean of 1/labels over the clips) and the rank of the correct name per clip; top-5 accuracy follows from the ranks. All need labelled clips with a class vocabulary matching the label set from the deployment domain that the caller must supply; Kinetics-400 and UCF101 are not bundled. The public `evaluation_report(results, correct_labels=None)` stage returns that report in machine-readable form: `top1_accuracy`, the `chance` baseline and one per-clip entry (top-1, its probability, the correct name, its rank) with the verdict `sample-sanity`, or the verdict `not-measurable` naming the labelled set that would be required when no classes are supplied. The upstream card's Kinetics-400 top-1 80.4 % / top-5 95.0 % (fully supervised, upstream-reported) is not reproduced or claimed by this pipeline.

###### Decision thresholds

No score threshold exists in the model: it scores every supplied name and the softmax always ranks one first; nothing is filtered or abstained. The decision parameter is the **label set** itself — which names are offered, how they are phrased, and whether the true class is among them — and the smoke run shows how much it matters: the same rolling-ball clip ranked `a ball standing still` at 0.69 under the tutorial's five descriptive names and `juggling balls` at 0.98 under five Kinetics-style names. `format_labels` applies one fixed normalisation (whitespace collapsed, lower-cased, trailing full stop removed) so the same name always produces the same query, and `sample_frames` fixes the temporal sampling to 8 evenly spaced frames including the first and last. A deployment owns its label set and its frame sampling, and decides how a ranking is verified against the clip before it is used.

###### Approaches to uncertainty and variability

This repository reports no central metric value and therefore no dispersion: the smoke run records timings and the rankings of five drawn clips, not accuracy. Run-to-run variability comes only from floating-point kernel selection across CPU builds and accelerators; there is no sampling and no seed to set, so a fixed input on fixed hardware is repeatable, but CPU and CUDA logits can differ in the low decimals and a close pair of names can swap ranks; the drawn clips use no text rendering, so their bytes do not depend on the Pillow build. On the five drawn clips the model ranked `a ball standing still` first every time (0.43–0.69), placing the correct name at rank 3, 3, 4 and 4 for the rolling, bouncing, growing and sunset clips — `top1_accuracy` 0.2 against a chance baseline of 0.2, one observation on five flat cartoons, not an estimate; under Kinetics-style names the rolling and sunset clips both ranked `juggling balls` first (0.98, 0.90); eight blank white frames ranked `a ball standing still` first (0.70) and eight noise frames `a ball bouncing up and down` (0.34), which is what an uncalibrated closed-set ranker with no abstention looks like. A caller who needs an accuracy estimate must supply labelled clips and compute top-1/top-5 over many clips or bootstrap resamples themselves; a caller who needs a confidence per clip has none from this model.

#### Ethical considerations and biases

No external ethics board, red-team, or population-specific clearance reviewed this repository or, to our knowledge, the upstream checkpoint; nothing below should be read as implying one.

###### Data

The upstream paper describes training on Kinetics-400 — about 240k training clips from YouTube videos of 400 human action classes, collected by DeepMind in 2017 with Mechanical Turk annotation — on top of CLIP's public weights, themselves trained on 400M web image–text pairs whose collection was not published; the clips show real, identifiable people performing actions, so personal data in the training corpus is present by construction, and the videos remain subject to their uploaders' rights. Neither was audited here. This repository distributes code, tests, and documentation; it does not distribute the 786,414,772-byte `model.safetensors`, which is staged locally under `weights/xclip-base-patch32/` and git-ignored, and it ships no footage — the tutorial clips are drawn in code. The operator must audit the clips they submit for personal, proprietary, or otherwise restricted content; the pipeline performs no such check and will rank a clip of a person against any names as readily as a cartoon.

###### Human Life

This pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, or housing, and it has not been validated or certified for any of them by this repository, the upstream authors, or any regulator. Foreseeable but unintended sensitive uses — detecting violence, falls, self-harm or medical events from footage, monitoring workers, students or patients, moderating user-uploaded video, tagging people's activities for profiling — would be admissible only with human review of every ranking against the clip (the model always crowns one of the supplied names and gives no signal), a locally measured accuracy on the deployment's own labelled clips stratified by depicted group, a documented label-set and sampling policy, an explicit list of names about people that are refused before they reach the model, and whatever regulatory clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** `MODEL_REVISION` is a 40-hex commit; `stage_missing_files` refuses a manifest whose `modelId`/`revision` differ from the package constants and fetches only manifest-listed files at that revision when `allow_download=True`; `verify_snapshot` then checks all 9 listed files' byte sizes and SHA-256 before any load; `from_pretrained` loads only from the verified directory with `local_files_only=True`, always passes `trust_remote_code=False`, and the smoke run loaded and classified with `HF_HUB_OFFLINE=1`. The upstream `pytorch_model.bin` (pickle) is neither listed nor loaded. A test flips one hex digit of a manifest digest and asserts the loader refuses; another asserts a foreign manifest is refused; the import-boundary tests assert that a missing or tampered snapshot is refused before `torch` or `transformers` is imported.
- **Input integrity:** the public `validate_inputs(clips, labels)` stage applies exactly the checks `classify` applies (both route through one shared private checker) and returns an input manifest recording the schema, the ceilings, each clip's observed frame count, mode and size, the normalised label set and the verdict; `validate_clip` rejects non-sequence inputs, wrong frame counts, mixed frame sizes and sides outside 16–4096 px; `format_labels` rejects non-list, too few, too many, empty, non-string, over-long or duplicate names; `classify` raises when the backend returns logits of the wrong shape or non-finite values; `evaluation_report` rejects mismatched or unknown correct labels.
- **Reproducibility:** exact `==` pins in `pyproject.toml`; deterministic forward pass and softmax; fixed name normalisation and frame sampling; the two sub-processors are called explicitly so the contract does not depend on the pinned `XCLIPProcessor`'s keyword handling; every result carries `model_id`, `model_revision`, the normalised labels, the frame count and size.
- **Refusals:** no video decoding, no batching across clips, no download without the explicit flag, no Hub access at inference time, no pickle deserialisation, no attempt to guess whether the label set contains the truth, and no filtering of label content — that policy is the operator's to implement around the pipeline.
- No statistical mitigation (class balancing, subsampling) applies: no training happens in this repository.

###### Risks and harms

- **Confident rankings of anything:** the model has no abstention — eight blank frames ranked a name first at 0.70 in the smoke run — so a clip that shows none of the supplied names, or is not footage at all, produces a confident top-1 with no signal; downstream consumers that trust `top1` (taggers, moderation rules, databases) inherit the error silently.
- **Label-set dependence:** the ranking is a function of the names offered and their phrasing (`a ball standing still` vs `juggling balls` for the same clip); a deployment that authors names carelessly authors the answers.
- **Rankings about people:** names describing people's behaviour, state or intent (`fighting`, `stealing`, `drunk`) are scored from appearance and motion with the same fluency as `playing soccer`; such rankings can cause direct harm when surfaced or acted on and are prohibited for profiling below.
- **Temporal blindness:** only 8 frames are seen; an action that happens between them, or a clip whose span is too short or too long, is ranked on the wrong evidence.
- **Automation bias:** a softmax that reads like a probability invites trust that a relative ranking has not earned.
- **Bias amplification:** any footage population or action vocabulary Kinetics under-represents (non-Western activities, non-human subjects, cartoons, non-English names) is reproduced as uneven accuracy, undetected because no per-group evaluation exists.
- **Resource use:** a 786 MB model and ~0.1 s per clip on the reference CPU; a large clip collection scales linearly, and the CUDA path was not measured.

###### Use cases

Prohibited even where the model would work: ranking clips of people in order to infer or record protected characteristics (race, ethnicity, religion, health, disability, sexual orientation), to identify, track, profile or surveil individuals, to monitor workers, students or patients, or to make or support decisions in employment, housing, credit, insurance, education, healthcare access, law enforcement or immigration; processing footage the operator has no right to process, including intimate imagery and licence-restricted material; presenting rankings as verified facts, evidence or moderation decisions without human review; and any use that violates the upstream MIT licence terms, the DIMER deployment terms, or the consent and data-protection obligations attached to the footage processed. Autonomous high-consequence actions triggered by unreviewed rankings are prohibited by the intended-use contract above.

## Immutable provenance

- Model: `microsoft/xclip-base-patch32`
- Revision: `a2e27a78a2b5d802e894b8a1ef14f3a8ce490963`
- Snapshot manifest: `weights/xclip-base-patch32/dimer-base-manifest.json`, 9 files, `totalBytes` 790034990
- `model.safetensors` SHA-256: `abf286e8cdd0612761c3e42d3a55eca998382dfa67a04a0f3fdcdfa4f150cdbb` (786,414,772 bytes, float32)
- `config.json` SHA-256: `13bb919d1ef16f3b80b03cdf5c575d688dfafd21d12566849cc557b00e058ce3` (4,718 bytes; `XClipModel`, `num_frames` 8, `patch_size` 32, `mit_num_hidden_layers` 1, `prompt_layers` 2)
- `preprocessor_config.json` SHA-256: `c14b2b5c8f26a754df62235ba79d1ca63cfdd9b3de76ee688e4a30ea1e5c6986` (309 bytes; `VideoMAEImageProcessor`, size 224, centre crop, ImageNet mean/std)
- Weight format: SafeTensors; loader `XCLIPModel.from_pretrained(<dir>, local_files_only=True, trust_remote_code=False, dtype=float32)` with `XCLIPProcessor` from the same directory, its `image_processor` called on the frame list and its `tokenizer` on the names, `pixel_values` of shape `(1, 8, 3, 224, 224)`. The upstream `pytorch_model.bin` is not part of the manifest and is never loaded.

## Input/output contract

- `XClipVideoClassificationPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)` — stages missing manifest files (only with `allow_download=True`), verifies digests, loads; `device` defaults to `cuda:0` when visible, else `cpu`; float32 on both.
- `classify(frames, labels) -> dict` with keys `predictions` (descending: `label`, `probability`, `logit`), `top1`, `labels` (normalised), `n_frames`, `frame_size`, `model_id`, `model_revision`.
- `format_labels(labels) -> list[str]`; `validate_clip(frames) -> list[Image]`; `sample_frames(frames, n=8) -> list[Image]`; `frames_from_animation(image) -> list[Image]`.
- Ceilings and constants: `NUM_FRAMES = 8`, `FRAME_SIZE = 224`, `MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `MIN_LABELS = 2`, `MAX_LABELS = 32`, `MAX_LABEL_CHARS = 64`, `MAX_TEXT_TOKENS = 77`, `INPUT_SCHEMA`.
- `validate_inputs(clips, labels, *, names) -> dict`; `evaluation_report(results, correct_labels=None, *, sample_kind) -> dict` where `correct_labels` holds one class name per result; `verify_snapshot(path=None) -> dict`; `stage_missing_files(path=None, *, allow_download=False, downloader=None) -> list[str]`.

## Runtime

- Pins: `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, `huggingface-hub==0.36.2`; Python 3.12.
- Precision: float32; preprocessing resizes each frame's shorter side to 224 and centre-crops 224×224 with ImageNet mean/std (`VideoMAEImageProcessor`, snapshot defaults) and tokenises each name with the CLIP BPE tokenizer; the model's `logits_per_video` are softmaxed over the supplied names.
- Measured 2026-09-14 in the Windows venv (`torch 2.14.0+cu130`) with `CUDA_VISIBLE_DEVICES=-1` and `HF_HUB_OFFLINE=1`, device `cpu`: `verify_snapshot` 0.41 s (9 files, 786 MB); load 5.18 s; `classify` on five synthetic 8-frame 320×240 cartoon clips (a ball rolling right, a ball bouncing, a square growing, a sun setting, a ball standing still) against their five names: rolling → `a ball standing still` 0.69 (0.36 s, first call; correct at rank 3), bouncing → `a ball standing still` 0.51 (rank 3), growing → `a ball standing still` 0.43 (rank 4), sunset → `a ball standing still` 0.46 (rank 4), static → `a ball standing still` 0.57 (correct); 0.11 s each after the first; `evaluation_report`: `top1_accuracy` 0.2, `chance` 0.2, verdict `sample-sanity`; Kinetics-style names (`playing soccer`, `bouncing on trampoline`, `sunset`, `juggling balls`, `drawing`): rolling → `juggling balls` 0.98, sunset → `juggling balls` 0.90; eight blank white 320×240 frames → `a ball standing still` 0.70; eight uniform-noise frames → `a ball bouncing up and down` 0.34; eight blank 4096×4096 frames 1.32 s; `sample_frames` on a 30-frame sequence → 8 frames; two names only → 0.59 / 0.41.
- Tutorial execution: `tutorials/xclip_video_classification_colab.ipynb` ran top-to-bottom in a fresh local kernel (all 8 code cells, 81.8 s including the 786 MB staging, same five rankings as the smoke run — `top1_accuracy` 0.2 at chance 0.2); recorded in `docs/release-verification.md` as pre-flight, not supported-runtime evidence.
- Tests: `pytest -q -o addopts= tests` — offline, no weights required; `ruff check src tests tools` clean.
- Not executed: CUDA path, real footage (only drawn clips, blank and noise frames), any accuracy measurement against labelled clips, clips of people, non-English names, the `videos=` keyword of the pinned processor (bypassed on purpose).

## References

- Ni et al. Expanding Language-Image Pretrained Models for General Video Recognition. ECCV 2022. https://arxiv.org/abs/2208.02816
- Kay et al. The Kinetics Human Action Video Dataset. 2017. https://arxiv.org/abs/1705.06950
- Radford et al. Learning Transferable Visual Models From Natural Language Supervision (CLIP). ICML 2021. https://arxiv.org/abs/2103.00020
- Upstream code: https://github.com/microsoft/VideoX/tree/master/X-CLIP
- Upstream card: https://huggingface.co/microsoft/xclip-base-patch32
- Transformers `X-CLIP` documentation: https://huggingface.co/docs/transformers/model_doc/xclip
