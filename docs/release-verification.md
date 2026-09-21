# Release verification

`tutorials/xclip_video_classification_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate record for the
notebook. The earlier `TASK-INFERENCE` notebook's Kaggle CPU run (2026-09-14, retained below) is history for a
superseded blob, not evidence for this one.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 9-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the HMDB51
  parquet-conversion revision `50bb2abb741074ef0868e59ba51af710c825cd1f` is the one other 40-hex revision the
  documents may cite);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `XClipVideoClassificationPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path,
  `read_corpus`, `build_sample_dataset(corpus, seed=SPLIT_SEED)` / `load_byod_dataset` + `split_dataset`,
  `validate_dataset` per split, `check_split_disjoint`, `write_dataset_csv`, the four dataset refusal probes, the
  ceiling print, `validate_inputs` with the three-frame refusal probe, `classify` for the five drawn clips with the
  structural checks and the `evaluation_report` against the drawn intentions, `chance_baseline`,
  `majority_baseline`, `pipe.evaluate` on the frozen model, `pipe.adapt` with its explicit hyperparameters,
  `pipe.evaluate` on the validation and test splits after adaptation with the two accuracy assertions, the drawn
  clips re-ranked after adaptation, the example panels, `pipe.save_artifact`,
  `XClipVideoClassificationPipeline.from_artifact` and the ranking-parity assertion, and the result fields
  `weight_file` / `weight_format` / `weight_sha256`, the `corpus` block), the seven expected `outputs/` paths, the
  learner-facing statements (MIT weights, the softmax over the label set, adaptation of a closed label set on
  labelled clips, the two non-adapted baselines, top-1 and top-3 accuracy, macro recall and F1, the chance and
  majority-label baselines, the closed-set cross-entropy, the frozen-tower cache, highest validation top-1, no
  dispersion estimate, the caller-owned label set, float32 on every device, the leakage guidance, the macro-over-micro
  guidance, the snapshot note, the troubleshooting block) and the gated-off BYOD default; forbidden patterns
  (credential-in-URL, any `git clone` / `github.com/kurtvalcorza` / repository import on the primary path, a mutable
  `revision='main'`, direct `from transformers import` / `XCLIPModel` / `XCLIPProcessor` / `logits_per_video` /
  `torch.inference_mode(` / `from huggingface_hub import` / `urllib.request` / `pyarrow` / `av.open(` / `safetensors`
  imports / `torch.optim` / `.backward(` / `requires_grad` / `pipe._model` / `pipe._processor` / `extractall(` use
  **outside the carried module cells**, `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `huggingface-hub`, `safetensors`, `numpy`,
`pillow`, `pyarrow` and `av`, the package with `--no-deps`, runs `ruff check src tests tools`, `tools/build_notebook.py
--check`, and the unit suite (`tests/`, including `test_adaptation.py`, `test_import_boundary.py`,
`test_role_helpers.py`, `test_notebook_parity.py`; injected runner and parquet opener, no weights —
`tests/test_model_backed.py` is skipped without the snapshot). These are source/provenance and unit checks. They are
**not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU or GPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Kaggle script kernel (pre-flight only) | Fresh GPU container that clones the candidate branch, installs the pins and runs `tests/test_model_backed.py` plus the package-API recipe probe | Builder pre-flight to catch defects and fix the recipe before spending a notebook run; **not** promotion evidence for the notebook blob |
| Local Windows-venv CPU run (pre-flight only) | Workstation venv `dimer-xclip`, `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1` | Builder pre-flight of the package API on the cached row group (the recipe sweep and the model-backed suite); **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/xclip-base-patch32/` or the row-group cache `weights/hmdb51/` (the standalone path writes the
   manifest itself, stages the missing files from the Hub and reads the pinned row group over a range request, so
   neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `EPOCHS = 8`, `LEARNING_RATE = 1e-5`, `BATCH_SIZE = 16`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`,
   `numpy==2.5.3`, `pillow==11.3.0`, `pyarrow==25.0.1`, `av==18.1.0` (an interpreter restart after the install is
   expected where the runtime's preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `XClipVideoClassificationPipeline`, `verify_snapshot`,
     `stage_missing_files`, `validate_inputs`, `evaluation_report`, `format_labels`, `validate_clip`,
     `sample_frames`, `frames_from_animation`, `classification_metrics`, `chance_baseline`, `majority_baseline`,
     `majority_label`, `fetch_corpus`, `read_corpus`, `decode_clip`, `source_of`, `build_sample_dataset`,
     `validate_dataset`, `check_split_disjoint`, `split_dataset`, `load_byod_dataset`, `write_dataset_csv`, the
     label constants and the ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting all 9 manifest entries fetched from `microsoft/xclip-base-patch32` at the
     immutable revision on a clean runtime, `verify_snapshot` returning its dict (9 files, the 786 MB
     `model.safetensors` re-hashed), and `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified
     directory in float32;
   - Section 4: `fetch_corpus` reading the pinned row group over an HTTPS range request with its SHA-256 and byte
     total matching (318 rows, about 113 MB), `read_corpus` decoding the 300 clips of the ten sample classes; the
     seeded source-grouped split into 181 / 53 / 66 with `check_split_disjoint` reporting no shared clip or source
     and the three dataset digests printed; `outputs/…_train.csv` and `outputs/…_example_clip.png` written; the four
     dataset refusal probes each raising `ValueError`;
   - Section 5: the ceilings surfaced; the five drawn clips rendered; the combined input manifest written to
     `outputs/…_input_manifest.json` (verdict `accepted`, one recorded rejection finding from the three-frame
     probe); the five clips ranked with every structural check `True`, `outputs/…_drawing_frozen.json` written and
     the `evaluation_report` verdict `sample-sanity` (the inference-only card recorded `top1_accuracy` 0.2 at
     chance 0.2 — an observation, not an assertion);
   - Section 6: the chance baseline (top-1 0.100 exactly), the majority-label baseline (≈ 0.09) and the frozen
     model's test rates (≈ 0.803 top-1 / 0.792 macro F1 in the Tesla T4 build record —
     the zero-shot prompts already rank 53 of the 66 clips first; the misses sit on `chewing` (3 of 7, the rest called `drawing a sword` or `brushing hair`) and `clapping hands` (3 of 6, called `brushing hair` or `doing a cartwheel`) — close-up upper-body actions the prompts do not separate) with four rankings printed under their references;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 10,247,680 trainable of 196,585,729 parameters,
     and an eight-epoch history with the validation top-1 rising (build record: 0.811 → 0.925 → 0.962 → 0.962 → 0.962 → 0.962 → 0.962 → 0.962 → 0.962, `best_epoch`
     2);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison, the per-label
     recall and `outputs/…_evaluation_report.json` written (the cell asserts the adapted test top-1 is at least the
     frozen one and above chance — 0.909 against 0.803 in the build record, macro F1
     0.792 → 0.905; the adapted model also clears the majority baseline, reported, not asserted);
   - Section 9: six example panels under `outputs/…_examples/`; the five drawn clips re-ranked by the adapted model
     with `outputs/…_drawing_adapted.json` (build record: before adaptation `a ball rolling to the right` → `a ball standing still` (0.69); `a ball bouncing up and down` → `a ball standing still` (0.51); `a square growing larger` → `a ball standing still` (0.43); `the sun setting` → `a ball standing still` (0.46); `a ball standing still` → `a ball standing still` (0.57) (top-1 0.20 over the five drawn labels); after adaptation `a ball rolling to the right` → `a ball standing still` (0.67); `a ball bouncing up and down` → `a ball standing still` (0.47); `a square growing larger` → `a ball bouncing up and down` (0.37); `the sun setting` → `a ball standing still` (0.41); `a ball standing still` → `a ball standing still` (0.55) (top-1 0.20) — a recorded observation, not an
     assertion); `pipe.save_artifact` writing `outputs/…_adapter/{adapter.safetensors,manifest.json}` (the fusion
     head, about 41 MB) and `XClipVideoClassificationPipeline.from_artifact` reloading it with 8/8 identical
     rankings on eight test clips (the cell asserts it); `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the
     model identity and licence, the snapshot block (`weight_file`, `weight_format`, `weight_sha256`), the `corpus`
     block, the inference-contract records before and after adaptation, the comparison, the artifact digest, the
     reload parity, the runtime versions, device and dtype;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the row-group cache were
   clean, outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or
   applicable `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `xclip_video_classification_colab.ipynb` (`E2E`) | `81fc31a` / `1e9956b7` | 2026-09-21 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-xclip-video-classification` v3; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`, float32) | **PASSED** — 11/11 code cells ok (1 restart after install cell); 21 files, 902 MB staged from the Hub into a clean cache; comparison {top1_accuracy: {chance: 0.1, majority: 0.091, frozen: 0.803, adapted: 0.909}, top3_accuracy: {chance: 0.3, majority: 0.288, frozen: 0.939, adapted: 0.985}, macro_recall: {chance: 0.1, majority: 0.1, frozen: 0.802, adapted: 0.91}, macro_f1: {chance: 0.1, majority: 0.017, frozen: 0.792, adapted: 0.905}, delta_vs_frozen: {top1_accuracy: 0.106, top3_accuracy: 0.045, macro_recall: 0.107, macro_f1: 0.112}, per_label_recall: {brushing hair: {frozen: 1, adapted: 1, support: 6}, doing a cartwheel: {frozen: 1, adapted: 1, support: 6}, catching a ball: {frozen: 1, adapted: 1, support: 7}, chewing: {frozen: 0.43, adapted: 0.86, support: 7}, clapping hands: {frozen: 0.5, adapted: 0.67, support: 6}, climbing: {frozen: 1, adapted: 1, support: 8}, climbing stairs: {frozen: 1, adapted: 1, support: 6}, diving into water: {frozen: 0.67, adapted: 1, support: 6}, drawing a sword: {frozen: 0.71, adapted: 1, support: 7}, dribbling a basketball: {frozen: 0.71, adapted: 0.57, support: 7}}}; drawing / scene / page check frozen vs adapted {frozen: [('a ball standing still', 0.69), ('a ball standing still', 0.51), ('a ball standing still', 0.43), ('a ball standing still', 0.46), ('a ball standing still', 0.57)], adapted: [('a ball standing still', 0.67), ('a ball standing still', 0.47), ('a ball bouncing up and down', 0.37), ('a ball standing still', 0.41), ('a ball standing still', 0.55)]}; reload parity {identical_rankings: 8, of: 8}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-xclip-video-classification/v3/evidence/` in the workspace |
| `xclip_video_classification_colab.ipynb` (`TASK-INFERENCE`, superseded) | `79b1285` / `51f0d714b933` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-xclip-video-classification` v1) | PASSED — 8/8 code cells, 20 files, 790 MB staged, 212.9 s; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/xclip_video_classification_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/xclip_video_classification_colab.ipynb`). Wall times, when recorded, are the sum of
per-cell times reported by the executor and include installs and the model download; they are measurements for the
stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-21 | `d24faf4` / `93a0accd` (pre-flight: the build-record placeholders still unfilled in the prose, code identical) | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-xclip-video-classification` v2) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout | 342.2 s | **PASSED** — 11/11 code cells ok (1 restart after install cell); 21 files, 902 MB staged; the metrics the build record quotes |
| 2026-09-21 | `81fc31a` / `1e9956b7` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-xclip-video-classification` v3; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`, float32) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 374.5 s | **PASSED** — 11/11 code cells ok (1 restart after install cell); 21 files, 902 MB staged from the Hub into a clean cache; comparison {top1_accuracy: {chance: 0.1, majority: 0.091, frozen: 0.803, adapted: 0.909}, top3_accuracy: {chance: 0.3, majority: 0.288, frozen: 0.939, adapted: 0.985}, macro_recall: {chance: 0.1, majority: 0.1, frozen: 0.802, adapted: 0.91}, macro_f1: {chance: 0.1, majority: 0.017, frozen: 0.792, adapted: 0.905}, delta_vs_frozen: {top1_accuracy: 0.106, top3_accuracy: 0.045, macro_recall: 0.107, macro_f1: 0.112}, per_label_recall: {brushing hair: {frozen: 1, adapted: 1, support: 6}, doing a cartwheel: {frozen: 1, adapted: 1, support: 6}, catching a ball: {frozen: 1, adapted: 1, support: 7}, chewing: {frozen: 0.43, adapted: 0.86, support: 7}, clapping hands: {frozen: 0.5, adapted: 0.67, support: 6}, climbing: {frozen: 1, adapted: 1, support: 8}, climbing stairs: {frozen: 1, adapted: 1, support: 6}, diving into water: {frozen: 0.67, adapted: 1, support: 6}, drawing a sword: {frozen: 0.71, adapted: 1, support: 7}, dribbling a basketball: {frozen: 0.71, adapted: 0.57, support: 7}}}; drawing / scene / page check frozen vs adapted {frozen: [('a ball standing still', 0.69), ('a ball standing still', 0.51), ('a ball standing still', 0.43), ('a ball standing still', 0.46), ('a ball standing still', 0.57)], adapted: [('a ball standing still', 0.67), ('a ball standing still', 0.47), ('a ball bouncing up and down', 0.37), ('a ball standing still', 0.41), ('a ball standing still', 0.55)]}; reload parity {identical_rankings: 8, of: 8}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-xclip-video-classification/v3/evidence/` in the workspace |
| 2026-09-21 | package API at `d24faf4` (pre-flight, not the notebook blob) | Kaggle Tesla T4 script kernel (`kurtvalcorza/dimer-probe-xclip-e2e` v2 — v1 died after its green pytest on an import-path slip in the probe script, not in the row; `torch 2.14.0+cu130`, `transformers 4.57.6`, Python 3.12, `cuda:0`, float32), branch cloned, pins installed, snapshot staged from the Hub | `tests/test_model_backed.py` (7 passed, 14 warnings in 52.92s) and the recipe probe: the pinned row group read over a range request (300 clips, digest match), chance and majority baselines, frozen model on the 66 test clips, `adapt(epochs=8, lr=1e-5, batch_size=16)` with validation-accuracy selection, adapted evaluation, artifact round trip | 418 s | **PASS** — 7 passed, 14 warnings in 52.92s; the notebook's 8 code cells re-executed through the package API in 94 s with peak CUDA memory 1.83 GB; the metrics it produced are the ones the notebook run above recorded (same seed, same split, same recipe) |
| 2026-09-20 | package API at the working tree of `feat/e2e-video-classification-adaptation` (pre-flight, not the notebook blob) | Windows venv `dimer-xclip` (`torch 2.14.0+cpu`, `transformers 4.57.6`, `av 18.1.0`, Python 3.12.10, `cpu`, float32), `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, row group cached | the CPU recipe sweep on the default split (181 / 53 / 66): frozen 0.803 top-1 / 0.939 top-3 / 0.792 macro F1; head-logits parity with the full forward 3.8e-6; lr 2e-5 / 5e-5 / 1e-4 × 10 epochs all peak in epoch 1 (test 0.909 / 0.894 / 0.848); lr 1e-5 × 8 rises 0.811 → 0.925 → 0.962 on validation (epoch 2 kept), test **0.909 / 0.985 / 0.905**; lr 5e-6 × 8 reaches the same test top-1 at epoch 5; artifact 40,996,184 B, reload parity 8/8; the model-backed suite 6 passed, 1 skipped (CUDA) | ~70 s decode + ~60 s per arm | PASS — pre-flight only; fixed the recipe at lr 1e-5 × 8; not promotion evidence |
| 2026-09-14 | `79b1285` / `51f0d714b933` (`TASK-INFERENCE`, superseded) | Kaggle CPU (`kurtvalcorza/dimer-nb2-xclip-video-classification` v1) | Default sample path, `Run all` from a fresh interpreter, no repository checkout | 212.9 s | PASSED — 8/8 code cells, 20 files, 790 MB staged; not evidence for the `E2E` blob |

## Current status

**Release-grade.** The `E2E` notebook blob `1e9956b7` (committed at `81fc31a`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-21 (11/11 ok (1 restart after install cell), 374.5 s, 21 files, 902 MB fetched from the Hub and digest-verified inside the notebook) with no repository checkout — the REL1/REL10 supported-runtime evidence this file gates on. The pre-flight rows above (the package-API probe and the notebook pre-flight of the previous blob) and the superseded TASK-INFERENCE run are history. Any later change to the carried modules or to the notebook produces a new blob, and the registry returns to **Candidate** until a clean run of that blob is recorded here.

Facts a reviewer should weigh: the sample is ten HMDB51 human actions cut from films and web videos — close to the
checkpoint's Kinetics-400 training distribution, which is why the frozen zero-shot model already ranks the right class
first for most held-out clips (the zero-shot prompts already rank 53 of the 66 clips first; the misses sit on `chewing` (3 of 7, the rest called `drawing a sword` or `brushing hair`) and `clapping hands` (3 of 6, called `brushing hair` or `doing a cartwheel`) — close-up upper-body actions the prompts do not separate) and why the gain is a modest closed-set specialisation, not a repair of
a domain gap; the rates are top-1 / top-3 accuracy, macro recall and macro F1 over one reference label per clip and the
notebook says so; the 53-clip validation split selects the epoch; the towers are frozen, so what the frame encoder
cannot see in 8 frames at 224 px stays unseen; the head that was tuned ranks every request, and the drawn clips
re-ranked after adaptation are the only evidence about what happened outside the label set. The forward pass is
deterministic on a fixed device and dtype, but the training of the head is not bit-reproducible across GPUs, and 66
clips make one clip 1.5 points, so a Kaggle number a few points off the build record is the expected spread, not a
finding. The Tesla T4 run reproduced the CPU sweep's numbers exactly (0.909 / 0.985 / 0.905 at epoch 2). This row has no sibling on its corpus — X-CLIP is the fleet's only video model — so the comparison is the frozen prompts against the tuned head on the same 66 clips: the gain comes from four labels (`chewing` 0.43 → 0.86, `diving into water` 0.67 → 1.00, `drawing a sword` 0.71 → 1.00, `clapping hands` 0.50 → 0.67) while `dribbling a basketball` lost one clip (0.71 → 0.57), and the five drawn cartoon clips stay at chance before and after (every clip but one is called `a ball standing still`): the head learned the ten HMDB51 actions, not motion in general.
