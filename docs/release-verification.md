# Release verification

`tutorials/xclip_video_classification_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the durable release-gate record for the
notebook. The earlier `TASK-INFERENCE` notebook's Kaggle CPU run (2026-09-14, retained below) is history for a
superseded blob, not evidence for this one.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, a §3.3 pedagogical mode,
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
| `xclip_video_classification_colab.ipynb` (`E2E`) | `1bf9b91` / `0c9b1f0f` | 2026-10-18 | Colab CLI 0.7.4, fresh Colab Tesla T4 (CLI sequential execution, not a browser Run all; forms and uploads not exercised) | **PASSED** — 14/14 code cells, 0 errors, 390.5 s; **pre-registered evaluation of record** (REL13): fresh test (669 clips, 242 films): frozen top-1 0.786 (526/669) → selected adapted (lr 1e-5) 0.895 (599/669), paired difference **+0.109**, film-bootstrap 95% interval **[0.066, 0.151]**, McNemar 17 frozen-only vs 90 selected-only (exact p 3.5e-13, film-clustered p 3.1e-6; both descriptive) → **adapted helps**; evidence `docs/execution-evidence/2026-10-18-1bf9b91/` |
| `xclip_video_classification_colab.ipynb` (`E2E`) | `81fc31a` (not on `main`; the same blob is on `main` via `79e91a2`) / `1e9956b7` | 2026-09-21 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-xclip-video-classification` v3; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`, float32) | **PASSED** — 11/11 code cells ok (1 restart after install cell); 21 files, 902 MB staged from the Hub into a clean cache; comparison {top1_accuracy: {chance: 0.1, majority: 0.091, frozen: 0.803, adapted: 0.909}, top3_accuracy: {chance: 0.3, majority: 0.288, frozen: 0.939, adapted: 0.985}, macro_recall: {chance: 0.1, majority: 0.1, frozen: 0.802, adapted: 0.91}, macro_f1: {chance: 0.1, majority: 0.017, frozen: 0.792, adapted: 0.905}, delta_vs_frozen: {top1_accuracy: 0.106, top3_accuracy: 0.045, macro_recall: 0.107, macro_f1: 0.112}, per_label_recall: {brushing hair: {frozen: 1, adapted: 1, support: 6}, doing a cartwheel: {frozen: 1, adapted: 1, support: 6}, catching a ball: {frozen: 1, adapted: 1, support: 7}, chewing: {frozen: 0.43, adapted: 0.86, support: 7}, clapping hands: {frozen: 0.5, adapted: 0.67, support: 6}, climbing: {frozen: 1, adapted: 1, support: 8}, climbing stairs: {frozen: 1, adapted: 1, support: 6}, diving into water: {frozen: 0.67, adapted: 1, support: 6}, drawing a sword: {frozen: 0.71, adapted: 1, support: 7}, dribbling a basketball: {frozen: 0.71, adapted: 0.57, support: 7}}}; drawing / scene / page check frozen vs adapted {frozen: [('a ball standing still', 0.69), ('a ball standing still', 0.51), ('a ball standing still', 0.43), ('a ball standing still', 0.46), ('a ball standing still', 0.57)], adapted: [('a ball standing still', 0.67), ('a ball standing still', 0.47), ('a ball bouncing up and down', 0.37), ('a ball standing still', 0.41), ('a ball standing still', 0.55)]}; reload parity {identical_rankings: 8, of: 8}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-xclip-video-classification/v3/evidence/` in the workspace |
| `xclip_video_classification_colab.ipynb` (`TASK-INFERENCE`, superseded) | `79b1285` / `51f0d714b933` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-xclip-video-classification` v1) | PASSED — 8/8 code cells, 20 files, 790 MB staged, 212.9 s; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/xclip_video_classification_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/xclip_video_classification_colab.ipynb`). Wall times, when recorded, are the sum of
per-cell times reported by the executor and include installs and the model download; they are measurements for the
stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-18 (08:25:59 UTC start) | commit `1bf9b91` / blob `0c9b1f0f9300` (`NOTEBOOK_SOURCE.repository_revision` `3baea09`, equal to `metadata.dimer.generated_from.revision`; `3baea09..1bf9b91` regenerates the notebook and adds a validator marker; `module_sha256` `4e34e8b557c8…`, generator `build_notebook.py/2.5`, `notebook_spec` 2.5) | Colab CLI 0.7.4 sequential execution (`colab exec -f`), fresh Colab Tesla T4, session `suite-xclip-1bf9b91-2472`; isolated hash-locked uv environment (49 packages, CPython 3.12.12, built in 62 s; kernel Python 3.13.15) | Default pre-registered path, one pass, no restart: 14/14 code cells, 0 errors (4 carried-module cells print nothing); snapshot 9 files / 790,034,990 bytes digest-verified; split 181/53/66; grid 1e-5 / 5e-6 / 2e-5 all 51/53 on validation, frozen 43/53, **1e-5 kept** (epoch 2); exposed development test frozen 0.803 → adapted 0.909 (top-3 0.985, macro F1 0.905); `preregistered` True, `settings_differing` []; fresh test (669 clips, 242 films): frozen top-1 0.786 (526/669) → selected adapted (lr 1e-5) 0.895 (599/669), paired difference **+0.109**, film-bootstrap 95% interval **[0.066, 0.151]**, McNemar 17 frozen-only vs 90 selected-only (exact p 3.5e-13, film-clustered p 3.1e-6; both descriptive) → **adapted helps**; reload parity 8/8; Section 10 skipped. Executed notebook SHA-256 `0855767678962a4f…`; evidence `docs/execution-evidence/2026-10-18-1bf9b91/` | 390.5 s wall | PASS — **pre-registered evaluation of record (REL13)**; not a browser Run all; BYOD (REL12), `SPLIT_SEED` re-run and Section 10 not exercised |
| 2026-10-09 (22:52:32 UTC start) | commit `adcb4e8` / blob `41caa483bb88` (`metadata.dimer.generated_from.revision` `1a3e196`, the generation-time HEAD; `1a3e196..adcb4e8` is the ModuleSpec/spec-2.2 commit itself; generator `build_notebook.py/2.2`, `notebook_spec` 2.2) | Colab CLI 0.7.4 sequential execution (`colab exec -f`), fresh Colab Tesla T4, session `suite-xclip-adcb4e8-2472`; isolated hash-locked uv environment (49 packages, CPython 3.12.12; kernel Python 3.13.15), nothing installed into the kernel | Default sample path, one pass, no restart: 13/13 code cells, 0 errors (3 carried-module cells print nothing); snapshot 9 files / 790,034,990 bytes digest-verified; HMDB51 row group 318 rows, 300 clips, split 181/53/66; drawn clips top-1 0.2; majority 0.091, frozen test top-1 0.803 / top-3 0.939 / macro F1 0.792; validation epoch 0 (frozen) 0.811, best epoch 2 (0.962), `started_from` pinned base; adapted test 0.909 / 0.985 / 0.905, all verdicts improved; reload parity 8/8; Section 10 skipped (default). Executed notebook SHA-256 `f6e483046884…`; evidence `docs/execution-evidence/2026-10-09-adcb4e8/` | 182.3 s wall | PASS — not a browser Run all; BYOD (REL12), `SPLIT_SEED` re-run and Section 10 not exercised |
| 2026-09-21 | `d24faf4` (not on `main`) / `93a0accd` (pre-flight: the build-record placeholders still unfilled in the prose, code identical) | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-xclip-video-classification` v2) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout | 342.2 s | **PASSED** — 11/11 code cells ok (1 restart after install cell); 21 files, 902 MB staged; the metrics the build record quotes |
| 2026-09-21 | `81fc31a` (not on `main`; the same blob is on `main` via `79e91a2`) / `1e9956b7` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-xclip-video-classification` v3; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`, float32) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 374.5 s | **PASSED** — 11/11 code cells ok (1 restart after install cell); 21 files, 902 MB staged from the Hub into a clean cache; comparison {top1_accuracy: {chance: 0.1, majority: 0.091, frozen: 0.803, adapted: 0.909}, top3_accuracy: {chance: 0.3, majority: 0.288, frozen: 0.939, adapted: 0.985}, macro_recall: {chance: 0.1, majority: 0.1, frozen: 0.802, adapted: 0.91}, macro_f1: {chance: 0.1, majority: 0.017, frozen: 0.792, adapted: 0.905}, delta_vs_frozen: {top1_accuracy: 0.106, top3_accuracy: 0.045, macro_recall: 0.107, macro_f1: 0.112}, per_label_recall: {brushing hair: {frozen: 1, adapted: 1, support: 6}, doing a cartwheel: {frozen: 1, adapted: 1, support: 6}, catching a ball: {frozen: 1, adapted: 1, support: 7}, chewing: {frozen: 0.43, adapted: 0.86, support: 7}, clapping hands: {frozen: 0.5, adapted: 0.67, support: 6}, climbing: {frozen: 1, adapted: 1, support: 8}, climbing stairs: {frozen: 1, adapted: 1, support: 6}, diving into water: {frozen: 0.67, adapted: 1, support: 6}, drawing a sword: {frozen: 0.71, adapted: 1, support: 7}, dribbling a basketball: {frozen: 0.71, adapted: 0.57, support: 7}}}; drawing / scene / page check frozen vs adapted {frozen: [('a ball standing still', 0.69), ('a ball standing still', 0.51), ('a ball standing still', 0.43), ('a ball standing still', 0.46), ('a ball standing still', 0.57)], adapted: [('a ball standing still', 0.67), ('a ball standing still', 0.47), ('a ball bouncing up and down', 0.37), ('a ball standing still', 0.41), ('a ball standing still', 0.55)]}; reload parity {identical_rankings: 8, of: 8}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-xclip-video-classification/v3/evidence/` in the workspace |
| 2026-09-21 | package API at `d24faf4` (pre-flight, not the notebook blob) | Kaggle Tesla T4 script kernel (`kurtvalcorza/dimer-probe-xclip-e2e` v2 — v1 died after its green pytest on an import-path slip in the probe script, not in the row; `torch 2.14.0+cu130`, `transformers 4.57.6`, Python 3.12, `cuda:0`, float32), branch cloned, pins installed, snapshot staged from the Hub | `tests/test_model_backed.py` (7 passed, 14 warnings in 52.92s) and the recipe probe: the pinned row group read over a range request (300 clips, digest match), chance and majority baselines, frozen model on the 66 test clips, `adapt(epochs=8, lr=1e-5, batch_size=16)` with validation-accuracy selection, adapted evaluation, artifact round trip | 418 s | **PASS** — 7 passed, 14 warnings in 52.92s; the notebook's 8 code cells re-executed through the package API in 94 s with peak CUDA memory 1.83 GB; the metrics it produced are the ones the notebook run above recorded (same seed, same split, same recipe) |
| 2026-09-20 | package API at the working tree of `feat/e2e-video-classification-adaptation` (pre-flight, not the notebook blob) | Windows venv `dimer-xclip` (`torch 2.14.0+cpu`, `transformers 4.57.6`, `av 18.1.0`, Python 3.12.10, `cpu`, float32), `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, row group cached | the CPU recipe sweep on the default split (181 / 53 / 66): frozen 0.803 top-1 / 0.939 top-3 / 0.792 macro F1; head-logits parity with the full forward 3.8e-6; lr 2e-5 / 5e-5 / 1e-4 × 10 epochs all peak in epoch 1 (test 0.909 / 0.894 / 0.848); lr 1e-5 × 8 rises 0.811 → 0.925 → 0.962 on validation (epoch 2 kept), test **0.909 / 0.985 / 0.905**; lr 5e-6 × 8 reaches the same test top-1 at epoch 5; artifact 40,996,184 B, reload parity 8/8; the model-backed suite 6 passed, 1 skipped (CUDA) | ~70 s decode + ~60 s per arm | PASS — pre-flight only; fixed the recipe at lr 1e-5 × 8; not promotion evidence |
| 2026-09-14 | `79b1285` / `51f0d714b933` (`TASK-INFERENCE`, superseded) | Kaggle CPU (`kurtvalcorza/dimer-nb2-xclip-video-classification` v1) | Default sample path, `Run all` from a fresh interpreter, no repository checkout | 212.9 s | PASSED — 8/8 code cells, 20 files, 790 MB staged; not evidence for the `E2E` blob |

## Pre-registered grid, fresh test and score-once (B1 review fix, 2026-10-18)

- **Disclosure (R113-M1).** The 2026-09-20 CPU pre-flight in the table above read the **test** scores of all five
  learning-rate arms (2e-5, 5e-5 and 1e-4 × 10 epochs; 1e-5 and 5e-6 × 8) before fixing lr 1e-5 × 8; validation was
  recorded for the chosen arm only. The 66 test clips are therefore the **exposed development test** from now on.
- **Pre-registration** `docs/xclip-preregistration.md`, committed alone first (`134af58`) before any grid run of the
  protocol or any fresh-test score: grid lr {1e-5 (default), 5e-6, 2e-5} × 8 epochs on validation with a two-clip
  margin and ties to the default, keep-frozen unless beaten by two clips, the fresh test (682 clips, list SHA-256
  `ac0144fa…bc04`, row groups 0–2 of `default/train/0000.parquet`), and the reading rule (paired top-1 difference,
  film bootstrap n_boot 2000 seed 113, McNemar counts; "adapted helps" only if the interval excludes 0).
- **Score-once.** The fresh test is scored only under the pre-registered settings (SCORE_ONCE_RULE). The first hosted
  T4 run with those settings is **the evaluation of record**; later identical runs are reproductions.
  `protocol.RECORDED_EVALUATION` is `None` until that run. **Evaluation of record: the 2026-10-18 Colab T4 run of
  `1bf9b91` / `0c9b1f0f` (section below). Reproductions: none yet.** `RECORDED_EVALUATION` is still `None` in the
  code at `1bf9b91`, so a later identical run labels itself "pre-registered evaluation (this runtime)"; this document,
  not that label, says which run is the record.
- **CPU development check, fresh test MASKED** (SCORE_ONCE_RULE 8; `DIMER_MASK_FRESH_TEST=1`), 2026-10-18, code at
  commit `6840787` (working-tree notebook regenerated from it), Windows venv `torch 2.14.0+cpu`, float32, 310 s.
  Nothing was read from the fresh test; no setting was changed after it. `settings_differing` = [] (the gate passes
  with real inputs). Grid, kept-epoch validation top-1 on 53 clips: 1e-5 51 (epoch 2), 5e-6 51 (epoch 5), 2e-5 51
  (epoch 1); frozen 43. **Selected: 1e-5** (no challenger won by two clips). Exposed development test (old = Colab T4
  `adcb4e8` → this CPU run): frozen top-1 0.803 → 0.803, adapted 0.909 → 0.909, top-3 0.985 → 0.985, macro F1
  0.905 → 0.905; majority baseline (now an expected value over the cartwheel / clapping tie) top-1 0.091 → 0.091,
  top-3 0.288 → 0.273, macro F1 0.017. Reload parity 8 / 8. A CPU build record pending the hosted run, not
  clean-runtime evidence.
- **R113-S2.** `docs/execution-evidence/2026-10-09-adcb4e8/exec.log`: two lines, the executor home-directory run path →
  `<local-run-dir>/` in commit `500c44c`. SHA-256 before
  `e66895d7b572d0a147413862eb0c1208bce28ed9d80fa7122b34d6a461df564a`, after
  `33c082aef15a974912ea961d884295117556919caf5c4b53b0f70c9448a7e5d1` (the executed notebook and run_summary.json are
  unchanged; run_summary.json carries no log digest).
- Spec: generator `build_notebook.py/2.5`, `notebook_spec` 2.5, `notebook_kind` TUTORIAL, use notice (DOC17).

## Amendment 1 and the peer-review round (2026-10-18)

- **Pre-registration amendment 1** `docs/xclip-preregistration-amendment-1.md`, committed alone (`56fd447`) before
  any fresh-test score. The fresh test has only ever been masked.
  - Film key normalised: case-fold, then delete everything outside 0-9 / a-z, then strip trailing digits (peer review
    F1/F6). Under the earlier key, 13 fresh clips of 4 films that are also in the 300 escaped exclusion:
    `Treppen_steigen` (9), `Basketball_Dribbling_Tips` (2), `Hechtsprung_1` (1) and `…(Part4)` (1).
  - The fresh test is now **669 clips** in 242 films, list SHA-256 `c651eb98…abba` (it was 682, `ac0144fa…bc04`).
    climb_stairs drops to 57, still at least 50.
  - Bootstrap clusters use the same key.
  - The MDE is stated: about 0.034–0.067 at 80 % power.
  - McNemar now also reports a cluster-adjusted (Durkalski) p. Both p values are descriptive only.
- **Masked fetch-and-decode check** (`DIMER_MASK_FRESH_TEST=1` plus `DIMER_FRESH_FETCH_CHECK=1`, or
  `samples.check_fresh_test`). It fetches the fresh test, verifies the pins and the list, and decodes every clip,
  with no model involved.
  Result: 669 clips verified against the row-group pins and the list digest, 669 decoded to 8 frames with PyAV 18.1.0 (fetch and verify 1.6 s from the cached row groups, decode 510.7 s). No model loaded, no prediction made.
- **Look ledger** (`outputs/*_test_looks.jsonl`):
  - An `attempt` entry is written before any model sees the fresh clips, so a crash mid-scoring still counts as a look.
  - The report is written before the `scored` entry, which carries the full reading.
  - An unparsable ledger line counts as a prior look.
  - Every look records the device and GPU name. The definition of the evaluation of record is unchanged.
- **Other changes:**
  - The sample split's film check refuses records without a film key, where it used to pass silently.
  - The fresh-test report carries the chance and majority baselines.
  - The majority baseline's per-label rates and confusion are now expected values.
  - Adapted drawings are written under the run suffix.
  - "without leakage" is replaced by the per-action statement.
  - Section 7 states that the grid sits at the validation ceiling.

## Fresh-test evaluation of record: hosted T4 (2026-10-18)

**Label (REL13): pre-registered evaluation of record.** It is the first hosted run of the pre-registered settings
with amendment 1, and the first time any run scored the 669-clip fresh test. The fixer's CPU checks ran with the
fresh test masked (development check at `6840787`; masked fetch-and-decode check, no model), so there was no
development look. No earlier hosted attempt at this head scored a fresh clip.

- **Identity.** Commit `1bf9b91abbc103bc581df5f9354dc556e0e19028`, notebook blob
  `0c9b1f0f930069c365412fb3c1f560a9d7de4feb`; the run's `source.ipynb` hashes to that blob and is byte-equal to the
  committed notebook. `inspect_run.py`: 39/39 cells, same ids and order, source identity OK; embedded revision
  `3baea09`, generator `build_notebook.py/2.5`, `module_sha256` `4e34e8b557c8…`, spec 2.5. The CLI writes no
  execution counts; 10 of 14 code cells have output.
- **Runtime.** Colab Tesla T4 (`colab status`: `Hardware: T4`), device `cuda:0` (printed by the model-load and ceilings cells),
  torch 2.14.0+cu130, transformers 4.57.6, av 18.1.0, isolated CPython 3.12.12. The GPU name goes only into the
  look ledger and `outputs/xclip_video_classification_evaluation_report.json`, which stayed on the released VM, so
  the committed evidence shows `cuda:0` and `Hardware: T4`, not the device-name string.
- **Score-once checks.** Section 8 printed `test_look` "pre-registered evaluation (this runtime)", `preregistered`
  True, `settings_differing` [], selected system "adapted (lr 1e-05)". Every input was the default.
- **Film check.** Section 4 printed `film_overlap` with 5 films shared across the sample splits under the
  amendment-1 key (40 clips): `goalkeepertrainingday` and `theboondocksaints` (train / validation), `chewinggum` and
  `theslowclap` (test / train), `returnoftheking` (test / validation). The masked CPU check at `6840787` printed `{}`
  under the earlier key. The fresh-test overlap gate (no fresh film in the 300) passed, since scoring went ahead.
- **Grid (validation, 53 clips, kept-epoch correct).** 1e-5: 51 (epoch 2); 5e-6: 51 (epoch 5); 2e-5: 51 (epoch 1);
  frozen 43. Selected **1e-5** ("no challenger beat 1e-05 by 2 validation clips"). All 43 grid, baseline and
  exposed-test lines are equal at printed precision to the masked CPU check (`6840787`; timings excluded).
- **Fresh test (669 clips, 242 films; the pre-registered reading rule).**

  | | frozen | selected (adapted, lr 1e-5) |
  |---|---|---|
  | top-1 | 0.786 (526) | 0.895 (599) |

  Paired difference **+0.109** (73 clips = 90 − 17), film-bootstrap 95% interval **[0.066, 0.151]**, which excludes 0:
  **adapted helps**. McNemar counts: 17 clips only the frozen model got right, 90 only the adapted head; exact
  two-sided p 3.5e-13 (assumes independent clips) and film-clustered p 3.1e-6 (Durkalski), both descriptive.
  Fresh-test chance and majority baselines, top-3 accuracy, macro recall / F1 and per-label recall are written to the
  evaluation report on the VM and were not printed, so they are not in the committed evidence. Derived from the
  printed per-class counts and the documented baseline rules: chance top-1 0.100 / top-3 0.300; majority (expected
  over the cartwheel / clapping tie) top-1 0.102, top-3 0.296, macro recall 0.100, macro F1 0.018 (the same rule
  reproduces the printed exposed-test values 0.091 / 0.273 / 0.017).
- **Exposed development test (66 clips; decides nothing).** chance 0.1, majority 0.091 (top-3 0.273), frozen 0.803 /
  0.939 / 0.802 / 0.792, adapted 0.909 / 0.985 / 0.910 / 0.905 (top-1 / top-3 / macro recall / macro F1); equal to
  the masked CPU check and, apart from the majority top-3 (0.288 → 0.273, the R113-S1 tie rule), to 2026-10-09.
- **Re-derivation.** 0.786 × 669 → 526, 0.895 × 669 → 599, difference 73/669 = 0.1091; McNemar net 90 − 17 = 73;
  the exact binomial p for 17 of 107 recomputes to 3.505779998119023e-13, equal to the printed value. The bootstrap
  and the clustered p were not re-run (per-clip predictions stayed on the VM).
- **Timing.** 390.5 s suite wall (exec 08:26:05 → 08:32:26 UTC), against the peer review's estimate of about 7 min.
  Printed parts: environment 62 s, sample fetch 9.2 s + decode 30.1 s, frozen scoring of 66 clips 3.8 s, tower cache
  6.7 s, final fit 35.2 s. The fresh fetch / decode / scoring time is only in the report on the VM.
- **Other.** Adapter 50 tensors, 40,996,184 B; reload parity 8/8; drawn clips 0.2 at chance 0.2 before and after;
  Section 10 skipped.
- **Evidence.** `docs/execution-evidence/2026-10-18-1bf9b91/`: the executed notebook (SHA-256
  `0855767678962a4f128b03b8ec254e22a7e88037d957ade9b9364231b9370257`) and `run_summary.json`
  (`4a83a60e535ac1fd467ec0089da5d0342b3a3cda6d15762edbd482dd54aa69c5`) byte-equal to the run directory. `exec.log`
  has the executor run path replaced by `<local-run-dir>/` on lines 1 and 95 (as in `500c44c`): captured
  `63cf7e8b83ed6ccb3392eadaa5a92b849531de5ec36817da55d267df1f109291`, committed
  `373ee3d38c1f71341ccf80fe16182ea0fd61f32a6ee27f0678e39acea7f30f86`.
- **Boundary.** CLI sequential execution, not a browser Run all; forms, uploads, BYOD (REL12), the `SPLIT_SEED`
  re-run and Section 10 not exercised. The status stays **Candidate**.

## Current status

**Candidate** (2026-10-18). **Fresh-test evaluation of record (2026-10-18):** commit `1bf9b91` / blob `0c9b1f0f` ran in one pass on a fresh Colab Tesla T4 (Colab CLI 0.7.4, 14/14 code cells, 0 errors, 390.5 s); on the 669-clip fresh test the selected adapted head (lr 1e-5) scored top-1 0.895 against the frozen model's 0.786, difference +0.109, film-bootstrap 95% interval [0.066, 0.151]: **adapted helps**; evidence `docs/execution-evidence/2026-10-18-1bf9b91/`. Earlier: **Colab T4 one-pass run of the isolated-environment revision (2026-10-09).** Commit `adcb4e8` / blob `41caa483bb88` (generator `build_notebook.py/2.2`, `notebook_spec` 2.2, hash-locked isolated uv environment, worker `google.colab` stubs with a `ModuleSpec`) executed in one pass with no restart and 0 errors on a fresh Colab Tesla T4 (Colab CLI 0.7.4 sequential execution, not a browser `Run all`; 13/13 code cells, 182.3 s wall, `cuda:0`); evidence in `docs/execution-evidence/2026-10-09-adcb4e8/`. The status stays Candidate (REL14): the BYOD journeys (REL12), the `SPLIT_SEED` re-run and the optional Section 10 experiment were not hosted-run. The 2026-09-21 Kaggle T4 run of blob `1e9956b7` (11/11 ok, one restart after the install cell; the blob is on `main` via `79e91a2`) is history.

Facts a reviewer should weigh: the sample is ten HMDB51 human actions cut from films and web videos — close to the
checkpoint's Kinetics-400 training distribution, which is why the frozen zero-shot model already ranks the right class
first for most held-out clips (the zero-shot prompts already rank 53 of the 66 clips first; the misses sit on `chewing` (3 of 7, the rest called `drawing a sword` or `brushing hair`) and `clapping hands` (3 of 6, called `brushing hair` or `doing a cartwheel`) — close-up upper-body actions the prompts do not separate) and why the gain is a modest closed-set specialisation, not a repair of
a domain gap; the rates are top-1 / top-3 accuracy, macro recall and macro F1 over one reference label per clip and the
notebook says so; the 53-clip validation split selects the epoch; the towers are frozen, so what the frame encoder
cannot see in 8 frames at 224 px stays unseen; the head that was tuned ranks every request, and the drawn clips
re-ranked after adaptation are the only evidence about what happened outside the label set. The forward pass is
deterministic on a fixed device and dtype, but the training of the head is not bit-reproducible across GPUs, and 66
clips make one clip 1.5 points, so a Kaggle number a few points off the build record is the expected spread, not a
finding. The Tesla T4 run reproduced the CPU sweep's exposed-development-test numbers (0.909 / 0.985 / 0.905 at epoch 2). This row has no sibling on its corpus — X-CLIP is the fleet's only video model — so the comparison is the frozen prompts against the tuned head on the same 66 clips: the gain comes from four labels (`chewing` 0.43 → 0.86, `diving into water` 0.67 → 1.00, `drawing a sword` 0.71 → 1.00, `clapping hands` 0.50 → 0.67) while `dribbling a basketball` lost one clip (0.71 → 0.57), and the five drawn cartoon clips stay at chance before and after (every clip but one is called `a ball standing still`): the head learned the ten HMDB51 actions, not motion in general.
