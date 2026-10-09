"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package (three modules,
carried verbatim in dependency order), and the model pin/stage/verify cells are produced by the generator from
repository sources so they cannot drift from the package.

This template configures an E2E closed-set video-classification adaptation workflow: the pinned
microsoft/xclip-base-patch32 snapshot is digest-verified and loaded, 300 CC BY 4.0 HMDB51 clips of ten action
classes are fetched as one digest-pinned parquet row group over an HTTPS range request and decoded into 8-frame
clips, the records are validated and split by source video, five drawn clips are ranked through the inference
contract, the frozen model is scored over the held-out clips (top-1 / top-3 accuracy, macro recall and F1) beside a
chance and a majority-label baseline, a bounded fine-tuning of the fusion head runs on cached tower features with
validation-accuracy epoch selection, the held-out split is scored again, six held-out clips and the drawn clips are
re-run with the adapted model, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "xclip_video_classification_pipeline",
    "repo_name": "xclip-video-classification-pipeline",
    "stem": "xclip_video_classification",
    "notebook_name": "xclip_video_classification_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "isolated_runtime": True,
    "infrastructure_labels": True,
    # The fleet's uv isolated-environment mechanism (bioclip2-biodiversity-pipeline): managed CPython, a size- and
    # SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "pipeline_class": "XClipVideoClassificationPipeline",
    "weights_key": "xclip-base-patch32",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "runtime_imports": ["torch", "transformers", "av"],
    "title": "X-CLIP base/32 — DIMER E2E video-classification adaptation tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/xclip-video-classification-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/xclip-video-classification-pipeline/blob/main/tutorials/xclip_video_classification_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-microsoft%2Fxclip--base--patch32-ffcc4d?style=flat",
            "https://huggingface.co/microsoft/xclip-base-patch32",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-microsoft%2FVideoX-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/microsoft/VideoX/tree/master/X-CLIP",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2208.02816-b31b1b.svg", "https://arxiv.org/abs/2208.02816"),
    ],
    "capability": "zero-shot video classification — one 8-frame clip plus 2–32 free-text class names → a ranking of those names with a softmax over them — and bounded supervised fine-tuning of the fusion head on labelled clips of a closed label set, using the pinned `microsoft/xclip-base-patch32` weights",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (nothing is "
        "installed into the notebook's own Python, so no restart is needed and Run all completes in one pass), stages and digest-verifies the "
        "pinned `microsoft/xclip-base-patch32` snapshot (a 786 MB `model.safetensors`; no pickle is opened anywhere), fetches "
        "the first row group of the HMDB51 test shard from the Hugging Face Hub at an immutable revision with one HTTPS range "
        "request (about 113 MB; refused on any SHA-256 or byte-total mismatch), decodes the 300 clips of the ten sample classes "
        "into 8 uniformly spaced frames each with PyAV, validates the records and splits them by source video into "
        "181 / 53 / 66, ranks five drawn clips through the inference contract with a combined input manifest and a rejection "
        "probe, scores the frozen model over the 66 held-out clips (top-1 and top-3 accuracy, macro recall and F1) beside a "
        "chance and a majority-label baseline, runs a bounded fine-tuning of the fusion head on cached tower features with "
        "validation-accuracy epoch selection, scores the held-out clips again, re-runs six held-out clips and the five drawn "
        "clips with the adapted model, exports the adapter as safetensors with a manifest, and reloads that artifact into a "
        "fresh pipeline to verify ranking parity. The default path needs no repository clone, no DIMER worker or service, no "
        "credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.2 §5). On a Tesla T4 the default path took "
        "about 2 minutes of cell time (eight epochs 12 s, frozen scoring of 66 clips "
        "3 s); a CUDA runtime is used automatically when present, and the path is practical on CPU too (the "
        "build venv decoded the 300 clips in about 70 s, scored the test split in 10 s and ran the eight epochs in about 60 s)."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and either set `BYOD_PATH` to a zip or folder "
        "in the runtime (Colab, Kaggle or Jupyter) or leave it empty to upload one zip in Colab, then choose **Run after** from "
        "that cell (it first puts the model back to the pinned base) to supply clips (AVI, MP4, MOV, MKV, WebM, GIF or WebP; at least 8 frames each) plus a `labels.csv` (`file`, `label`, optional "
        "`id` and `source`; one row per clip — at least **24 clips for two labels** (12 per label), 27 for three, 24 for four "
        "(`min_byod_records(n_labels)`); a `source` keeps all its clips, of every label, in one split). The "
        "records pass through the same validation, source-disjoint split, baselines, fine-tuning, held-out evaluation, "
        "artifact export and reload-parity cells as the HMDB51 sample. Uploaded files stay inside this runtime. BYOD is "
        "optional and never part of the default path."
    ),
    "intro": (
        "X-CLIP base/32 is a video–text model: a CLIP ViT-B/32 frame encoder with cross-frame attention embeds each of the "
        "8 frames of a clip, a one-layer multi-frame integration transformer fuses the 8 frame embeddings into one video "
        "embedding, a CLIP text encoder embeds each class name, and a video-specific prompt generator conditions the text "
        "embeddings on the clip's patch features; the clip is scored against every name and the carried module softmaxes the "
        "scores over the names you supplied (196,585,729 parameters in all, trained fully supervised on Kinetics-400, "
        "published under the **MIT** licence). **The probabilities are a softmax over your label set**: a relative ranking "
        "that sums to 1, not a calibrated probability, and a set that omits the true class still yields a confident "
        "top-1.\n\n"
        "What this notebook adds to inference is **adaptation of a closed label set on labelled clips**. The clips are 300 "
        "human-action clips of ten HMDB51 classes — brushing hair, doing a cartwheel, catching a ball, chewing, clapping "
        "hands, climbing, climbing stairs, diving into water, drawing a sword, dribbling a basketball — cut from films and "
        "web videos; the frozen model already ranks the right class first for **0.803** of the 66 held-out clips in "
        "the build record (chance is 0.100), so the honest question is narrow: does a bounded fine-tuning of the fusion head "
        "— the frame-integration transformer, the two visual projections and the prompt generator, 10,247,680 of the "
        "parameters — on 181 clips move the held-out **top-1 accuracy**, **top-3 accuracy**, **macro recall** and **macro F1** "
        "on a source-disjoint test split past the frozen model and two **non-adapted baselines**, and what does it do to the "
        "drawn clips the same head ranks? Nothing here is a claim about your videos or your classes: it is one seeded split "
        "of one small labelled set.\n\n"
        "**Snapshot note:** the pinned revision ships `model.safetensors` (a 9-file manifest with the tokenizer and processor "
        "files) — no pickle is opened anywhere in this notebook. Section 3 stages and digest-verifies those files before the "
        "processor or the model is constructed. The pipeline runs in **float32 on every device**: the adapter is trained in "
        "float32 and overlays without a cast, and CPU, Tesla-class and consumer GPUs then run the same arithmetic."
    ),
    "guided": {
        "opening": [
            (
                "**Who this notebook is for.** A learner who knows basic Python, has used Colab or Jupyter, and wants to see how a pretrained video–text model is adapted to a closed label set with a small labelled clip set, and how to read the result honestly — including what the pre-training already covered. No prior experience with video models or fine-tuning is assumed; each term is explained where it first matters and again in the **Glossary** at the end. CPU is adequate; a GPU is faster.\n\n**Input → Model → Output.**\n\n| | Ranking a clip | Bounded fine-tuning |\n|---|---|---|\n| Input | one clip of 8 frames and 2–32 free-text class names | labelled clips: 181 training and 53 validation HMDB51 clips in the sample, split by source video |\n| Model | X-CLIP base/32: a CLIP frame encoder with cross-frame attention, a frame-integration transformer, a text encoder and a prompt generator | the fusion head (10.2 M parameters) trained with cross-entropy over the label set on cached tower features; validation top-1 chooses the epoch |\n| Output | a ranking of the names with a softmax over them — relative, not calibrated | a safetensors adapter, and held-out top-1 / top-3, macro recall and macro F1 beside two baselines |\n\n**How to use this notebook.** Choose a runtime (CPU works; a GPU is faster), then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated environment, the carried package and the model snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the recorded run. Before each principal result the notebook asks you to **Predict**; after it come **What to notice** and a collapsible **Check your reasoning** with a worked answer that names the run it quotes — the Kaggle T4 release run of 21 September 2026. Section 10 is a **change-one-thing experiment**, off by default. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 HMDB51 clips and a source-grouped split *(evaluation practice)* → 5 ranking drawn clips through the inference contract *(core concept: a softmax over your labels)* → 6 baselines and the pre-trained model, with its Kinetics-400 overlap *(evaluation practice)* → 7 fine-tuning the fusion head *(core concept)* → 8 held-out evaluation → 9 clips, drawings, export and reload *(engineering)* → 10 change one thing (optional) → conclude."
            )
        ]
    },
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the immutable "
        "upstream snapshot; fetch a digest-pinned labelled clip set, decode it into 8-frame clips, validate it and split it "
        "by source video without leakage; rank drawn clips through the public API and read the output contract correctly (a "
        "softmax over the supplied names, no calibrated probability, a `sample-sanity` report only when the true classes are "
        "known); measure the frozen model's held-out top-1 / top-3 accuracy, macro recall and F1 beside two non-adapted "
        "baselines; run a bounded fine-tuning of the fusion head with the cross-entropy over the closed label set, explicit "
        "hyperparameters and validation-based epoch selection; evaluate on a source-disjoint test split; look at the adapted "
        "rankings next to the frozen ones and the references, and at what the drawn clips do after the shared head was "
        "tuned; and export a safetensors adapter that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "temporal localisation or per-frame labels (one ranking per clip), open-set or abstaining classification (the "
        "softmax always picks one of your names), video–text retrieval over a corpus, fine-tuning of the vision tower, the "
        "text tower, the text projection or the logit scale, evaluation on Kinetics-400 or the full HMDB51 protocol (only one "
        "seeded 300-clip sample of ten classes is scored here), non-English class names, and any claim that ten HMDB51 "
        "actions stand in for your videos. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Learner:** basic Python and Colab or Jupyter familiarity; no prior experience with video models or fine-tuning. The notebook explains cross-frame attention, the frame-integration transformer, the prompt generator, the frozen-tower cache, top-k accuracy, macro recall and source grouping where they are first used; the Glossary repeats them.",
        "- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or Linux Jupyter; CPU or CUDA). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels (PyAV included), so the kernel's own Python version does not matter and nothing is installed into it. The default path uses CUDA automatically when present. The vision tower runs `EVAL_BATCH_SIZE` clips (8 × 8 frames) per forward and the build record measured 3 s to score 66 clips and 12 s for the eight epochs (caching the tower features for 181 + 53 clips took 6 s) on a Tesla T4, about 2 minutes of cell time for the whole path including the pinned install and the downloads; the build venv's CPU ran the same path in a few minutes. The pinned `torch==2.14.0` install, the 786 MB checkpoint and the 113 MB row group are the large downloads of the run.",
        "- **Knowledge:** basic Python, NumPy and PIL; what a contrastive video–text model scores and why a softmax over a label set you chose is a ranking and not a probability; what top-1 accuracy, macro recall and macro F1 measure on a closed label set and why 66 clips give no dispersion; why clips cut from one source video must stay in one split.",
        "- **Data contract:** records are `{id, frames, label}` — `frames` exactly `NUM_FRAMES` (8) PIL frames of one size with sides within 16..4,096 px (a longer clip is subsampled uniformly by `sample_frames`; a container file is decoded by `decode_clip`), `label` one of the closed label set (normalised like the candidate names: stripped, lower-cased, trailing full stop removed), and an optional `source` naming the video the clip was cut from. Ids match `[A-Za-z0-9_.:-]{1,64}` and are unique; a dataset needs 16..5,000 records, at least two labels and at least two clips per label; splitting de-duplicates by decoded pixels and keeps every clip of one source in one split. BYOD accepts one zip (or directory) of clips plus a `labels.csv` in the layout named above.",
        "- **Validation is structural, not semantic:** every clip is decoded and every label checked against the set, but nothing checks that a label describes its clip — a mislabelled set is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path reads row group 0 of `default/test/0000.parquet` from `https://huggingface.co/datasets/mteb/HMDB51/resolve/<revision>/` at the immutable parquet-conversion revision `50bb2abb…` with HTTPS range requests (the parquet footer plus about 113 MB of row-group bytes out of a 481 MB shard), pinned by SHA-256 and byte total in the carried `samples.py` and refused on any mismatch. HMDB51 is published under the CC BY 4.0 licence (Serre Lab, Brown University; Kuehne et al. 2011); the `mteb/HMDB51` repository is a parquet repack of it; nothing is redistributed by this repository.",
    ],
    "cells": [
        {
            "md": (
                "## 4. HMDB51 clips, the labels and the source-grouped split\n\n"
                "`fetch_corpus` returns the pinned row group from the cache under `weights/hmdb51/` or the Hub at the pinned "
                "parquet-conversion revision — `pyarrow` reads the shard's footer and exactly that row group over HTTPS range "
                "requests; the cached file is re-hashed and a fetched row group refused on any SHA-256 or byte-total mismatch — "
                "and `read_corpus` turns each row of the ten sample classes into a record: the AVI decoded by PyAV into 8 "
                "uniformly spaced frames (mostly 320 × 240), the class rendered as the natural-language label the text tower "
                "is asked to rank, and the source video the clip was cut from. `build_sample_dataset` draws a seeded "
                "**source-grouped** split: per class, whole source videos go to the test split until it holds six clips, then "
                "to validation until four, and the rest train (181 / 53 / 66 in the build record). `validate_dataset` then "
                "checks every record against the contract and the label set, `check_split_disjoint` asserts no clip (by "
                "decoded-pixel digest) and no source video is shared, and the training split's summary table is written to "
                "`outputs/{stem}_train.csv`.\n\n"
                "Look for: 300 clips of ten labels, three digests, and four refusal probes — a duplicate id, a label outside "
                "the set, a clip with three frames, and a dataset too small to use — each rejected before the model does "
                "anything.\n\n"
                "*Evaluation practice.* **Bring your own data (optional):** set `USE_BYOD = True` and either `BYOD_PATH` (a zip or "
                "a folder holding `labels.csv` and the clips, as a path in this runtime — this works on Colab, Kaggle and "
                "Jupyter) or leave `BYOD_PATH` empty to upload exactly one zip through the Colab dialog; then choose **Run after** "
                "from this cell. This cell first puts the model back to the pinned base, so Section 6 scores the pre-trained "
                "head — also after a `SPLIT_SEED` change, when clips the adapted head trained on could land in the test split. "
                "The effective minimum is 24 clips for two labels.\n\n"
                "**Predict before running:** several HMDB51 clips are cut from one film. What would a split by clip measure?"
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "# A re-run after Section 7 (BYOD, a new split or new phrasing): Section 6 and Section 7's epoch 0 must read the pinned base.\n"
                "had_adapter = pipe.adapter is not None\n"
                "restored_tensors = pipe.restore_base()\n"
                "if had_adapter or restored_tensors:\n"
                "    print({{'restored_pinned_base': len(restored_tensors), 'note': 'the fine-tuned fusion head was put back to the checkpoint; Sections 5-7 start from it again'}})\n"
                "if USE_BYOD:\n"
                "    if BYOD_PATH.strip():\n"
                "        byod_zip = Path(BYOD_PATH.strip()).expanduser()\n"
                "        if not byod_zip.exists():\n"
                "            raise FileNotFoundError(f'BYOD_PATH {{BYOD_PATH!r}} does not exist (relative paths start at {{Path.cwd()}}): give a .zip or a folder holding labels.csv and the clips.')\n"
                "        file_name = byod_zip.name\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise RuntimeError('USE_BYOD is True but BYOD_PATH is empty, and the upload dialog exists only in Google Colab: on Kaggle or Jupyter put the zip (or folder) in the runtime and set BYOD_PATH to its path.') from None\n"
                "        uploaded = files.upload() or {{}}\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'Upload exactly one .zip file (received {{len(uploaded)}}; a cancelled dialog sends none): run this cell again.')\n"
                "        file_name, payload = next(iter(uploaded.items()))\n"
                "        if not file_name.lower().endswith('.zip'):\n"
                "            raise ValueError(f'{{file_name}}: upload one .zip holding labels.csv and the clips.')\n"
                "        byod_zip = Path('work') / 'byod.zip'\n"
                "        byod_zip.parent.mkdir(parents=True, exist_ok=True)\n"
                "        byod_zip.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_zip)\n"
                "    LABELS = validate_dataset(records)['labels']\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records), 'labels': len(LABELS), 'effective_minimum': min_byod_records(len(LABELS))['total']}}\n"
                "    if len(splits['test']) < 5 * len(LABELS):\n"
                "        print({{'caution': f\"only {{len(splits['test'])}} held-out test clips for {{len(LABELS)}} labels (fewer than 5 per label): rates move in large steps; add clips before reading them\"}})\n"
                "else:\n"
                "    t0 = time.perf_counter()\n"
                "    corpus_groups = fetch_corpus(cache_dir='weights/hmdb51')\n"
                "    fetch_seconds = round(time.perf_counter() - t0, 1)\n"
                "    corpus = read_corpus(corpus_groups)\n"
                "    LABELS = list(SAMPLE_LABELS)\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} @ {{CORPUS_REVISION[:12]}} ({{CORPUS_LICENSE}})'\n"
                "    raw_rows = {{'row_groups': len(corpus_groups), 'rows': sum(len(v) for v in corpus_groups.values()), 'clips_kept': len(corpus), 'bytes': sum(len(r['video']) for v in corpus_groups.values() for r in v), 'fetch_seconds': fetch_seconds, 'decode_seconds': round(time.perf_counter() - t0 - fetch_seconds, 1)}}\n"
                "dataset_manifests = {{name: validate_dataset(part, LABELS, min_records=1, min_per_class=1) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "validate_dataset(train_records, LABELS)  # the training split must satisfy the full record bounds\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint, 'labels': LABELS}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'per_label': manifest['per_label'], 'sources': manifest['n_sources'], 'width': manifest['frame_width'], 'height': manifest['frame_height'], 'digest': manifest['digest'][:16] + '...'}}}})\n\n\n"
                "def frame_strip(frames, height=120):\n"
                "    tiles = [f.resize((round(f.width * height / f.height), height)) for f in frames]\n"
                "    strip = Image.new('RGB', (sum(t.width for t in tiles) + 4 * (len(tiles) - 1), height), (255, 255, 255))\n"
                "    x = 0\n"
                "    for tile in tiles:\n"
                "        strip.paste(tile, (x, 0))\n"
                "        x += tile.width + 4\n"
                "    return strip\n\n\n"
                "example = train_records[0]\n"
                "frame_strip(example['frames']).save('outputs/{stem}_example_clip.png')\n"
                "print({{'example': {{'id': example['id'], 'frames': len(example['frames']), 'frame_size': list(example['frames'][0].size), 'label': example['label'], 'source': example.get('source')}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:16]],\n"
                "    'label outside the set': [{{**train_records[0], 'label': 'juggling'}}, *train_records[1:16]],\n"
                "    'clip with three frames': [{{**train_records[0], 'frames': train_records[0]['frames'][:3]}}, *train_records[1:16]],\n"
                "    'too small': train_records[:8],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe, LABELS)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "**What to notice:** 300 clips of ten labels, 181 / 53 / 66, `sources` per split, the three digests and the four refusals.\n\n<details><summary>Check your reasoning</summary>How well the model recognises a film it has already seen: clips cut from one source share its scene, lighting and actor. The sample's split keeps a source together *per action*, so one film can still contribute different actions to different splits; BYOD's `split_dataset` keeps a source together across labels.</details>"
            ),
        },
        {
            "md": (
                "## 5. Rank five drawn clips through the inference contract\n\n"
                "*Core concept.* The probabilities are a softmax over the names you supplied: they always sum to one, so a set "
                "that omits the true class still gets a confident top-1. **Predict before running:** on simple cartoon "
                "clips, will the human-action model rank the right motion first?\n\n"
                "The inference contract is exercised as the inference-only tutorial exercised it: five deterministic 8-frame "
                "clips at 320 × 240 drawn with Pillow — a ball rolling right, a ball bouncing, a square growing, a sun setting, "
                "a ball standing still — with the five class names that describe them; a different clip family from the "
                "human-action videos, and clips the adapted model will be asked to rank again in Section 9. `validate_inputs` "
                "applies exactly the checks `classify` applies (exactly `NUM_FRAMES` frames of one size within the side "
                "ceilings, 2..`MAX_LABELS` distinct names of at most `MAX_LABEL_CHARS` characters) and one combined manifest "
                "records the request; a three-frame clip is validated too and its rejection recorded as a finding. `classify` "
                "returns one probability per supplied name — **the probabilities are a softmax over your label set**, and "
                "the label set is a **caller-owned request parameter**. The per-request `evaluation_report` with the "
                "intended classes is `sample-sanity`: `top1_accuracy` against the chance baseline on five cartoons is plumbing "
                "evidence, not a video-classification benchmark — video-classification accuracy needs labelled clips of the "
                "deployment domain, which Section 6 supplies. The inference-only card recorded 1/5 (chance 1/5): the model "
                "was trained on human-action video and does not read the motion of flat drawn shapes, which the notebook "
                "keeps as a finding rather than tuning the drawings until they pass."
            ),
            "code": (
                "def synthetic_clip(kind, n=8, size=(320, 240)):\n"
                "    \"\"\"An 8-frame cartoon clip drawn with Pillow (no text): sky, green ground and one moving element.\"\"\"\n"
                "    frames = []\n"
                "    for index in range(n):\n"
                "        t = index / (n - 1)\n"
                "        sky = (135, 206, 235)\n"
                "        if kind == 'the sun setting':\n"
                "            sky = (int(135 * (1 - t) + 30 * t), int(206 * (1 - t) + 40 * t), int(235 * (1 - t) + 80 * t))\n"
                "        frame = Image.new('RGB', size, sky)\n"
                "        d = ImageDraw.Draw(frame)\n"
                "        d.rectangle([0, 170, 320, 240], fill=(60, 179, 75))  # ground\n"
                "        if kind == 'a ball rolling to the right':\n"
                "            x = 30 + t * 230\n"
                "            d.ellipse([x, 130, x + 40, 170], fill=(220, 40, 40))\n"
                "        elif kind == 'a ball bouncing up and down':\n"
                "            y = 130 - abs(np.sin(t * np.pi * 2)) * 100\n"
                "            d.ellipse([140, y, 180, y + 40], fill=(220, 40, 40))\n"
                "        elif kind == 'a square growing larger':\n"
                "            s = 10 + t * 90\n"
                "            d.rectangle([160 - s / 2, 120 - s / 2, 160 + s / 2, 120 + s / 2], fill=(40, 70, 200))\n"
                "        elif kind == 'the sun setting':\n"
                "            y = 30 + t * 140\n"
                "            d.ellipse([240, y, 290, y + 50], fill=(255, 215, 0))\n"
                "        elif kind == 'a ball standing still':\n"
                "            d.ellipse([140, 130, 180, 170], fill=(220, 40, 40))\n"
                "        frames.append(frame)\n"
                "    return frames\n\n\n"
                "DRAWN_LABELS = ['a ball rolling to the right', 'a ball bouncing up and down', 'a square growing larger', 'the sun setting', 'a ball standing still']\n"
                "drawn_clips = [synthetic_clip(kind) for kind in DRAWN_LABELS]\n"
                "drawn_names = [f\"synthetic_{{kind.replace(' ', '_')}}_8x320x240\" for kind in DRAWN_LABELS]\n"
                "drawn_digests = {{name: hashlib.sha256(b''.join(np.asarray(frame.convert('RGB')).tobytes() for frame in clip)).hexdigest() for name, clip in zip(drawn_names, drawn_clips, strict=True)}}\n"
                "print({{'ceilings': {{'NUM_FRAMES': NUM_FRAMES, 'FRAME_SIZE': FRAME_SIZE, 'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MIN_LABELS': MIN_LABELS, 'MAX_LABELS': MAX_LABELS, 'MAX_LABEL_CHARS': MAX_LABEL_CHARS, 'MAX_TEXT_TOKENS': MAX_TEXT_TOKENS, 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS, 'MIN_PER_CLASS': MIN_PER_CLASS, 'EVAL_BATCH_SIZE': EVAL_BATCH_SIZE, 'device': pipe.device}}}})\n"
                "input_manifest = validate_inputs(drawn_clips, DRAWN_LABELS, names=drawn_names)\n"
                "try:\n"
                "    validate_inputs([drawn_clips[0][:3]], DRAWN_LABELS)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'three-frame-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'drawn_clips': len(drawn_clips), 'manifest_verdict': input_manifest['verdict'], 'findings': len(input_manifest['findings'])}})\n\n\n"
                "def rank_drawings(pipeline, label):\n"
                "    results, timings = [], []\n"
                "    for name, clip in zip(drawn_names, drawn_clips, strict=True):\n"
                "        started = time.perf_counter()\n"
                "        result = pipeline.classify(clip, DRAWN_LABELS)\n"
                "        timings.append(round(time.perf_counter() - started, 3))\n"
                "        results.append({{**result, 'clip': name, 'rgb_sha256': drawn_digests[name]}})\n"
                "    checks = {{\n"
                "        'probabilities_sum_to_one': all(abs(sum(p['probability'] for p in r['predictions']) - 1.0) < 1e-6 for r in results),\n"
                "        'one_entry_per_label': all([p['label'] for p in r['predictions']] and sorted(p['label'] for p in r['predictions']) == sorted(r['labels']) for r in results),\n"
                "        'ranked_descending': all(all(a['probability'] >= b['probability'] for a, b in zip(r['predictions'], r['predictions'][1:], strict=False)) for r in results),\n"
                "        'identity_reported': all(r['model_id'] == MODEL_ID and r['model_revision'] == MODEL_REVISION for r in results),\n"
                "    }}\n"
                "    if not all(checks.values()):\n"
                "        raise RuntimeError(f'classify output failed a sanity check: {{checks}}')\n"
                "    report = evaluation_report(results, DRAWN_LABELS, sample_kind='synthetic (drawn in this notebook)')\n"
                "    with open(f'outputs/{stem}_drawing_{{label}}.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump({{'results': results, 'report': report}}, handle, indent=2, ensure_ascii=False)\n"
                "    print({{label: {{'seconds': timings, 'checks': checks, 'top1': [(r['clip'].split('_8x')[0], r['top1'], round(r['predictions'][0]['probability'], 3)) for r in results], 'verdict': report['verdict'], 'top1_accuracy': report['metrics'][0]['value'], 'chance': report['baselines'][0]['value']}}}})\n"
                "    return results, timings, checks, report\n\n\n"
                "frozen_drawn, frozen_drawn_timings, frozen_drawn_checks, frozen_drawn_report = rank_drawings(pipe, 'frozen')"
            ),
        },
        {
            "md": (
                "**What to notice:** each drawn clip's top-1 and its probability, and the drawn-label top-1 rate.\n\n<details><summary>Check your reasoning</summary>No. In the Kaggle T4 release run the pre-trained model ranked `a ball standing still` first for all five drawn clips (probabilities 0.43–0.69), a drawn-label top-1 of 0.20. A softmax over five names you chose will always name one of them, confidently; the drawings are plumbing evidence, not a measurement.</details>"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model on the test clips\n\n"
                "Two non-adapted baselines frame the adaptation, each scored by `classification_metrics` (carried in "
                "`metrics.py`): **top-1 accuracy** (the highest-scoring name is the reference label), **top-3 accuracy** (the "
                "reference is among the three highest), **macro recall** (the mean per-label recall, so a class the model never "
                "predicts counts fully) and **macro F1**, with the per-label confusion beside them. The **chance** baseline "
                "guesses uniformly: top-1 = 1 / 10 by construction. The **majority-label** baseline predicts the most frequent "
                "training label for every clip: what the label distribution buys without looking at the frames. The **frozen "
                "model** is scored by `pipe.evaluate`, which ranks the closed label set for every clip in batches of "
                "`EVAL_BATCH_SIZE` and returns the rankings with the rates. Call it **the pre-trained model scored on our label "
                "phrasing** rather than zero-shot: X-CLIP was trained fully supervised on Kinetics-400, and most of these ten "
                "actions are in that training vocabulary. Four labels have an exact or near-exact Kinetics-400 class "
                "(`brushing hair`, `doing a cartwheel`, `clapping hands`, `dribbling a basketball`), four have related classes "
                "(catching or throwing a ball, several kinds of climbing and diving, sword fighting), and only `chewing` and "
                "`climbing stairs` have none. HMDB51 and Kinetics both draw on web video, so overlap at the video level cannot "
                "be ruled out either. **What to look for:** the per-label recall against that overlap.\n\n"
                "**Predict before running:** which labels will the pre-trained model find hardest — those with a Kinetics-400 "
                "counterpart or those without?"
            ),
            "code": (
                "METRICS = ('top1_accuracy', 'top3_accuracy', 'macro_recall', 'macro_f1')\n\n"
                "baseline_chance = chance_baseline(test_records, LABELS)\n"
                "baseline_majority = majority_baseline(train_records, test_records, LABELS)\n"
                "print({{'chance_baseline': {{k: round(baseline_chance[k], 3) for k in METRICS}}, 'n': baseline_chance['n'], 'n_labels': baseline_chance['n_labels']}})\n"
                "print({{'majority_baseline': {{k: round(baseline_majority[k], 3) for k in METRICS}}, 'label': baseline_majority['label']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, LABELS, batch_size=EVAL_BATCH_SIZE)\n"
                "print({{'frozen_model_test': {{k: round(frozen_test[k], 3) for k in METRICS}}, 'n': frozen_test['n'], 'mean_rank': round(frozen_test['mean_rank'], 2), 'verdict': frozen_test['verdict'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'per_label_recall': {{label: round(row['recall'], 2) for label, row in frozen_test['per_label'].items()}}}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "for record, ranking, probability in zip(test_records[:4], frozen_test['rankings'][:4], frozen_test['top1_probability'][:4], strict=True):\n"
                "    print({{'id': record['id'], 'reference': record['label'], 'frozen_top3': ranking[:3], 'top1_probability': round(probability, 3)}})\n"
                "frozen_verdict = 'pre-trained model above both baselines' if frozen_test['top1_accuracy'] > max(baseline_chance['top1_accuracy'], baseline_majority['top1_accuracy']) else 'a baseline matches or beats the pre-trained model'\n"
                "print({{'frozen_vs_baselines': frozen_verdict}})"
            ),
        },
        {
            "md": (
                "**What to notice:** the two baselines, the frozen top-1 / top-3 / macro F1, and the per-label recall.\n\n<details><summary>Check your reasoning</summary>In the Kaggle T4 release run (21 September 2026) the pre-trained model ranked the right label first for 0.803 of the 66 test clips (top-3 0.939, macro F1 0.792), against chance at 0.100 and the majority label at 0.091 — below chance because the training split's most frequent label is under-represented in the test split. Its weakest labels were `chewing` (0.43), which has no Kinetics-400 counterpart, `clapping hands` (0.50), `diving into water` (0.67) and `drawing a sword` (0.71). Much of the 0.803 is recognition of classes the model was trained on, not transfer.</details>"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the fusion head\n\n"
                "`pipe.adapt` trains only the fusion head — the multi-frame integration transformer, the visual projection, "
                "the prompt-side visual layer norm and projection, and the video-specific prompt generator: 10,247,680 of "
                "196,585,729 parameters — while the ViT-B/32 vision tower with its cross-frame attention, the CLIP text "
                "tower, the text projection and the logit scale stay frozen. The loss is the **cross-entropy over the closed "
                "label set**: the model's own contrastive scores, read as a classifier over the ten names. Because both towers "
                "are frozen, their outputs — the pooled CLS and the 49 patch features of every frame, and the ten label "
                "embeddings — are computed once under no gradient and cached (the **frozen-tower cache**), and each step runs "
                "only the head on those cached features: the logits equal the full model's exactly, at a fraction of the "
                "cost. AdamW without weight decay at a fixed learning rate, gradient clipping at 1.0, seeded shuffling, no "
                "scheduler, no augmentation. Epoch 0 records the frozen model's validation rates; every epoch is scored on the "
                "53 validation clips, and the epoch with the **highest validation top-1 accuracy** (the earliest on ties) is "
                "kept.\n\n"
                "Watch the validation top-1 rise from 0.811 to 0.962 (epoch 2 in the build "
                "record) while the loss drops from about 0.44 to 0.00: the adapted head ranks 60 of 66 held-out clips first (frozen 53), and the reference is in the top three for 65. The learning rate is "
                "deliberately small — a head this size memorises 181 clips within an epoch or two at 1e-4, and the validation "
                "curve is then flat from the first epoch.\n\n"
                "*Core concept.* Every call to `pipe.adapt` starts from the **pinned base**: head tensors an earlier call (or an "
                "artifact) changed are put back first, so epoch 0 is always the pre-trained model and re-running Sections 7–8 "
                "with a changed field repeats the comparison validly. To compare a change side by side without replacing the "
                "default exports, use Section 10.\n\n"
                "**Predict before running:** will validation keep the last epoch, or an early one?"
            ),
            "code": (
                "EPOCHS = 8  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 16  # @param {{type:\"integer\"}}\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_' + k: round(entry['val'][k], 3) for k in METRICS}})\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "settings = {{'epochs': EPOCHS, 'lr': LEARNING_RATE, 'batch_size': BATCH_SIZE}}\n"
                "if settings != {{'epochs': 8, 'lr': 1e-5, 'batch_size': 16}}:\n"
                "    print({{'note': 'changed settings: this run starts again from the pinned base and replaces the default results of Sections 8-9; Section 10 compares a change side by side instead', 'settings': settings}})\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, LABELS, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'labels': adapt_result['labels'], 'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'objective': adapt_result['objective'], 'cache_seconds': adapt_result['cache_seconds'], 'started_from': adapt_result['started_from'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                '**What to notice:** the validation top-1 per epoch, the training loss, and `best_epoch`.\n\n<details><summary>Check your reasoning</summary>An early one: in the release run validation top-1 rose from 0.811 (epoch 0, the pre-trained model) to 0.962 and epoch 2 was kept, while the training loss fell towards zero — the head was memorising 181 clips. The earliest best epoch wins ties, so later epochs that only fit the training clips better are not chosen.</details>'
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test clips were never used for training or epoch selection; no clip appears in two splits, and no source "
                "video appears in two splits *for the same action* (the sample's split key is per action). The adapted model is scored exactly as the frozen model was in Section 6 and the four systems are "
                "put side by side. Read it in this order: **top-1 accuracy** first (the measure the epoch was selected on — the "
                "build record measured 0.803 → **0.909**), then **macro F1** (0.792 → "
                "0.905), then **top-3 accuracy** (0.939 → 0.985), then the per-label recall to "
                "see which classes moved (`chewing` 0.43 → 0.86, `diving into water` 0.67 → 1.00, `drawing a sword` 0.71 → 1.00; down: `dribbling a basketball` 0.71 → 0.57). The cell records verdicts — "
                "`improved`, `no gain` or `worse` against the frozen model, and whether the adapted model beats both baselines — "
                "instead of asserting them: the epoch is chosen on validation, so the test split can still move the other way, "
                "and export, reload and the result still run. Sixty-six clips from one seeded split give **no dispersion estimate** — one clip is 1.5 "
                "points — so the deltas are sample-sanity evidence that the adaptation contract works, not a benchmark, and "
                "a result on ten HMDB51 actions says nothing about other actions, other cameras or your videos until you "
                "measure them.\n\n"
                "**Predict before running:** which labels will fine-tuning help most — those the pre-trained model already "
                "handled, or those it missed?"
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, LABELS, batch_size=EVAL_BATCH_SIZE)\n"
                "adapted_val = pipe.evaluate(val_records, LABELS, batch_size=EVAL_BATCH_SIZE)\n"
                "comparison = {{metric: {{'chance': round(baseline_chance[metric], 3), 'majority': round(baseline_majority[metric], 3), 'frozen': round(frozen_test[metric], 3), 'adapted': round(adapted_test[metric], 3)}} for metric in METRICS}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 3) for metric in METRICS}}\n"
                "comparison['per_label_recall'] = {{label: {{'frozen': round(frozen_test['per_label'][label]['recall'], 2), 'adapted': round(adapted_test['per_label'][label]['recall'], 2), 'support': adapted_test['per_label'][label]['support']}} for label in LABELS}}\n"
                "def direction(new, old):\n"
                "    return 'improved' if new > old else ('no gain' if new == old else 'worse')\n"
                "# Reported verdicts, not assertions: a fine-tune that does not help on the test split is a result to record.\n"
                "comparison['verdicts'] = {{'frozen_vs_baselines': frozen_verdict, **{{f'adapted_vs_frozen_{{metric}}': direction(adapted_test[metric], frozen_test[metric]) for metric in METRICS}}, 'adapted_above_both_baselines': bool(adapted_test['top1_accuracy'] > max(baseline_chance['top1_accuracy'], baseline_majority['top1_accuracy']))}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'labels': LABELS,\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'baselines': {{'chance': baseline_chance, 'majority': {{k: v for k, v in baseline_majority.items() if k not in ('definitions',)}}}},\n"
                "    'frozen_test': {{k: v for k, v in frozen_test.items() if k != 'definitions'}},\n"
                "    'validation_metrics': {{k: v for k, v in adapted_val.items() if k != 'definitions'}},\n"
                "    'test_metrics': adapted_test,\n"
                "    'per_clip': [{{'id': r['id'], 'reference': r['label'], 'source': r.get('source'), 'frozen_top1': f, 'frozen_ranking': fr, 'adapted_top1': a, 'adapted_ranking': ar}} for r, f, fr, a, ar in zip(test_records, frozen_test['predictions'], frozen_test['rankings'], adapted_test['predictions'], adapted_test['rankings'], strict=True)],\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "print({{'verdicts': comparison['verdicts']}})\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json', 'adapted_beats_both_baselines': adapted_test['top1_accuracy'] > max(baseline_chance['top1_accuracy'], baseline_majority['top1_accuracy'])}})"
            ),
        },
        {
            "md": (
                '**What to notice:** `delta_vs_frozen`, the per-label recall rows and the `verdicts`.\n\n<details><summary>Check your reasoning</summary>Mostly the ones it missed. In the release run top-1 rose from 0.803 to 0.909 (macro F1 0.792 → 0.905, top-3 0.939 → 0.985). The gain came from four labels — `chewing` 0.43 → 0.86, `diving into water` 0.67 → 1.00, `drawing a sword` 0.71 → 1.00, `clapping hands` 0.50 → 0.67 — while `dribbling a basketball` lost one clip (0.71 → 0.57). On 66 clips one clip is 1.5 points, and `chewing`, the label with no Kinetics-400 counterpart, is where tuning had most to add.</details>'
            ),
        },
        {
            "md": (
                "## 9. Look at the clips, rank the drawings again, export the adapter and reload it\n\n"
                "Six held-out clips are written as panels (`outputs/{stem}_examples/`: the 8-frame strip with the reference "
                "label, the frozen top-3 and the adapted top-3 beneath it) so the numbers can be checked by eye. The five "
                "drawn clips from Section 5 are then ranked again by the adapted model — the fusion head that was tuned ranks "
                "every request, so this is a small look at what the adaptation did *outside* its label set and its corpus: "
                "the build record measured before adaptation `a ball rolling to the right` → `a ball standing still` (0.69); `a ball bouncing up and down` → `a ball standing still` (0.51); `a square growing larger` → `a ball standing still` (0.43); `the sun setting` → `a ball standing still` (0.46); `a ball standing still` → `a ball standing still` (0.57) (top-1 0.20 over the five drawn labels); after adaptation `a ball rolling to the right` → `a ball standing still` (0.67); `a ball bouncing up and down` → `a ball standing still` (0.47); `a square growing larger` → `a ball bouncing up and down` (0.37); `the sun setting` → `a ball standing still` (0.41); `a ball standing still` → `a ball standing still` (0.55) (top-1 0.20) — five cartoons of evidence, not a measurement.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the fusion head, about 41 MB in float32 — as "
                "`adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and revision, "
                "the digest of the base `model.safetensors`, the tensor names, the file size and SHA-256, the label set the "
                "head was trained on, the training configuration and the epoch history (OUT8). "
                "`XClipVideoClassificationPipeline.from_artifact` re-verifies the base snapshot, checks the artifact manifest, "
                "its digest and its exact tensor set **before** deserialising, refuses any tensor outside the fusion head, and "
                "overlays the tensors onto a freshly loaded base — a new object from files, not the in-memory model (VER2). "
                "The cell asserts identical rankings on eight test clips (VER4)."
            ),
            "code": (
                "import shutil\n\n"
                "examples_dir = Path('outputs/{stem}_examples')\n"
                "shutil.rmtree(examples_dir, ignore_errors=True)\n"
                "examples_dir.mkdir(parents=True)\n"
                "caption_font = ImageFont.load_default(size=18)\n"
                "for record, frozen_ranking, adapted_ranking in zip(test_records[:6], frozen_test['rankings'][:6], adapted_test['rankings'][:6], strict=True):\n"
                "    strip = frame_strip(record['frames'], height=120)\n"
                "    sheet = Image.new('RGB', (max(strip.width, 1400), strip.height + 96), (255, 255, 255))\n"
                "    sheet.paste(strip, (0, 0))\n"
                "    marker = ImageDraw.Draw(sheet)\n"
                "    for i, (tag, text) in enumerate((('REF', record['label']), ('FROZEN', ' > '.join(frozen_ranking[:3])), ('ADAPTED', ' > '.join(adapted_ranking[:3])))):\n"
                "        marker.text((8, strip.height + 6 + i * 28), f'{{tag}}: {{text[:140]}}', fill=(20, 20, 20) if tag != 'FROZEN' else (150, 40, 40), font=caption_font)\n"
                "    sheet.save(examples_dir / f\"{{record['id']}}.png\")\n"
                "print({{'examples': sorted(p.name for p in examples_dir.iterdir()), 'rows': ['reference', 'frozen top-3', 'adapted top-3']}})\n\n"
                "adapted_drawn, adapted_drawn_timings, adapted_drawn_checks, adapted_drawn_report = rank_drawings(pipe, 'adapted')\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...', 'labels': artifact_manifest['adapter']['labels'], 'best_epoch': artifact_manifest['adapter']['best_epoch']}})\n\n"
                "reloaded = XClipVideoClassificationPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = [[p['label'] for p in item['predictions']] for item in pipe.classify_batch([r['frames'] for r in test_records[:8]], LABELS)]\n"
                "after = [[p['label'] for p in item['predictions']] for item in reloaded.classify_batch([r['frames'] for r in test_records[:8]], LABELS)]\n"
                "parity = {{'identical_rankings': sum(a == b for a, b in zip(before, after, strict=True)), 'of': len(before)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_rankings'] == parity['of']  # a contract check: the artifact reloads ranking-for-ranking\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'total_bytes': snapshot.get('total_bytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHTS_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': pipe.weight_sha256}},\n"
                "    'data_source': data_source,\n"
                "    'labels': LABELS,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'repo': CORPUS_REPO, 'revision': CORPUS_REVISION, 'file': CORPUS_FILE, 'license': CORPUS_LICENSE, 'row_groups': sorted(ROW_GROUP_PINS), 'shard_bytes': CORPUS_BYTES, 'classes': list(SAMPLE_CLASS_TEXT)}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'drawn': {{'names': drawn_names, 'labels': DRAWN_LABELS, 'digests': drawn_digests}}, 'frozen': {{'results': frozen_drawn, 'seconds': frozen_drawn_timings, 'checks': frozen_drawn_checks, 'report': frozen_drawn_report}}, 'adapted': {{'results': adapted_drawn, 'seconds': adapted_drawn_timings, 'checks': adapted_drawn_checks, 'report': adapted_drawn_report}}, 'output_files': ['outputs/{stem}_drawing_frozen.json', 'outputs/{stem}_drawing_adapted.json']}},\n"
                "    'comparison': comparison,\n"
                "    'examples': 'outputs/{stem}_examples',\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'av': av.__version__, 'device': pipe.device, 'dtype': 'float32'}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
        {
            "md": (
                "**What to notice:** the six panels in `outputs/{stem}_examples/`, the drawn clips' rankings before and after, and the reload parity line.\n\n<details><summary>Check your reasoning</summary>In the release run the drawn clips stayed almost all `a ball standing still` before and after adaptation — the tuned head was never shown cartoons — with one ranking changed. Reload parity held: eight of eight identical rankings.</details>"
            ),
        },
        {
            "md": (
                "## 10. Change one thing: a ten-times-higher learning rate (optional)\n\n"
                "*Evaluation practice.* A **Predict → Change one thing → Run → Observe → Explain** activity, off by default so "
                "Run all is unaffected. Set `RUN_EXPERIMENT = True`, change **one** field — by default the learning rate goes from "
                "1e-5 to 1e-4 — and run this cell after Sections 4–9. The experiment loads its **own** pipeline from the verified "
                "snapshot, so it starts from the checkpoint and never touches the default `pipe`; it writes only to "
                "`outputs/{stem}_experiment/`, prints the default and the changed run side by side (epoch 0 must match), and "
                "checks that the default exports are byte-identical afterwards.\n\n"
                "**Predict:** at 1e-4, at which epoch will validation top-1 peak?"
            ),
            "code": (
                "RUN_EXPERIMENT = False  # @param {{type:\"boolean\"}}\n"
                "EXPERIMENT_LEARNING_RATE = 1e-4  # @param {{type:\"number\"}}\n"
                "EXPERIMENT_EPOCHS = 8  # @param {{type:\"integer\"}}\n\n"
                "if not RUN_EXPERIMENT:\n"
                "    print({{'experiment': 'skipped (RUN_EXPERIMENT = False); the default path above is complete'}})\n"
                "else:\n"
                "    canonical_files = {{'adapter': artifact_dir / 'adapter.safetensors', 'evaluation_report': Path('outputs/{stem}_evaluation_report.json'), 'result': Path('outputs/{stem}_result.json')}}\n"
                "    canonical = {{name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in canonical_files.items()}}\n"
                "    experiment_dir = Path('outputs/{stem}_experiment')\n"
                "    shutil.rmtree(experiment_dir, ignore_errors=True)\n"
                "    experiment_dir.mkdir(parents=True)\n"
                "    # Its own pipeline from the verified snapshot: the experiment starts from the checkpoint and the default pipe is untouched.\n"
                "    experiment_pipe = XClipVideoClassificationPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "    experiment_result = experiment_pipe.adapt(train_records, val_records, LABELS, epochs=EXPERIMENT_EPOCHS, lr=EXPERIMENT_LEARNING_RATE, batch_size=BATCH_SIZE, progress=report)\n"
                "    experiment_test = experiment_pipe.evaluate(test_records, LABELS, batch_size=EVAL_BATCH_SIZE)\n"
                "    side_by_side = {{\n"
                "        'settings': {{'default': {{'lr': adapt_result['lr'], 'epochs': adapt_result['epochs']}}, 'experiment': {{'lr': EXPERIMENT_LEARNING_RATE, 'epochs': EXPERIMENT_EPOCHS}}}},\n"
                "        'validation_top1_by_epoch': {{'default': [round((h['val'] or {{}}).get('top1_accuracy', float('nan')), 3) for h in adapt_result['history']], 'experiment': [round((h['val'] or {{}}).get('top1_accuracy', float('nan')), 3) for h in experiment_result['history']]}},\n"
                "        'best_epoch': {{'default': adapt_result['best_epoch'], 'experiment': experiment_result['best_epoch']}},\n"
                "        'test': {{metric: {{'frozen': round(frozen_test[metric], 3), 'default': round(adapted_test[metric], 3), 'experiment': round(experiment_test[metric], 3)}} for metric in METRICS}},\n"
                "    }}\n"
                "    for key, row in side_by_side.items():\n"
                "        print({{key: row}})\n"
                "    with open(experiment_dir / 'experiment_report.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump({{'side_by_side': side_by_side, 'history': experiment_result['history']}}, handle, indent=2, ensure_ascii=False, default=str)\n"
                "    unchanged = {{name: hashlib.sha256(path.read_bytes()).hexdigest() == canonical[name] for name, path in canonical_files.items()}}\n"
                "    if not all(unchanged.values()):\n"
                "        raise RuntimeError(f'the experiment changed a default export: {{unchanged}}')\n"
                "    print({{'default_exports_unchanged': unchanged, 'experiment_outputs': str(experiment_dir)}})\n"
                "    del experiment_pipe"
            ),
        },
        {
            "md": (
                "**Observe → Explain.** Compare the two validation curves (epoch 0 must be equal) and the `test` rows.\n\n"
                "<details><summary>Check your reasoning</summary>The build record found that at 1e-4 a head this size memorises "
                "181 clips within an epoch or two, so validation top-1 peaks at epoch 1 and the curve is then flat; the earliest "
                "best epoch is kept. A faster climb is not a better model: compare the test rows, and read a difference of one "
                "or two clips (1.5 points each) as noise. No experiment run is recorded on the release runtime.</details>"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "A video–text model trained on Kinetics-400 — whose vocabulary already covers most of these ten actions — ranks the "
        "right one of ten HMDB51 actions first for 0.803 of the held-out clips in the Kaggle T4 release run (21 September "
        "2026); a bounded fine-tuning of its fusion head on 181 clips moves that to "
        "0.909 top-1 and 0.905 macro F1 in the build record, with a 41 MB adapter that reloads "
        "ranking-for-ranking. That is the claim: the adaptation contract works end to end on a closed label set with a real "
        "labelled set, and the numbers it produces are read as top-1 / top-3 accuracy, macro recall and macro F1 against two "
        "non-adapted baselines and the frozen model, with the per-label recall beside them rather than in isolation. "
        "The Tesla T4 run reproduced the CPU sweep's numbers exactly (0.909 / 0.985 / 0.905 at epoch 2). This row has no sibling on its corpus — X-CLIP is the fleet's only video model — so the comparison is the frozen prompts against the tuned head on the same 66 clips: the gain comes from four labels (`chewing` 0.43 → 0.86, `diving into water` 0.67 → 1.00, `drawing a sword` 0.71 → 1.00, `clapping hands` 0.50 → 0.67) while `dribbling a basketball` lost one clip (0.71 → 0.57), and the five drawn cartoon clips stay at chance before and after (every clip but one is called `a ball standing still`): the head learned the ten HMDB51 actions, not motion in general.\n\n"
        "The test split is 66 clips from one seeded, source-grouped draw of one 300-clip sample, the validation split that "
        "picks the epoch is 53, and every rate is over one reference label per clip — not a benchmark, not the HMDB51 "
        "protocol (three splits over all 51 classes), not a measure of temporal localisation. So a result here says the "
        "contract works on ten actions cut from films, not that the adapted model handles other actions, other cameras or "
        "your clips. The head that was tuned ranks every request: the drawn clips re-ranked in Section 9 are five cartoons "
        "of evidence about what the tuning did outside its label set (before adaptation `a ball rolling to the right` → `a ball standing still` (0.69); `a ball bouncing up and down` → `a ball standing still` (0.51); `a square growing larger` → `a ball standing still` (0.43); `the sun setting` → `a ball standing still` (0.46); `a ball standing still` → `a ball standing still` (0.57) (top-1 0.20 over the five drawn labels); after adaptation `a ball rolling to the right` → `a ball standing still` (0.67); `a ball bouncing up and down` → `a ball standing still` (0.47); `a square growing larger` → `a ball bouncing up and down` (0.37); `the sun setting` → `a ball standing still` (0.41); `a ball standing still` → `a ball standing still` (0.55) (top-1 0.20)), not a measurement, and a "
        "deployment that ranks other label sets must measure them after adapting. The towers were not adapted: what the "
        "frame encoder cannot see stays unseen, and **the probabilities remain a softmax over your label set**.\n\n"
        "Three things to carry to real data. **Baselines first:** the chance and majority rates on *your* labels, and the "
        "frozen model's per-label recall, are the numbers to read before any adapted one. **Macro over micro:** a class the "
        "model never predicts costs a full tenth of macro recall while barely moving top-1 on a skewed set; read both. "
        "**Leakage:** keep every clip in one split (the contract de-duplicates by decoded pixels) and split by source video, "
        "camera or session when your clips come from few recordings — clips cut from one video share its scene and actor.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this standalone notebook, "
        "can acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real labelled clip set, validate "
        "the demonstrated dataset contract without leakage, execute the inference contract for five clips and a bounded "
        "fine-tuning of the fusion head with the closed-set cross-entropy, evaluate against two non-adapted baselines and "
        "the frozen model on a source-disjoint split, and emit the shown machine-readable artifacts — without the "
        "repository being reachable. It does **not** establish benchmark superiority, classification quality on any other "
        "action set or video domain, ranking quality on other label sets after adaptation, or production fitness.\n\n"
        "**Optional experiments (off by default; each names its field and what to run):** Section 10 runs a ten-times-higher "
        "learning rate in its own pipeline and prints it beside the default run — change `EXPERIMENT_LEARNING_RATE` or "
        "`EXPERIMENT_EPOCHS` there and run that cell again. Changing `EPOCHS` or `LEARNING_RATE` and choosing **Run after** from "
        "Section 7, or `SPLIT_SEED` and **Run after** from Section 4, also starts from the pinned base — every `adapt` and "
        "Section 4 put it back first — but replaces the default results and exports. Editing `SAMPLE_CLASS_TEXT`'s phrasing "
        "in the carried module and running from Section 4 shows how much the pre-trained model depends on the wording of a "
        "label. BYOD: `USE_BYOD` and `BYOD_PATH` in Section 4, then **Run after** from Section 4, and read the two baselines "
        "before the adapted number.\n\n"
        "## Troubleshooting\n\n"
'- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — you are on Windows, macOS or an ARM machine. Use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused, an incomplete one is finished. If it repeats, the network is blocking or altering `files.pythonhosted.org` or `pypi.org`.\n- **"The isolated environment\'s Python process exited"** — usually out of memory. Restart the session and choose **Run all**; leave the optional experiment off on a small runtime.\n- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable, so the cells after it keep working. After a session restart, run from the top.\n- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message names the file. Delete it from the snapshot folder Section 3 prints and run Section 3 again; the snapshot comes from `huggingface.co`.\n- **Section 4 names the parquet row group in a `sha256` error** — the cached `weights/hmdb51/test-rg0.parquet` is incomplete; delete it and run Section 4 again (the default path needs `huggingface.co`).\n- **Out of memory** — lower `BATCH_SIZE` in Section 7, or restart the session and choose **Run all**; leave Section 10 off on a small runtime.\n- **BYOD: "BYOD_PATH … does not exist"** — the path is relative to the working directory printed in the message.\n- **BYOD: "the upload dialog exists only in Google Colab"** — on Kaggle or Jupyter, put the zip in the runtime (or attach it as a dataset) and set `BYOD_PATH`.\n- **BYOD: "Upload exactly one .zip file"** — the dialog was cancelled or several files were chosen; run the cell again.\n- **BYOD: "labels.csv line N (file …): names a missing clip"** — fix the `file` column of that row, or add the clip to the zip.\n- **BYOD: "… not a decodable clip of at least 8 frames"** — the file on that line is corrupt, too short, or in a container PyAV cannot read.\n- **BYOD: "the train split holds …" or "the … split holds no clip of …"** — add clips or sources; the message names the minimum for your label count.\n'
        "## Glossary\n\n"
        "- **Cross-frame attention** — each frame's encoder attends to the other frames, so motion informs every frame "
        "embedding.\n"
        "- **Frame-integration transformer** — a one-layer transformer that fuses the 8 frame embeddings into one video "
        "embedding.\n"
        "- **Prompt generator** — conditions the class-name embeddings on the clip's patch features.\n"
        "- **Fusion head** — the integration transformer, the visual projections and the prompt generator: the only "
        "trained part.\n"
        "- **Frozen-tower cache** — the frozen towers' outputs computed once and reused, so each step runs only the head.\n"
        "- **Top-k accuracy** — the reference label is among the k highest-ranked names.\n"
        "- **Macro recall / macro F1** — the mean per-label recall or F1, so a never-predicted label costs a full share.\n"
        "- **Source grouping** — clips cut from one source video stay in one split.\n"
        "- **Kinetics-400 overlap** — labels whose action the model was trained on; the pre-trained score on them is not "
        "zero-shot transfer.\n"
        "- **Adapter / reload parity** — the trained tensors only (safetensors) overlaid on the pinned base; the reloaded "
        "pipeline ranks identically.\n"
        "- **BYOD** — bring your own data: your labelled clips through the same cells.\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional — fill in from **your** run, not the recorded one:\n\n"
        "- The task was ___ labels on ___ test clips from ___ sources.\n"
        "- Labels with a Kinetics-400 counterpart: ___; without: ___.\n"
        "- Pre-trained top-1 ___ (weakest label ___ at ___); chance ___, majority ___.\n"
        "- After fine-tuning (epoch ___ kept): top-1 ___, macro F1 ___; labels that moved: ___.\n"
        "- What I would need before claiming the fine-tune helps on my clips: ___ (for example labels outside Kinetics-400, more test clips, several seeds).\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/xclip-video-classification-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/xclip-video-classification-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/xclip-video-classification-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/microsoft/VideoX/tree/master/X-CLIP\n"
        "- Expanding Language-Image Pretrained Models for General Video Recognition (Ni et al., ECCV 2022): https://arxiv.org/abs/2208.02816\n"
        "- HMDB51 (Serre Lab, CC BY 4.0): https://huggingface.co/datasets/Serrelab/hmdb51 — Kuehne, Jhuang, Garrote, Poggio, Serre, HMDB: A Large Video Database for Human Motion Recognition (ICCV 2011); parquet repack read here: https://huggingface.co/datasets/mteb/HMDB51\n"
        "- DIMER Notebook Specification 2.2 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
