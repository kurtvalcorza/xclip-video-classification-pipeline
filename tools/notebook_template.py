"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

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
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
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
        "credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5). On a Tesla T4 the default path took "
        "about 2 minutes of cell time (eight epochs 12 s, frozen scoring of 66 clips "
        "3 s); a CUDA runtime is used automatically when present, and the path is practical on CPU too (the "
        "build venv decoded the 300 clips in about 70 s, scored the test split in 10 s and ran the eight epochs in about 60 s)."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to upload one zip "
        "of clips (AVI, MP4, MOV, MKV, WebM, GIF or WebP; at least 8 frames each) plus a `labels.csv` (`file`, `label`, optional "
        "`id` and `source`; one row per clip, at least sixteen clips of at least two labels with at least two clips each). The "
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
        "- **Runtime:** a fresh supported runtime (Google Colab or Kaggle, Python 3.12; CPU or CUDA). The default path uses CUDA automatically when present. The vision tower runs `EVAL_BATCH_SIZE` clips (8 × 8 frames) per forward and the build record measured 3 s to score 66 clips and 12 s for the eight epochs (caching the tower features for 181 + 53 clips took 6 s) on a Tesla T4, about 2 minutes of cell time for the whole path including the pinned install and the downloads; the build venv's CPU ran the same path in a few minutes. The pinned `torch==2.14.0` install, the 786 MB checkpoint and the 113 MB row group are the large downloads of the run.",
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
                "anything."
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_zip = Path('work') / 'byod.zip'\n"
                "    byod_zip.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_zip.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_zip)\n"
                "    LABELS = validate_dataset(records)['labels']\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records)}}\n"
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
                "## 5. Rank five drawn clips through the inference contract\n\n"
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
                "## 6. Baselines and the frozen model on the test clips\n\n"
                "Two non-adapted baselines frame the adaptation, each scored by `classification_metrics` (carried in "
                "`metrics.py`): **top-1 accuracy** (the highest-scoring name is the reference label), **top-3 accuracy** (the "
                "reference is among the three highest), **macro recall** (the mean per-label recall, so a class the model never "
                "predicts counts fully) and **macro F1**, with the per-label confusion beside them. The **chance** baseline "
                "guesses uniformly: top-1 = 1 / 10 by construction. The **majority-label** baseline predicts the most frequent "
                "training label for every clip: what the label distribution buys without looking at the frames. The **frozen "
                "model** is scored by `pipe.evaluate`, which ranks the closed label set for every clip in batches of "
                "`EVAL_BATCH_SIZE` and returns the rankings with the rates. Expect the frozen model **well above both "
                "baselines** — it is a zero-shot action classifier and these are human actions: the build record measured "
                "**0.803** top-1 on the 66 held-out clips, with the per-label recall showing which classes it "
                "misses (`chewing` 0.43, `clapping hands` 0.50, `diving into water` 0.67); read four rankings under their references."
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
                "    print({{'id': record['id'], 'reference': record['label'], 'frozen_top3': ranking[:3], 'top1_probability': round(probability, 3)}})"
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
                "curve is then flat from the first epoch."
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
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, LABELS, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'labels': adapt_result['labels'], 'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'objective': adapt_result['objective'], 'cache_seconds': adapt_result['cache_seconds'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test clips were never used for training or epoch selection, and no clip and no source video appears in "
                "two splits. The adapted model is scored exactly as the frozen model was in Section 6 and the four systems are "
                "put side by side. Read it in this order: **top-1 accuracy** first (the measure the epoch was selected on — the "
                "build record measured 0.803 → **0.909**), then **macro F1** (0.792 → "
                "0.905), then **top-3 accuracy** (0.939 → 0.985), then the per-label recall to "
                "see which classes moved (`chewing` 0.43 → 0.86, `diving into water` 0.67 → 1.00, `drawing a sword` 0.71 → 1.00; down: `dribbling a basketball` 0.71 → 0.57). The cell asserts the adapted top-1 is at least the frozen one "
                "and above chance. Sixty-six clips from one seeded split give **no dispersion estimate** — one clip is 1.5 "
                "points — so the deltas are sample-sanity evidence that the adaptation contract works, not a benchmark, and "
                "a result on ten HMDB51 actions says nothing about other actions, other cameras or your videos until you "
                "measure them."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, LABELS, batch_size=EVAL_BATCH_SIZE)\n"
                "adapted_val = pipe.evaluate(val_records, LABELS, batch_size=EVAL_BATCH_SIZE)\n"
                "comparison = {{metric: {{'chance': round(baseline_chance[metric], 3), 'majority': round(baseline_majority[metric], 3), 'frozen': round(frozen_test[metric], 3), 'adapted': round(adapted_test[metric], 3)}} for metric in METRICS}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 3) for metric in METRICS}}\n"
                "comparison['per_label_recall'] = {{label: {{'frozen': round(frozen_test['per_label'][label]['recall'], 2), 'adapted': round(adapted_test['per_label'][label]['recall'], 2), 'support': adapted_test['per_label'][label]['support']}} for label in LABELS}}\n"
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
                "assert adapted_test['top1_accuracy'] >= frozen_test['top1_accuracy']\n"
                "assert adapted_test['top1_accuracy'] > baseline_chance['top1_accuracy']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json', 'adapted_beats_both_baselines': adapted_test['top1_accuracy'] > max(baseline_chance['top1_accuracy'], baseline_majority['top1_accuracy'])}})"
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
                "assert parity['identical_rankings'] == parity['of']\n\n"
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
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "A zero-shot video–text model trained on Kinetics-400 already ranks the right one of ten HMDB51 actions first for "
        "0.803 of the held-out clips; a bounded fine-tuning of its fusion head on 181 clips moves that to "
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
        "**Optional experiments (they do not affect the default path):** raise `EPOCHS` and watch the validation top-1 pick "
        "the epoch; set `LEARNING_RATE` to `1e-4` and read a curve that peaks in the first epoch and then flattens as the head "
        "memorises the training clips; change `SPLIT_SEED` and read how much 66 clips move; edit `SAMPLE_CLASS_TEXT`'s "
        "phrasing in the carried module and rerun from Section 4 to see how much the frozen model depends on the wording of "
        "the label; or bring your own labelled clips through BYOD and read the two baselines before the adapted number.\n\n"
        "**Troubleshooting.** `RuntimeError: Core dependencies changed while older modules were loaded` in Section 1: the "
        "pinned install replaced a package the runtime had pre-imported — restart the runtime and rerun from the top. "
        "`FileNotFoundError: snapshot file missing` or a `sha256`/`size` `ValueError` in Section 3: a staged file is "
        "incomplete or altered — delete it from `weights/xclip-base-patch32/` and rerun Section 3. A `sha256` `ValueError` "
        "naming the parquet row group in Section 4: the cached `weights/hmdb51/test-rg0.parquet` is incomplete — delete it "
        "and rerun Section 4.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/xclip-video-classification-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/xclip-video-classification-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/xclip-video-classification-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/microsoft/VideoX/tree/master/X-CLIP\n"
        "- Expanding Language-Image Pretrained Models for General Video Recognition (Ni et al., ECCV 2022): https://arxiv.org/abs/2208.02816\n"
        "- HMDB51 (Serre Lab, CC BY 4.0): https://huggingface.co/datasets/Serrelab/hmdb51 — Kuehne, Jhuang, Garrote, Poggio, Serre, HMDB: A Large Video Database for Human Motion Recognition (ICCV 2011); parquet repack read here: https://huggingface.co/datasets/mteb/HMDB51\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
