"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "xclip_video_classification_pipeline",
    "repo_name": "xclip-video-classification-pipeline",
    "stem": "xclip_video_classification",
    "notebook_name": "xclip_video_classification_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "pipeline_class": "XClipVideoClassificationPipeline",
    "weights_key": "xclip-base-patch32",
    "runtime_imports": ["torch", "transformers"],
    "title": "X-CLIP base/32 — DIMER zero-shot video classification tutorial (standalone)",
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
    "capability": "zero-shot video classification — one 8-frame clip plus 2–32 free-text class names → a ranking of those names with a softmax over them — using the pinned `microsoft/xclip-base-patch32` weights",
    "intro": (
        "At inference the X-CLIP model (a CLIP ViT-B/32 frame encoder with cross-frame attention, a one-layer multi-frame "
        "integration transformer that fuses the 8 frame embeddings into one video embedding, a CLIP text encoder, and a "
        "video-specific prompt generator; about 197M parameters, trained fully supervised on Kinetics-400) embeds the clip "
        "and each class name and scores every pair; the carried module softmaxes the scores over the names you supplied. "
        "**No adaptation occurs:** no training, fine-tuning, in-context conditioning, or preprocessing fitting happens in "
        "this notebook — the upstream checkpoint supplies the weights, processor and tokenizer, and the carried module adds "
        "snapshot verification, the input contract (exactly 8 frames of one size within the side ceilings, 2–32 distinct "
        "class names up to 64 characters), a fixed output contract (probability, logit and rank per name, the top-1 name), "
        "and the `sample_frames`, `frames_from_animation`, `validate_inputs` and `evaluation_report` helpers. The default "
        "sample is five 8-frame clips of moving shapes drawn in code with their intended class names, so `top1_accuracy` "
        "against the chance baseline is demonstration (plumbing) evidence for cartoons, not a Kinetics benchmark — and the "
        "model gets one of the five right, which the notebook keeps and explains: it was trained on human-action video and "
        "does not read the motion of flat drawn shapes."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, draw synthetic 8-frame clips (or upload your own animated GIF/WebP and "
        "subsample it) and validate them into an input manifest, name the candidate classes, run the supported task, read "
        "the ranking correctly (a softmax over your own label set, not a calibrated probability), exercise an optional "
        "BYOD path, produce an evaluation report that is `sample-sanity` with `top1_accuracy` and a chance baseline only "
        "when the true classes are known and `not-measurable` otherwise, and export the rankings, a contact sheet and "
        "provenance."
    ),
    "exclusions": (
        "Video decoding from container formats (no `.mp4`/`.avi` reader is pinned; the notebook accepts frames or an "
        "animated image Pillow can open), temporal localisation or per-frame labels (one ranking per clip), open-set or "
        "abstaining classification (the softmax always picks one of your names), video–text retrieval over a corpus, "
        "batch throughput, evaluation on Kinetics-400 or UCF101 (not bundled; only drawn clips are scored here), and any "
        "training. The model was trained on human-action video at 8 frames; drawn shapes, static scenes, screen "
        "recordings and non-English class names are outside what this notebook measures, and a confident ranking carries "
        "no signal."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both. CPU is adequate: the repository's model card records 5.2 s to load and about 0.1 s per 8-frame clip with five names in the Windows venv (Intel Core Ultra 9 275HX). The pinned `torch==2.14.0` install and the 786 MB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python, NumPy and PIL; what a contrastive video–text model scores; why a softmax over a label set you chose is a ranking and not a probability; what top-1 accuracy against a chance baseline does and does not show on five clips.",
        "- **Data:** the default sample is five deterministic 8-frame clips at 320×240 drawn in code with Pillow (a ball rolling right, a ball bouncing, a square growing, a sun setting with a darkening sky, a ball standing still; no text rendering, so their digests are stable across Pillow builds) with the five class names that describe them, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one animated GIF/WebP/APNG that Pillow can open (at least 8 frames; subsampled uniformly to 8), any colour mode, sides between 16 and 4096 px, plus your own class names typed into the form field; the true class is unknown for uploads, so their report is `not-measurable`. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Draw the synthetic clips or optional BYOD\n\n"
                "The default sample is **synthetic** and carries its own references: five 8-frame clips at 320×240 — a red "
                "ball rolling left to right along green ground, a ball bouncing in place, a blue square growing, a sun "
                "sinking while the sky darkens, and a ball standing still — are drawn with Pillow, the same clips the "
                "repository's smoke run used, and the five class names describing them are the label set. The intended class "
                "of each clip is the reference for the `top1_accuracy` sanity check later. They are not a labelled dataset, so "
                "nothing here is a Kinetics measurement — and the smoke run ranked `a ball standing still` first for every "
                "clip except the sunset's runner-up, scoring 1/5 (chance 1/5): the model does not read the motion of flat "
                "drawn shapes, which the notebook keeps as a recorded finding rather than tuning the drawings until they "
                "pass. Each clip's digest is printed for the record. BYOD is optional and disabled by default; when enabled, "
                "upload one animated image and type your class names — its true class is unknown, so the evaluation report "
                "will be `not-measurable`.\n\n"
                "The label set is a **caller-owned request parameter**: the softmax ranks only the names you supply, so a "
                "set that omits the true class still yields a confident top-1. Nothing is validated in this cell — the next "
                "section hands the clips and the names to the pipeline's own validation stage, which is the only checker. "
                "Look for one dictionary per clip naming the sample kind, frame count, size and digest, plus the label set."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "byod_labels = 'playing soccer\\ndancing\\ncooking'  # @param {{type:\"string\"}}\n\n\n"
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
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    clip_name = next(iter(uploaded))\n"
                "    animation = Image.open(io.BytesIO(uploaded[clip_name]))\n"
                "    clips = [sample_frames(frames_from_animation(animation))]\n"
                "    clip_names = [clip_name]\n"
                "    labels = [line.strip() for line in byod_labels.splitlines() if line.strip()]\n"
                "    correct_labels = None\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Deterministic drawings: no randomness and no text rendering, so no seed is needed and the digests are stable.\n"
                "    labels = ['a ball rolling to the right', 'a ball bouncing up and down', 'a square growing larger', 'the sun setting', 'a ball standing still']\n"
                "    clips = [synthetic_clip(kind) for kind in labels]\n"
                "    clip_names = [f\"synthetic_{{kind.replace(' ', '_')}}_8x320x240\" for kind in labels]\n"
                "    correct_labels = list(labels)  # clip i was drawn to depict labels[i]\n"
                "    sample_kind = 'synthetic'\n\n"
                "digests = {{name: hashlib.sha256(b''.join(np.asarray(frame.convert('RGB')).tobytes() for frame in clip)).hexdigest() for name, clip in zip(clip_names, clips)}}\n"
                "for name, clip in zip(clip_names, clips):\n"
                "    print({{'sample_kind': sample_kind, 'name': name, 'n_frames': len(clip), 'frame_size': clip[0].size, 'rgb_sha256': digests[name]}})\n"
                "print({{'labels': labels, 'has_correct_labels': correct_labels is not None}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the request → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `classify` applies — "
                "each clip exactly `NUM_FRAMES` PIL frames of one size with sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` px, and "
                "`MIN_LABELS`..`MAX_LABELS` distinct non-empty class names of at most `MAX_LABEL_CHARS` characters (normalised "
                "by `format_labels`) — and returns an **input manifest** naming the schema (including the resize/centre-crop "
                "preprocessing and the softmax rule), each clip's observed frame count, mode and size, the normalised label "
                "set and the verdict. The manifest is written to `outputs/{stem}_input_manifest.json`. To show what rejection "
                "looks like, the cell also validates a 3-frame clip and records the pipeline's own error message as a "
                "finding. Inside the pipeline each frame is converted to RGB, resized on its shorter side to 224 and "
                "centre-cropped; nothing else is dropped or altered. The pipeline cannot tell whether a clip shows an action "
                "or whether your label set contains its true class: that contract is the caller's."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'NUM_FRAMES': NUM_FRAMES, 'FRAME_SIZE': FRAME_SIZE, 'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MIN_LABELS': MIN_LABELS, 'MAX_LABELS': MAX_LABELS, 'MAX_LABEL_CHARS': MAX_LABEL_CHARS, 'MAX_TEXT_TOKENS': MAX_TEXT_TOKENS}}}})\n"
                "input_manifest = validate_inputs(clips, labels, names=clip_names)\n"
                "# Demonstrate rejection on a request that breaks the contract; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs([clips[0][:3]], labels)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'three-frame-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Classify the clips and read the output correctly\n\n"
                "`classify` returns, per clip, `predictions` ordered by descending probability (each with the normalised "
                "`label`, its `probability` and raw `logit`), the `top1` name, the normalised `labels`, the frame count and "
                "size, and the model identity. **The probabilities are a softmax over your label set**: they sum to 1 and "
                "rank the names you supplied, they are not calibrated, and a label set without the true class still yields a "
                "confident winner. The forward pass is deterministic on a fixed device and dtype; CUDA kernels can shift "
                "logits slightly, so GPU and CPU rankings need not agree on close pairs. Each clip costs one 8-frame encoding "
                "plus one text encoding per name (about 0.1 s on the reference CPU). As recorded in the model card, the "
                "repository's CPU smoke on these same clips ranked `a ball standing still` first for all five (0.43–0.69), "
                "putting the correct name at rank 3–4 for the four moving clips, and ranked the same name first for eight "
                "blank white frames (0.70): the model always produces a ranking, whether or not the clip shows an action."
            ),
            "code": (
                "import time\n\n"
                "results, seconds = [], []\n"
                "for name, clip in zip(clip_names, clips):\n"
                "    t0 = time.time()\n"
                "    result = pipe.classify(clip, labels)\n"
                "    result['clip'] = name\n"
                "    results.append(result)\n"
                "    seconds.append(round(time.time() - t0, 2))\n"
                "print({{'device': pipe.device, 'seconds_per_clip': seconds, 'n_labels': len(results[0]['labels'])}})\n"
                "for result in results:\n"
                "    ranking = ', '.join(f\"{{entry['label']}}={{entry['probability']:.2f}}\" for entry in result['predictions'])\n"
                "    print(f\"{{result['clip']}}\\n   top-1: {{result['top1']!r}}  |  {{ranking}}\")"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. No accuracy is "
                "reported by default: video-classification accuracy needs labelled clips from the deployment domain with a "
                "matching class vocabulary, and this repository ships none (Kinetics-400 and UCF101 are not bundled). When the "
                "true class of each clip is supplied the report carries `top1_accuracy` over the clips, a `chance` baseline "
                "(the mean of 1/labels over the clips) and one entry per clip with the rank of the correct name, with the "
                "verdict `sample-sanity`. On the synthetic path those classes are intentions **you drew yourself**, so the score "
                "proves only that the input contract, preprocessing, forward pass and softmax round-trip — and the four "
                "recorded misses show what chance-level ranking looks like in the report. On BYOD the true class is unknown, "
                "the verdict is `not-measurable`, and the report states what would make the task measurable. The report is "
                "written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(results, correct_labels, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{k: v for k, v in report.items() if k not in ('metrics', 'per_clip', 'baselines')}}, indent=2))\n"
                "for metric in report['metrics']:\n"
                "    print(f\"{{metric['id']:15}} {{metric['value']:.3f}}  ({{metric['estimation']}})\")\n"
                "for baseline in report['baselines']:\n"
                "    print(f\"{{baseline['id']:15}} {{baseline['value']:.3f}}  ({{baseline['note']}})\")\n"
                "for entry in report.get('per_clip', []):\n"
                "    print(f\"  {{'OK  ' if entry['correct'] else 'MISS'}}  {{entry['clip']}} -> {{entry['top1']!r}} {{entry['top1_probability']:.2f}} (correct {{entry['correct_label']!r}} at rank {{entry['rank_of_correct']}})\")\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('The true class of this clip is not known to the notebook, so nothing is scored; judge the ranking yourself.')"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves every ranking (clip, ordered predictions with probability and logit, top-1, "
                "the label set), the evaluation report, the input manifest, the sample identities and digests, the "
                "notebook's source (repository, revision, embedded module digest, generator), the model identifier, the "
                "immutable model revision, the model licence, and the runtime identity (Python, `torch`, `transformers`, "
                "device). The rankings are also written as CSV with explicit `clip`, `rank`, `label`, `probability`, "
                "`logit` columns so ordering survives downstream use, and a contact-sheet PNG shows the 8 frames of every "
                "clip with its top-1 name for visual inspection (a supplement to, not a replacement for, the machine-readable "
                "files). No credentials are recorded."
            ),
            "code": (
                "import csv\n\n"
                "from PIL import ImageFont\n\n"
                "thumb_w, thumb_h, row_h = 120, 90, 90 + 28\n"
                "sheet = Image.new('RGB', (thumb_w * NUM_FRAMES, row_h * len(clips)), 'white')\n"
                "draw = ImageDraw.Draw(sheet)\n"
                "panel_font = ImageFont.load_default(size=14)\n"
                "for row, (clip, result) in enumerate(zip(clips, results)):\n"
                "    for col, frame in enumerate(clip):\n"
                "        thumb = frame.convert('RGB').copy()\n"
                "        thumb.thumbnail((thumb_w, thumb_h))\n"
                "        sheet.paste(thumb, (col * thumb_w, row * row_h))\n"
                "    draw.text((6, row * row_h + thumb_h + 6), f\"{{result['clip']}}  ->  {{result['top1']}} ({{result['predictions'][0]['probability']:.2f}})\", fill=(40, 90, 220), font=panel_font)\n"
                "sheet.save('outputs/{stem}_contact_sheet.png')\n"
                "payload = {{\n"
                "    'predictions': results,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'names': clip_names, 'rgb_sha256': digests, 'labels': labels, 'correct_labels': correct_labels}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_rankings.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['clip', 'rank', 'label', 'probability', 'logit'])\n"
                "    for result in results:\n"
                "        for rank, entry in enumerate(result['predictions'], start=1):\n"
                "            writer.writerow([result['clip'], rank, entry['label'], f\"{{entry['probability']:.6f}}\", f\"{{entry['logit']:.4f}}\"])\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The rankings are softmaxes over the class names you supplied; nothing in the output says whether any of those "
        "names fits the clip, the probabilities are relative and uncalibrated, and the model ranks every clip — including "
        "eight blank frames — with equal confidence. On the drawn clips the `top1_accuracy` in the evaluation report compares "
        "the top-1 name with the class you drew each clip to depict and the verdict is `sample-sanity`, which proves only "
        "that the input contract, preprocessing, forward pass and softmax work (the repository's smoke run scored 1/5 against "
        "a chance baseline of 0.2, ranking `a ball standing still` first for every clip: a Kinetics-trained model does not "
        "read the motion of flat cartoon shapes); it says nothing about real footage of people, sports or everyday actions, "
        "clips longer or shorter than the 8 sampled frames, fine-grained action distinctions, or non-English names, and a "
        "BYOD result is a single-clip observation with the verdict `not-measurable`. **The label set is part of the "
        "request**: the same clip ranked `juggling balls` at 0.98 under Kinetics-style names in the smoke run, so choose "
        "names that cover what the clip could show and treat a confident top-1 for a clip that shows none of them as the "
        "expected failure mode, not an exception. The pipeline provides no video decoding, no temporal localisation, no "
        "abstention, no benchmark evaluation and no training capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, can "
        "acquire and digest-verify the pinned model, validate the demonstrated request, execute the public pipeline path, and "
        "emit the shown machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** "
        "establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on "
        "an unseen domain.\n\n"
        "**Next experiments:** replace the label set with Kinetics-style action names (`playing soccer`, `juggling balls`, "
        "`sunset`) and watch the ranking change entirely; drop `a ball standing still` from the set and see which name "
        "inherits the clips; enable `USE_BYOD` with an animated GIF of a real action you know, then pass its class as "
        "`correct_labels` to `evaluation_report` to see the verdict switch to `sample-sanity`.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/xclip-video-classification-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/xclip-video-classification-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/xclip-video-classification-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/microsoft/VideoX/tree/master/X-CLIP\n"
        "- Expanding Language-Image Pretrained Models for General Video Recognition (Ni et al., 2022): https://arxiv.org/abs/2208.02816\n"
        "- The Kinetics Human Action Video Dataset (Kay et al., 2017): https://arxiv.org/abs/1705.06950"
    ),
}
