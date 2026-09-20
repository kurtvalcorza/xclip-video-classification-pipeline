"""Labelled clip datasets for the adaptation contract: the digest-pinned HMDB51 sample, the record contract and its
structural validation, source-disjoint splitting, and the BYOD loader.

A record is ``{id, frames, label}`` where ``frames`` is a list of exactly ``NUM_FRAMES`` PIL frames of one size
(uniformly sampled from a clip; sides within the pipeline's ceilings) and ``label`` the class name the clip is asked
to be ranked under — one of the caller's closed label set, normalised like ``format_labels`` normalises the
candidate names. An optional ``source`` names the video the clip was cut from; splits are made disjoint on it.

The default sample is drawn from HMDB51 (Kuehne et al. 2011; Serre Lab, Brown University; **CC BY 4.0**) as
repacked into parquet on the Hugging Face Hub (``mteb/HMDB51``) at an immutable revision: the first row group of
the test shard is read with one HTTPS range request (about 113 MB; the shard's declared size is checked first and
the row group's content is refused unless its SHA-256 matches the pin). It holds every test clip of the first ten
action classes (30 each, 300 clips) plus 18 clips of the eleventh, which are dropped so every class is complete.
Clips are MPEG-4 AVI files of 76..647 frames at 30 fps, mostly 320×240, decoded with PyAV into ``NUM_FRAMES``
uniformly spaced frames. Several clips are cut from one source video (its name is the prefix of the file name), so
the sample is split by source, never by clip.
"""
# ruff: noqa: E501  -- record and pin literals are kept on single lines

from __future__ import annotations

import csv
import hashlib
import io
import random
import re
import urllib.request
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .pipeline import MODEL_ID, NUM_FRAMES, format_labels, sample_frames, validate_clip

CORPUS_NAME = "HMDB51 (test shard), first parquet row group: the ten first action classes"
CORPUS_REPO = "mteb/HMDB51"
CORPUS_REVISION = "50bb2abb741074ef0868e59ba51af710c825cd1f"  # refs/convert/parquet commit on the Hub
CORPUS_FILE = "default/test/0000.parquet"
CORPUS_BYTES = 480_591_821
CORPUS_ROWS = 1_530
CORPUS_ROW_GROUPS = 1  # of 5; 318 clips
CORPUS_LICENSE = "CC BY 4.0 (HMDB51, Serre Lab, Brown University; Kuehne, Jhuang, Garrote, Poggio, Serre, ICCV 2011; parquet repack mteb/HMDB51)"
CORPUS_URL = f"https://huggingface.co/datasets/{CORPUS_REPO}/resolve/{CORPUS_REVISION}/{CORPUS_FILE}"
# SHA-256 over the concatenated AVI bytes + UTF-8 label index of each row, in row order, and that byte total.
ROW_GROUP_PINS: dict[int, tuple[str, int]] = {
    0: ("c1eb3e9cd30abc8d3deadb1e82b982a57d2baf2035bfdf99d1343dcdb1c09bc1", 112_858_960),
}
DEFAULT_CACHE_DIR = Path("weights") / "hmdb51"

# The HMDB51 class vocabulary (parquet `label` feature names) and the ten classes the sample keeps, with the
# natural-language label the CLIP text tower is asked to rank (the upstream zero-shot recipe uses plain phrases).
HMDB51_CLASSES = ("brush_hair", "cartwheel", "catch", "chew", "clap", "climb", "climb_stairs", "dive", "draw_sword", "dribble", "drink", "eat", "fall_floor", "fencing", "flic_flac", "golf", "handstand", "hit", "hug", "jump", "kick", "kick_ball", "kiss", "laugh", "pick", "pour", "pullup", "punch", "push", "pushup", "ride_bike", "ride_horse", "run", "shake_hands", "shoot_ball", "shoot_bow", "shoot_gun", "sit", "situp", "smile", "smoke", "somersault", "stand", "swing_baseball", "sword", "sword_exercise", "talk", "throw", "turn", "walk", "wave")
SAMPLE_CLASS_TEXT: dict[str, str] = {
    "brush_hair": "brushing hair",
    "cartwheel": "doing a cartwheel",
    "catch": "catching a ball",
    "chew": "chewing",
    "clap": "clapping hands",
    "climb": "climbing",
    "climb_stairs": "climbing stairs",
    "dive": "diving into water",
    "draw_sword": "drawing a sword",
    "dribble": "dribbling a basketball",
}
SAMPLE_LABELS: tuple[str, ...] = tuple(SAMPLE_CLASS_TEXT.values())
SAMPLE_CLIPS_PER_CLASS = 30

SAMPLE_SEED = 42
SAMPLE_SPLIT = {"test": 6, "validation": 4}  # clips per class the source-grouped split reserves at least; the rest train
SAMPLE_DIGEST = "ddf8c1730d64ff80b5ec691522f09461220c6817dee70b55197cfd12b9191f4e"  # dataset_digest over the three default splits together; tests pin it
MIN_RECORDS = 16
MAX_RECORDS = 5_000
MIN_PER_CLASS = 2
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_CLIP_SUFFIXES = (".avi", ".mp4", ".mov", ".mkv", ".webm", ".gif", ".webp")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _HttpRangeFile(io.RawIOBase):
    """A seekable read-only view of one HTTPS object served with `Range` requests (what `pyarrow` needs to read a
    parquet footer and one row group without downloading the file)."""

    def __init__(self, url: str, size: int) -> None:
        self.url, self.size, self.pos = url, size, 0
        self.fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}", "User-Agent": "xclip-video-classification-pipeline"})
        with urllib.request.urlopen(request, timeout=600) as response:  # noqa: S310 (pinned https URL)
            if response.status != 206:
                raise ValueError(f"{self.url}: server ignored the Range request (HTTP {response.status})")
            data = response.read()
        self.fetched += len(data)
        self.pos += len(data)
        return data

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def _declared_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "xclip-video-classification-pipeline"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 (pinned https URL)
        length = response.headers.get("Content-Length")
    if length is None:
        raise ValueError(f"{url}: no Content-Length in the HEAD response")
    return int(length)


def _group_digest(rows: Sequence[Mapping[str, Any]]) -> tuple[str, int]:
    digest, total = hashlib.sha256(), 0
    for row in rows:
        data = row["video"]["bytes"]
        label = str(row["label"]).encode("utf-8")
        digest.update(data)
        digest.update(label)
        total += len(data) + len(label)
    return digest.hexdigest(), total


def fetch_corpus(
    *, cache_dir: str | Path | None = None, groups: Sequence[int] | None = None, opener: Any = None
) -> dict[int, list[dict[str, Any]]]:
    """Return the pinned row groups as lists of `{video, path, label}` (AVI bytes, file name, class index), from the
    cache (one parquet file per row group) or the Hub (footer + the row group, over range requests). Every row group's
    content is refused unless its SHA-256 and byte total match `ROW_GROUP_PINS`; a fresh fetch also checks the shard's
    declared size and row count."""
    import pyarrow.parquet as pq

    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    wanted = list(groups) if groups is not None else sorted(ROW_GROUP_PINS)
    out: dict[int, list[dict[str, Any]]] = {}
    reader = None
    for group in wanted:
        if group not in ROW_GROUP_PINS:
            raise ValueError(f"row group {group} has no pin; pinned groups are {sorted(ROW_GROUP_PINS)}")
        local = cache / f"test-rg{group}.parquet"
        rows: list[dict[str, Any]] | None = None
        if local.is_file():
            rows = pq.read_table(local).to_pylist()
            if _group_digest(rows) != ROW_GROUP_PINS[group]:
                rows = None  # stale or corrupt cache: refetch
        if rows is None:
            if reader is None:
                if opener is not None:
                    reader = pq.ParquetFile(opener(CORPUS_URL))
                else:
                    declared = _declared_size(CORPUS_URL)
                    if declared != CORPUS_BYTES:
                        raise ValueError(f"{CORPUS_FILE}: declared size {declared} != pinned {CORPUS_BYTES}")
                    reader = pq.ParquetFile(io.BufferedReader(_HttpRangeFile(CORPUS_URL, CORPUS_BYTES), buffer_size=1 << 20))
                if reader.metadata.num_rows != CORPUS_ROWS:
                    raise ValueError(f"{CORPUS_FILE}: {reader.metadata.num_rows} rows, pinned {CORPUS_ROWS}")
            table = reader.read_row_group(group, columns=["video", "label"])
            rows = table.to_pylist()
            digest, total = _group_digest(rows)
            if (digest, total) != ROW_GROUP_PINS[group]:
                raise ValueError(f"{CORPUS_FILE} row group {group}: sha256 {digest} / {total} bytes != pinned {ROW_GROUP_PINS[group]}")
            pq.write_table(table, local)
        out[group] = [{"video": r["video"]["bytes"], "path": str(r["video"]["path"] or ""), "label": int(r["label"])} for r in rows]
    return out


def decode_clip(data: bytes, n: int = NUM_FRAMES) -> list[Image.Image]:
    """Decode a video file held in memory with PyAV and return `n` uniformly spaced RGB frames (first and last
    included). Every frame is decoded (the container's declared frame count is not trusted); only the chosen ones
    are kept."""
    import av

    frames: list[Image.Image] = []
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            frames.append(frame.to_image())
    if len(frames) < n:
        raise ValueError(f"clip holds {len(frames)} frames, fewer than the {n} a clip needs")
    return sample_frames(frames, n)


def source_of(path: str, class_name: str) -> str:
    """The source an HMDB51 clip was cut from, keyed per action: `<source video>/<class>` from the file name
    `<source>_<class>_<tags>_<n>.avi`. Clips cut from one source video for one action share its scene and actor, so
    a split is made disjoint on this key (one film can still contribute different actions to different splits)."""
    match = re.match(rf"^(.*)_{re.escape(class_name)}_(.*)_(\d+)\.avi$", path)
    return f"{match.group(1) if match else Path(path).stem}/{class_name}"


def read_corpus(groups: Mapping[int, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Decode the verified row groups into records (one per clip of the ten sample classes; the partial eleventh
    class is dropped)."""
    out = []
    for group in sorted(groups):
        for index, row in enumerate(groups[group]):
            class_name = HMDB51_CLASSES[row["label"]]
            if class_name not in SAMPLE_CLASS_TEXT:
                continue
            out.append(
                {
                    "id": f"hmdb51-test-{group}-{index}",
                    "frames": decode_clip(row["video"]),
                    "label": SAMPLE_CLASS_TEXT[class_name],
                    "class": class_name,
                    "source": source_of(row["path"], class_name),
                    "source_row_group": group,
                }
            )
    return out


def build_sample_dataset(
    records: Sequence[Mapping[str, Any]], *, seed: int = SAMPLE_SEED, sizes: Mapping[str, int] | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Seeded source-grouped draw, per class: whole source videos go to the test split until it holds at least
    `sizes['test']` clips of that class, then to validation until `sizes['validation']`, and the rest train. Sources
    are taken smallest first (seeded order among equals), so the held-out splits stay near their targets and the
    largest sources train."""
    sizes = dict(sizes or SAMPLE_SPLIT)
    rng = random.Random(seed)
    by_class: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_class[record["label"]][record.get("source") or record["id"]].append(dict(record))
    out: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for label in sorted(by_class):
        sources = sorted(by_class[label])
        rng.shuffle(sources)
        sources.sort(key=lambda source: len(by_class[label][source]))  # stable: seeded order among equal sizes
        counts = {"test": 0, "validation": 0, "train": 0}
        for source in sources:
            clips = by_class[label][source]
            if counts["test"] < sizes["test"]:
                out["test"].extend(clips)
                counts["test"] += len(clips)
            elif counts["validation"] < sizes["validation"]:
                out["validation"].extend(clips)
                counts["validation"] += len(clips)
            else:
                out["train"].extend(clips)
                counts["train"] += len(clips)
        if counts["train"] < MIN_PER_CLASS or counts["test"] < sizes["test"]:
            raise ValueError(f"class {label!r} has too few sources to reserve {sizes} clips and keep a training source")
    return out


def fetch_sample_dataset(*, cache_dir: str | Path | None = None, seed: int = SAMPLE_SEED) -> dict[str, list[dict[str, Any]]]:
    return build_sample_dataset(read_corpus(fetch_corpus(cache_dir=cache_dir)), seed=seed)


# ---------------------------------------------------------------------------------------------------------
# Record contract
# ---------------------------------------------------------------------------------------------------------


def _check_record(record: Any, index: int, labels: Sequence[str] | None) -> dict[str, Any]:
    where = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{where} must be a mapping with id/frames/label")
    for key in ("id", "frames", "label"):
        if key not in record:
            raise ValueError(f"{where} is missing {key!r}")
    rid = record["id"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{where}: id must match {_ID_RE.pattern}")
    try:
        frames = validate_clip(record["frames"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: {exc}") from exc
    if not isinstance(record["label"], str) or not record["label"].strip():
        raise ValueError(f"{where}: label must be a non-empty str")
    label = format_labels([record["label"], "__other__"])[0]
    if labels is not None and label not in labels:
        raise ValueError(f"{where}: label {label!r} is not in the label set {list(labels)}")
    item = {"id": rid, "frames": frames, "label": label}
    for key in ("class", "source", "source_row_group"):
        if key in record:
            item[key] = record[key]
    return item


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    labels: Sequence[str] | None = None,
    *,
    min_records: int = MIN_RECORDS,
    max_records: int = MAX_RECORDS,
    min_per_class: int = MIN_PER_CLASS,
) -> dict[str, Any]:
    """Structural validation of a labelled-clip dataset; raises ValueError before any model import. With `labels`
    every record's label must belong to that closed set; without, the set is the labels the records carry."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ValueError("records must be a list of {id, frames, label} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    label_set = list(format_labels(labels)) if labels is not None else None
    checked, ids = [], set()
    for index, record in enumerate(records):
        item = _check_record(record, index, label_set)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        checked.append(item)
    counts: dict[str, int] = defaultdict(int)
    for item in checked:
        counts[item["label"]] += 1
    if label_set is None:
        label_set = sorted(counts)
        if len(label_set) < 2:
            raise ValueError("a labelled-clip dataset needs at least two distinct labels")
    thin = [label for label, n in counts.items() if n < min_per_class]
    if thin:
        raise ValueError(f"every label needs at least {min_per_class} clips; too few for {sorted(thin)}")
    widths = [r["frames"][0].width for r in checked]
    heights = [r["frames"][0].height for r in checked]
    return {
        "records": checked,
        "n_records": len(checked),
        "labels": list(label_set),
        "per_label": {label: counts.get(label, 0) for label in label_set},
        "n_sources": len({r.get("source") or r["id"] for r in checked}),
        "frames_per_clip": NUM_FRAMES,
        "frame_width": {"min": min(widths), "max": max(widths)},
        "frame_height": {"min": min(heights), "max": max(heights)},
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def clip_digest(frames: Sequence[Image.Image]) -> str:
    """SHA-256 of the decoded RGB pixels of every frame (size-prefixed) — the identity a split is made disjoint on."""
    digest = hashlib.sha256()
    for frame in frames:
        rgb = frame.convert("RGB")
        digest.update(f"{rgb.width}x{rgb.height}:".encode())
        digest.update(rgb.tobytes())
    return digest.hexdigest()


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """Order-independent SHA-256 over (id, clip digest, label)."""
    parts = sorted(f"{r['id']}:{clip_digest(r['frames'])}:{r['label']}" for r in records)
    return _sha256_bytes("\n".join(parts).encode("utf-8"))


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no clip (by decoded-pixel digest) and no source video appears in two splits (leakage check)."""
    seen_clip: dict[str, str] = {}
    seen_source: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = clip_digest(record["frames"])
            if key in seen_clip and seen_clip[key] != name:
                raise ValueError(f"clip {record['id']!r} appears in both {seen_clip[key]} and {name}")
            seen_clip[key] = name
            source = record.get("source")
            if source:
                if source in seen_source and seen_source[source] != name:
                    raise ValueError(f"source video {source!r} has clips in both {seen_source[source]} and {name}")
                seen_source[source] = name
    return {name: len(records) for name, records in splits.items()}


def split_dataset(
    records: Sequence[Mapping[str, Any]], *, val_fraction: float = 0.15, test_fraction: float = 0.2, seed: int = 0
) -> dict[str, list[dict[str, Any]]]:
    """Seeded, label-stratified shuffle of a BYOD dataset into train/validation/test after de-duplicating clips;
    records that carry a `source` keep every clip of one source in one split."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records)["records"]
    seen: set[str] = set()
    unique = []
    for record in checked:
        key = clip_digest(record["frames"])
        if key not in seen:
            seen.add(key)
            unique.append(record)
    rng = random.Random(seed)
    out: dict[str, list[dict[str, Any]]] = {"test": [], "validation": [], "train": []}
    by_label: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in unique:
        by_label[record["label"]][record.get("source") or record["id"]].append(record)
    for label in sorted(by_label):
        groups = sorted(by_label[label])
        rng.shuffle(groups)
        n = sum(len(by_label[label][g]) for g in groups)
        n_test, n_val = max(1, round(n * test_fraction)), round(n * val_fraction)
        counts = {"test": 0, "validation": 0}
        for group in groups:
            clips = by_label[label][group]
            if counts["test"] < n_test:
                out["test"].extend(clips)
                counts["test"] += len(clips)
            elif counts["validation"] < n_val:
                out["validation"].extend(clips)
                counts["validation"] += len(clips)
            else:
                out["train"].extend(clips)
    if not out["train"]:
        raise ValueError(f"{len(unique)} distinct clips are too few to split into train/validation/test")
    return out


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Records from a directory or zip holding video clips (any container PyAV or Pillow decodes: AVI, MP4, MOV,
    MKV, WebM, GIF, WebP) and a `labels.csv` with the columns `file` and `label` (and optionally `id` and `source`);
    every clip file must have a label row and every row a clip. Each clip is decoded into NUM_FRAMES uniform frames."""
    source = Path(path)
    members: dict[str, bytes] = {}
    if source.is_dir():
        for file in sorted(source.rglob("*")):
            if file.is_file():
                members[file.name] = file.read_bytes()
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    members[Path(info.filename).name] = archive.read(info)  # flattened; no extractall
    else:
        raise ValueError(f"{source} is neither a directory nor a zip file")
    if "labels.csv" not in members:
        raise ValueError("BYOD data must include labels.csv with the columns file and label")
    rows = list(csv.DictReader(io.StringIO(members["labels.csv"].decode("utf-8-sig"))))
    if not rows or "file" not in rows[0] or "label" not in rows[0]:
        raise ValueError("labels.csv must have the columns file and label")
    out = []
    for row in rows:
        name = Path(str(row.get("file", "")).strip()).name
        if name not in members:
            raise ValueError(f"labels.csv names a missing clip: {name}")
        try:
            if name.lower().endswith((".gif", ".webp")):
                from .pipeline import frames_from_animation

                frames = sample_frames(frames_from_animation(Image.open(io.BytesIO(members[name]))))
            else:
                frames = decode_clip(members[name])
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"BYOD file is not a decodable clip of at least {NUM_FRAMES} frames: {name}") from exc
        rid = str(row.get("id", "") or "").strip()
        item = {"id": rid or re.sub(r"[^A-Za-z0-9_.:-]", "_", Path(name).stem)[:64], "frames": frames, "label": str(row.get("label", ""))}
        if str(row.get("source", "") or "").strip():
            item["source"] = str(row["source"]).strip()
        out.append(item)
    listed = {Path(str(r.get("file", "")).strip()).name for r in rows}
    unlisted = [n for n in members if n != "labels.csv" and n.lower().endswith(_CLIP_SUFFIXES) and n not in listed]
    if unlisted:
        raise ValueError(f"{len(unlisted)} clip file(s) have no labels.csv row, e.g. {unlisted[0]}")
    return out


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """A summary table (id, frame size, label, class, source, provenance) in the BYOD `labels.csv` column layout plus
    extras (`file` names the id; the frames themselves are not written)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "file", "width", "height", "frames", "label", "class", "source", "source_row_group"])
        for r in records:
            writer.writerow([r["id"], f"{r['id']}.avi", r["frames"][0].width, r["frames"][0].height, len(r["frames"]), r["label"], r.get("class", ""), r.get("source", ""), r.get("source_row_group", "")])
    return out
