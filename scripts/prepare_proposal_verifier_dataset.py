"""Build a leakage-safe 9-material + background proposal-crop dataset.

Unlike the original verifier dataset, every crop in this dataset comes from an
actual YOLO prediction.  Images with more than one ground-truth object are
excluded before inference.  A proposal is assigned to the single GT material
when its IoU is at least ``positive_iou``; it is assigned to ``background``
when the image has no GT or its IoU is at most ``negative_iou``.  Proposals in
between are deliberately skipped rather than pseudo-labelled.

The train/validation split from the supplied YOLO data YAML is preserved.  The
output is selected deterministically before crops are written, so class caps do
not depend on filesystem or inference order.  Ultralytics is imported only
after all output and input preflight checks pass, which keeps the pure helpers
and safety checks unit-testable without loading YOLO.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import heapq
import json
import math
import os
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import cv2
import numpy as np


CLASS_NAMES = (
    "can", "pet", "paper", "plastic", "styrofoam",
    "vinyl", "glass", "battery", "fluorescent",
)
BACKGROUND_CLASS_ID = len(CLASS_NAMES)
OUTPUT_CLASS_NAMES = CLASS_NAMES + ("background",)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
GIB = 1024 ** 3


@dataclass(frozen=True)
class GroundTruth:
    class_id: int
    # Normalized YOLO centre-x, centre-y, width, height.
    xywhn: tuple[float, float, float, float]

    def xyxy(self, width: int, height: int) -> tuple[float, float, float, float]:
        cx, cy, box_w, box_h = self.xywhn
        return (
            (cx - box_w / 2) * width,
            (cy - box_h / 2) * height,
            (cx + box_w / 2) * width,
            (cy + box_h / 2) * height,
        )


@dataclass(frozen=True)
class SourceRecord:
    path: Path
    split: str
    source_id: str
    ground_truth: GroundTruth | None


@dataclass(frozen=True)
class Proposal:
    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float]
    class_name: str = ""


@dataclass(frozen=True)
class PredictedFrame:
    source: SourceRecord
    width: int
    height: int
    proposals: tuple[Proposal, ...]


@dataclass(frozen=True)
class Candidate:
    source: SourceRecord
    proposal_index: int
    proposal: Proposal
    material: int
    category: str
    matched_iou: float
    assignment: str
    gt_bbox: tuple[float, float, float, float] | None

    @property
    def identity(self) -> str:
        bbox = ",".join(f"{value:.6f}" for value in self.proposal.bbox)
        return (
            f"{self.source.split}|{self.source.source_id}|"
            f"{self.proposal.class_id}|{self.proposal.confidence:.9f}|{bbox}"
        )


def bbox_iou(
    first: Sequence[float], second: Sequence[float]
) -> float:
    """Return IoU for two ``x1, y1, x2, y2`` boxes."""
    if len(first) != 4 or len(second) != 4:
        raise ValueError("bbox must contain four coordinates")
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    if not all(math.isfinite(value) for value in (*first, *second)):
        raise ValueError("bbox coordinates must be finite")
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return 0.0
    intersection_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_w * intersection_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def assign_proposal(
    proposal_bbox: Sequence[float],
    *,
    gt_bbox: Sequence[float] | None,
    gt_class_id: int | None,
    positive_iou: float,
    negative_iou: float,
) -> tuple[int | None, float, str]:
    """Assign a proposal at inclusive IoU boundaries.

    Returns ``(material_id, iou, reason)``.  ``material_id`` is ``None`` for
    the ambiguous band, which must not be used for training.
    """
    if not 0 <= negative_iou < positive_iou <= 1:
        raise ValueError("IoU thresholds must satisfy 0 <= negative < positive <= 1")
    if gt_bbox is None:
        return BACKGROUND_CLASS_ID, 0.0, "no_ground_truth"
    if gt_class_id is None or not 0 <= gt_class_id < len(CLASS_NAMES):
        raise ValueError("valid gt_class_id is required when gt_bbox is present")
    overlap = bbox_iou(proposal_bbox, gt_bbox)
    if overlap >= positive_iou:
        return gt_class_id, overlap, "positive_iou"
    if overlap <= negative_iou:
        return BACKGROUND_CLASS_ID, overlap, "low_iou"
    return None, overlap, "ambiguous_iou"


def parse_yolo_label_text(text: str) -> tuple[GroundTruth | None, str | None]:
    """Parse a zero-or-one-object YOLO label.

    Empty labels are valid negative frames.  A multi-object or malformed label
    returns a rejection reason and must be excluded before model inference.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None, None
    if len(lines) != 1:
        return None, "not_single_object"
    parts = lines[0].split()
    if len(parts) != 5:
        return None, "invalid_column_count"
    try:
        raw_class, cx, cy, width, height = (float(value) for value in parts)
    except ValueError:
        return None, "non_numeric_label"
    class_id = int(raw_class)
    if raw_class != class_id or not 0 <= class_id < len(CLASS_NAMES):
        return None, "invalid_class_id"
    values = (cx, cy, width, height)
    if not all(math.isfinite(value) for value in values):
        return None, "non_finite_bbox"
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
        return None, "invalid_bbox"
    # Exported YOLO labels are commonly rounded at the image boundary.  Match
    # the detector dataset audit tolerance instead of discarding valid edge
    # objects for sub-pixel excursions.
    if cx - width / 2 < -0.01 or cx + width / 2 > 1.01:
        return None, "bbox_outside_image"
    if cy - height / 2 < -0.01 or cy + height / 2 > 1.01:
        return None, "bbox_outside_image"
    return GroundTruth(class_id, values), None


def _stable_score(candidate: Candidate, seed: int) -> int:
    digest = hashlib.blake2b(
        f"{seed}|{candidate.identity}".encode("utf-8", errors="surrogateescape"),
        digest_size=16,
    ).digest()
    return int.from_bytes(digest, "big")


def _cap_for(
    candidate: Candidate,
    *,
    max_per_class: int,
    val_max_per_class: int,
    max_background: int,
    val_max_background: int,
) -> int:
    validation = candidate.source.split == "validation"
    if candidate.material == BACKGROUND_CLASS_ID:
        return val_max_background if validation else max_background
    return val_max_per_class if validation else max_per_class


def select_deterministic_candidates(
    candidates: Iterable[Candidate],
    *,
    max_per_class: int,
    val_max_per_class: int,
    max_background: int,
    val_max_background: int,
    seed: int,
) -> list[Candidate]:
    """Apply bounded deterministic sampling independently per split/class."""
    limits = (max_per_class, val_max_per_class, max_background, val_max_background)
    if any(value < 1 for value in limits):
        raise ValueError("all proposal caps must be positive")
    heaps: dict[tuple[str, int], list[tuple[int, str, Candidate]]] = {}
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.source.split not in {"training", "validation"}:
            raise ValueError(f"unsupported split: {candidate.source.split}")
        if candidate.identity in seen:
            continue
        seen.add(candidate.identity)
        key = (candidate.source.split, candidate.material)
        heap = heaps.setdefault(key, [])
        limit = _cap_for(
            candidate,
            max_per_class=max_per_class,
            val_max_per_class=val_max_per_class,
            max_background=max_background,
            val_max_background=val_max_background,
        )
        score = _stable_score(candidate, seed)
        # The min-heap root is the largest selected score because scores are negated.
        item = (-score, candidate.identity, candidate)
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif score < -heap[0][0]:
            heapq.heapreplace(heap, item)

    selected = [item[2] for heap in heaps.values() for item in heap]
    selected.sort(
        key=lambda item: (
            item.source.split,
            item.material,
            _stable_score(item, seed),
            item.identity,
        )
    )
    return selected


def ensure_empty_output(output_dir: Path) -> None:
    """Refuse to overwrite any existing output artifact."""
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"output path already exists and is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")


def _existing_ancestor(path: Path) -> Path:
    current = path.resolve(strict=False)
    while not current.exists():
        if current.parent == current:
            raise FileNotFoundError(f"no existing ancestor for output: {path}")
        current = current.parent
    return current


def check_storage_limits(
    path: Path,
    *,
    written_bytes: int,
    next_bytes: int = 0,
    min_free_gb: float,
    max_output_gb: float,
) -> None:
    free_bytes = shutil.disk_usage(_existing_ancestor(path)).free
    if min_free_gb > 0 and free_bytes - next_bytes < min_free_gb * GIB:
        raise RuntimeError(
            f"free space guard: {(free_bytes - next_bytes) / GIB:.1f}GB "
            f"< {min_free_gb:.1f}GB"
        )
    if max_output_gb > 0 and written_bytes + next_bytes > max_output_gb * GIB:
        raise RuntimeError(
            f"crop output guard: {(written_bytes + next_bytes) / GIB:.2f}GB "
            f"> {max_output_gb:.2f}GB"
        )


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    if image.size == 0 or size < 1:
        raise ValueError("letterbox requires a nonempty image and positive size")
    height, width = image.shape[:2]
    scale = size / max(height, width)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale <= 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return canvas


def _read_image(path: Path) -> np.ndarray | None:
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    image_indexes = [index for index, value in enumerate(parts) if value.lower() == "images"]
    if not image_indexes:
        raise ValueError(f"image path has no images directory: {image_path}")
    parts[image_indexes[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def _source_id(split: str, path: Path, dataset_dir: Path) -> str:
    try:
        relative = path.resolve(strict=False).relative_to(dataset_dir.resolve(strict=False))
        key = relative.as_posix()
    except ValueError:
        key = path.resolve(strict=False).as_posix()
    digest = hashlib.sha1(
        f"{split}|{key}".encode("utf-8", errors="surrogateescape")
    ).hexdigest()
    return digest[:24]


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to read the YOLO data file") from error
    with path.open(encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"YOLO data YAML must contain an object: {path}")
    names = value.get("names")
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=lambda item: int(item))]
    if not isinstance(names, list) or len(names) != len(CLASS_NAMES):
        raise ValueError("YOLO data YAML must define exactly nine material classes")
    normalized_names = tuple(str(name).strip().lower() for name in names)
    if normalized_names != CLASS_NAMES:
        raise ValueError(
            "YOLO data class order must be: " + ", ".join(CLASS_NAMES)
        )
    return value


def _remap_entry(
    value: str | Path,
    *,
    dataset_dir: Path,
    configured_root: Path | None,
) -> Path:
    raw = Path(value)
    if not raw.is_absolute():
        return dataset_dir / raw
    if configured_root is not None:
        try:
            return dataset_dir / raw.relative_to(configured_root)
        except ValueError:
            pass
    return raw


def _images_from_entry(
    entry: Path,
    *,
    dataset_dir: Path,
    configured_root: Path | None,
) -> list[Path]:
    if entry.is_dir():
        return sorted(
            path for path in entry.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    if entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES:
        return [entry]
    if entry.is_file() and entry.suffix.lower() == ".txt":
        images = []
        for line in entry.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            candidate = _remap_entry(
                line.strip(), dataset_dir=dataset_dir, configured_root=configured_root
            )
            if not candidate.is_absolute():
                candidate = entry.parent / candidate
            if not candidate.exists():
                alternate = entry.parent / line.strip().removeprefix("./")
                candidate = alternate if alternate.exists() else candidate
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                images.append(candidate)
        return images
    raise FileNotFoundError(f"YOLO split entry does not exist: {entry}")


def resolve_split_images(
    data_path: Path, dataset_dir: Path
) -> dict[str, list[Path]]:
    config = _load_yaml(data_path)
    raw_root = config.get("path")
    configured_root = None
    if raw_root:
        configured_root = Path(raw_root)
        if not configured_root.is_absolute():
            configured_root = data_path.parent / configured_root
        configured_root = configured_root.resolve(strict=False)
    split_images: dict[str, list[Path]] = {}
    for yaml_name, split in (("train", "training"), ("val", "validation")):
        raw_entries = config.get(yaml_name)
        if raw_entries is None:
            raise ValueError(f"YOLO data YAML is missing '{yaml_name}'")
        if not isinstance(raw_entries, list):
            raw_entries = [raw_entries]
        images = []
        for raw_entry in raw_entries:
            entry = _remap_entry(
                str(raw_entry),
                dataset_dir=dataset_dir,
                configured_root=configured_root,
            )
            images.extend(
                _images_from_entry(
                    entry,
                    dataset_dir=dataset_dir,
                    configured_root=configured_root,
                )
            )
        unique = {path.resolve(strict=False).as_posix(): path for path in images}
        if not unique:
            raise RuntimeError(f"no images found for YOLO split: {yaml_name}")
        split_images[split] = [unique[key] for key in sorted(unique)]
    return split_images


def collect_sources(
    split_images: dict[str, list[Path]], dataset_dir: Path
) -> tuple[list[SourceRecord], Counter]:
    records = []
    rejected = Counter()
    source_splits: dict[str, str] = {}
    for split in ("training", "validation"):
        for image_path in split_images.get(split, []):
            canonical = image_path.resolve(strict=False).as_posix()
            previous_split = source_splits.get(canonical)
            if previous_split is not None and previous_split != split:
                raise RuntimeError(f"source image crosses train/validation splits: {image_path}")
            source_splits[canonical] = split
            try:
                label_path = _label_path(image_path)
            except ValueError:
                rejected["unresolved_label_path"] += 1
                continue
            try:
                text = label_path.read_text(encoding="utf-8") if label_path.is_file() else ""
            except (OSError, UnicodeError):
                rejected["unreadable_label"] += 1
                continue
            ground_truth, reason = parse_yolo_label_text(text)
            if reason is not None:
                rejected[reason] += 1
                continue
            records.append(
                SourceRecord(
                    path=image_path,
                    split=split,
                    source_id=_source_id(split, image_path, dataset_dir),
                    ground_truth=ground_truth,
                )
            )
    return records, rejected


def iter_yolo_predictions(
    records: Sequence[SourceRecord],
    *,
    model_path: Path,
    device: str,
    batch: int,
    imgsz: int,
    conf: float,
) -> Iterator[PredictedFrame]:
    """Run YOLO lazily and expose only its actual proposal boxes."""
    from ultralytics import YOLO

    # Export backends such as NCNN do not implement list batching consistently
    # in Ultralytics (some versions index a one-item result with the source-list
    # index).  They are used for production-faithful evaluation, so keep those
    # backends sequential while retaining batched GPU inference for .pt models.
    exported_backend = model_path.is_dir() or model_path.suffix.lower() != ".pt"
    model = YOLO(str(model_path), task="detect")
    chunk_size = 1 if exported_backend else max(batch, min(1024, batch * 32))
    started = time.monotonic()
    processed = 0
    for start in range(0, len(records), chunk_size):
        chunk = records[start:start + chunk_size]
        source: str | list[str]
        source = str(chunk[0].path) if exported_backend else [
            str(record.path) for record in chunk
        ]
        results = model.predict(
            source=source,
            device=device,
            batch=1 if exported_backend else batch,
            imgsz=imgsz,
            conf=conf,
            stream=True,
            save=False,
            verbose=False,
        )
        yielded = 0
        for result in results:
            if yielded >= len(chunk):
                raise RuntimeError("YOLO returned more results than requested images")
            source = chunk[yielded]
            yielded += 1
            height, width = (int(value) for value in result.orig_shape)
            names = result.names
            proposals = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.detach().cpu().tolist()
                classes = result.boxes.cls.detach().cpu().tolist()
                confidences = result.boxes.conf.detach().cpu().tolist()
                for bbox, raw_class, confidence in zip(boxes, classes, confidences):
                    class_id = int(raw_class)
                    class_name = (
                        str(names.get(class_id, class_id))
                        if isinstance(names, dict)
                        else str(names[class_id])
                    )
                    proposals.append(
                        Proposal(
                            class_id=class_id,
                            confidence=float(confidence),
                            bbox=tuple(float(value) for value in bbox),
                            class_name=class_name,
                        )
                    )
            yield PredictedFrame(source, width, height, tuple(proposals))
            processed += 1
            if processed % 1000 == 0 or processed == len(records):
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"proposal inference={processed:,}/{len(records):,} "
                    f"rate={processed / elapsed:.1f} images/s",
                    flush=True,
                )
        if yielded != len(chunk):
            raise RuntimeError(
                f"YOLO returned {yielded} results for {len(chunk)} requested images"
            )


def candidates_from_frames(
    frames: Iterable[PredictedFrame],
    *,
    positive_iou: float,
    negative_iou: float,
    stats: Counter | None = None,
) -> Iterator[Candidate]:
    stats = stats if stats is not None else Counter()
    for frame in frames:
        gt = frame.source.ground_truth
        gt_bbox = gt.xyxy(frame.width, frame.height) if gt is not None else None
        for proposal_index, proposal in enumerate(frame.proposals):
            if (
                len(proposal.bbox) != 4
                or not all(math.isfinite(value) for value in proposal.bbox)
                or proposal.bbox[2] <= proposal.bbox[0]
                or proposal.bbox[3] <= proposal.bbox[1]
            ):
                stats["invalid_proposal_bbox"] += 1
                continue
            material, overlap, assignment = assign_proposal(
                proposal.bbox,
                gt_bbox=gt_bbox,
                gt_class_id=gt.class_id if gt is not None else None,
                positive_iou=positive_iou,
                negative_iou=negative_iou,
            )
            stats[assignment] += 1
            if material is None:
                continue
            yield Candidate(
                source=frame.source,
                proposal_index=proposal_index,
                proposal=proposal,
                material=material,
                category=OUTPUT_CLASS_NAMES[material],
                matched_iou=overlap,
                assignment=assignment,
                gt_bbox=gt_bbox,
            )


def _encode_path(path: Path) -> str:
    return base64.urlsafe_b64encode(os.fsencode(str(path))).decode("ascii")


def _crop_bounds(
    bbox: Sequence[float], width: int, height: int, padding: float
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    box_w, box_h = x2 - x1, y2 - y1
    left = max(0, math.floor(x1 - box_w * padding))
    top = max(0, math.floor(y1 - box_h * padding))
    right = min(width, math.ceil(x2 + box_w * padding))
    bottom = min(height, math.ceil(y2 + box_h * padding))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


MANIFEST_FIELDS = (
    "filepath", "split", "source_id", "material", "category",
    "dent", "label", "foreign_material", "source_object_count",
    "source_path_b64", "proposal_index", "assignment", "matched_iou",
    "gt_class_id", "gt_class_name", "gt_bbox_x1", "gt_bbox_y1",
    "gt_bbox_x2", "gt_bbox_y2", "predicted_class_id",
    "predicted_class_name", "predicted_confidence", "predicted_bbox_x1",
    "predicted_bbox_y1", "predicted_bbox_x2", "predicted_bbox_y2",
    "crop_x1", "crop_y1", "crop_x2", "crop_y2", "source_width",
    "source_height", "crop_bytes",
)


def write_selected_crops(
    selected: Sequence[Candidate],
    output_dir: Path,
    *,
    crop_size: int,
    padding: float,
    jpeg_quality: int,
    min_free_gb: float,
    max_output_gb: float,
) -> tuple[list[dict[str, object]], int, Counter]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rejected = Counter()
    written_bytes = 0
    by_source: dict[Path, list[Candidate]] = {}
    for candidate in selected:
        by_source.setdefault(candidate.source.path, []).append(candidate)

    for source_path in sorted(by_source, key=lambda path: path.as_posix()):
        image = _read_image(source_path)
        if image is None:
            rejected["unreadable_source_image"] += len(by_source[source_path])
            continue
        height, width = image.shape[:2]
        for candidate in sorted(by_source[source_path], key=lambda item: item.identity):
            bounds = _crop_bounds(candidate.proposal.bbox, width, height, padding)
            if bounds is None:
                rejected["empty_crop"] += 1
                continue
            left, top, right, bottom = bounds
            crop = letterbox(image[top:bottom, left:right], crop_size)
            ok, encoded = cv2.imencode(
                ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            if not ok:
                rejected["jpeg_encode_failed"] += 1
                continue
            crop_bytes = int(encoded.nbytes)
            check_storage_limits(
                output_dir,
                written_bytes=written_bytes,
                next_bytes=crop_bytes,
                min_free_gb=min_free_gb,
                max_output_gb=max_output_gb,
            )
            filename = hashlib.sha1(
                candidate.identity.encode("utf-8", errors="surrogateescape")
            ).hexdigest()[:24] + ".jpg"
            relative = Path(candidate.source.split) / candidate.category / filename
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            encoded.tofile(destination)
            written_bytes += crop_bytes

            gt = candidate.source.ground_truth
            gt_bbox = candidate.gt_bbox or ("", "", "", "")
            predicted_bbox = candidate.proposal.bbox
            rows.append(
                {
                    "filepath": relative.as_posix(),
                    "split": candidate.source.split,
                    "source_id": candidate.source.source_id,
                    "material": candidate.material,
                    "category": candidate.category,
                    "dent": -1,
                    "label": -1,
                    "foreign_material": -1,
                    "source_object_count": 1 if gt is not None else 0,
                    "source_path_b64": _encode_path(source_path),
                    "proposal_index": candidate.proposal_index,
                    "assignment": candidate.assignment,
                    "matched_iou": f"{candidate.matched_iou:.8f}",
                    "gt_class_id": gt.class_id if gt is not None else "",
                    "gt_class_name": CLASS_NAMES[gt.class_id] if gt is not None else "",
                    "gt_bbox_x1": gt_bbox[0],
                    "gt_bbox_y1": gt_bbox[1],
                    "gt_bbox_x2": gt_bbox[2],
                    "gt_bbox_y2": gt_bbox[3],
                    "predicted_class_id": candidate.proposal.class_id,
                    "predicted_class_name": candidate.proposal.class_name,
                    "predicted_confidence": f"{candidate.proposal.confidence:.8f}",
                    "predicted_bbox_x1": predicted_bbox[0],
                    "predicted_bbox_y1": predicted_bbox[1],
                    "predicted_bbox_x2": predicted_bbox[2],
                    "predicted_bbox_y2": predicted_bbox[3],
                    "crop_x1": left,
                    "crop_y1": top,
                    "crop_x2": right,
                    "crop_y2": bottom,
                    "source_width": width,
                    "source_height": height,
                    "crop_bytes": crop_bytes,
                }
            )
    rows.sort(key=lambda row: (str(row["split"]), int(row["material"]), str(row["filepath"])))
    return rows, written_bytes, rejected


PredictionProvider = Callable[[Sequence[SourceRecord]], Iterable[PredictedFrame]]


def build_proposal_verifier_dataset(
    *,
    model_path: Path,
    data_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    device: str,
    batch: int,
    imgsz: int,
    conf: float,
    positive_iou: float,
    negative_iou: float,
    crop_size: int,
    padding: float,
    max_per_class: int,
    val_max_per_class: int,
    max_background: int,
    val_max_background: int,
    seed: int,
    min_free_gb: float,
    max_output_gb: float,
    jpeg_quality: int,
    prediction_provider: PredictionProvider | None = None,
) -> dict:
    # This must remain the first preflight: a bad invocation can never load a
    # multi-GB model or touch an existing dataset before overwrite refusal.
    ensure_empty_output(output_dir)
    if not 0 <= negative_iou < positive_iou <= 1:
        raise ValueError("IoU thresholds must satisfy 0 <= negative < positive <= 1")
    if batch < 1 or imgsz < 1 or crop_size < 1:
        raise ValueError("batch, imgsz and crop-size must be positive")
    if not 0 <= conf <= 1 or padding < 0 or not 1 <= jpeg_quality <= 100:
        raise ValueError("conf, padding or jpeg-quality is outside its valid range")
    check_storage_limits(
        output_dir,
        written_bytes=0,
        min_free_gb=min_free_gb,
        max_output_gb=max_output_gb,
    )

    split_images = resolve_split_images(data_path, dataset_dir)
    sources, source_rejections = collect_sources(split_images, dataset_dir)
    if not sources:
        raise RuntimeError("no valid zero-or-one-object source images found")

    if prediction_provider is None:
        prediction_provider = lambda items: iter_yolo_predictions(
            items,
            model_path=model_path,
            device=device,
            batch=batch,
            imgsz=imgsz,
            conf=conf,
        )
    proposal_stats = Counter()
    candidates = candidates_from_frames(
        prediction_provider(sources),
        positive_iou=positive_iou,
        negative_iou=negative_iou,
        stats=proposal_stats,
    )
    selected = select_deterministic_candidates(
        candidates,
        max_per_class=max_per_class,
        val_max_per_class=val_max_per_class,
        max_background=max_background,
        val_max_background=val_max_background,
        seed=seed,
    )
    if not selected:
        raise RuntimeError("proposal assignment produced no usable crops")
    selected_splits = {candidate.source.split for candidate in selected}
    missing_splits = {"training", "validation"} - selected_splits
    if missing_splits:
        raise RuntimeError(
            "proposal assignment produced no usable crops for: "
            + ", ".join(sorted(missing_splits))
        )

    rows, written_bytes, write_rejections = write_selected_crops(
        selected,
        output_dir,
        crop_size=crop_size,
        padding=padding,
        jpeg_quality=jpeg_quality,
        min_free_gb=min_free_gb,
        max_output_gb=max_output_gb,
    )
    if not rows:
        raise RuntimeError("all selected proposal crops failed to write")

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter((str(row["split"]), str(row["category"])) for row in rows)
    summary = {
        "model": str(model_path),
        "data": str(data_path),
        "dataset_dir": str(dataset_dir),
        "manifest": str(manifest_path),
        "classes": list(OUTPUT_CLASS_NAMES),
        "source_policy": "zero or one GT object; multi-object sources excluded",
        "crop_source": "actual YOLO predicted bbox with runtime-style padding and letterbox",
        "eligible_sources": len(sources),
        "source_rejections": dict(source_rejections),
        "proposal_assignments": dict(proposal_stats),
        "selected_before_write": len(selected),
        "written_crops": len(rows),
        "written_bytes": written_bytes,
        "write_rejections": dict(write_rejections),
        "counts": {
            f"{split}/{category}": count
            for (split, category), count in sorted(counts.items())
        },
        "inference": {
            "device": device,
            "batch": batch,
            "imgsz": imgsz,
            "conf": conf,
        },
        "assignment": {
            "positive_iou_inclusive": positive_iou,
            "negative_iou_inclusive": negative_iou,
            "ambiguous_iou_skipped": True,
        },
        "selection": {
            "max_per_class": max_per_class,
            "val_max_per_class": val_max_per_class,
            "max_background": max_background,
            "val_max_background": val_max_background,
            "seed": seed,
        },
        "crop": {
            "size": crop_size,
            "padding": padding,
            "jpeg_quality": jpeg_quality,
        },
        "storage_guards": {
            "min_free_gb": min_free_gb,
            "max_output_gb": max_output_gb,
        },
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--positive-iou", type=float, default=0.50)
    parser.add_argument("--negative-iou", type=float, default=0.10)
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--max-per-class", type=int, default=10_000)
    parser.add_argument("--val-max-per-class", type=int, default=2_000)
    parser.add_argument("--max-background", type=int, default=10_000)
    parser.add_argument("--val-max-background", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--min-free-gb", type=float, default=100.0)
    parser.add_argument("--max-output-gb", type=float, default=30.0)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    args = parser.parse_args()
    if not 0 <= args.negative_iou < args.positive_iou <= 1:
        parser.error("IoU thresholds must satisfy 0 <= negative < positive <= 1")
    if min(
        args.batch,
        args.imgsz,
        args.crop_size,
        args.max_per_class,
        args.val_max_per_class,
        args.max_background,
        args.val_max_background,
    ) < 1:
        parser.error("batch, sizes and caps must be positive")
    if not 0 <= args.conf <= 1 or args.padding < 0:
        parser.error("conf must be in [0, 1] and padding must be non-negative")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("jpeg-quality must be in [1, 100]")
    if args.min_free_gb < 0 or args.max_output_gb < 0:
        parser.error("storage guards must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    build_proposal_verifier_dataset(
        model_path=args.model,
        data_path=args.data,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch=args.batch,
        imgsz=args.imgsz,
        conf=args.conf,
        positive_iou=args.positive_iou,
        negative_iou=args.negative_iou,
        crop_size=args.crop_size,
        padding=args.padding,
        max_per_class=args.max_per_class,
        val_max_per_class=args.val_max_per_class,
        max_background=args.max_background,
        val_max_background=args.val_max_background,
        seed=args.seed,
        min_free_gb=args.min_free_gb,
        max_output_gb=args.max_output_gb,
        jpeg_quality=args.jpeg_quality,
    )


if __name__ == "__main__":
    main()
