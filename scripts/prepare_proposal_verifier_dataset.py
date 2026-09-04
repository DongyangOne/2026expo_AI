"""Build a leakage-safe 9-material + background proposal-crop dataset.

Unlike the original verifier dataset, every crop in this dataset comes from an
actual YOLO prediction.  Images with more than one ground-truth object are
excluded before inference.  A proposal is assigned to the single GT material
when its IoU is at least ``positive_iou``; it is assigned to ``background``
when the image has no GT or its IoU is at most ``negative_iou``.  Proposals in
between are deliberately skipped rather than pseudo-labelled.

The train/validation split from the supplied YOLO data YAML is preserved.  The
optional ``runtime-top1`` mode keeps only the highest-confidence proposal above
the configured confidence floor, matching the production selection rule.  The
optional ``no-ground-truth-only`` policy prevents low-IoU proposals from a
labelled frame from becoming noisy background pseudo-labels.  The stricter
``strict-zero-intersection`` policy also accepts a labelled frame only when the
*final padded verifier crop* has no intersection with the GT box expanded by a
frozen safety margin.  This produces useful hard negatives without teaching a
partially visible recyclable object as background.

Output is selected deterministically before crops are written, so class caps do
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
import io
import json
import math
import os
import shutil
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import cv2
import numpy as np

try:
    from scripts.verifier_preprocessing_contract import (
        letterbox_bgr as _contract_letterbox_bgr,
        padded_clipped_bbox as _contract_padded_clipped_bbox,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from verifier_preprocessing_contract import (  # type: ignore[no-redef]
        letterbox_bgr as _contract_letterbox_bgr,
        padded_clipped_bbox as _contract_padded_clipped_bbox,
    )


CLASS_NAMES = (
    "can", "pet", "paper", "plastic", "styrofoam",
    "vinyl", "glass", "battery", "fluorescent",
)
BACKGROUND_CLASS_ID = len(CLASS_NAMES)
OUTPUT_CLASS_NAMES = CLASS_NAMES + ("background",)
PROPOSAL_SELECTION_MODES = ("all", "runtime-top1")
BACKGROUND_POLICIES = (
    "low-iou-or-no-ground-truth",
    "no-ground-truth-only",
    "strict-zero-intersection",
)
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
    # Only populated by the fully revalidated source-evidence adapter. This
    # reference box is a pseudo-annotation, NEVER the runtime detector crop.
    operational_evidence: dict | None = None
    # Original file/annotation lineage, distinct from the resized replay input.
    audited_aihub_metadata: dict[str, str] | None = None


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


def boxes_intersect(first: Sequence[float], second: Sequence[float]) -> bool:
    """Return whether two valid ``xyxy`` boxes overlap with positive area."""

    if len(first) != 4 or len(second) != 4:
        raise ValueError("bbox must contain four coordinates")
    values = tuple(float(value) for value in (*first, *second))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bbox coordinates must be finite")
    ax1, ay1, ax2, ay2, bx1, by1, bx2, by2 = values
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return False
    return min(ax2, bx2) > max(ax1, bx1) and min(ay2, by2) > max(ay1, by1)


def expanded_clipped_bbox(
    bbox: Sequence[float],
    *,
    width: int,
    height: int,
    margin: float,
) -> tuple[float, float, float, float]:
    """Expand a GT box by a per-side ratio and clip it to the source image."""

    if len(bbox) != 4:
        raise ValueError("bbox must contain four coordinates")
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError("bbox coordinates must be finite")
    if width <= 0 or height <= 0 or x2 <= x1 or y2 <= y1:
        raise ValueError("source dimensions and bbox must have positive area")
    if not math.isfinite(margin) or not 0 <= margin <= 1:
        raise ValueError("margin must be between zero and one")
    box_width, box_height = x2 - x1, y2 - y1
    return (
        max(0.0, x1 - box_width * margin),
        max(0.0, y1 - box_height * margin),
        min(float(width), x2 + box_width * margin),
        min(float(height), y2 + box_height * margin),
    )


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
    canvas, *_ = _contract_letterbox_bgr(image, size=size, fill=114)
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
    """Return a split/path-independent identity for the source image bytes."""
    del split, dataset_dir
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    split_images: dict[str, list[Path]], dataset_dir: Path,
    *, source_hashes: dict[Path, str] | None = None,
) -> tuple[list[SourceRecord], Counter]:
    records = []
    rejected = Counter()
    source_splits: dict[str, str] = {}
    canonical_sources: dict[str, SourceRecord] = {}
    conflicted_sources: set[str] = set()
    cross_split_sources: set[str] = set()
    for split in ("training", "validation"):
        for image_path in split_images.get(split, []):
            source_id = (source_hashes[image_path] if source_hashes is not None
                         else _source_id(split, image_path, dataset_dir))
            if source_id in cross_split_sources:
                rejected["duplicate_source_content_cross_split"] += 1
                continue
            previous_split = source_splits.get(source_id)
            if previous_split is not None and previous_split != split:
                # Quarantine both sides instead of allowing a copied image to
                # contaminate validation.  Cleaning a large source collection
                # is safe; emitting either copy across roles would not be.
                cross_split_sources.add(source_id)
                canonical_sources.pop(source_id, None)
                rejected["duplicate_source_content_cross_split"] += 2
                continue
            source_splits[source_id] = split
            try:
                label_path = _label_path(image_path)
            except ValueError:
                rejected["unresolved_label_path"] += 1
                continue
            if not label_path.is_file():
                # Only an existing, intentionally empty YOLO label file is
                # authoritative evidence that the source frame is empty.  A
                # missing sidecar is unknown annotation state, not a negative.
                rejected["missing_label_file"] += 1
                continue
            try:
                text = label_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                rejected["unreadable_label"] += 1
                continue
            ground_truth, reason = parse_yolo_label_text(text)
            if reason is not None:
                rejected[reason] += 1
                continue
            record = SourceRecord(
                path=image_path,
                split=split,
                source_id=source_id,
                ground_truth=ground_truth,
            )
            if source_id in conflicted_sources:
                rejected["duplicate_source_content_conflicting_ground_truth"] += 1
                continue
            previous = canonical_sources.get(source_id)
            if previous is not None:
                if previous.ground_truth != record.ground_truth:
                    # Neither label has enough authority to win automatically.
                    # Quarantine every byte-identical copy of this source and
                    # keep the rest of a large preparation run usable.
                    conflicted_sources.add(source_id)
                    canonical_sources.pop(source_id, None)
                    rejected["duplicate_source_content_conflicting_ground_truth"] += 2
                    continue
                # Mixed replay lists can contain renamed byte-identical copies.
                # Infer a content identity once so duplicates cannot waste GPU
                # time or silently increase one object's sampling weight.
                rejected["duplicate_source_content_same_split"] += 1
                continue
            canonical_sources[source_id] = record
            records.append(record)
    quarantined_sources = conflicted_sources | cross_split_sources
    if quarantined_sources:
        records = [
            record for record in records if record.source_id not in quarantined_sources
        ]
    return records, rejected


def _reject_operational_material_hold(bundle_dir: Path) -> None:
    # A sealed format/provenance bundle is not proof of correct material labels.
    # Keep semantic quarantine outside that immutable bundle and never interpret
    # malformed/empty markers as clearance, including dangling symlink markers.
    marker = bundle_dir.parent / "material_semantics_hold.json"
    if os.path.lexists(marker):
        raise ValueError("operational source evidence is on material semantics hold")


def _operational_bundle_reader(bundle_dir: Path) -> list[dict]:
    _reject_operational_material_hold(bundle_dir)
    try:
        from scripts.build_operational_source_evidence import validate_source_evidence_bundle
    except ModuleNotFoundError:
        from build_operational_source_evidence import validate_source_evidence_bundle
    records = validate_source_evidence_bundle(bundle_dir)
    _reject_operational_material_hold(bundle_dir)
    return records


def _operational_bundle_binding(bundle_dir: Path) -> dict[str, str]:
    _reject_operational_material_hold(bundle_dir)
    reject_symlinks, _ = _mixed_file_helpers()
    reject_symlinks(bundle_dir, description="operational source evidence bundle")
    for filename in ("source_evidence_receipt.json", "sources.jsonl", "source_evidence.sha256"):
        reject_symlinks(bundle_dir / filename, description="operational source evidence binding")
    return {
        "bundle_dir": bundle_dir.resolve(strict=True).as_posix(),
        **{
            field: _source_id("", bundle_dir / filename, bundle_dir)
            for field, filename in (
                ("receipt_sha256", "source_evidence_receipt.json"),
                ("index_sha256", "sources.jsonl"),
                ("marker_sha256", "source_evidence.sha256"),
            )
        },
    }


def _operational_bundle_input_roots(bundle_dir: Path, binding: dict[str, str]) -> list[Path]:
    """Protect complete sealed input directories, not only accepted-image leaves."""
    receipt_path = bundle_dir / "source_evidence_receipt.json"
    content = receipt_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != binding["receipt_sha256"]:
        raise RuntimeError("operational source evidence changed while resolving protected roots")
    receipt = json.loads(content)
    roots = {bundle_dir, Path(receipt["image_root"])}
    roots.update(
        Path(item["path"]).parent for name, item in receipt["inputs"].items()
        if name.startswith(("teacher_output_", "quality_"))
    )
    reject_symlinks, _ = _mixed_file_helpers()
    for root in roots:
        reject_symlinks(root, description="operational protected input root")
    return sorted({root.resolve(strict=True) for root in roots})


def append_operational_sources(
    sources: Sequence[SourceRecord], evidence: Sequence[dict],
    *, all_split_images: dict[str, list[Path]], dataset_dir: Path,
    source_hashes: dict[Path, str] | None = None,
) -> list[SourceRecord]:
    """Add train-only positive references, rejecting *all* base-source overlap.

    Include quarantined/missing-label base images in this check: failing base
    collection must not let an operational copy sneak into the other split.
    Near-duplicate/group/protected-cohort audits are still downstream gates.
    """
    base_hashes = {
        source_hashes[path] if source_hashes is not None else _source_id("", path, dataset_dir)
        for images in all_split_images.values() for path in images
    }
    result = list(sources)
    seen: set[str] = set()
    for row in evidence:
        sha = row["source_sha256"]
        if sha in base_hashes or sha in seen:
            raise ValueError("operational source duplicates a base or operational source")
        if row.get("role") != "train" or row.get("annotation_authority") != "vlm_teacher_pseudo_label_train_only":
            raise ValueError("operational source must be a train-only pseudo-annotation")
        if (type(row.get("source_object_count")) is not int or row["source_object_count"] != 1
                or type(row.get("material")) is not int or row["material"] not in range(9)):
            raise ValueError("operational source must contain one positive material")
        path = Path(row["source_filepath"])
        if not path.is_absolute() or _source_id("", path, dataset_dir) != sha:
            raise ValueError("operational source path/hash mismatch")
        width, height = row["source_width"], row["source_height"]
        x1, y1, x2, y2 = row["source_bbox_xyxy"]
        if (type(width) is not int or type(height) is not int or min(width, height) <= 0
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for value in (x1, y1, x2, y2))
                or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height)):
            raise ValueError("operational pseudo-annotation geometry is invalid")
        gt, error = parse_yolo_label_text(
            f"{row['material']} {(x1 + x2) / (2 * width)} {(y1 + y2) / (2 * height)} "
            f"{(x2 - x1) / width} {(y2 - y1) / height}"
        )
        if error or gt is None:
            raise ValueError("operational pseudo-annotation geometry is invalid")
        seen.add(sha)
        result.append(SourceRecord(path, "training", sha, gt, dict(row)))
    return result


def iter_yolo_predictions(
    records: Sequence[SourceRecord],
    *,
    model_path: Path,
    device: str,
    batch: int,
    imgsz: int,
    conf: float,
    nms_iou: float,
) -> Iterator[PredictedFrame]:
    """Run YOLO lazily and expose only its actual proposal boxes."""
    from ultralytics import YOLO

    # Export backends such as NCNN do not implement list batching consistently
    # in Ultralytics (some versions index a one-item result with the source-list
    # index). Keep those backends sequential. For .pt models, hand Ultralytics
    # exactly one batch at a time: very large source lists can make its predictor
    # retain oversized intermediate buffers even though ``batch`` is smaller,
    # which caused repeatable CUDA OOMs on the QNAP 16 GiB GPU.
    exported_backend = model_path.is_dir() or model_path.suffix.lower() != ".pt"
    model = YOLO(str(model_path), task="detect")
    chunk_size = 1 if exported_backend else batch
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
            iou=nms_iou,
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


def eager_initialize_cuda_context(device: str) -> object | None:
    """Open CUDA in this long-lived process before source hashing can fragment RAM.

    QNAP's NVIDIA 575 driver allocates a per-client fault buffer when the first
    real CUDA tensor is created.  Merely observing ``device_count`` (or probing
    CUDA in another short-lived process) does not reserve that buffer for the
    process that will later run YOLO.

    The production NAS preparation command uses logical GPU 0.  CPU and custom
    prediction-provider paths deliberately bypass this helper so unit tests and
    offline CPU use do not acquire PyTorch/CUDA as an incidental dependency.
    """

    normalized = str(device).strip().lower()
    if normalized not in {"0", "cuda", "cuda:0"}:
        return None

    import torch

    try:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "torch.cuda.is_available() is false "
                f"(device_count={torch.cuda.device_count()})"
            )
        guard = torch.ones(1, device="cuda:0") + 1
        torch.cuda.synchronize(0)
        if guard.item() != 2:
            raise RuntimeError("CUDA tensor smoke result was not 2")
        print(
            "eager CUDA context ready: " + torch.cuda.get_device_name(0),
            flush=True,
        )
        return guard
    except Exception as exc:
        raise RuntimeError(
            "failed to eagerly initialize CUDA in the YOLO preparation process"
        ) from exc


def candidates_from_frames(
    frames: Iterable[PredictedFrame],
    *,
    positive_iou: float,
    negative_iou: float,
    proposal_selection: str = "all",
    background_policy: str = "low-iou-or-no-ground-truth",
    min_confidence: float = 0.0,
    verifier_crop_padding: float = 0.08,
    background_gt_margin: float = 0.10,
    stats: Counter | None = None,
    policy_stats: Counter | None = None,
) -> Iterator[Candidate]:
    stats = stats if stats is not None else Counter()
    policy_stats = policy_stats if policy_stats is not None else Counter()
    if proposal_selection not in PROPOSAL_SELECTION_MODES:
        raise ValueError(f"unsupported proposal selection: {proposal_selection}")
    if background_policy not in BACKGROUND_POLICIES:
        raise ValueError(f"unsupported background policy: {background_policy}")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be in [0, 1]")
    if not 0 <= verifier_crop_padding <= 1:
        raise ValueError("verifier_crop_padding must be in [0, 1]")
    if not 0 <= background_gt_margin <= 1:
        raise ValueError("background_gt_margin must be in [0, 1]")

    for frame in frames:
        gt = frame.source.ground_truth
        gt_bbox = gt.xyxy(frame.width, frame.height) if gt is not None else None
        indexed_proposals = list(enumerate(frame.proposals))
        policy_stats["frames_seen"] += 1
        policy_stats["proposals_seen"] += len(indexed_proposals)
        if proposal_selection == "runtime-top1":
            eligible = []
            for proposal_index, proposal in indexed_proposals:
                if not math.isfinite(proposal.confidence):
                    policy_stats["invalid_proposal_confidence"] += 1
                    continue
                if proposal.confidence < min_confidence:
                    policy_stats["below_min_confidence"] += 1
                    continue
                eligible.append((proposal_index, proposal))
            if not eligible:
                policy_stats["frames_without_eligible_proposal"] += 1
                continue
            # Match the runtime rule: take exactly one highest-confidence bbox.
            # The original proposal order is the deterministic tie-breaker.
            indexed_proposals = [max(
                eligible,
                key=lambda item: (item[1].confidence, -item[0]),
            )]
            policy_stats["discarded_by_runtime_top1"] += len(eligible) - 1
        policy_stats["proposals_selected"] += len(indexed_proposals)

        for proposal_index, proposal in indexed_proposals:
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
            if frame.source.operational_evidence is not None and material == BACKGROUND_CLASS_ID:
                # VLM localization is not an exhaustive negative annotation.
                # A different proposal may contain an unlabelled real object.
                policy_stats["operational_background_not_authorized"] += 1
                continue
            if material == BACKGROUND_CLASS_ID and gt is not None:
                if background_policy == "no-ground-truth-only":
                    # A low-IoU proposal in a labelled frame can be another
                    # valid, unlabelled object. It is not safe automatic data.
                    policy_stats["background_rejected_gt_present"] += 1
                    continue
                if background_policy == "strict-zero-intersection":
                    crop_bounds = _crop_bounds(
                        proposal.bbox,
                        frame.width,
                        frame.height,
                        verifier_crop_padding,
                    )
                    if crop_bounds is None:
                        policy_stats["background_rejected_invalid_crop"] += 1
                        continue
                    exclusion = expanded_clipped_bbox(
                        gt_bbox,
                        width=frame.width,
                        height=frame.height,
                        margin=background_gt_margin,
                    )
                    if boxes_intersect(crop_bounds, exclusion):
                        policy_stats["background_rejected_gt_intersection"] += 1
                        continue
                    policy_stats["background_accepted_zero_intersection"] += 1
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
    try:
        return _contract_padded_clipped_bbox(
            bbox, width=width, height=height, padding=padding
        )
    except ValueError:
        return None


MANIFEST_FIELDS = (
    "filepath", "split", "source_id", "material", "category",
    "dent", "label", "foreign_material", "source_object_count",
    "crop_object_count",
    "source_path_b64", "proposal_index", "assignment", "matched_iou",
    "gt_class_id", "gt_class_name", "gt_bbox_x1", "gt_bbox_y1",
    "gt_bbox_x2", "gt_bbox_y2", "predicted_class_id",
    "predicted_class_name", "predicted_confidence", "predicted_bbox_x1",
    "predicted_bbox_y1", "predicted_bbox_x2", "predicted_bbox_y2",
    "crop_x1", "crop_y1", "crop_x2", "crop_y2", "source_width",
    "source_height", "crop_bytes",
)

AUDITED_AIHUB_FIELDS = (
    "original_source_id", "original_source_sha256", "original_annotation_sha256",
    "original_source_path_b64", "original_annotation_path_b64", "materializer_report_sha256",
)


def _audited_aihub_reader():
    try:
        from scripts.audited_aihub_snapshot import load_audited_aihub_snapshot
    except ModuleNotFoundError:
        from audited_aihub_snapshot import load_audited_aihub_snapshot
    return load_audited_aihub_snapshot


CANONICAL_FIELDS = (
    "sample_id", "role", "fold", "source_sha256", "image_sha256",
    "object_group", "capture_session", "origin", "source_filepath",
    "captured_at", "auditor_sha256", "teacher_output_sha256",
    "localizer_output_sha256", "annotation_authority", "source_evidence_ref",
    "source_foreign_material",
)


def _canonical_row_metadata(candidate: Candidate, destination: Path, aihub_origin: str) -> dict:
    source = candidate.source
    evidence = source.operational_evidence
    sha = _source_id("", source.path, source.path.parent)
    if sha != source.source_id:
        raise ValueError("source changed between collection and crop writing")
    image_sha = _source_id("", destination, destination.parent)
    role = {"training": "train", "validation": "model_validation"}[source.split]
    group = evidence["object_group"] if evidence else f"source_sha256:{sha}"
    payload = json.dumps(
        {"source_sha256": sha, "image_sha256": image_sha, "object_group": group},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return {
        "filepath": destination.resolve(strict=True).as_posix(),
        "source_filepath": source.path.resolve(strict=True).as_posix(),
        "sample_id": "proposal_" + hashlib.sha256(payload).hexdigest()[:24],
        "role": role, "fold": role, "source_sha256": sha, "image_sha256": image_sha,
        "object_group": group,
        "capture_session": evidence["capture_session"] if evidence else group,
        "origin": evidence["origin"] if evidence else aihub_origin,
        "annotation_authority": (
            "vlm_teacher_pseudo_label_train_only" if evidence else
            "aihub_annotation_geometry_development_only"
        ),
        **{field: evidence[field] if evidence else "" for field in (
            "captured_at", "auditor_sha256", "teacher_output_sha256",
            "localizer_output_sha256", "source_evidence_ref",
        )},
        # Full-frame teacher states cannot assert presence inside a YOLO crop.
        "source_foreign_material": evidence["foreign_material"] if evidence else "",
    }


def write_selected_crops(
    selected: Sequence[Candidate],
    output_dir: Path,
    *,
    crop_size: int,
    padding: float,
    jpeg_quality: int,
    min_free_gb: float,
    max_output_gb: float,
    canonical_aihub_origin: str | None = None,
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
            if canonical_aihub_origin is not None:
                reject_symlinks, _ = _mixed_file_helpers()
                reject_symlinks(destination, description="mixed crop output")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if canonical_aihub_origin is None:
                encoded.tofile(destination)
            else:
                with destination.open("xb") as stream:
                    stream.write(encoded.tobytes())
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
                    # Objectness describes the final verifier crop, while the
                    # source count truthfully retains whether the full frame
                    # contained its single annotated object.  A v4 hard
                    # negative is therefore source=1, crop=0.
                    "crop_object_count": (
                        0 if candidate.material == BACKGROUND_CLASS_ID else 1
                    ),
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
            if canonical_aihub_origin is not None:
                rows[-1].update(_canonical_row_metadata(candidate, destination, canonical_aihub_origin))
            if candidate.source.audited_aihub_metadata is not None:
                rows[-1].update(candidate.source.audited_aihub_metadata)
    rows.sort(key=lambda row: (str(row["split"]), int(row["material"]), str(row["filepath"])))
    return rows, written_bytes, rejected


PredictionProvider = Callable[[Sequence[SourceRecord]], Iterable[PredictedFrame]]


def _mixed_file_helpers():
    # Reuse the bounded-memory, path-identity-aware reader used by the source
    # adapter.  This optional path does not change legacy AIHub preparation.
    try:
        from scripts.assemble_operational_quality_exclusions import (
            _reject_symlink_components, _stable_file_sha256,
        )
    except ModuleNotFoundError:
        from assemble_operational_quality_exclusions import (
            _reject_symlink_components, _stable_file_sha256,
        )
    return _reject_symlink_components, _stable_file_sha256


def _mixed_input_snapshot(
    data_path: Path, split_images: dict[str, list[Path]], model_path: Path | None,
) -> dict[Path, tuple | None]:
    reject_symlinks, stable_hash = _mixed_file_helpers()
    paths = {data_path}
    optional_labels = set()
    for images in split_images.values():
        for path in images:
            paths.add(path)
            try:
                optional_labels.add(_label_path(path))
            except ValueError:
                pass  # The established collector quarantines unresolved labels.
    paths.update(optional_labels)
    if model_path is not None:
        reject_symlinks(model_path, description="mixed detector model")
        if model_path.is_dir():
            model_files = []
            for path in model_path.rglob("*"):
                reject_symlinks(path, description="mixed detector model member")
                if path.is_file():
                    model_files.append(path)
            if not model_files:
                raise ValueError("mixed detector model directory is empty")
            paths.update(model_files)
        else:
            paths.add(model_path)
    snapshot = {}
    for path in sorted(paths):
        reject_symlinks(path, description="mixed generation input")
        if path in optional_labels and not path.exists():
            snapshot[path] = None
            continue
        _, sha = stable_hash(path, description="mixed generation input")
        stat = path.stat()
        snapshot[path] = (sha, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    return snapshot


def _mixed_snapshot_unchanged(snapshot: dict[Path, tuple | None]) -> None:
    """Cheap boundary check; publication also repeats the complete SHA scan."""
    reject_symlinks, _ = _mixed_file_helpers()
    for path, previous in snapshot.items():
        reject_symlinks(path, description="mixed generation input")
        if previous is None:
            if path.exists():
                raise RuntimeError("mixed generation input changed during preparation")
        else:
            stat = path.stat()
            if previous[1:] != (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns):
                raise RuntimeError("mixed generation input changed during preparation")


def _claim_mixed_output(output_dir: Path, protected_roots: Sequence[Path]) -> tuple[int, int]:
    reject_symlinks, _ = _mixed_file_helpers()
    reject_symlinks(output_dir, description="mixed output")
    resolved = output_dir.resolve(strict=False)
    for root in protected_roots:
        if resolved.is_relative_to(root.resolve(strict=True)):
            raise ValueError("mixed output cannot be nested in generation inputs")
    # A fresh directory also keeps a concurrent run from sharing crop paths.
    output_dir.mkdir(parents=True, exist_ok=False)
    stat = output_dir.stat()
    return stat.st_dev, stat.st_ino


def _mark_mixed_failure(output_dir: Path, identity: tuple[int, int]) -> None:
    reject_symlinks, _ = _mixed_file_helpers()
    reject_symlinks(output_dir, description="mixed output")
    stat = output_dir.stat()
    if (stat.st_dev, stat.st_ino) != identity:
        return  # Never modify a foreign replacement directory.
    try:
        with (output_dir / "failed.json").open("x", encoding="utf-8") as stream:
            stream.write('{"status":"failed","stage":"mixed_preparation"}\n')
    except FileExistsError:
        pass


def _publish_mixed_metadata(
    output_dir: Path, rows: list[dict], summary: dict, *,
    validate: Callable[[bool], None], identity: tuple[int, int],
    extra_fields: tuple[str, ...] = (),
) -> None:
    try:
        reject_symlinks, _ = _mixed_file_helpers()
        reject_symlinks(output_dir, description="mixed output")
        stat = output_dir.stat()
        if (stat.st_dev, stat.st_ino) != identity:
            raise RuntimeError("mixed output directory ownership changed")
        validate(False)
        csv_buffer = io.StringIO(newline="")
        writer = csv.DictWriter(csv_buffer, fieldnames=MANIFEST_FIELDS + CANONICAL_FIELDS + extra_fields)
        writer.writeheader()
        writer.writerows(rows)
        metadata = {
            "manifest.csv": csv_buffer.getvalue(),
            "dataset_info.json": json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        }
        for name, content in metadata.items():
            with (output_dir / name).open("x", encoding="utf-8", newline="") as stream:
                stream.write(content)
        # Metadata publication is not success if any upstream input changed at
        # that boundary.  Consumers reject failed.json even if metadata exists.
        validate(True)
        _, stable_hash = _mixed_file_helpers()
        for name, content in metadata.items():
            _, sha = stable_hash(output_dir / name, description="mixed published metadata")
            if sha != hashlib.sha256(content.encode("utf-8")).hexdigest():
                raise RuntimeError("mixed metadata changed during publication")
        for row in rows:
            _, sha = stable_hash(Path(row["filepath"]), description="mixed published crop")
            if sha != row["image_sha256"]:
                raise RuntimeError("mixed crop changed during publication")
    except Exception:
        _mark_mixed_failure(output_dir, identity)
        raise


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
    proposal_selection: str = "all",
    background_policy: str = "low-iou-or-no-ground-truth",
    background_gt_margin: float = 0.10,
    nms_iou: float = 0.70,
    prediction_provider: PredictionProvider | None = None,
    operational_source_evidence_dir: Path | None = None,
    aihub_origin: str | None = None,
    audited_aihub_report: Path | None = None,
    audited_aihub_report_sha256: str | None = None,
    audited_aihub_cohort: Path | None = None,
    audited_aihub_diagnostic: bool = False,
) -> dict:
    # This must remain the first preflight: a bad invocation can never load a
    # multi-GB model or touch an existing dataset before overwrite refusal.
    ensure_empty_output(output_dir)
    audited_args = (audited_aihub_report, audited_aihub_report_sha256, audited_aihub_cohort)
    if any(value is not None for value in audited_args) and not all(value is not None for value in audited_args):
        raise ValueError("audited AIHub report, report SHA256 and cohort must be supplied together")
    if type(audited_aihub_diagnostic) is not bool or (audited_aihub_diagnostic and audited_aihub_report is None):
        raise ValueError("audited AIHub diagnostic requires an explicit audited snapshot")
    canonical_mode = operational_source_evidence_dir is not None or audited_aihub_report is not None
    if operational_source_evidence_dir is not None:
        _reject_operational_material_hold(operational_source_evidence_dir)
    if canonical_mode and (not aihub_origin or not aihub_origin.strip()):
        raise ValueError("explicit aihub_origin is required for mixed canonical manifests")
    if not 0 <= negative_iou < positive_iou <= 1:
        raise ValueError("IoU thresholds must satisfy 0 <= negative < positive <= 1")
    if batch < 1 or imgsz < 1 or crop_size < 1:
        raise ValueError("batch, imgsz and crop-size must be positive")
    if (
        not 0 <= conf <= 1
        or not 0 <= nms_iou <= 1
        or padding < 0
        or not 0 <= background_gt_margin <= 1
        or not 1 <= jpeg_quality <= 100
    ):
        raise ValueError("conf, padding or jpeg-quality is outside its valid range")
    if proposal_selection not in PROPOSAL_SELECTION_MODES:
        raise ValueError(f"unsupported proposal selection: {proposal_selection}")
    if background_policy not in BACKGROUND_POLICIES:
        raise ValueError(f"unsupported background policy: {background_policy}")
    check_storage_limits(
        output_dir,
        written_bytes=0,
        min_free_gb=min_free_gb,
        max_output_gb=max_output_gb,
    )

    # Keep this tensor referenced in the build frame through all source hashing
    # and crop writing.  On QNAP this reserves the same process's fault buffer;
    # a separate docker/preflight process is not an equivalent guarantee.
    _cuda_context_guard = (
        eager_initialize_cuda_context(device)
        if prediction_provider is None
        else None
    )

    mixed_snapshot = None
    actual_model_path = model_path if prediction_provider is None else None
    if canonical_mode:
        _, stable_hash = _mixed_file_helpers()
        _, initial_data_sha = stable_hash(data_path, description="mixed data YAML")
    split_images = resolve_split_images(data_path, dataset_dir)
    source_hashes = None
    if canonical_mode:
        mixed_snapshot = _mixed_input_snapshot(data_path, split_images, actual_model_path)
        if mixed_snapshot[data_path][0] != initial_data_sha:
            raise RuntimeError("mixed data YAML changed during source collection")
        source_hashes = {path: mixed_snapshot[path][0] for images in split_images.values() for path in images}
    sources, source_rejections = collect_sources(
        split_images, dataset_dir,
        **({"source_hashes": source_hashes} if source_hashes is not None else {}),
    )
    audited_snapshot = None
    audited_binding = None
    audited_input_roots = []
    if audited_aihub_report is not None:
        audited_snapshot = _audited_aihub_reader()(
            audited_aihub_report, audited_aihub_report_sha256,
            cohort_path=audited_aihub_cohort, require_full_cohort=not audited_aihub_diagnostic,
        )
        audited_snapshot.assert_source_membership(split_images)
        audited_binding = audited_snapshot.binding()
        sources = [replace(source, audited_aihub_metadata=audited_snapshot.metadata_for(source.path)) for source in sources]
        audited_input_roots = [audited_aihub_report.parent, audited_aihub_cohort.parent]
        for source in sources:
            for field in ("original_source_path_b64", "original_annotation_path_b64"):
                value = source.audited_aihub_metadata[field]
                original_path = Path(os.fsdecode(base64.urlsafe_b64decode(value)))
                # The reader validates the original root/official-split/data-dir/
                # category/file layout. Protect the entire input root, including
                # source classes excluded by quality or crop selection.
                audited_input_roots.append(original_path.parents[3])
        audited_input_roots = sorted(set(audited_input_roots))
    operational_records = None
    operational_binding = None
    operational_input_roots = []
    if operational_source_evidence_dir is not None:
        operational_binding = _operational_bundle_binding(operational_source_evidence_dir)
        operational_records = _operational_bundle_reader(operational_source_evidence_dir)
        if _operational_bundle_binding(operational_source_evidence_dir) != operational_binding:
            raise RuntimeError("operational source evidence changed during initial validation")
        operational_input_roots = _operational_bundle_input_roots(
            operational_source_evidence_dir, operational_binding,
        )
        sources = append_operational_sources(
            sources, operational_records, all_split_images=split_images, dataset_dir=dataset_dir,
            source_hashes=source_hashes,
        )
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
            nms_iou=nms_iou,
        )
    proposal_stats = Counter()
    proposal_policy_stats = Counter()
    candidates = candidates_from_frames(
        prediction_provider(sources),
        positive_iou=positive_iou,
        negative_iou=negative_iou,
        proposal_selection=proposal_selection,
        background_policy=background_policy,
        min_confidence=conf,
        verifier_crop_padding=padding,
        background_gt_margin=background_gt_margin,
        stats=proposal_stats,
        policy_stats=proposal_policy_stats,
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

    mixed_output_identity = None
    if canonical_mode:
        if operational_source_evidence_dir is not None:
            _reject_operational_material_hold(operational_source_evidence_dir)
        if audited_snapshot is not None:
            audited_snapshot.recheck()
        _mixed_snapshot_unchanged(mixed_snapshot)
        mixed_output_identity = _claim_mixed_output(
            output_dir, [dataset_dir, *operational_input_roots, *audited_input_roots]
            + sorted({Path(row["source_filepath"]).parent for row in (operational_records or [])})
            + ([model_path] if actual_model_path is not None and model_path.is_dir() else []),
        )
    try:
        rows, written_bytes, write_rejections = write_selected_crops(
            selected,
            output_dir,
            crop_size=crop_size,
            padding=padding,
            jpeg_quality=jpeg_quality,
            min_free_gb=min_free_gb,
            max_output_gb=max_output_gb,
            **({"canonical_aihub_origin": aihub_origin} if canonical_mode else {}),
        )
        if not rows:
            raise RuntimeError("all selected proposal crops failed to write")
        if canonical_mode and {row["split"] for row in rows} != {"training", "validation"}:
            raise RuntimeError("mixed crop writing did not preserve both source splits")
    except Exception:
        if mixed_output_identity is not None:
            _mark_mixed_failure(output_dir, mixed_output_identity)
        raise

    manifest_path = output_dir / "manifest.csv"
    if not canonical_mode:
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
        "proposal_policy": {
            "selection_mode": proposal_selection,
            "minimum_confidence": (
                conf if proposal_selection == "runtime-top1" else None
            ),
            "background_policy": background_policy,
            **(
                {
                    "background_gt_margin": background_gt_margin,
                    "background_intersection_basis": (
                        "final_padded_verifier_crop_vs_expanded_gt"
                    ),
                }
                if background_policy == "strict-zero-intersection"
                else {}
            ),
        },
        "proposal_policy_stats": dict(proposal_policy_stats),
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
            "nms_iou": nms_iou,
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
    if operational_binding is not None:
        summary["operational_source_evidence"] = operational_binding
        summary["operational_sources"] = len(operational_records)
        summary["operational_crop_state_targets"] = "all_unknown_minus_one"
        summary["source_policy"] = "AIHub zero/one annotation plus verified train-only positive operational pseudo-annotations"
    if audited_binding is not None:
        summary["audited_aihub_snapshot"] = audited_binding
        summary["original_identity_semantics"] = "original file and annotation lineage; source-based object_group/capture_session do not prove physical object or capture session identity"
    if not canonical_mode:
        (output_dir / "dataset_info.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    else:
        def validate_mixed_generation(full_rehash: bool) -> None:
            _mixed_snapshot_unchanged(mixed_snapshot)
            if operational_records is not None and _operational_bundle_binding(operational_source_evidence_dir) != operational_binding:
                raise RuntimeError("operational source evidence changed during preparation")
            if audited_snapshot is not None:
                audited_snapshot.recheck()
                if audited_snapshot.binding() != audited_binding:
                    raise RuntimeError("audited AIHub snapshot binding changed during preparation")
            if full_rehash:
                if (resolve_split_images(data_path, dataset_dir) != split_images
                        or _mixed_input_snapshot(data_path, split_images, actual_model_path) != mixed_snapshot):
                    raise RuntimeError("mixed generation input changed during preparation")
                if operational_records is not None and _operational_bundle_reader(operational_source_evidence_dir) != operational_records:
                    raise RuntimeError("operational source evidence changed during preparation")
        _publish_mixed_metadata(
            output_dir, rows, summary, validate=validate_mixed_generation, identity=mixed_output_identity,
            **({"extra_fields": AUDITED_AIHUB_FIELDS} if audited_snapshot is not None else {}),
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--operational-source-evidence-dir", type=Path)
    parser.add_argument("--aihub-origin", help="Explicit base origin for optional canonical mixed manifest; not a license approval")
    parser.add_argument("--audited-aihub-report", type=Path)
    parser.add_argument("--audited-aihub-report-sha256")
    parser.add_argument("--audited-aihub-cohort", type=Path)
    parser.add_argument("--audited-aihub-diagnostic", action="store_true", help="Explicitly allow a partial audited snapshot; replay must remain diagnostic-only")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.70,
        help="운영 run_main과 동일한 YOLO NMS IoU",
    )
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
    parser.add_argument(
        "--proposal-selection",
        choices=PROPOSAL_SELECTION_MODES,
        default="all",
        help=(
            "all은 모든 proposal을 사용하고, runtime-top1은 conf 이상 중 "
            "최고 confidence bbox 하나만 사용합니다."
        ),
    )
    parser.add_argument(
        "--background-policy",
        choices=BACKGROUND_POLICIES,
        default="low-iou-or-no-ground-truth",
        help=(
            "no-ground-truth-only는 GT가 없는 source의 proposal만 "
            "background로 허용하고, strict-zero-intersection은 최종 padded "
            "crop이 확장 GT와 전혀 겹치지 않을 때만 허용합니다."
        ),
    )
    parser.add_argument(
        "--background-gt-margin",
        type=float,
        default=0.10,
        help="strict-zero-intersection에서 GT 각 변에 추가하는 안전 여백 비율",
    )
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
    if (
        not 0 <= args.conf <= 1
        or not 0 <= args.nms_iou <= 1
        or args.padding < 0
        or not 0 <= args.background_gt_margin <= 1
    ):
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
        nms_iou=args.nms_iou,
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
        proposal_selection=args.proposal_selection,
        background_policy=args.background_policy,
        background_gt_margin=args.background_gt_margin,
        operational_source_evidence_dir=args.operational_source_evidence_dir,
        aihub_origin=args.aihub_origin,
        audited_aihub_report=args.audited_aihub_report,
        audited_aihub_report_sha256=args.audited_aihub_report_sha256,
        audited_aihub_cohort=args.audited_aihub_cohort,
        audited_aihub_diagnostic=args.audited_aihub_diagnostic,
    )


if __name__ == "__main__":
    main()
