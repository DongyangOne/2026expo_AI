"""Mine runtime-shaped background crops from trusted empty operational scenes.

The input is a *source inventory*, not a crop manifest.  Every row must be a
high-confidence, train-only VLM ``negative`` whose raw image bytes and capture
lineage are still available.  The frozen detector is then run exactly like the
runtime candidate stage (confidence 0.10, NMS IoU 0.70, highest confidence with
original proposal order as the tie-breaker).  Only an actual detector proposal
is cropped; an easy empty scene with no proposal is reported but never turned
into a full-frame background sample.

All validation and overwrite guards run before Ultralytics is imported.  The
``prediction_provider`` hook keeps the safety contract unit-testable without a
GPU or Ultralytics installation.  Successful output is built in a sibling
temporary directory and atomically published so a failed run cannot leave a
partially valid dataset at the requested path.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import cv2
import numpy as np

try:
    from scripts.verifier_preprocessing_contract import (
        VerifierCropContract,
        letterbox_bgr as _contract_letterbox_bgr,
        padded_clipped_bbox as _contract_padded_clipped_bbox,
        validate_crop_preprocessing_spec,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from verifier_preprocessing_contract import (  # type: ignore[no-redef]
        VerifierCropContract,
        letterbox_bgr as _contract_letterbox_bgr,
        padded_clipped_bbox as _contract_padded_clipped_bbox,
        validate_crop_preprocessing_spec,
    )


CLASS_NAMES = (
    "can",
    "pet",
    "paper",
    "plastic",
    "styrofoam",
    "vinyl",
    "glass",
    "battery",
    "fluorescent",
)
BACKGROUND_CLASS_ID = len(CLASS_NAMES)
BACKGROUND_CLASS_NAME = "background"
DETECTOR_CONFIDENCE = 0.10
DETECTOR_NMS_IOU = 0.70
PROPOSAL_SELECTION = "highest_confidence_then_original_order"
CROP_PADDING = 0.08
CROP_SIZE = 320
EXPECTED_ORIGIN = "operational_empty_scene_vlm_teacher_source"
EXPECTED_AUTHORITY = "vlm_teacher_pseudo_label_train_only"
DEFAULT_MINIMUM_TEACHER_CONFIDENCE = 0.80
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIB = 1024**3

ARTIFACT_NAMES = {
    "csv": "operational_empty_scene_proposals.csv",
    "jsonl": "operational_empty_scene_proposals.jsonl",
    "no_proposal": "operational_empty_scene_no_proposal.json",
    "lineage": "operational_empty_scene_lineage.json",
}

REQUIRED_INVENTORY_FIELDS = frozenset(
    {
        "filepath",
        "source_sha256",
        "object_group",
        "capture_session",
        "role",
        "fold",
        "origin",
        "material",
        "category",
        "source_object_count",
        "training_crop_ready",
        "teacher_material",
        "teacher_minimum_confidence",
        "teacher_consensus_votes",
        "teacher_pass_count",
        "pseudo_label",
        "ground_truth_authority",
        "blind_test_eligible",
    }
)

MANIFEST_FIELDS = (
    "sample_id",
    "role",
    "split_role",
    "fold",
    "filepath",
    "split",
    "source_id",
    "source_sha256",
    "image_sha256",
    "content_identity",
    "object_group",
    "capture_session",
    "origin",
    "source_origin",
    "selection_reason",
    "material",
    "category",
    "dent",
    "label",
    "foreign_material",
    "source_object_count",
    "source_image",
    "source_path_b64",
    "source_bbox_x",
    "source_bbox_y",
    "source_bbox_w",
    "source_bbox_h",
    "source_width",
    "source_height",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_source",
    "proposal_index",
    "assignment",
    "predicted_class_id",
    "predicted_class_name",
    "predicted_confidence",
    "predicted_bbox_x1",
    "predicted_bbox_y1",
    "predicted_bbox_x2",
    "predicted_bbox_y2",
    "crop_bbox_x1",
    "crop_bbox_y1",
    "crop_bbox_x2",
    "crop_bbox_y2",
    "crop_width",
    "crop_height",
    "crop_padding_ratio",
    "letterbox_size",
    "letterbox_scale",
    "letterbox_resized_width",
    "letterbox_resized_height",
    "letterbox_offset_x",
    "letterbox_offset_y",
    "letterbox_fill",
    "jpeg_quality",
    "crop_transform_version",
    "crop_bytes",
    "teacher_material",
    "teacher_minimum_confidence",
    "teacher_consensus_votes",
    "teacher_pass_count",
    "pseudo_label",
    "ground_truth_authority",
    "blind_test_eligible",
    "training_crop_ready",
    "detector_model_sha256",
    "inference_spec_sha256",
    "source_inventory_sha256",
)


@dataclass(frozen=True)
class SourceRecord:
    path: Path
    source_sha256: str
    source_reference: str
    object_group: str
    capture_session: str
    fold: str
    source_width: int
    source_height: int
    teacher_minimum_confidence: float
    teacher_consensus_votes: int
    teacher_pass_count: int
    input_line: int


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
class FrozenInferenceSpec:
    raw: dict[str, Any]
    crop: VerifierCropContract
    input_size: int
    jpeg_quality: int


PredictionProvider = Callable[[Sequence[SourceRecord]], Iterable[PredictedFrame]]


def _canonical_json(value: object, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_artifact(path: Path) -> str:
    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(f"detector model does not exist: {path}")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"detector model directory is empty: {path}")
    digest = hashlib.sha256(b"detector-directory-v1\0")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(item)))
    return digest.hexdigest()


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return _canonical_json(value)
    return str(value).strip()


def _parse_bool(value: object, *, field: str) -> bool:
    normalized = _as_text(value).casefold()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"{field} must be an explicit boolean")


def _parse_int(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    text = _as_text(value)
    try:
        result = int(text)
    except ValueError as error:
        raise ValueError(f"{field} must be an integer") from error
    if text not in {str(result), f"+{result}"}:
        raise ValueError(f"{field} must be an integer")
    return result


def _parse_probability(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite probability")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite probability") from error
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{field} must be between zero and one")
    return result


def _normalized_sha(value: object, *, field: str) -> str:
    normalized = _as_text(value).casefold()
    if not SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return normalized


def _load_inventory(path: Path) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    if not path.is_file():
        raise FileNotFoundError(f"source inventory does not exist: {path}")
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            headers = tuple(reader.fieldnames or ())
            if not headers or len(headers) != len(set(headers)) or any(not item for item in headers):
                raise ValueError("source inventory has empty or duplicate CSV columns")
            rows: list[dict[str, object]] = []
            for line, row in enumerate(reader, start=2):
                if None in row:
                    raise ValueError(f"source inventory CSV row {line} has extra columns")
                rows.append({**row, "_input_line": line})
    elif suffix == ".jsonl":
        rows = []
        header_set: set[str] = set()
        for line, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at source inventory line {line}") from error
            if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
                raise ValueError(f"source inventory line {line} must be a JSON object")
            header_set.update(value)
            rows.append({**value, "_input_line": line})
        headers = tuple(sorted(header_set))
    else:
        raise ValueError("source inventory must be .csv or .jsonl")
    if not rows:
        raise ValueError("source inventory is empty")
    missing = sorted(REQUIRED_INVENTORY_FIELDS - set(headers))
    if missing:
        raise ValueError(f"source inventory is missing required fields: {missing}")
    return rows, headers


def _resolve_source(row: Mapping[str, object], *, inventory_dir: Path, line: int) -> tuple[Path, str]:
    references = []
    for field in ("filepath", "source_image"):
        value = _as_text(row.get(field))
        if value:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = inventory_dir / candidate
            references.append((field, candidate.resolve(strict=False)))
    if not references:
        raise ValueError(f"source inventory line {line}: filepath is empty")
    if any(path != references[0][1] for _, path in references[1:]):
        raise ValueError(f"source inventory line {line}: filepath and source_image conflict")
    path = references[0][1]
    if not path.is_file():
        raise FileNotFoundError(f"source inventory line {line}: source image does not exist: {path}")
    return path, path.as_posix()


def _read_image(path: Path) -> np.ndarray:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as error:
        raise ValueError(f"could not read source image: {path}") from error
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"could not decode source image: {path}")
    return image


def _contains_forbidden_partition(value: object) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", _as_text(value).casefold()).strip()
    compact = normalized.replace(" ", "")
    tokens = set(normalized.split())
    return (
        "hardware41" in compact
        or "calibration" in tokens
        or "blind" in tokens
        or "blindtest" in compact
        or "validation" in tokens
        or "modelvalidation" in compact
    )


def _validate_inventory_row(
    row: Mapping[str, object],
    *,
    inventory_dir: Path,
    minimum_teacher_confidence: float,
) -> SourceRecord:
    line = int(row["_input_line"])
    prefix = f"source inventory line {line}"
    for field in REQUIRED_INVENTORY_FIELDS:
        if field not in row:
            raise ValueError(f"{prefix}: missing {field}")

    role = _as_text(row.get("role")).casefold()
    if role != "train":
        raise ValueError(f"{prefix}: role must be train; calibration/blind/hardware inputs are forbidden")
    split_role = _as_text(row.get("split_role")).casefold()
    if split_role and split_role != "train":
        raise ValueError(f"{prefix}: split_role must be train")
    split = _as_text(row.get("split")).casefold()
    if split and split not in {"train", "training"}:
        raise ValueError(f"{prefix}: split must be training")
    for field in ("role", "split_role", "split", "fold", "origin", "object_group", "capture_session"):
        if _contains_forbidden_partition(row.get(field)):
            raise ValueError(f"{prefix}: forbidden calibration/blind/hardware41 lineage in {field}")

    if _as_text(row.get("origin")) != EXPECTED_ORIGIN:
        raise ValueError(f"{prefix}: origin must be {EXPECTED_ORIGIN}")
    if _parse_int(row.get("material"), field=f"{prefix} material") != BACKGROUND_CLASS_ID:
        raise ValueError(f"{prefix}: material must be {BACKGROUND_CLASS_ID}")
    if _as_text(row.get("category")).casefold() != BACKGROUND_CLASS_NAME:
        raise ValueError(f"{prefix}: category must be background")
    if _parse_int(row.get("source_object_count"), field=f"{prefix} source_object_count") != 0:
        raise ValueError(f"{prefix}: source_object_count must be zero")
    if _parse_bool(row.get("training_crop_ready"), field=f"{prefix} training_crop_ready"):
        raise ValueError(f"{prefix}: source inventory rows must not already be crop-ready")
    if _as_text(row.get("teacher_material")).casefold() != "negative":
        raise ValueError(f"{prefix}: teacher_material must be negative")
    if "teacher_consensus" in row and not _parse_bool(
        row.get("teacher_consensus"), field=f"{prefix} teacher_consensus"
    ):
        raise ValueError(f"{prefix}: teacher_consensus must be true")
    confidence = _parse_probability(
        row.get("teacher_minimum_confidence"),
        field=f"{prefix} teacher_minimum_confidence",
    )
    if confidence < minimum_teacher_confidence:
        raise ValueError(
            f"{prefix}: teacher confidence {confidence:.4f} is below "
            f"{minimum_teacher_confidence:.4f}"
        )
    votes = _parse_int(row.get("teacher_consensus_votes"), field=f"{prefix} teacher_consensus_votes")
    passes = _parse_int(row.get("teacher_pass_count"), field=f"{prefix} teacher_pass_count")
    if votes < 2 or passes < votes:
        raise ValueError(f"{prefix}: teacher consensus requires at least two supporting votes")
    if not _parse_bool(row.get("pseudo_label"), field=f"{prefix} pseudo_label"):
        raise ValueError(f"{prefix}: pseudo_label must be true")
    if _as_text(row.get("ground_truth_authority")) != EXPECTED_AUTHORITY:
        raise ValueError(f"{prefix}: unsupported ground_truth_authority")
    if _parse_bool(row.get("blind_test_eligible"), field=f"{prefix} blind_test_eligible"):
        raise ValueError(f"{prefix}: teacher negatives can never be blind-test eligible")

    fold = _as_text(row.get("fold"))
    object_group = _as_text(row.get("object_group"))
    capture_session = _as_text(row.get("capture_session"))
    if not all((fold, object_group, capture_session)):
        raise ValueError(f"{prefix}: fold/object_group/capture_session must be nonempty")

    # A source inventory must not smuggle a full-frame or stale detector bbox
    # into this run.  Geometry authority comes exclusively from the frozen
    # detector invocation below.
    for field, value in row.items():
        normalized_field = field.casefold()
        if (
            "bbox" in normalized_field
            or normalized_field in {"proposal_index", "assignment"}
        ) and _as_text(value):
            raise ValueError(f"{prefix}: {field} must be blank in a source inventory")

    source_sha = _normalized_sha(row.get("source_sha256"), field=f"{prefix} source_sha256")
    for field in ("source_id", "image_sha256"):
        value = _as_text(row.get(field)).casefold()
        if value and value != source_sha:
            raise ValueError(f"{prefix}: {field} must equal source_sha256")
    identity = _as_text(row.get("content_identity")).casefold()
    if identity and identity != f"sha256:{source_sha}":
        raise ValueError(f"{prefix}: content_identity must identify source_sha256")
    path, source_reference = _resolve_source(row, inventory_dir=inventory_dir, line=line)
    if _sha256_file(path) != source_sha:
        raise ValueError(f"{prefix}: source_sha256 mismatch")
    image = _read_image(path)
    source_height, source_width = (int(value) for value in image.shape[:2])
    for field, actual in (("source_width", source_width), ("source_height", source_height)):
        value = _as_text(row.get(field))
        if value and _parse_int(value, field=f"{prefix} {field}") != actual:
            raise ValueError(f"{prefix}: {field} mismatch")

    return SourceRecord(
        path=path,
        source_sha256=source_sha,
        source_reference=source_reference,
        object_group=object_group,
        capture_session=capture_session,
        fold=fold,
        source_width=source_width,
        source_height=source_height,
        teacher_minimum_confidence=confidence,
        teacher_consensus_votes=votes,
        teacher_pass_count=passes,
        input_line=line,
    )


def _load_and_validate_sources(
    inventory_path: Path,
    *,
    minimum_teacher_confidence: float,
) -> list[SourceRecord]:
    rows, _ = _load_inventory(inventory_path)
    sources = [
        _validate_inventory_row(
            row,
            inventory_dir=inventory_path.parent,
            minimum_teacher_confidence=minimum_teacher_confidence,
        )
        for row in rows
    ]
    seen: dict[str, int] = {}
    for source in sources:
        previous = seen.get(source.source_sha256)
        if previous is not None:
            raise ValueError(
                "duplicate source content in inventory at lines "
                f"{previous} and {source.input_line}"
            )
        seen[source.source_sha256] = source.input_line
    sources.sort(key=lambda item: (item.source_sha256, item.object_group, item.capture_session))
    return sources


def _load_frozen_inference_spec(path: Path) -> FrozenInferenceSpec:
    if not path.is_file():
        raise FileNotFoundError(f"inference spec does not exist: {path}")
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("inference spec is not valid JSON") from error
    if not isinstance(spec, dict):
        raise ValueError("inference spec must be a JSON object")
    if spec.get("format_version") != 1:
        raise ValueError("inference spec format_version must be 1")
    if spec.get("artifact_role") != "offline_candidate_spec_not_production_authorization":
        raise ValueError("inference spec is not an offline-candidate contract")
    detector = spec.get("detector")
    if not isinstance(detector, dict) or detector.get("task") != "detect":
        raise ValueError("inference spec detector task must be detect")
    confidence = _parse_probability(detector.get("candidate_confidence"), field="candidate_confidence")
    nms_iou = _parse_probability(detector.get("nms_iou"), field="nms_iou")
    if not math.isclose(confidence, DETECTOR_CONFIDENCE, abs_tol=1e-12):
        raise ValueError("empty-scene mining requires detector confidence=0.10")
    if not math.isclose(nms_iou, DETECTOR_NMS_IOU, abs_tol=1e-12):
        raise ValueError("empty-scene mining requires detector NMS IoU=0.70")
    if detector.get("proposal_selection") != PROPOSAL_SELECTION:
        raise ValueError(f"inference spec proposal_selection must be {PROPOSAL_SELECTION}")
    input_size = detector.get("input_size")
    if not isinstance(input_size, int) or isinstance(input_size, bool) or input_size != 640:
        raise ValueError("empty-scene mining requires detector input_size=640")
    if tuple(spec.get("detector_classes", ())) != CLASS_NAMES:
        raise ValueError("inference spec detector class order does not match the nine-class contract")
    crop = validate_crop_preprocessing_spec(spec)
    if not math.isclose(crop.padding_ratio, CROP_PADDING, abs_tol=1e-12):
        raise ValueError("empty-scene mining requires crop padding=0.08")
    if crop.size != CROP_SIZE:
        raise ValueError("empty-scene mining requires crop size=320")
    raw_crop = spec.get("crop")
    assert isinstance(raw_crop, dict)
    jpeg_quality = raw_crop.get("jpeg_quality")
    if not isinstance(jpeg_quality, int) or isinstance(jpeg_quality, bool) or not 1 <= jpeg_quality <= 100:
        raise ValueError("inference spec crop.jpeg_quality must be an integer in 1..100")
    return FrozenInferenceSpec(spec, crop, input_size, jpeg_quality)


def _ensure_new_output(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output path: {output_dir}")


def _existing_ancestor(path: Path) -> Path:
    current = path.resolve(strict=False)
    while not current.exists():
        if current.parent == current:
            raise FileNotFoundError(f"no existing ancestor for output: {path}")
        current = current.parent
    return current


def _check_storage(
    path: Path,
    *,
    written_bytes: int,
    next_bytes: int,
    minimum_free_bytes: int,
    max_output_bytes: int,
) -> None:
    if minimum_free_bytes < 0 or max_output_bytes < 0:
        raise ValueError("storage guards must be non-negative")
    free = shutil.disk_usage(_existing_ancestor(path)).free
    if minimum_free_bytes and free - next_bytes < minimum_free_bytes:
        raise RuntimeError("free-space guard would be violated")
    if max_output_bytes and written_bytes + next_bytes > max_output_bytes:
        raise RuntimeError("output would exceed max_output_bytes")


def eager_initialize_cuda_context(device: str) -> object | None:
    """Reserve QNAP GPU0's CUDA fault buffer in this long-lived process."""

    if str(device).strip().casefold() not in {"0", "cuda", "cuda:0"}:
        return None
    import torch

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false")
        guard = torch.ones(1, device="cuda:0") + 1
        torch.cuda.synchronize(0)
        if guard.item() != 2:
            raise RuntimeError("CUDA tensor smoke failed")
        return guard
    except Exception as error:
        raise RuntimeError("failed to eagerly initialize CUDA for empty-scene mining") from error


def _normalized_model_names(names: object) -> tuple[str, ...] | None:
    if isinstance(names, dict):
        try:
            return tuple(str(names[index]) for index in range(len(names)))
        except (KeyError, TypeError):
            return None
    if isinstance(names, (list, tuple)):
        return tuple(str(value) for value in names)
    return None


def iter_detector_predictions(
    records: Sequence[SourceRecord],
    *,
    model_path: Path,
    device: str,
    batch: int,
    imgsz: int,
) -> Iterator[PredictedFrame]:
    """Import Ultralytics lazily and yield predictions in source order."""

    from ultralytics import YOLO

    model = YOLO(str(model_path), task="detect")
    names = _normalized_model_names(getattr(model, "names", None))
    if names is not None and names != CLASS_NAMES:
        raise RuntimeError("detector model class order does not match the inference spec")
    exported_backend = model_path.is_dir() or model_path.suffix.casefold() != ".pt"
    chunk_size = 1 if exported_backend else batch
    for start in range(0, len(records), chunk_size):
        chunk = records[start : start + chunk_size]
        source_arg: str | list[str]
        source_arg = str(chunk[0].path) if exported_backend else [str(item.path) for item in chunk]
        results = model.predict(
            source=source_arg,
            device=device,
            batch=1 if exported_backend else batch,
            imgsz=imgsz,
            conf=DETECTOR_CONFIDENCE,
            iou=DETECTOR_NMS_IOU,
            stream=True,
            save=False,
            verbose=False,
        )
        yielded = 0
        for result in results:
            if yielded >= len(chunk):
                raise RuntimeError("detector returned more frames than requested")
            source = chunk[yielded]
            yielded += 1
            height, width = (int(value) for value in result.orig_shape)
            result_names = _normalized_model_names(result.names)
            if result_names != CLASS_NAMES:
                raise RuntimeError("detector result class order does not match the inference spec")
            proposals: list[Proposal] = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.detach().cpu().tolist()
                classes = result.boxes.cls.detach().cpu().tolist()
                confidences = result.boxes.conf.detach().cpu().tolist()
                for bbox, raw_class, confidence in zip(boxes, classes, confidences):
                    class_id = int(raw_class)
                    proposals.append(
                        Proposal(
                            class_id=class_id,
                            confidence=float(confidence),
                            bbox=tuple(float(value) for value in bbox),  # type: ignore[arg-type]
                            class_name=CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else "",
                        )
                    )
            yield PredictedFrame(source, width, height, tuple(proposals))
        if yielded != len(chunk):
            raise RuntimeError(f"detector returned {yielded} frames for {len(chunk)} sources")


def _select_runtime_top1(
    frame: PredictedFrame,
    *,
    stats: Counter[str],
) -> tuple[int, Proposal] | None:
    eligible: list[tuple[int, Proposal]] = []
    stats["proposals_seen"] += len(frame.proposals)
    for index, proposal in enumerate(frame.proposals):
        try:
            class_id = int(proposal.class_id)
            if isinstance(proposal.class_id, bool) or class_id != proposal.class_id:
                raise ValueError
            confidence = float(proposal.confidence)
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError
            bbox = tuple(float(value) for value in proposal.bbox)
            if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
                raise ValueError
            if not 0 <= class_id < len(CLASS_NAMES):
                raise ValueError
            if proposal.class_name and proposal.class_name != CLASS_NAMES[class_id]:
                raise RuntimeError("detector proposal class name conflicts with frozen class order")
            _contract_padded_clipped_bbox(
                bbox,
                width=frame.width,
                height=frame.height,
                padding=CROP_PADDING,
            )
        except RuntimeError:
            raise
        except (TypeError, ValueError):
            stats["invalid_proposals"] += 1
            continue
        if confidence < DETECTOR_CONFIDENCE:
            stats["below_candidate_confidence"] += 1
            continue
        eligible.append(
            (
                index,
                Proposal(class_id, confidence, bbox, CLASS_NAMES[class_id]),
            )
        )
    if not eligible:
        stats["frames_without_eligible_proposal"] += 1
        return None
    selected = max(eligible, key=lambda item: (item[1].confidence, -item[0]))
    stats["discarded_by_runtime_top1"] += len(eligible) - 1
    stats["proposals_selected"] += 1
    return selected


def _sample_id(
    source: SourceRecord,
    *,
    proposal_index: int,
    proposal: Proposal,
    model_sha256: str,
    spec_sha256: str,
) -> str:
    identity = {
        "bbox": [f"{value:.8f}" for value in proposal.bbox],
        "confidence": f"{proposal.confidence:.8f}",
        "model_sha256": model_sha256,
        "object_group": source.object_group,
        "proposal_index": proposal_index,
        "source_sha256": source.source_sha256,
        "spec_sha256": spec_sha256,
        "transform": "operational_empty_scene_runtime_crop_v1",
    }
    return "opempty_" + _sha256_bytes(_canonical_json(identity).encode("utf-8"))[:24]


def _encode_jpeg(image: np.ndarray, *, quality: int) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            quality,
            cv2.IMWRITE_JPEG_OPTIMIZE,
            0,
            cv2.IMWRITE_JPEG_PROGRESSIVE,
            0,
        ],
    )
    if not ok:
        raise RuntimeError("failed to encode detector proposal crop")
    return encoded.tobytes()


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in MANIFEST_FIELDS} for row in rows)
    path.write_text(output.getvalue(), encoding="utf-8", newline="")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    text = "".join(_canonical_json(dict(row)) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8", newline="")


def mine_operational_empty_scene_proposals(
    *,
    input_inventory: Path,
    detector_model: Path,
    inference_spec: Path,
    output_dir: Path,
    device: str = "cpu",
    batch: int = 16,
    minimum_teacher_confidence: float = DEFAULT_MINIMUM_TEACHER_CONFIDENCE,
    minimum_free_bytes: int = 0,
    max_output_bytes: int = 0,
    prediction_provider: PredictionProvider | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build train-only runtime proposal backgrounds and return lineage counts."""

    # The overwrite refusal is intentionally first.  A bad call cannot hash a
    # multi-GB model, initialize CUDA, or import Ultralytics before protecting
    # a pre-existing dataset.
    _ensure_new_output(output_dir)
    if batch < 1:
        raise ValueError("batch must be positive")
    minimum_teacher_confidence = _parse_probability(
        minimum_teacher_confidence,
        field="minimum_teacher_confidence",
    )
    _check_storage(
        output_dir,
        written_bytes=0,
        next_bytes=0,
        minimum_free_bytes=minimum_free_bytes,
        max_output_bytes=max_output_bytes,
    )
    frozen = _load_frozen_inference_spec(inference_spec)
    model_sha256 = _sha256_artifact(detector_model)
    spec_sha256 = _sha256_file(inference_spec)
    inventory_sha256 = _sha256_file(input_inventory)
    sources = _load_and_validate_sources(
        input_inventory,
        minimum_teacher_confidence=minimum_teacher_confidence,
    )

    preflight = {
        "format_version": 1,
        "artifact_role": "train_only_pseudo_label_not_deployment_authorization",
        "dry_run": bool(dry_run),
        "validated_sources": len(sources),
        "input_hashes": {
            "source_inventory_sha256": inventory_sha256,
            "detector_model_sha256": model_sha256,
            "inference_spec_sha256": spec_sha256,
        },
        "runtime_contract": {
            "candidate_confidence": DETECTOR_CONFIDENCE,
            "nms_iou": DETECTOR_NMS_IOU,
            "proposal_selection": PROPOSAL_SELECTION,
            "crop_padding": CROP_PADDING,
            "crop_size": CROP_SIZE,
            "letterbox_fill": frozen.crop.fill,
            "jpeg_quality": frozen.jpeg_quality,
        },
    }
    if dry_run:
        return preflight

    output_parent = output_dir.resolve(strict=False).parent
    output_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_parent))
    rows: list[dict[str, str]] = []
    no_proposal: list[dict[str, object]] = []
    stats: Counter[str] = Counter()
    written_bytes = 0
    try:
        # Keep the guard alive until all inference/crop work is complete.  An
        # injected provider deliberately bypasses both torch and Ultralytics.
        _cuda_context_guard = (
            eager_initialize_cuda_context(device) if prediction_provider is None else None
        )
        if prediction_provider is None:
            prediction_provider = lambda items: iter_detector_predictions(
                items,
                model_path=detector_model,
                device=device,
                batch=batch,
                imgsz=frozen.input_size,
            )
        frames = iter(prediction_provider(sources))
        for source in sources:
            try:
                frame = next(frames)
            except StopIteration as error:
                raise RuntimeError("prediction provider returned fewer frames than sources") from error
            if not isinstance(frame, PredictedFrame):
                raise TypeError("prediction provider must yield PredictedFrame values")
            if frame.source.source_sha256 != source.source_sha256:
                raise RuntimeError("prediction provider did not preserve deterministic source order")
            if (frame.width, frame.height) != (source.source_width, source.source_height):
                raise RuntimeError("detector result dimensions do not match validated source bytes")
            # Re-bind every prediction result to the preflight bytes, including
            # frames that yield no proposal and therefore never enter the crop
            # path below.  Otherwise a concurrently replaced source could be
            # reported as a no-proposal frame under a stale digest.
            if _sha256_file(source.path) != source.source_sha256:
                raise RuntimeError("source image changed after preflight")
            stats["frames_seen"] += 1
            before_invalid = stats["invalid_proposals"]
            before_below = stats["below_candidate_confidence"]
            selected = _select_runtime_top1(frame, stats=stats)
            if selected is None:
                reasons = []
                if not frame.proposals:
                    reasons.append("no_detector_proposals")
                if stats["below_candidate_confidence"] > before_below:
                    reasons.append("all_valid_proposals_below_confidence")
                if stats["invalid_proposals"] > before_invalid:
                    reasons.append("invalid_proposals_excluded")
                no_proposal.append(
                    {
                        "source_sha256": source.source_sha256,
                        "object_group": source.object_group,
                        "capture_session": source.capture_session,
                        "reasons": reasons or ["no_eligible_runtime_proposal"],
                    }
                )
                continue

            proposal_index, proposal = selected
            # Revalidate bytes after inference to prevent a source replacement
            # race from binding a crop to the preflight digest.
            if _sha256_file(source.path) != source.source_sha256:
                raise RuntimeError("source image changed after preflight")
            image = _read_image(source.path)
            if image.shape[:2] != (source.source_height, source.source_width):
                raise RuntimeError("source image dimensions changed after preflight")
            left, top, right, bottom = _contract_padded_clipped_bbox(
                proposal.bbox,
                width=source.source_width,
                height=source.source_height,
                padding=frozen.crop.padding_ratio,
            )
            boxed, scale, resized_width, resized_height, offset_x, offset_y = _contract_letterbox_bgr(
                image[top:bottom, left:right],
                size=frozen.crop.size,
                fill=frozen.crop.fill,
            )
            jpeg = _encode_jpeg(boxed, quality=frozen.jpeg_quality)
            _check_storage(
                stage,
                written_bytes=written_bytes,
                next_bytes=len(jpeg),
                minimum_free_bytes=minimum_free_bytes,
                max_output_bytes=max_output_bytes,
            )
            sample_id = _sample_id(
                source,
                proposal_index=proposal_index,
                proposal=proposal,
                model_sha256=model_sha256,
                spec_sha256=spec_sha256,
            )
            relative = Path("crops") / "train" / BACKGROUND_CLASS_NAME / f"{sample_id}.jpg"
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(jpeg)
            written_bytes += len(jpeg)
            image_sha256 = _sha256_bytes(jpeg)
            x1, y1, x2, y2 = proposal.bbox
            rows.append(
                {
                    "sample_id": sample_id,
                    "role": "train",
                    "split_role": "train",
                    "fold": source.fold,
                    "filepath": relative.as_posix(),
                    "split": "training",
                    "source_id": source.source_sha256,
                    "source_sha256": source.source_sha256,
                    "image_sha256": image_sha256,
                    "content_identity": f"sha256:{image_sha256}",
                    "object_group": source.object_group,
                    "capture_session": source.capture_session,
                    "origin": "operational_empty_scene_v4",
                    "source_origin": EXPECTED_ORIGIN,
                    "selection_reason": "high_confidence_teacher_negative_runtime_top1_false_positive",
                    "material": str(BACKGROUND_CLASS_ID),
                    "category": BACKGROUND_CLASS_NAME,
                    "dent": "-1",
                    "label": "-1",
                    "foreign_material": "-1",
                    "source_object_count": "0",
                    "source_image": source.source_reference,
                    "source_path_b64": base64.urlsafe_b64encode(os.fsencode(str(source.path))).decode("ascii"),
                    "source_bbox_x": f"{x1:.8f}",
                    "source_bbox_y": f"{y1:.8f}",
                    "source_bbox_w": f"{x2 - x1:.8f}",
                    "source_bbox_h": f"{y2 - y1:.8f}",
                    "source_width": str(source.source_width),
                    "source_height": str(source.source_height),
                    "bbox_x1": f"{x1:.8f}",
                    "bbox_y1": f"{y1:.8f}",
                    "bbox_x2": f"{x2:.8f}",
                    "bbox_y2": f"{y2:.8f}",
                    "bbox_source": "runtime_top1_detector_proposal",
                    "proposal_index": str(proposal_index),
                    "assignment": "no_ground_truth",
                    "predicted_class_id": str(proposal.class_id),
                    "predicted_class_name": proposal.class_name,
                    "predicted_confidence": f"{proposal.confidence:.8f}",
                    "predicted_bbox_x1": f"{x1:.8f}",
                    "predicted_bbox_y1": f"{y1:.8f}",
                    "predicted_bbox_x2": f"{x2:.8f}",
                    "predicted_bbox_y2": f"{y2:.8f}",
                    "crop_bbox_x1": str(left),
                    "crop_bbox_y1": str(top),
                    "crop_bbox_x2": str(right),
                    "crop_bbox_y2": str(bottom),
                    "crop_width": str(right - left),
                    "crop_height": str(bottom - top),
                    "crop_padding_ratio": f"{frozen.crop.padding_ratio:.8f}",
                    "letterbox_size": str(frozen.crop.size),
                    "letterbox_scale": f"{scale:.12f}",
                    "letterbox_resized_width": str(resized_width),
                    "letterbox_resized_height": str(resized_height),
                    "letterbox_offset_x": str(offset_x),
                    "letterbox_offset_y": str(offset_y),
                    "letterbox_fill": str(frozen.crop.fill),
                    "jpeg_quality": str(frozen.jpeg_quality),
                    "crop_transform_version": "operational_empty_scene_runtime_crop_v1",
                    "crop_bytes": str(len(jpeg)),
                    "teacher_material": "negative",
                    "teacher_minimum_confidence": f"{source.teacher_minimum_confidence:.8f}",
                    "teacher_consensus_votes": str(source.teacher_consensus_votes),
                    "teacher_pass_count": str(source.teacher_pass_count),
                    "pseudo_label": "true",
                    "ground_truth_authority": EXPECTED_AUTHORITY,
                    "blind_test_eligible": "false",
                    "training_crop_ready": "true",
                    "detector_model_sha256": model_sha256,
                    "inference_spec_sha256": spec_sha256,
                    "source_inventory_sha256": inventory_sha256,
                }
            )
        try:
            next(frames)
        except StopIteration:
            pass
        else:
            raise RuntimeError("prediction provider returned more frames than sources")

        rows.sort(key=lambda row: (row["source_sha256"], row["sample_id"]))
        no_proposal.sort(key=lambda row: str(row["source_sha256"]))
        _write_csv(stage / ARTIFACT_NAMES["csv"], rows)
        _write_jsonl(stage / ARTIFACT_NAMES["jsonl"], rows)
        no_proposal_report = {
            "format_version": 1,
            "policy": "report_only_never_full_frame_crop",
            "count": len(no_proposal),
            "frames": no_proposal,
        }
        (stage / ARTIFACT_NAMES["no_proposal"]).write_text(
            _canonical_json(no_proposal_report, pretty=True),
            encoding="utf-8",
            newline="",
        )
        counts = {
            "validated_sources": len(sources),
            "written_background_crops": len(rows),
            "no_eligible_proposal_frames": len(no_proposal),
            "written_crop_bytes": written_bytes,
        }
        artifact_hashes = {
            name: _sha256_file(stage / filename)
            for name, filename in ARTIFACT_NAMES.items()
            if name != "lineage"
        }
        lineage = {
            **preflight,
            "dry_run": False,
            "builder": "scripts/mine_operational_empty_scene_proposals.py",
            "source_names": {
                "source_inventory": input_inventory.name,
                "detector_model": detector_model.name,
                "inference_spec": inference_spec.name,
            },
            "safety": {
                "accepted_role": "train_only",
                "hardware41_calibration_input_forbidden": True,
                "blind_input_forbidden": True,
                "full_frame_crop_forbidden": True,
                "no_proposal_full_frame_crop_created": False,
                "production_authorization": False,
            },
            "counts": counts,
            "proposal_stats": dict(sorted(stats.items())),
            "artifact_hashes": artifact_hashes,
        }
        (stage / ARTIFACT_NAMES["lineage"]).write_text(
            _canonical_json(lineage, pretty=True),
            encoding="utf-8",
            newline="",
        )
        _ensure_new_output(output_dir)
        os.replace(stage, output_dir)
        result = dict(lineage)
        result["output_dir"] = str(output_dir)
        return result
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Mine runtime-top1 background proposals from trusted empty scenes"
    )
    parser.add_argument("--input-inventory", required=True, type=Path)
    parser.add_argument("--detector-model", required=True, type=Path)
    parser.add_argument("--inference-spec", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--minimum-teacher-confidence",
        type=float,
        default=DEFAULT_MINIMUM_TEACHER_CONFIDENCE,
    )
    parser.add_argument("--minimum-free-gb", type=float, default=0.0)
    parser.add_argument("--max-output-gb", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.minimum_free_gb < 0 or args.max_output_gb < 0:
        parser.error("storage guards must be non-negative")
    result = mine_operational_empty_scene_proposals(
        input_inventory=args.input_inventory,
        detector_model=args.detector_model,
        inference_spec=args.inference_spec,
        output_dir=args.output_dir,
        device=args.device,
        batch=args.batch,
        minimum_teacher_confidence=args.minimum_teacher_confidence,
        minimum_free_bytes=int(args.minimum_free_gb * GIB),
        max_output_bytes=int(args.max_output_gb * GIB),
        dry_run=args.dry_run,
    )
    print(_canonical_json(result, pretty=True), end="", flush=True)


if __name__ == "__main__":
    main()
