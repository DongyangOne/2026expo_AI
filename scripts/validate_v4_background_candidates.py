"""Freeze and validate v4 proposal crops before lineage upgrade or training.

The v4 background miner can deliberately take a detector proposal from a
single-object source frame when the *final padded verifier crop* is completely
outside the annotated object plus a safety margin.  Such a crop has two
different object counts:

* ``source_object_count`` describes the complete source frame;
* ``crop_object_count`` describes the verifier crop used for training.

This validator preserves that distinction and rescues an already generated
legacy-shaped proposal manifest without mutating it.  It also rejects missing
YOLO label files (missing annotation is not evidence of an empty scene),
recomputes every geometry decision, binds the detector/spec/source/crop bytes,
and publishes an immutable validated CSV plus a report.  The result remains a
development candidate and can never serve as blind-test truth or deployment
authorization.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np

try:
    from scripts.prepare_proposal_verifier_dataset import (
        BACKGROUND_CLASS_ID,
        CLASS_NAMES,
        GroundTruth,
        PredictedFrame,
        Proposal,
        SourceRecord,
        _label_path,
        _reject_operational_material_hold,
        assign_proposal,
        bbox_iou,
        boxes_intersect,
        eager_initialize_cuda_context,
        expanded_clipped_bbox,
        iter_yolo_predictions,
        parse_yolo_label_text,
    )
    from scripts.verifier_preprocessing_contract import (
        CONTRACT_VERSION,
        crop_and_letterbox_bgr,
        padded_clipped_bbox,
        validate_crop_preprocessing_spec,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from prepare_proposal_verifier_dataset import (  # type: ignore[no-redef]
        BACKGROUND_CLASS_ID,
        CLASS_NAMES,
        GroundTruth,
        PredictedFrame,
        Proposal,
        SourceRecord,
        _label_path,
        _reject_operational_material_hold,
        assign_proposal,
        bbox_iou,
        boxes_intersect,
        eager_initialize_cuda_context,
        expanded_clipped_bbox,
        iter_yolo_predictions,
        parse_yolo_label_text,
    )
    from verifier_preprocessing_contract import (  # type: ignore[no-redef]
        CONTRACT_VERSION,
        crop_and_letterbox_bgr,
        padded_clipped_bbox,
        validate_crop_preprocessing_spec,
    )


SCHEMA_VERSION = "proposal_verifier.v4.bgfix.v1"
AUTHORITATIVE_ARTIFACT_ROLE = (
    "v4_development_candidates_not_blind_or_deployment_authority"
)
RUNTIME_DIAGNOSTIC_ARTIFACT_ROLE = (
    "v4_runtime_replay_diagnostic_not_lineage_blind_or_deployment_authority"
)
CUSTOM_PROVIDER_ARTIFACT_ROLE = (
    "v4_custom_provider_diagnostics_not_lineage_blind_or_deployment_authority"
)
EXPECTED_BACKGROUND_POLICY = "strict-zero-intersection"
EXPECTED_BACKGROUND_MARGIN = 0.10
EXPECTED_SELECTION_MODE = "runtime-top1"
EXPECTED_CONFIDENCE = 0.10
EXPECTED_NMS_IOU = 0.70
EXPECTED_CROP_SIZE = 320
EXPECTED_PADDING = 0.08
EXPECTED_LETTERBOX_FILL = 114
EXPECTED_JPEG_QUALITY = 92
EXPECTED_POSITIVE_IOU = 0.50
EXPECTED_NEGATIVE_IOU = 0.10
REPLAY_CONFIDENCE_ABS_TOLERANCE = 1e-6
REPLAY_BBOX_ABS_TOLERANCE = 1e-4
OPERATIONAL_ANNOTATION_AUTHORITY = "vlm_teacher_pseudo_label_train_only"
AIHUB_ANNOTATION_AUTHORITY = "aihub_annotation_geometry_development_only"
OPERATIONAL_ORIGIN = "operational_capture_vlm_teacher"
ADDED_FIELDS = (
    "manifest_schema_version",
    "crop_object_count",
    "background_exclusion_policy",
    "background_gt_margin",
    "crop_transform_version",
    "detector_model_sha256",
    "inference_spec_sha256",
    "source_annotation_sha256",
    "source_sha256",
    "image_sha256",
    "blind_test_eligible",
    "ground_truth_authority",
)
PredictionProvider = Callable[[Sequence[SourceRecord]], Iterable[PredictedFrame]]


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_read_bytes(path: Path, *, description: str) -> bytes:
    """Read one file descriptor and reject replacement or mutation while reading."""

    with path.open("rb") as file:
        before = os.fstat(file.fileno())
        content = file.read()
        after = os.fstat(file.fileno())
    path_after = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    path_identity = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
    )
    if os.name != "nt":
        before_identity += (before.st_ctime_ns,)
        after_identity += (after.st_ctime_ns,)
        path_identity += (path_after.st_ctime_ns,)
    if before_identity != after_identity or after_identity != path_identity:
        raise ValueError(f"{description} changed while being read: {path}")
    if len(content) != before.st_size:
        raise ValueError(f"{description} size changed while being read: {path}")
    return content


def _write_snapshot(path: Path, content: bytes) -> None:
    """Create one private replay snapshot without permitting overwrite.

    These files are ephemeral same-process replay inputs, so closing the file is
    sufficient visibility and avoids two fsync calls per source on the QNAP.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        file.write(content)


@dataclass(frozen=True)
class _OperationalEvidence:
    root: Path
    records: dict[str, dict]
    files: dict[str, bytes]
    reader_source: Path
    reader_source_bytes: bytes
    output_paths: tuple[Path, ...]
    protected_roots: tuple[Path, ...]


def _operational_adapter():
    # Keep the existing AIHub-only path independent of this optional adapter.
    try:
        from scripts import build_operational_source_evidence as adapter
    except ModuleNotFoundError:
        import build_operational_source_evidence as adapter
    return adapter


def _no_symlink_components(path: Path, description: str) -> None:
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError(f"{description} must not contain symlinks")


def _operational_tree(root: Path) -> dict[str, bytes]:
    _no_symlink_components(root, "operational evidence directory")
    if not root.is_dir():
        raise ValueError("operational evidence directory is missing")
    files = {}
    for path in sorted(root.rglob("*")):
        _no_symlink_components(path, "operational evidence file")
        if path.is_dir():
            files[path.relative_to(root).as_posix() + "/"] = b""
            continue
        if not path.is_file():
            raise ValueError("operational evidence must contain only regular files")
        files[path.relative_to(root).as_posix()] = _stable_read_bytes(
            path, description="operational evidence file"
        )
    return files


def _load_operational_evidence(
    root: Path, info: Mapping[str, object], manifest_dir: Path,
    output_paths: Sequence[Path] = (),
) -> _OperationalEvidence:
    _reject_operational_material_hold(root)
    _no_symlink_components(root, "operational evidence directory")
    root = root.resolve(strict=True)
    if manifest_dir.resolve().is_relative_to(root):
        raise ValueError("validator output must not be inside operational evidence")
    _check_operational_output_paths(output_paths, (root,))
    adapter = _operational_adapter()
    reader_source = Path(adapter.__file__).resolve(strict=True)
    reader_source_bytes = _stable_read_bytes(reader_source, description="operational reader")
    files = _operational_tree(root)
    records = adapter.validate_source_evidence_bundle(root)
    if files != _operational_tree(root):
        raise ValueError("operational evidence changed during initial validation")
    if _stable_read_bytes(reader_source, description="operational reader") != reader_source_bytes:
        raise ValueError("operational evidence reader changed during validation")
    # The generation event must already have bound the same complete bundle.
    expected_binding = {
        "bundle_dir": root.as_posix(),
        "receipt_sha256": _sha256_bytes(files[adapter.FILES["receipt"]]),
        "index_sha256": _sha256_bytes(files[adapter.FILES["index"]]),
        "marker_sha256": _sha256_bytes(files[adapter.FILES["marker"]]),
    }
    if info.get("operational_source_evidence") != expected_binding:
        raise ValueError("dataset_info operational source evidence binding mismatch")
    receipt = json.loads(files[adapter.FILES["receipt"]])
    protected_roots = (root, *sorted({
        Path(item["path"]).parent.resolve()
        for name, item in receipt["inputs"].items()
        if name.startswith(("teacher_output_", "quality_"))
    }))
    _check_operational_output_paths(output_paths, protected_roots)
    indexed: dict[str, dict] = {}
    for record in records:
        sha = record["source_sha256"]
        if sha in indexed:
            raise ValueError("duplicate operational source evidence SHA")
        indexed[sha] = json.loads(_json_bytes(record))
    if not indexed:
        raise ValueError("operational source evidence contains no accepted sources")
    _reject_operational_material_hold(root)
    return _OperationalEvidence(root, indexed, files, reader_source, reader_source_bytes,
                                tuple(output_paths), protected_roots)


def _check_operational_output_paths(paths: Sequence[Path], roots: Sequence[Path]) -> None:
    for path in paths:
        _no_symlink_components(path, "operational validator output")
        if any(path.resolve(strict=False).is_relative_to(root) for root in roots):
            raise ValueError("validator output must not be inside operational evidence")


def _verify_operational_evidence(evidence: _OperationalEvidence) -> None:
    _reject_operational_material_hold(evidence.root)
    _check_operational_output_paths(evidence.output_paths, evidence.protected_roots)
    if _operational_tree(evidence.root) != evidence.files:
        raise ValueError("operational evidence changed after validation")
    if _stable_read_bytes(evidence.reader_source, description="operational reader") != evidence.reader_source_bytes:
        raise ValueError("operational evidence reader changed after validation")
    records = _operational_adapter().validate_source_evidence_bundle(evidence.root)
    current = {record["source_sha256"]: record for record in records}
    if len(current) != len(records) or current != evidence.records:
        raise ValueError("operational source evidence records changed after validation")
    for record in evidence.records.values():
        path = Path(record["source_filepath"])
        _no_symlink_components(path, "operational source image")
        if _sha256_bytes(_stable_read_bytes(path, description="operational source image")) != record["source_sha256"]:
            raise ValueError("operational source image changed after validation")
    if _operational_tree(evidence.root) != evidence.files:
        raise ValueError("operational evidence changed during revalidation")
    if _stable_read_bytes(evidence.reader_source, description="operational reader") != evidence.reader_source_bytes:
        raise ValueError("operational evidence reader changed during revalidation")
    _reject_operational_material_hold(evidence.root)


def _row_annotation(
    row: Mapping[str, str], source_path: Path, *, location: str,
    operational_evidence: _OperationalEvidence | None,
) -> tuple[GroundTruth | None, bytes, str]:
    record = None
    if operational_evidence is not None:
        record = operational_evidence.records.get(row["source_id"].strip())
        if record is None:
            record = next((item for item in operational_evidence.records.values()
                           if Path(item["source_filepath"]).resolve() == source_path), None)
    declared_operational = (
        row.get("origin", "").strip() == OPERATIONAL_ORIGIN
        or row.get("annotation_authority", "").strip() == OPERATIONAL_ANNOTATION_AUTHORITY
        or any(row.get(field, "").strip() for field in (
            "auditor_sha256", "teacher_output_sha256", "localizer_output_sha256", "source_evidence_ref",
        ))
    )
    if record is None:
        if declared_operational:
            raise ValueError(f"{location}: verified operational source evidence is required")
        if row.get("annotation_authority", "").strip() not in {"", AIHUB_ANNOTATION_AUTHORITY}:
            raise ValueError(f"{location}: unsupported annotation authority")
        label_path = _label_path(source_path)
        if not label_path.is_file():
            raise ValueError(f"{location}: explicit YOLO label file is required; missing annotation is not an empty-scene ground truth")
        try:
            annotation_bytes = label_path.read_bytes()
            label_text = annotation_bytes.decode("utf-8")
        except (OSError, UnicodeError) as error:
            raise ValueError(f"{location}: YOLO label file is unreadable") from error
        ground_truth, reason = parse_yolo_label_text(label_text)
        if reason is not None:
            raise ValueError(f"{location}: unsupported YOLO annotation: {reason}")
        return ground_truth, annotation_bytes, AIHUB_ANNOTATION_AUTHORITY

    if row.get("split") != "training" or row.get("role") != "train" or row.get("fold") != "train":
        raise ValueError(f"{location}: operational pseudoannotations are train-only")
    if row.get("annotation_authority") != OPERATIONAL_ANNOTATION_AUTHORITY:
        raise ValueError(f"{location}: operational pseudoannotation cannot claim AIHub authority")
    for field in (
        "source_id", "source_sha256", "origin", "captured_at", "object_group", "capture_session",
        "teacher_output_sha256", "localizer_output_sha256", "auditor_sha256", "source_evidence_ref",
    ):
        if row.get(field) != str(record[field]):
            raise ValueError(f"{location}: operational {field} evidence mismatch")
    expected_path = Path(record["source_filepath"])
    _no_symlink_components(expected_path, "operational source image")
    if source_path != expected_path.resolve() or row.get("source_filepath") != expected_path.as_posix():
        raise ValueError(f"{location}: operational source path evidence mismatch")
    raw_source_path = Path(os.fsdecode(base64.urlsafe_b64decode(row["source_path_b64"].encode("ascii"))))
    if not raw_source_path.is_absolute():
        raise ValueError(f"{location}: operational source path must be absolute")
    _no_symlink_components(raw_source_path, "operational manifest source image")
    for field in ("material", "source_width", "source_height", "source_object_count"):
        if _integer(row.get(field), field=f"{location} {field}") != record[field]:
            raise ValueError(f"{location}: operational {field} evidence mismatch")
    if row.get("category") != record["category"] or record["material"] not in range(len(CLASS_NAMES)):
        raise ValueError(f"{location}: operational evidence permits material positives only")
    if any(row.get(field) != "-1" for field in ("dent", "label", "foreign_material")):
        raise ValueError(f"{location}: operational crop state targets must remain -1")
    if _integer(row.get("source_foreign_material"), field=f"{location} source_foreign_material") != record["foreign_material"]:
        raise ValueError(f"{location}: source foreign-material provenance mismatch")
    width, height = record["source_width"], record["source_height"]
    left, top, right, bottom = record["source_bbox_xyxy"]
    ground_truth = GroundTruth(record["material"], (
        (left + right) / (2 * width), (top + bottom) / (2 * height),
        (right - left) / width, (bottom - top) / height,
    ))
    assert operational_evidence is not None
    annotation_bytes = operational_evidence.files[record["source_evidence_ref"]]
    if _sha256_bytes(annotation_bytes) != record["auditor_sha256"]:
        raise ValueError(f"{location}: operational annotation bytes mismatch")
    return ground_truth, annotation_bytes, OPERATIONAL_ANNOTATION_AUTHORITY


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        result = int(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be an integer") from error
    return result


def _decode_source_path(value: str, *, location: str) -> Path:
    try:
        raw = base64.urlsafe_b64decode(value.encode("ascii"))
        return Path(os.fsdecode(raw)).resolve()
    except (UnicodeError, ValueError, binascii.Error) as error:
        raise ValueError(f"{location}: invalid source_path_b64") from error


def _read_json(path: Path, *, description: str) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid {description} JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {description} must be a JSON object")
    return value, raw


def _read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]], bytes]:
    raw = path.read_bytes()
    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    except (UnicodeError, csv.Error) as error:
        raise ValueError(f"{path}: invalid UTF-8 CSV") from error
    if not fields or any(not field for field in fields) or len(fields) != len(set(fields)):
        raise ValueError(f"{path}: manifest has empty or duplicate columns")
    required = {
        "filepath", "split", "source_id", "source_path_b64", "material",
        "category", "source_object_count", "assignment", "gt_class_id",
        "gt_class_name",
        "gt_bbox_x1", "gt_bbox_y1", "gt_bbox_x2", "gt_bbox_y2",
        "predicted_bbox_x1", "predicted_bbox_y1", "predicted_bbox_x2",
        "predicted_bbox_y2", "crop_x1", "crop_y1", "crop_x2", "crop_y2",
        "source_width", "source_height", "matched_iou", "proposal_index",
        "predicted_class_id", "predicted_class_name", "predicted_confidence",
        "crop_bytes",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"{path}: manifest is missing fields {missing}")
    if not rows:
        raise ValueError(f"{path}: manifest is empty")
    if any(None in row for row in rows):
        raise ValueError(f"{path}: manifest contains an unnamed extra CSV column")
    return fields, [
        {field: "" if row.get(field) is None else str(row[field]) for field in fields}
        for row in rows
    ], raw


def _validate_dataset_contract(info: Mapping[str, object], spec: Mapping[str, object]) -> None:
    policy = info.get("proposal_policy")
    inference = info.get("inference")
    crop = info.get("crop")
    if not isinstance(policy, Mapping) or not isinstance(inference, Mapping):
        raise ValueError("dataset_info is missing proposal_policy/inference")
    if not isinstance(crop, Mapping):
        raise ValueError("dataset_info is missing crop")
    expected = {
        "selection_mode": EXPECTED_SELECTION_MODE,
        "background_policy": EXPECTED_BACKGROUND_POLICY,
    }
    for field, value in expected.items():
        if policy.get(field) != value:
            raise ValueError(f"dataset_info proposal_policy.{field} must be {value!r}")
    if _finite(policy.get("background_gt_margin"), field="background_gt_margin") != EXPECTED_BACKGROUND_MARGIN:
        raise ValueError("dataset_info background_gt_margin must be 0.10")
    if _finite(inference.get("conf"), field="inference.conf") != EXPECTED_CONFIDENCE:
        raise ValueError("dataset_info inference.conf must be 0.10")
    if _finite(inference.get("nms_iou"), field="inference.nms_iou") != EXPECTED_NMS_IOU:
        raise ValueError("dataset_info inference.nms_iou must be 0.70")
    if _integer(inference.get("imgsz"), field="inference.imgsz") != 640:
        raise ValueError("dataset_info inference.imgsz must be 640")
    if _integer(crop.get("size"), field="crop.size") != EXPECTED_CROP_SIZE:
        raise ValueError("dataset_info crop.size must be 320")
    if _finite(crop.get("padding"), field="crop.padding") != EXPECTED_PADDING:
        raise ValueError("dataset_info crop.padding must be 0.08")
    if _integer(crop.get("jpeg_quality"), field="crop.jpeg_quality") != EXPECTED_JPEG_QUALITY:
        raise ValueError("dataset_info crop.jpeg_quality must be 92")
    assignment = info.get("assignment")
    if not isinstance(assignment, Mapping):
        raise ValueError("dataset_info is missing assignment")
    if (
        _finite(
            assignment.get("positive_iou_inclusive"),
            field="assignment.positive_iou_inclusive",
        )
        != EXPECTED_POSITIVE_IOU
    ):
        raise ValueError("dataset_info positive IoU must be 0.50")
    if (
        _finite(
            assignment.get("negative_iou_inclusive"),
            field="assignment.negative_iou_inclusive",
        )
        != EXPECTED_NEGATIVE_IOU
    ):
        raise ValueError("dataset_info negative IoU must be 0.10")
    if assignment.get("ambiguous_iou_skipped") is not True:
        raise ValueError("dataset_info must skip ambiguous IoU proposals")

    contract = validate_crop_preprocessing_spec(spec)
    if (
        contract.size != EXPECTED_CROP_SIZE
        or contract.padding_ratio != EXPECTED_PADDING
        or contract.fill != EXPECTED_LETTERBOX_FILL
    ):
        raise ValueError(
            "inference spec crop size/padding/fill does not match v4 generation"
        )
    detector = spec.get("detector")
    if not isinstance(detector, Mapping):
        raise ValueError("inference spec is missing detector")
    if _finite(detector.get("candidate_confidence"), field="detector confidence") != EXPECTED_CONFIDENCE:
        raise ValueError("inference spec detector confidence does not match v4 generation")
    if _finite(detector.get("nms_iou"), field="detector nms_iou") != EXPECTED_NMS_IOU:
        raise ValueError("inference spec detector nms_iou does not match v4 generation")
    if _integer(detector.get("input_size"), field="detector input_size") != 640:
        raise ValueError("inference spec detector input size must be 640")
    if detector.get("proposal_selection") != "highest_confidence_then_original_order":
        raise ValueError("inference spec proposal selection is not runtime top1")


def _xyxy(row: Mapping[str, str], prefix: str, *, location: str) -> tuple[float, float, float, float]:
    return tuple(
        _finite(row[f"{prefix}_{axis}"], field=f"{location} {prefix}_{axis}")
        for axis in ("x1", "y1", "x2", "y2")
    )  # type: ignore[return-value]


def _resolve_crop(manifest: Path, value: str, *, location: str) -> Path:
    raw = value.strip()
    if not raw:
        raise ValueError(f"{location}: filepath is empty")
    path = Path(raw)
    if not path.is_absolute():
        path = manifest.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{location}: crop does not exist: {path}")
    return path


def _replay_source_records(
    manifest_path: Path,
    rows: Sequence[Mapping[str, str]],
    operational_evidence: _OperationalEvidence | None = None,
) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    seen_paths: set[Path] = set()
    for line, row in enumerate(rows, start=2):
        location = f"{manifest_path}:{line}"
        source_path = _decode_source_path(row["source_path_b64"], location=location)
        if source_path in seen_paths:
            raise ValueError(
                f"{location}: runtime-top1 manifest contains duplicate source rows"
            )
        seen_paths.add(source_path)
        if not source_path.is_file():
            raise FileNotFoundError(
                f"{location}: source image does not exist: {source_path}"
            )
        ground_truth, _, _ = _row_annotation(
            row, source_path, location=location,
            operational_evidence=operational_evidence,
        )
        split = row["split"].strip()
        if split not in {"training", "validation"}:
            raise ValueError(f"{location}: split must be training or validation")
        records.append(
            SourceRecord(
                path=source_path,
                split=split,
                source_id=row["source_id"].strip(),
                ground_truth=ground_truth,
            )
        )
    return records


def validate_detector_replay(
    manifest_path: Path,
    rows: Sequence[Mapping[str, str]],
    *,
    prediction_provider: PredictionProvider,
    records: Sequence[SourceRecord] | None = None,
    provider_kind: str,
    runtime_detector_executed: bool,
    operational_evidence: _OperationalEvidence | None = None,
) -> dict[str, object]:
    """Replay frozen detector top1 and require every emitted proposal to match."""

    if records is None:
        records = _replay_source_records(manifest_path, rows, operational_evidence)
    elif len(records) != len(rows):
        raise ValueError("detector replay snapshot count differs from manifest rows")
    expected = {record.path.resolve(): row for record, row in zip(records, rows, strict=True)}
    observed: set[Path] = set()
    for frame in prediction_provider(records):
        source_path = frame.source.path.resolve()
        if source_path not in expected:
            raise ValueError(
                f"detector replay returned an unexpected source: {source_path}"
            )
        if source_path in observed:
            raise ValueError(
                f"detector replay returned duplicate results for: {source_path}"
            )
        observed.add(source_path)
        row = expected[source_path]
        location = f"{manifest_path} source {source_path}"
        if frame.width != _integer(row["source_width"], field=f"{location} width"):
            raise ValueError(f"{location}: detector replay width differs from manifest")
        if frame.height != _integer(row["source_height"], field=f"{location} height"):
            raise ValueError(f"{location}: detector replay height differs from manifest")
        eligible = [
            (index, proposal)
            for index, proposal in enumerate(frame.proposals)
            if math.isfinite(proposal.confidence)
            and proposal.confidence >= EXPECTED_CONFIDENCE
        ]
        if not eligible:
            raise ValueError(f"{location}: detector replay has no eligible proposal")
        replay_index, replay = max(
            eligible, key=lambda item: (item[1].confidence, -item[0])
        )
        if replay_index != _integer(
            row["proposal_index"], field=f"{location} proposal_index"
        ):
            raise ValueError(f"{location}: detector replay proposal_index mismatch")
        if replay.class_id != _integer(
            row["predicted_class_id"], field=f"{location} predicted_class_id"
        ):
            raise ValueError(f"{location}: detector replay class mismatch")
        if replay.class_name != row["predicted_class_name"].strip():
            raise ValueError(f"{location}: detector replay class name mismatch")
        declared_confidence = _finite(
            row["predicted_confidence"], field=f"{location} predicted_confidence"
        )
        if not math.isclose(
            replay.confidence,
            declared_confidence,
            rel_tol=0.0,
            abs_tol=REPLAY_CONFIDENCE_ABS_TOLERANCE,
        ):
            raise ValueError(f"{location}: detector replay confidence mismatch")
        declared_bbox = _xyxy(row, "predicted_bbox", location=location)
        if any(
            not math.isclose(
                left,
                right,
                rel_tol=0.0,
                abs_tol=REPLAY_BBOX_ABS_TOLERANCE,
            )
            for left, right in zip(replay.bbox, declared_bbox)
        ):
            raise ValueError(f"{location}: detector replay bbox mismatch")
    missing = sorted(str(path) for path in set(expected) - observed)
    if missing:
        raise ValueError(
            f"detector replay omitted {len(missing)} manifest source(s): {missing[:3]}"
        )
    return {
        "sources": len(records),
        "provider_kind": provider_kind,
        "runtime_detector_executed": runtime_detector_executed,
        "runtime_top1_replayed": runtime_detector_executed,
        "provided_top1_predictions_matched": True,
        "proposal_class_confidence_bbox_matched": True,
        "confidence_abs_tolerance": REPLAY_CONFIDENCE_ABS_TOLERANCE,
        "bbox_abs_tolerance": REPLAY_BBOX_ABS_TOLERANCE,
        "original_generation_event_cryptographically_attested": False,
        "authority": (
            "development_only_current_detector_reproduction"
            if runtime_detector_executed
            else "custom_provider_diagnostics_only"
        ),
    }


def validate_rows(
    manifest_path: Path,
    rows: Sequence[Mapping[str, str]],
    *,
    detector_sha256: str,
    spec_sha256: str,
    replay_snapshot_dir: Path | None = None,
    operational_evidence: _OperationalEvidence | None = None,
) -> tuple[list[dict[str, str]], Counter, list[SourceRecord] | None]:
    validated: list[dict[str, str]] = []
    counts: Counter = Counter()
    replay_records: list[SourceRecord] | None = (
        [] if replay_snapshot_dir is not None else None
    )
    for line, raw in enumerate(rows, start=2):
        location = f"{manifest_path}:{line}"
        row = dict(raw)
        material = _integer(row["material"], field=f"{location} material")
        if material not in range(BACKGROUND_CLASS_ID + 1):
            raise ValueError(f"{location}: material must be 0..{BACKGROUND_CLASS_ID}")
        expected_category = "background" if material == BACKGROUND_CLASS_ID else CLASS_NAMES[material]
        if row["category"].strip() != expected_category:
            raise ValueError(f"{location}: category does not match material")

        source_path = _decode_source_path(row["source_path_b64"], location=location)
        if not source_path.is_file():
            raise FileNotFoundError(f"{location}: source image does not exist: {source_path}")
        ground_truth, annotation_bytes, annotation_authority = _row_annotation(
            row, source_path, location=location,
            operational_evidence=operational_evidence,
        )
        source_object_count = 1 if ground_truth is not None else 0
        if _integer(row["source_object_count"], field=f"{location} source_object_count") != source_object_count:
            raise ValueError(f"{location}: source_object_count disagrees with explicit annotation")

        source_bytes = source_path.read_bytes()
        source_image = cv2.imdecode(
            np.frombuffer(source_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if source_image is None:
            raise ValueError(f"{location}: source image cannot be decoded")
        height, width = source_image.shape[:2]
        if _integer(row["source_width"], field=f"{location} source_width") != width:
            raise ValueError(f"{location}: source_width does not match source pixels")
        if _integer(row["source_height"], field=f"{location} source_height") != height:
            raise ValueError(f"{location}: source_height does not match source pixels")

        proposal_index = _integer(
            row["proposal_index"], field=f"{location} proposal_index"
        )
        if proposal_index < 0:
            raise ValueError(f"{location}: proposal_index must be non-negative")
        predicted_class_id = _integer(
            row["predicted_class_id"], field=f"{location} predicted_class_id"
        )
        if predicted_class_id not in range(len(CLASS_NAMES)):
            raise ValueError(f"{location}: predicted_class_id must be 0..8")
        if row["predicted_class_name"].strip() != CLASS_NAMES[predicted_class_id]:
            raise ValueError(
                f"{location}: predicted_class_name disagrees with predicted_class_id"
            )
        predicted_confidence = _finite(
            row["predicted_confidence"], field=f"{location} predicted_confidence"
        )
        if not EXPECTED_CONFIDENCE <= predicted_confidence <= 1.0:
            raise ValueError(
                f"{location}: predicted_confidence is outside the frozen detector range"
            )

        predicted_bbox = _xyxy(row, "predicted_bbox", location=location)
        expected_crop_image, expected_crop = crop_and_letterbox_bgr(
            source_image,
            predicted_bbox,
            padding=EXPECTED_PADDING,
            size=EXPECTED_CROP_SIZE,
            fill=EXPECTED_LETTERBOX_FILL,
        )
        recorded_crop = tuple(
            _integer(row[f"crop_{axis}"], field=f"{location} crop_{axis}")
            for axis in ("x1", "y1", "x2", "y2")
        )
        if recorded_crop != expected_crop:
            raise ValueError(f"{location}: crop bounds do not match frozen transform")

        if ground_truth is not None:
            gt_bbox = _xyxy(row, "gt_bbox", location=location)
            expected_gt_bbox = ground_truth.xyxy(width, height)
            if any(
                abs(left - right) > 1e-6
                for left, right in zip(gt_bbox, expected_gt_bbox)
            ):
                raise ValueError(f"{location}: GT bbox disagrees with annotation")
            if row["gt_class_id"].strip() != str(ground_truth.class_id):
                raise ValueError(f"{location}: gt_class_id disagrees with annotation")
            if row["gt_class_name"].strip() != CLASS_NAMES[ground_truth.class_id]:
                raise ValueError(f"{location}: gt_class_name disagrees with annotation")
            direct_iou = bbox_iou(predicted_bbox, expected_gt_bbox)
        else:
            gt_bbox = None
            direct_iou = 0.0
            gt_fields = (
                "gt_class_id", "gt_class_name", "gt_bbox_x1", "gt_bbox_y1",
                "gt_bbox_x2", "gt_bbox_y2",
            )
            if any(row[name].strip() for name in gt_fields):
                raise ValueError(f"{location}: empty annotation row contains GT evidence")

        assigned_material, assigned_iou, assigned_reason = assign_proposal(
            predicted_bbox,
            gt_bbox=gt_bbox,
            gt_class_id=ground_truth.class_id if ground_truth is not None else None,
            positive_iou=EXPECTED_POSITIVE_IOU,
            negative_iou=EXPECTED_NEGATIVE_IOU,
        )
        if not math.isclose(
            assigned_iou, direct_iou, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(f"{location}: frozen IoU implementations disagree")
        declared_iou = _finite(row["matched_iou"], field=f"{location} matched_iou")
        if not math.isclose(
            declared_iou, assigned_iou, rel_tol=0.0, abs_tol=1e-8
        ):
            raise ValueError(f"{location}: matched_iou disagrees with frozen geometry")
        if assigned_material is None:
            raise ValueError(f"{location}: ambiguous-IoU proposal must not be emitted")
        if assigned_material != material or row["assignment"].strip() != assigned_reason:
            raise ValueError(
                f"{location}: material/assignment disagrees with frozen IoU policy"
            )

        if material == BACKGROUND_CLASS_ID:
            crop_object_count = 0
            if ground_truth is not None:
                exclusion = expanded_clipped_bbox(
                    expected_gt_bbox,
                    width=width,
                    height=height,
                    margin=EXPECTED_BACKGROUND_MARGIN,
                )
                if boxes_intersect(recorded_crop, exclusion):
                    raise ValueError(
                        f"{location}: background crop intersects expanded ground truth"
                    )
        else:
            if ground_truth is None:
                raise ValueError(f"{location}: positive crop has no source ground truth")
            crop_object_count = 1

        crop_path = _resolve_crop(manifest_path, row["filepath"], location=location)
        crop_bytes = crop_path.read_bytes()
        crop_image = cv2.imdecode(
            np.frombuffer(crop_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if crop_image is None or crop_image.shape[:2] != (EXPECTED_CROP_SIZE, EXPECTED_CROP_SIZE):
            raise ValueError(f"{location}: crop is not a decodable 320x320 image")
        encoded_ok, expected_encoded = cv2.imencode(
            ".jpg",
            expected_crop_image,
            [cv2.IMWRITE_JPEG_QUALITY, EXPECTED_JPEG_QUALITY],
        )
        if not encoded_ok:
            raise RuntimeError(f"{location}: failed to encode deterministic crop")
        expected_crop_bytes = expected_encoded.tobytes()
        if crop_bytes != expected_crop_bytes:
            raise ValueError(
                f"{location}: crop bytes do not match the frozen source/bbox transform"
            )
        declared_crop_bytes = _integer(
            row["crop_bytes"], field=f"{location} crop_bytes"
        )
        if declared_crop_bytes != len(crop_bytes):
            raise ValueError(f"{location}: crop_bytes disagrees with actual crop")
        source_sha = _sha256_bytes(source_bytes)
        annotation_sha = _sha256_bytes(annotation_bytes)
        image_sha = _sha256_bytes(crop_bytes)
        if row["source_id"].strip().casefold() != source_sha:
            raise ValueError(
                f"{location}: source_id is not bound to the source image SHA-256"
            )
        additions = {
            "manifest_schema_version": SCHEMA_VERSION,
            "crop_object_count": str(crop_object_count),
            "background_exclusion_policy": EXPECTED_BACKGROUND_POLICY,
            "background_gt_margin": f"{EXPECTED_BACKGROUND_MARGIN:.2f}",
            "crop_transform_version": CONTRACT_VERSION,
            "detector_model_sha256": detector_sha256,
            "inference_spec_sha256": spec_sha256,
            "source_annotation_sha256": annotation_sha,
            "source_sha256": source_sha,
            "image_sha256": image_sha,
            "blind_test_eligible": "false",
            "ground_truth_authority": annotation_authority,
        }
        for field, value in additions.items():
            existing = row.get(field, "").strip()
            if existing and existing != value:
                raise ValueError(f"{location}: declared {field} conflicts with recomputed value")
            row[field] = value
        if replay_records is not None:
            source_suffix = source_path.suffix
            snapshot_stem = f"source-{line - 1:08d}-{source_sha[:16]}"
            snapshot_shard = source_sha[:2]
            snapshot_source = (
                replay_snapshot_dir
                / "images"
                / snapshot_shard
                / f"{snapshot_stem}{source_suffix}"
            )
            snapshot_label = (
                replay_snapshot_dir
                / "labels"
                / snapshot_shard
                / f"{snapshot_stem}.txt"
            )
            _write_snapshot(snapshot_source, source_bytes)
            _write_snapshot(snapshot_label, annotation_bytes)
            replay_records.append(
                SourceRecord(
                    path=snapshot_source,
                    split=row["split"].strip(),
                    source_id=source_sha,
                    ground_truth=ground_truth,
                )
            )
        counts[(row["split"].strip(), expected_category)] += 1
        validated.append(row)
    return validated, counts, replay_records


def _verify_replay_snapshot_bytes(
    records: Sequence[SourceRecord],
    rows: Sequence[Mapping[str, str]],
) -> None:
    """Verify that the private bytes replayed by YOLO stayed hash-bound."""

    if len(records) != len(rows):
        raise ValueError("detector replay snapshot count differs from validated rows")
    for record, row in zip(records, rows, strict=True):
        if _sha256_file(record.path) != row["source_sha256"]:
            raise ValueError(f"detector replay source snapshot changed: {record.path}")
        label_path = _label_path(record.path)
        if _sha256_file(label_path) != row["source_annotation_sha256"]:
            raise ValueError(f"detector replay label snapshot changed: {label_path}")


def _render_csv(fields: Sequence[str], rows: Sequence[Mapping[str, str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return output.getvalue().encode("utf-8")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _stage(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        return temporary
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _publish_pair(output: Path, output_bytes: bytes, report: Path, report_bytes: bytes) -> None:
    if output.resolve(strict=False) == report.resolve(strict=False):
        raise ValueError("validated manifest and report paths must differ")
    existing = [path for path in (output, report) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing artifacts: {existing}")
    output_temp = _stage(output, output_bytes)
    report_temp = _stage(report, report_bytes)
    output_digest = _sha256_bytes(output_bytes)
    published_output = False
    try:
        os.link(output_temp, output)
        published_output = True
        os.link(report_temp, report)
    except BaseException:
        if published_output and output.is_file() and _sha256_file(output) == output_digest:
            output.unlink()
        raise
    finally:
        output_temp.unlink(missing_ok=True)
        report_temp.unlink(missing_ok=True)


def validate_manifest(
    *,
    input_manifest: Path,
    dataset_info: Path,
    detector_model: Path,
    inference_spec: Path,
    output_manifest: Path,
    output_report: Path,
    prediction_provider: PredictionProvider | None = None,
    diagnostic_only: bool = False,
    operational_source_evidence_dir: Path | None = None,
) -> dict:
    paths = [input_manifest, dataset_info, detector_model, inference_spec]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"required input does not exist: {path}")
    if output_manifest.parent.resolve() != input_manifest.parent.resolve():
        raise ValueError("validated manifest must be adjacent to its source manifest")
    if dataset_info.parent.resolve() != input_manifest.parent.resolve():
        raise ValueError("dataset_info must be adjacent to its source manifest")
    fields, rows, manifest_bytes = _read_manifest(input_manifest)
    info, info_bytes = _read_json(dataset_info, description="dataset info")
    spec, spec_bytes = _read_json(inference_spec, description="inference spec")
    _validate_dataset_contract(info, spec)
    operational_evidence = None
    if operational_source_evidence_dir is not None:
        _reject_operational_material_hold(operational_source_evidence_dir)
        if (dataset_info.parent / "failed.json").exists():
            raise ValueError("operational dataset generation has a failure marker")
    elif "operational_source_evidence" in info:
        raise ValueError("dataset_info requires an operational source evidence directory")
    declared_model = str(info.get("model", "")).strip()
    if not declared_model:
        raise ValueError("dataset_info must declare the detector model path")
    if Path(declared_model).resolve() != detector_model.resolve():
        raise ValueError("dataset_info model path does not match supplied detector model")
    declared_manifest = str(info.get("manifest", "")).strip()
    if not declared_manifest:
        raise ValueError("dataset_info must declare the generated manifest path")
    if Path(declared_manifest).resolve() != input_manifest.resolve():
        raise ValueError("dataset_info manifest path does not match supplied manifest")
    spec_sha = _sha256_bytes(spec_bytes)
    detector_spec = spec.get("detector")
    assert isinstance(detector_spec, Mapping)
    model_reference = str(detector_spec.get("model_reference", "")).strip()
    if not model_reference or Path(model_reference).name != detector_model.name:
        raise ValueError(
            "inference spec detector model reference does not name the supplied model"
        )
    inference = info["inference"]
    assert isinstance(inference, Mapping)
    device = str(inference.get("device", "")).strip()
    authoritative = prediction_provider is None
    batch = 0
    if authoritative:
        if not device:
            raise ValueError("dataset_info inference.device is required for replay")
        batch = _integer(inference.get("batch"), field="inference.batch")
        if batch <= 0:
            raise ValueError("dataset_info inference.batch must be positive")

    with ExitStack() as stack:
        accelerator_guard = (
            eager_initialize_cuda_context(device) if authoritative else None
        )
        if operational_source_evidence_dir is not None:
            # The adapter rehashes large model inputs and scans source images;
            # reserve the same-process QNAP CUDA context before that work too.
            operational_evidence = _load_operational_evidence(
                operational_source_evidence_dir, info, input_manifest.parent,
                (output_manifest, output_report),
            )
        detector_snapshot: Path | None = None
        replay_snapshot_dir: Path | None = None
        if authoritative:
            replay_root = Path(
                stack.enter_context(
                    tempfile.TemporaryDirectory(
                        prefix=".v4-detector-replay-",
                        dir=input_manifest.parent,
                    )
                )
            )
            detector_content = _stable_read_bytes(
                detector_model, description="detector model"
            )
            detector_sha = _sha256_bytes(detector_content)
            detector_snapshot = replay_root / "detector" / detector_model.name
            _write_snapshot(detector_snapshot, detector_content)
            if _sha256_file(detector_snapshot) != detector_sha:
                raise ValueError("detector snapshot differs from stable source bytes")
            replay_snapshot_dir = replay_root / "sources"
        else:
            detector_sha = _sha256_file(detector_model)

        optional_bindings = {
            "model_sha256": detector_sha,
            "manifest_sha256": _sha256_bytes(manifest_bytes),
            "inference_spec_sha256": spec_sha,
        }
        for field, actual in optional_bindings.items():
            declared = str(info.get(field, "")).strip().casefold()
            if declared and declared != actual:
                raise ValueError(f"dataset_info {field} conflicts with supplied artifact")

        validated, counts, replay_records = validate_rows(
            input_manifest,
            rows,
            detector_sha256=detector_sha,
            spec_sha256=spec_sha,
            replay_snapshot_dir=replay_snapshot_dir,
            **({"operational_evidence": operational_evidence} if operational_evidence is not None else {}),
        )
        if authoritative:
            assert detector_snapshot is not None
            assert replay_records is not None

            def frozen_yolo_provider(
                records: Sequence[SourceRecord],
            ) -> Iterable[PredictedFrame]:
                # Retain the QNAP CUDA client through the complete replay.
                _ = accelerator_guard
                return iter_yolo_predictions(
                    records,
                    model_path=detector_snapshot,
                    device=device,
                    batch=batch,
                    imgsz=640,
                    conf=EXPECTED_CONFIDENCE,
                    nms_iou=EXPECTED_NMS_IOU,
                )

            replay_provider = frozen_yolo_provider
            provider_kind = "frozen_yolo_runtime"
        else:
            assert prediction_provider is not None
            replay_provider = prediction_provider
            provider_kind = "custom_non_authoritative"

        detector_replay = validate_detector_replay(
            input_manifest,
            rows,
            prediction_provider=replay_provider,
            records=replay_records,
            provider_kind=provider_kind,
            runtime_detector_executed=authoritative,
            **({"operational_evidence": operational_evidence} if operational_evidence is not None else {}),
        )
        if authoritative:
            assert replay_records is not None
            assert detector_snapshot is not None
            _verify_replay_snapshot_bytes(replay_records, validated)
            if _sha256_file(detector_snapshot) != detector_sha:
                raise ValueError("detector replay snapshot changed during validation")
            detector_original_end = _stable_read_bytes(
                detector_model, description="detector model final rehash"
            )
            if _sha256_bytes(detector_original_end) != detector_sha:
                raise ValueError("original detector model changed during validation")

        output_fields = [
            *fields,
            *(field for field in ADDED_FIELDS if field not in fields),
        ]
        output_bytes = _render_csv(output_fields, validated)
        report = {
            "schema_version": 1,
            "artifact_role": (
                (
                    RUNTIME_DIAGNOSTIC_ARTIFACT_ROLE
                    if diagnostic_only
                    else AUTHORITATIVE_ARTIFACT_ROLE
                )
                if authoritative
                else CUSTOM_PROVIDER_ARTIFACT_ROLE
            ),
            "ready_for_lineage_upgrade": authoritative and not diagnostic_only,
            **(
                {"lineage_execution_authorized": False}
                if diagnostic_only
                else {}
            ),
            "blind_test_eligible": False,
            "production_deployment_authorized": False,
            "rows": len(validated),
            "counts": {
                f"{split}/{category}": count
                for (split, category), count in sorted(counts.items())
            },
            "contract": {
                "manifest_schema_version": SCHEMA_VERSION,
                "background_policy": EXPECTED_BACKGROUND_POLICY,
                "background_gt_margin": EXPECTED_BACKGROUND_MARGIN,
                "explicit_label_file_required": True,
                "source_object_count_semantics": "complete_source_frame",
                "crop_object_count_semantics": "final_padded_verifier_crop",
                "visual_judge_still_required": True,
                "proposal_provenance": {
                    **detector_replay,
                    "cuda_client_initialized_before_source_crop_scan": (
                        accelerator_guard is not None
                    ),
                    "detector_artifact_bytes_bound": authoritative,
                    "detector_replay_used_unique_snapshot": authoritative,
                    "source_and_label_replay_used_unique_snapshots": authoritative,
                    "replay_snapshots_verified_after_inference": authoritative,
                    "original_detector_bytes_unchanged_through_validation": authoritative,
                    "inference_spec_bytes_bound": True,
                    "dataset_info_bytes_bound": True,
                    "source_bbox_crop_bytes_recomputed": True,
                    "production_or_blind_authority": False,
                },
            },
            "bindings": {
                "input_manifest_sha256": _sha256_bytes(manifest_bytes),
                "dataset_info_sha256": _sha256_bytes(info_bytes),
                "detector_model_sha256": detector_sha,
                "inference_spec_sha256": spec_sha,
                "validated_manifest_sha256": _sha256_bytes(output_bytes),
            },
        }
        report_bytes = _json_bytes(report)
        def recheck_operational_inputs() -> None:
            if operational_evidence is None:
                return
            if (dataset_info.parent / "failed.json").exists():
                raise ValueError("operational dataset generation has a failure marker")
            _verify_operational_evidence(operational_evidence)
            for path, expected in ((input_manifest, manifest_bytes), (dataset_info, info_bytes), (inference_spec, spec_bytes)):
                if _stable_read_bytes(path, description="operational validation input") != expected:
                    raise ValueError("operational validation input changed during replay")

        recheck_operational_inputs()
        _publish_pair(output_manifest, output_bytes, output_report, report_bytes)
        try:
            recheck_operational_inputs()
        except BaseException:
            # Only remove our exact newly published bytes, never unrelated data.
            for path, expected in ((output_manifest, output_bytes), (output_report, report_bytes)):
                if path.is_file() and not path.is_symlink() and path.read_bytes() == expected:
                    path.unlink()
            raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--dataset-info", required=True, type=Path)
    parser.add_argument("--detector-model", required=True, type=Path)
    parser.add_argument("--inference-spec", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--operational-source-evidence-dir", type=Path)
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help=(
            "execute the frozen runtime replay but publish evidence that cannot "
            "authorize lineage, blind evaluation, or deployment"
        ),
    )
    args = parser.parse_args()
    report = validate_manifest(
        input_manifest=args.input_manifest,
        dataset_info=args.dataset_info,
        detector_model=args.detector_model,
        inference_spec=args.inference_spec,
        output_manifest=args.output_manifest,
        output_report=args.output_report,
        diagnostic_only=args.diagnostic_only,
        operational_source_evidence_dir=args.operational_source_evidence_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
