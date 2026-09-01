"""Measure historical-manifest replay drift without downstream authority.

This audit intentionally does not apply a numeric pass/fail tolerance.  It
reuses the v4 validator's byte and dataset contracts, replays every manifest
source from a private snapshot, and publishes one diagnostic-only JSON report.
It never modifies the manifest, crops, training inputs, or deployment state.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import tempfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

try:
    from scripts.prepare_proposal_verifier_dataset import (
        BACKGROUND_CLASS_ID,
        PredictedFrame,
        SourceRecord,
        _label_path,
        assign_proposal,
        bbox_iou,
        boxes_intersect,
        eager_initialize_cuda_context,
        expanded_clipped_bbox,
        iter_yolo_predictions,
    )
    from scripts.validate_v4_background_candidates import (
        EXPECTED_BACKGROUND_MARGIN,
        EXPECTED_CONFIDENCE,
        EXPECTED_CROP_SIZE,
        EXPECTED_LETTERBOX_FILL,
        EXPECTED_NEGATIVE_IOU,
        EXPECTED_NMS_IOU,
        EXPECTED_PADDING,
        EXPECTED_POSITIVE_IOU,
        _decode_source_path,
        _finite,
        _integer,
        _read_json,
        _read_manifest,
        _resolve_crop,
        _sha256_bytes,
        _sha256_file,
        _stable_read_bytes,
        _validate_dataset_contract,
        _verify_replay_snapshot_bytes,
        _write_snapshot,
        _xyxy,
        validate_rows,
    )
    from scripts.verifier_preprocessing_contract import padded_clipped_bbox
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from prepare_proposal_verifier_dataset import (  # type: ignore[no-redef]
        BACKGROUND_CLASS_ID,
        PredictedFrame,
        SourceRecord,
        _label_path,
        assign_proposal,
        bbox_iou,
        boxes_intersect,
        eager_initialize_cuda_context,
        expanded_clipped_bbox,
        iter_yolo_predictions,
    )
    from validate_v4_background_candidates import (  # type: ignore[no-redef]
        EXPECTED_BACKGROUND_MARGIN,
        EXPECTED_CONFIDENCE,
        EXPECTED_CROP_SIZE,
        EXPECTED_LETTERBOX_FILL,
        EXPECTED_NEGATIVE_IOU,
        EXPECTED_NMS_IOU,
        EXPECTED_PADDING,
        EXPECTED_POSITIVE_IOU,
        _decode_source_path,
        _finite,
        _integer,
        _read_json,
        _read_manifest,
        _resolve_crop,
        _sha256_bytes,
        _sha256_file,
        _stable_read_bytes,
        _validate_dataset_contract,
        _verify_replay_snapshot_bytes,
        _write_snapshot,
        _xyxy,
        validate_rows,
    )
    from verifier_preprocessing_contract import (  # type: ignore[no-redef]
        padded_clipped_bbox,
    )


ARTIFACT_ROLE = (
    "v4_historical_manifest_replay_drift_diagnostic_only_"
    "not_lineage_training_blind_or_deploy_authority"
)
REPORT_SCHEMA_VERSION = 1
MAX_EXAMPLES = 25
PredictionProvider = Callable[[Sequence[SourceRecord]], Iterable[PredictedFrame]]


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _publish_exclusive(path: Path, content: bytes) -> None:
    """Publish one JSON file atomically and never overwrite an existing path."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing audit report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _quantiles(values: Sequence[float | int]) -> dict[str, object]:
    """Return deterministic nearest-rank quantiles without judging them."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "method": "nearest_rank"}

    def select(q: float) -> float:
        if q <= 0:
            return ordered[0]
        index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
        return ordered[index]

    return {
        "count": len(ordered),
        "method": "nearest_rank",
        "min": ordered[0],
        "p50": select(0.50),
        "p90": select(0.90),
        "p95": select(0.95),
        "p99": select(0.99),
        "max": ordered[-1],
    }


def _bounded_examples(
    examples: Sequence[Mapping[str, object]],
    *,
    metric: str,
    reverse: bool,
) -> list[dict[str, object]]:
    return [
        dict(item)
        for item in sorted(
            examples,
            key=lambda item: (
                -float(item[metric]) if reverse else float(item[metric]),
                int(item.get("manifest_row", 0)),
                str(item.get("source_id", "")),
            ),
        )[:MAX_EXAMPLES]
    ]


def _nearest_threshold_examples(
    examples: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        dict(item)
        for item in sorted(
            examples,
            key=lambda item: (
                float(item["distance"]),
                int(item.get("manifest_row", 0)),
                str(item.get("source_id", "")),
            ),
        )[:MAX_EXAMPLES]
    ]


def _runtime_metadata(*, authoritative: bool, device: str, batch: int) -> dict[str, object]:
    """Describe the replay runtime without granting the report authority."""

    metadata: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "requested_device": device,
        "requested_batch": batch,
    }
    for distribution in ("torch", "ultralytics"):
        try:
            metadata[f"{distribution}_version"] = importlib.metadata.version(
                distribution
            )
        except importlib.metadata.PackageNotFoundError:
            metadata[f"{distribution}_version"] = None
    if not authoritative:
        metadata["cuda_observed"] = False
        metadata["runtime_identity_authoritative"] = False
        return metadata
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        properties = torch.cuda.get_device_properties(0) if cuda_available else None
        device_uuid = getattr(properties, "uuid", None)
        metadata.update(
            {
                "cuda_observed": cuda_available,
                "torch_cuda_version": torch.version.cuda,
                "cudnn_version": (
                    torch.backends.cudnn.version() if cuda_available else None
                ),
                "cuda_device_count": int(torch.cuda.device_count()),
                "cuda_device_name": (
                    torch.cuda.get_device_name(0) if cuda_available else None
                ),
                "cuda_device_uuid": (
                    str(device_uuid)
                    if properties is not None and device_uuid is not None
                    else None
                ),
                "cuda_total_memory_bytes": (
                    int(properties.total_memory)
                    if properties is not None
                    else None
                ),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "deterministic_algorithms_enabled": bool(
                    torch.are_deterministic_algorithms_enabled()
                ),
                "runtime_identity_authoritative": cuda_available,
            }
        )
    except Exception as error:  # Metadata must not hide a completed drift replay.
        metadata.update(
            {
                "cuda_observed": False,
                "runtime_identity_authoritative": False,
                "runtime_metadata_error": f"{type(error).__name__}: {error}",
            }
        )
    return metadata


def _loaded_code_bindings() -> tuple[dict[str, str], dict[Path, bytes]]:
    """Bind every local module that defines the audit's executed semantics."""

    script_dir = Path(__file__).resolve().parent
    paths = {
        "audit_v4_detector_replay_drift.py": Path(__file__).resolve(),
        "prepare_proposal_verifier_dataset.py": (
            script_dir / "prepare_proposal_verifier_dataset.py"
        ),
        "validate_v4_background_candidates.py": (
            script_dir / "validate_v4_background_candidates.py"
        ),
        "verifier_preprocessing_contract.py": (
            script_dir / "verifier_preprocessing_contract.py"
        ),
    }
    contents = {
        path: _stable_read_bytes(path, description=f"loaded code {name}")
        for name, path in paths.items()
    }
    return (
        {
            name: _sha256_bytes(contents[path])
            for name, path in sorted(paths.items())
        },
        contents,
    )


def _verify_loaded_code_bindings(
    bindings: Mapping[str, str],
    original_contents: Mapping[Path, bytes],
) -> None:
    for path, original in original_contents.items():
        current = _stable_read_bytes(
            path, description=f"loaded code final rehash {path.name}"
        )
        if current != original or _sha256_bytes(current) != bindings[path.name]:
            raise ValueError(f"loaded code changed during replay: {path.name}")


def _strict_background_state(
    material: int | None,
    *,
    ground_truth_bbox: Sequence[float] | None,
    crop_bounds: Sequence[int] | None,
    width: int,
    height: int,
) -> str:
    if material is None:
        return "ambiguous_assignment"
    if material != BACKGROUND_CLASS_ID:
        return "not_background"
    if crop_bounds is None:
        return "invalid_crop"
    if ground_truth_bbox is None:
        return "accepted_empty_source"
    exclusion = expanded_clipped_bbox(
        ground_truth_bbox,
        width=width,
        height=height,
        margin=EXPECTED_BACKGROUND_MARGIN,
    )
    return (
        "rejected_gt_intersection"
        if boxes_intersect(crop_bounds, exclusion)
        else "accepted_zero_intersection"
    )


def _row_identity(row_index: int, row: Mapping[str, str]) -> dict[str, object]:
    return {
        "manifest_row": row_index + 2,
        "source_id": row["source_id"].strip().casefold(),
        "split": row["split"].strip(),
        "declared_material": _integer(
            row["material"], field=f"manifest row {row_index + 2} material"
        ),
    }


def _audit_replay(
    rows: Sequence[Mapping[str, str]],
    records: Sequence[SourceRecord],
    prediction_provider: PredictionProvider,
) -> dict[str, object]:
    expected = {
        record.path.resolve(): (index, row, record)
        for index, (row, record) in enumerate(zip(rows, records, strict=True))
    }
    observed: set[Path] = set()
    hard_counts: Counter[str] = Counter()
    hard_examples: dict[str, list[dict[str, object]]] = defaultdict(list)
    mismatch_rows: set[int] = set()
    proposal_counts: list[int] = []
    proposal_count_histogram: Counter[int] = Counter()
    confidence_signed_drifts: list[float] = []
    confidence_drifts: list[float] = []
    confidence_examples: list[dict[str, object]] = []
    bbox_max_drifts: list[float] = []
    bbox_ious: list[float] = []
    bbox_examples: list[dict[str, object]] = []
    crop_bound_drifts: list[int] = []
    crop_examples: list[dict[str, object]] = []
    assignment_transitions: Counter[str] = Counter()
    strict_transitions: Counter[str] = Counter()
    threshold_crossings: Counter[str] = Counter()
    confidence_near: list[dict[str, object]] = []
    iou_near_010: list[dict[str, object]] = []
    iou_near_050: list[dict[str, object]] = []
    stratum_rows: dict[str, set[int]] = defaultdict(set)
    stratum_confidence: dict[str, list[float]] = defaultdict(list)
    stratum_bbox: dict[str, list[float]] = defaultdict(list)
    stratum_crop: dict[str, list[int]] = defaultdict(list)
    full_metric_rows: set[int] = set()
    metric_skip_reasons: Counter[str] = Counter()
    for row_index, row in enumerate(rows):
        stratum_rows[f"{row['split'].strip()}/{row['category'].strip()}"].add(
            row_index
        )

    def hard(name: str, row_index: int, detail: Mapping[str, object]) -> None:
        hard_counts[name] += 1
        mismatch_rows.add(row_index)
        if len(hard_examples[name]) < MAX_EXAMPLES:
            hard_examples[name].append(dict(detail))

    for frame in prediction_provider(records):
        resolved = frame.source.path.resolve()
        entry = expected.get(resolved)
        if entry is None:
            hard_counts["unexpected_source"] += 1
            if len(hard_examples["unexpected_source"]) < MAX_EXAMPLES:
                hard_examples["unexpected_source"].append(
                    {"snapshot_name": frame.source.path.name}
                )
            continue
        row_index, row, record = entry
        identity = _row_identity(row_index, row)
        stratum = f"{row['split'].strip()}/{row['category'].strip()}"
        if resolved in observed:
            hard("duplicate_source", row_index, identity)
            continue
        observed.add(resolved)
        declared_width = _integer(row["source_width"], field="source_width")
        declared_height = _integer(row["source_height"], field="source_height")
        if frame.width != declared_width:
            hard(
                "width_changed",
                row_index,
                {**identity, "declared": declared_width, "replayed": frame.width},
            )
        if frame.height != declared_height:
            hard(
                "height_changed",
                row_index,
                {**identity, "declared": declared_height, "replayed": frame.height},
            )

        proposal_count = len(frame.proposals)
        proposal_counts.append(proposal_count)
        proposal_count_histogram[proposal_count] += 1
        if proposal_count == 0:
            hard("no_proposal", row_index, identity)
            metric_skip_reasons["no_proposal"] += 1
            continue
        for proposal_index, proposal in enumerate(frame.proposals):
            if not math.isfinite(proposal.confidence):
                hard(
                    "nonfinite_proposal_confidence",
                    row_index,
                    {**identity, "proposal_index": proposal_index},
                )
            try:
                coordinates = tuple(float(value) for value in proposal.bbox)
                valid_bbox = (
                    len(coordinates) == 4
                    and all(math.isfinite(value) for value in coordinates)
                    and coordinates[2] > coordinates[0]
                    and coordinates[3] > coordinates[1]
                )
            except (TypeError, ValueError):
                valid_bbox = False
            if not valid_bbox:
                hard(
                    "invalid_proposal_bbox",
                    row_index,
                    {
                        **identity,
                        "proposal_index": proposal_index,
                        "bbox": list(proposal.bbox),
                    },
                )
        finite = [
            (index, proposal)
            for index, proposal in enumerate(frame.proposals)
            if math.isfinite(proposal.confidence)
        ]
        replay_peak = max((proposal.confidence for _, proposal in finite), default=None)
        declared_confidence = _finite(row["predicted_confidence"], field="predicted_confidence")
        if replay_peak is not None:
            confidence_near.append(
                {
                    **identity,
                    "threshold": EXPECTED_CONFIDENCE,
                    "declared": declared_confidence,
                    "replayed_peak": replay_peak,
                    "distance": min(
                        abs(declared_confidence - EXPECTED_CONFIDENCE),
                        abs(replay_peak - EXPECTED_CONFIDENCE),
                    ),
                }
            )
            if (declared_confidence >= EXPECTED_CONFIDENCE) != (
                replay_peak >= EXPECTED_CONFIDENCE
            ):
                threshold_crossings["confidence_0.1"] += 1
        eligible = [
            (index, proposal)
            for index, proposal in finite
            if proposal.confidence >= EXPECTED_CONFIDENCE
        ]
        if not eligible:
            hard(
                "no_eligible_proposal",
                row_index,
                {**identity, "replayed_peak_confidence": replay_peak},
            )
            metric_skip_reasons["no_eligible_proposal"] += 1
            continue
        replay_index, replay = max(
            eligible, key=lambda item: (item[1].confidence, -item[0])
        )
        declared_index = _integer(row["proposal_index"], field="proposal_index")
        if replay_index != declared_index:
            hard(
                "top1_index_changed",
                row_index,
                {**identity, "declared": declared_index, "replayed": replay_index},
            )
        declared_class = _integer(row["predicted_class_id"], field="predicted_class_id")
        if replay.class_id != declared_class:
            hard(
                "class_changed",
                row_index,
                {**identity, "declared": declared_class, "replayed": replay.class_id},
            )
        declared_class_name = row["predicted_class_name"].strip()
        if replay.class_name != declared_class_name:
            hard(
                "class_name_changed",
                row_index,
                {
                    **identity,
                    "declared": declared_class_name,
                    "replayed": replay.class_name,
                },
            )

        confidence_signed_drift = replay.confidence - declared_confidence
        confidence_drift = abs(confidence_signed_drift)
        confidence_signed_drifts.append(confidence_signed_drift)
        confidence_drifts.append(confidence_drift)
        stratum_confidence[stratum].append(confidence_drift)
        confidence_examples.append(
            {
                **identity,
                "declared": declared_confidence,
                "replayed": replay.confidence,
                "signed_delta": confidence_signed_drift,
                "abs_drift": confidence_drift,
            }
        )
        declared_bbox = _xyxy(row, "predicted_bbox", location=f"row {row_index + 2}")
        try:
            replay_bbox = tuple(float(value) for value in replay.bbox)
            bbox_overlap = bbox_iou(declared_bbox, replay_bbox)
            bbox_max_drift = max(
                abs(left - right)
                for left, right in zip(declared_bbox, replay_bbox, strict=True)
            )
            replay_crop = padded_clipped_bbox(
                replay_bbox,
                width=frame.width,
                height=frame.height,
                padding=EXPECTED_PADDING,
            )
        except (TypeError, ValueError) as error:
            hard(
                "invalid_replayed_bbox",
                row_index,
                {**identity, "error": str(error), "bbox": list(replay.bbox)},
            )
            metric_skip_reasons["invalid_selected_bbox"] += 1
            continue
        bbox_max_drifts.append(bbox_max_drift)
        bbox_ious.append(bbox_overlap)
        stratum_bbox[stratum].append(bbox_max_drift)
        bbox_examples.append(
            {
                **identity,
                "declared_bbox": list(declared_bbox),
                "replayed_bbox": list(replay_bbox),
                "bbox_max_abs_drift": bbox_max_drift,
                "bbox_iou": bbox_overlap,
            }
        )
        declared_crop = tuple(
            _integer(row[f"crop_{axis}"], field=f"crop_{axis}")
            for axis in ("x1", "y1", "x2", "y2")
        )
        crop_drift = max(
            abs(left - right)
            for left, right in zip(declared_crop, replay_crop, strict=True)
        )
        crop_bound_drifts.append(crop_drift)
        stratum_crop[stratum].append(crop_drift)
        crop_examples.append(
            {
                **identity,
                "declared_crop": list(declared_crop),
                "replayed_crop": list(replay_crop),
                "crop_bounds_max_abs_drift": crop_drift,
            }
        )
        if replay_crop != declared_crop:
            hard(
                "crop_bounds_changed",
                row_index,
                crop_examples[-1],
            )

        ground_truth = record.ground_truth
        replay_gt_bbox = (
            ground_truth.xyxy(frame.width, frame.height)
            if ground_truth is not None
            else None
        )
        replay_material, replay_iou, replay_assignment = assign_proposal(
            replay_bbox,
            gt_bbox=replay_gt_bbox,
            gt_class_id=ground_truth.class_id if ground_truth is not None else None,
            positive_iou=EXPECTED_POSITIVE_IOU,
            negative_iou=EXPECTED_NEGATIVE_IOU,
        )
        declared_material = _integer(row["material"], field="material")
        declared_assignment = row["assignment"].strip()
        assignment_transitions[
            f"{declared_material}/{declared_assignment}"
            f"->{replay_material}/{replay_assignment}"
        ] += 1
        if replay_material != declared_material:
            hard(
                "assignment_material_changed",
                row_index,
                {
                    **identity,
                    "declared": declared_material,
                    "replayed": replay_material,
                    "replayed_iou": replay_iou,
                },
            )
        if replay_assignment != declared_assignment:
            hard(
                "assignment_reason_changed",
                row_index,
                {
                    **identity,
                    "declared": declared_assignment,
                    "replayed": replay_assignment,
                    "replayed_iou": replay_iou,
                },
            )
        declared_iou = _finite(row["matched_iou"], field="matched_iou")
        for threshold, bucket in (
            (EXPECTED_NEGATIVE_IOU, iou_near_010),
            (EXPECTED_POSITIVE_IOU, iou_near_050),
        ):
            bucket.append(
                {
                    **identity,
                    "threshold": threshold,
                    "declared": declared_iou,
                    "replayed": replay_iou,
                    "distance": min(
                        abs(declared_iou - threshold),
                        abs(replay_iou - threshold),
                    ),
                }
            )
            if threshold == EXPECTED_NEGATIVE_IOU:
                crossed = (declared_iou <= threshold) != (replay_iou <= threshold)
            else:
                crossed = (declared_iou >= threshold) != (replay_iou >= threshold)
            if crossed:
                threshold_crossings[f"assignment_iou_{threshold:.1f}"] += 1

        declared_gt_bbox = (
            ground_truth.xyxy(declared_width, declared_height)
            if ground_truth is not None
            else None
        )
        declared_state = _strict_background_state(
            declared_material,
            ground_truth_bbox=declared_gt_bbox,
            crop_bounds=declared_crop,
            width=declared_width,
            height=declared_height,
        )
        replay_state = _strict_background_state(
            replay_material,
            ground_truth_bbox=replay_gt_bbox,
            crop_bounds=replay_crop,
            width=frame.width,
            height=frame.height,
        )
        strict_transitions[f"{declared_state}->{replay_state}"] += 1
        if replay_state != declared_state:
            hard(
                "strict_zero_intersection_decision_changed",
                row_index,
                {**identity, "declared": declared_state, "replayed": replay_state},
            )
        full_metric_rows.add(row_index)

    missing = sorted(set(expected) - observed, key=lambda path: expected[path][0])
    for path in missing:
        row_index, row, _ = expected[path]
        hard("omitted_source", row_index, _row_identity(row_index, row))

    stratum_summary = {
        stratum: {
            "rows": len(indexes),
            "hard_semantic_mismatch_sources": len(indexes & mismatch_rows),
            "full_metric_coverage_rows": len(indexes & full_metric_rows),
            "confidence_abs_drift": _quantiles(stratum_confidence[stratum]),
            "bbox_max_abs_drift": _quantiles(stratum_bbox[stratum]),
            "crop_bounds_max_abs_drift": _quantiles(stratum_crop[stratum]),
        }
        for stratum, indexes in sorted(stratum_rows.items())
    }
    numeric_drift_detected = any(value != 0.0 for value in confidence_drifts) or any(
        value != 0.0 for value in bbox_max_drifts
    )
    incomplete_counts = {
        name: hard_counts.get(name, 0)
        for name in ("unexpected_source", "duplicate_source", "omitted_source")
    }
    source_replay_complete = (
        len(observed) == len(expected)
        and not any(incomplete_counts.values())
    )
    return {
        "expected_sources": len(rows),
        "observed_unique_sources": len(observed),
        "completion": {
            "source_replay_complete": source_replay_complete,
            "full_metric_coverage": len(full_metric_rows) == len(rows),
            "full_metric_coverage_rows": len(full_metric_rows),
            "metric_skip_reasons": dict(sorted(metric_skip_reasons.items())),
            "source_completeness_errors": incomplete_counts,
        },
        "drift_detected": numeric_drift_detected or bool(hard_counts),
        "numeric_drift_detected": numeric_drift_detected,
        "contract_impacting_drift": bool(hard_counts),
        "hard_semantic_mismatch_sources": len(mismatch_rows),
        "hard_semantic_mismatch_counts": dict(sorted(hard_counts.items())),
        "hard_semantic_mismatch_examples": {
            name: sorted(
                examples,
                key=lambda item: (
                    int(item.get("manifest_row", 0)),
                    str(item.get("source_id", "")),
                ),
            )
            for name, examples in sorted(hard_examples.items())
        },
        "replayed_proposal_count": {
            "declared_count_available": False,
            "distribution": _quantiles(proposal_counts),
            "histogram": {
                str(count): occurrences
                for count, occurrences in sorted(proposal_count_histogram.items())
            },
        },
        "confidence_abs_drift": {
            "distribution": _quantiles(confidence_drifts),
            "max_examples": _bounded_examples(
                confidence_examples, metric="abs_drift", reverse=True
            ),
        },
        "confidence_signed_drift": {
            "distribution": _quantiles(confidence_signed_drifts),
        },
        "bbox_max_abs_drift": {
            "distribution": _quantiles(bbox_max_drifts),
            "max_examples": _bounded_examples(
                bbox_examples, metric="bbox_max_abs_drift", reverse=True
            ),
        },
        "bbox_iou": {
            "distribution": _quantiles(bbox_ious),
            "lowest_examples": _bounded_examples(
                bbox_examples, metric="bbox_iou", reverse=False
            ),
        },
        "declared_vs_replayed_crop_bounds": {
            "max_abs_drift_distribution": _quantiles(crop_bound_drifts),
            "max_examples": _bounded_examples(
                crop_examples,
                metric="crop_bounds_max_abs_drift",
                reverse=True,
            ),
        },
        "assignment_transition_counts": dict(sorted(assignment_transitions.items())),
        "strict_zero_intersection_transition_counts": dict(
            sorted(strict_transitions.items())
        ),
        "fixed_threshold_diagnostics": {
            "crossing_counts": dict(sorted(threshold_crossings.items())),
            "confidence_0.1_nearest_examples": _nearest_threshold_examples(
                confidence_near
            ),
            "assignment_iou_0.1_nearest_examples": _nearest_threshold_examples(
                iou_near_010
            ),
            "assignment_iou_0.5_nearest_examples": _nearest_threshold_examples(
                iou_near_050
            ),
        },
        "strata": stratum_summary,
    }


def _artifact_binding_sha256(rows: Sequence[Mapping[str, str]]) -> str:
    payload = [
        {
            "manifest_row": index + 2,
            "source_sha256": row["source_sha256"],
            "source_annotation_sha256": row["source_annotation_sha256"],
            "crop_sha256": row["image_sha256"],
        }
        for index, row in enumerate(rows)
    ]
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _verify_original_source_label_crop_bytes(
    manifest: Path,
    rows: Sequence[Mapping[str, str]],
) -> None:
    for index, row in enumerate(rows, start=2):
        source = _decode_source_path(row["source_path_b64"], location=f"row {index}")
        label = _label_path(source)
        crop = _resolve_crop(manifest, row["filepath"], location=f"row {index}")
        checks = (
            (source, row["source_sha256"], "source"),
            (label, row["source_annotation_sha256"], "source label"),
            (crop, row["image_sha256"], "crop"),
        )
        for path, expected, description in checks:
            content = _stable_read_bytes(
                path, description=f"row {index} original {description} final rehash"
            )
            if _sha256_bytes(content) != expected:
                raise ValueError(
                    f"row {index}: original {description} changed during replay"
                )


def audit_detector_replay_drift(
    *,
    input_manifest: Path,
    dataset_info: Path,
    detector_model: Path,
    inference_spec: Path,
    output_report: Path,
    prediction_provider: PredictionProvider | None = None,
) -> dict[str, object]:
    if output_report.exists():
        raise FileExistsError(
            f"refusing to overwrite existing audit report: {output_report}"
        )
    if output_report.suffix.lower() != ".json":
        raise ValueError("output report must use the .json extension")
    for path in (input_manifest, dataset_info, detector_model, inference_spec):
        if not path.is_file():
            raise FileNotFoundError(f"required input does not exist: {path}")
    if dataset_info.parent.resolve() != input_manifest.parent.resolve():
        raise ValueError("dataset_info must be adjacent to its source manifest")
    if output_report.parent.resolve() == input_manifest.parent.resolve():
        raise ValueError(
            "diagnostic output report must use a separate audit directory"
        )

    fields, rows, manifest_bytes = _read_manifest(input_manifest)
    del fields
    code_bindings, code_contents = _loaded_code_bindings()
    info, info_bytes = _read_json(dataset_info, description="dataset info")
    spec, spec_bytes = _read_json(inference_spec, description="inference spec")
    _validate_dataset_contract(info, spec)
    declared_model = str(info.get("model", "")).strip()
    if not declared_model or Path(declared_model).resolve() != detector_model.resolve():
        raise ValueError("dataset_info model path does not match supplied detector model")
    declared_manifest = str(info.get("manifest", "")).strip()
    if not declared_manifest or Path(declared_manifest).resolve() != input_manifest.resolve():
        raise ValueError("dataset_info manifest path does not match supplied manifest")
    detector_spec = spec.get("detector")
    if not isinstance(detector_spec, Mapping):
        raise ValueError("inference spec is missing detector")
    model_reference = str(detector_spec.get("model_reference", "")).strip()
    if not model_reference or Path(model_reference).name != detector_model.name:
        raise ValueError("inference spec model reference differs from detector model")
    inference = info.get("inference")
    assert isinstance(inference, Mapping)
    device = str(inference.get("device", "")).strip()
    batch = _integer(inference.get("batch"), field="inference.batch")
    if batch <= 0:
        raise ValueError("dataset_info inference.batch must be positive")
    authoritative = prediction_provider is None
    if authoritative:
        if device.strip().lower() not in {"0", "cuda", "cuda:0"}:
            raise ValueError(
                "authoritative diagnostic CLI requires logical CUDA device 0"
            )

    with ExitStack() as stack:
        accelerator_guard = (
            eager_initialize_cuda_context(device) if authoritative else None
        )
        runtime = _runtime_metadata(
            authoritative=authoritative,
            device=device,
            batch=batch,
        )
        if authoritative and (
            accelerator_guard is None
            or runtime.get("cuda_observed") is not True
            or runtime.get("runtime_identity_authoritative") is not True
        ):
            raise RuntimeError(
                "CUDA runtime identity was not verified before source/crop scan"
            )
        replay_root = Path(
            stack.enter_context(
                tempfile.TemporaryDirectory(
                    prefix=".v4-detector-drift-audit-",
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
        optional_bindings = {
            "model_sha256": detector_sha,
            "manifest_sha256": _sha256_bytes(manifest_bytes),
            "inference_spec_sha256": _sha256_bytes(spec_bytes),
        }
        for field, actual in optional_bindings.items():
            declared = str(info.get(field, "")).strip().casefold()
            if declared and declared != actual:
                raise ValueError(f"dataset_info {field} conflicts with supplied artifact")

        validated, counts, replay_records = validate_rows(
            input_manifest,
            rows,
            detector_sha256=detector_sha,
            spec_sha256=_sha256_bytes(spec_bytes),
            replay_snapshot_dir=replay_root / "sources",
        )
        assert replay_records is not None
        if authoritative:

            def frozen_yolo_provider(
                records: Sequence[SourceRecord],
            ) -> Iterable[PredictedFrame]:
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
            provider_kind = "custom_test_provider_non_authoritative"

        replay = _audit_replay(validated, replay_records, replay_provider)
        _verify_replay_snapshot_bytes(replay_records, validated)
        _verify_original_source_label_crop_bytes(input_manifest, validated)
        if _sha256_file(detector_snapshot) != detector_sha:
            raise ValueError("detector replay snapshot changed during audit")
        detector_end = _stable_read_bytes(
            detector_model, description="detector model final rehash"
        )
        if _sha256_bytes(detector_end) != detector_sha:
            raise ValueError("original detector model changed during audit")
        immutable_inputs = (
            (input_manifest, manifest_bytes, "input manifest"),
            (dataset_info, info_bytes, "dataset info"),
            (inference_spec, spec_bytes, "inference spec"),
        )
        for path, expected_bytes, description in immutable_inputs:
            current = _stable_read_bytes(
                path, description=f"{description} final rehash"
            )
            if current != expected_bytes:
                raise ValueError(f"{description} changed during replay")
        _verify_loaded_code_bindings(code_bindings, code_contents)

        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "artifact_role": ARTIFACT_ROLE,
            "completion": replay["completion"],
            "diagnostic_only": True,
            "ready_for_lineage_upgrade": False,
            "training_eligible": False,
            "blind_test_eligible": False,
            "production_deployment_authorized": False,
            "authority": {
                "provider_kind": provider_kind,
                "runtime_detector_executed": authoritative,
                "cuda_runtime_verified": (
                    authoritative and runtime.get("cuda_observed") is True
                ),
                "input_validator_report_bound": False,
                "input_authority": (
                    "source_manifest_contract_checked_but_not_validator_approved"
                ),
                "lineage_authority": False,
                "training_authority": False,
                "blind_test_authority": False,
                "deployment_authority": False,
            },
            "interpretation": {
                "numeric_tolerance_pass_fail_applied": False,
                "hard_semantic_mismatches_are_diagnostic_counts_only": True,
                "comparison_scope": (
                    "one current batch-N replay versus historical manifest top1; "
                    "does not isolate batch causality or retain historical non-top1"
                ),
                "historical_non_top1_available": False,
                "fixed_thresholds": {
                    "detector_confidence": EXPECTED_CONFIDENCE,
                    "assignment_negative_iou": EXPECTED_NEGATIVE_IOU,
                    "assignment_positive_iou": EXPECTED_POSITIVE_IOU,
                },
            },
            "static_contract": {
                "rows": len(validated),
                "counts": {
                    f"{split}/{category}": count
                    for (split, category), count in sorted(counts.items())
                },
                "crop_size": EXPECTED_CROP_SIZE,
                "crop_padding": EXPECTED_PADDING,
                "letterbox_fill": EXPECTED_LETTERBOX_FILL,
                "detector_confidence": EXPECTED_CONFIDENCE,
                "detector_nms_iou": EXPECTED_NMS_IOU,
                "requested_device": device,
                "requested_batch": batch,
                "cuda_client_initialized_before_source_crop_scan": (
                    accelerator_guard is not None
                ),
                "detector_snapshot_created": True,
                "detector_replay_used_unique_snapshot": authoritative,
                "source_and_label_replay_used_unique_snapshots": True,
                "crop_private_snapshot_used": False,
                "original_source_label_crop_rehashed_after_replay": True,
                "detector_and_loaded_code_reverified_after_replay": True,
            },
            "bindings": {
                "loaded_code_sha256": code_bindings,
                "input_manifest_sha256": _sha256_bytes(manifest_bytes),
                "dataset_info_sha256": _sha256_bytes(info_bytes),
                "detector_model_sha256": detector_sha,
                "inference_spec_sha256": _sha256_bytes(spec_bytes),
                "source_label_crop_binding_sha256": _artifact_binding_sha256(
                    validated
                ),
            },
            "runtime": runtime,
            "replay": replay,
        }
        _publish_exclusive(output_report, _json_bytes(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--dataset-info", required=True, type=Path)
    parser.add_argument("--detector-model", required=True, type=Path)
    parser.add_argument("--inference-spec", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    args = parser.parse_args()
    report = audit_detector_replay_drift(
        input_manifest=args.input_manifest,
        dataset_info=args.dataset_info,
        detector_model=args.detector_model,
        inference_spec=args.inference_spec,
        output_report=args.output_report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
