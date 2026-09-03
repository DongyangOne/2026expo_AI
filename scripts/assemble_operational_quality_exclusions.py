"""Assemble teacher-verified operational capture-quality exclusions.

Only exact post-cutoff v3 teacher decisions that mark a capture unusable for
one of the four subjective capture-quality reasons are eligible.  The emitted
quality manifest remains SHA/reason-only and grants no training, calibration,
blind-test, or deployment authority.  Objective failures removed before the
teacher queue are intentionally out of scope because the current queue summary
does not retain per-source evidence for them.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

try:
    from scripts.build_operational_teacher_manifest import (
        ARTIFACT_NAMES,
        EMPTY_SCENE_INVENTORY_FIELDS,
        EXTREME_EXPOSURE_FRACTION,
        MANIFEST_FIELDS,
        MINIMUM_IMAGE_HEIGHT,
        MINIMUM_IMAGE_WIDTH,
        OPERATIONAL_CAPTURE_CUTOFF_KST,
        OPERATIONAL_CAPTURE_CUTOFF_UTC,
        OVEREXPOSED_LUMA_MIN,
        QUALITY_REASONS as TEACHER_DECISION_QUALITY_REASONS,
        TEACHER_LABEL_SCHEMA_VERSION,
        UNDEREXPOSED_LUMA_MAX,
        _teacher_consensus,
    )
    from scripts.build_v4_quality_exclusion_manifest import (
        QUALITY_EXCLUSION_CONTRACT,
        QUALITY_EXCLUSION_REASON_ALIASES,
        QUALITY_EXCLUSION_REASONS,
        _reject_symlink_components,
        _resolve_source,
        _stable_bytes,
        build_quality_exclusion_manifest,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from build_operational_teacher_manifest import (  # type: ignore[no-redef]
        ARTIFACT_NAMES,
        EMPTY_SCENE_INVENTORY_FIELDS,
        EXTREME_EXPOSURE_FRACTION,
        MANIFEST_FIELDS,
        MINIMUM_IMAGE_HEIGHT,
        MINIMUM_IMAGE_WIDTH,
        OPERATIONAL_CAPTURE_CUTOFF_KST,
        OPERATIONAL_CAPTURE_CUTOFF_UTC,
        OVEREXPOSED_LUMA_MIN,
        QUALITY_REASONS as TEACHER_DECISION_QUALITY_REASONS,
        TEACHER_LABEL_SCHEMA_VERSION,
        UNDEREXPOSED_LUMA_MAX,
        _teacher_consensus,
    )
    from build_v4_quality_exclusion_manifest import (  # type: ignore[no-redef]
        QUALITY_EXCLUSION_CONTRACT,
        QUALITY_EXCLUSION_REASON_ALIASES,
        QUALITY_EXCLUSION_REASONS,
        _reject_symlink_components,
        _resolve_source,
        _stable_bytes,
        build_quality_exclusion_manifest,
    )


ASSEMBLY_SCHEMA = "operational_quality_exclusion_assembly.v1"
ASSEMBLY_ROLE = (
    "operational_quality_exclusion_assembly_selection_only_"
    "not_ground_truth_or_authority"
)
ASSEMBLY_STATUS = "operational_quality_exclusions_assembled"
ASSEMBLY_FILES = {
    "manifest": "operational_quality_exclusions.json",
    "receipt": "operational_quality_exclusion_assembly.json",
    "marker": "assembly.sha256",
}
TEACHER_QUALITY_REASONS = frozenset(
    {
        "severe_frame_crop",
        "person_occlusion_or_dominance",
        "clutter_or_multiple_objects",
        "boundary_unreadable",
    }
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
KNOWN_AUDIT_SPLITS = frozenset({"train", "validation", "protected_validation"})
AUTHORITY_FIELDS = (
    "selection",
    "ground_truth",
    "replay",
    "training",
    "calibration",
    "blind_test",
    "deployment",
)
LINEAGE_FIELDS = {
    "builder",
    "portable",
    "local_only_contains_absolute_paths",
    "policy",
    "inputs",
    "counts",
    "output_digests",
}
LINEAGE_POLICY_FIELDS = {
    "allowed_roles",
    "blind_test_eligible",
    "ground_truth_authority",
    "teacher_label_schema_version",
    "negative_source_role",
    "negative_source_is_training_crop",
    "negative_source_requires_runtime_proposal_mining",
    "role",
    "fold",
    "minimum_confidence",
    "burst_gap_seconds",
    "minimum_bbox_area_ratio",
    "maximum_bbox_area_ratio",
    "operational_capture_cutoff_kst",
    "operational_capture_cutoff_utc",
    "minimum_image_width",
    "minimum_image_height",
    "extreme_exposure_fraction",
    "underexposed_luma_max",
    "overexposed_luma_min",
    "blur_filter_enabled",
    "deployed_prediction_filter_enabled",
}
LINEAGE_INPUT_FIELDS = {
    "teacher_queue_sha256",
    "teacher_labels_sha256",
    "capture_inventory_sha256",
    "known_audit_sha256",
    "provider_a_manifest_sha256",
    "provider_a_model_sha256",
    "provider_a_spec_sha256",
    "provider_b_manifest_sha256",
    "provider_b_model_sha256",
    "provider_b_spec_sha256",
    "provider_a_name",
    "provider_b_name",
}
LINEAGE_COUNT_FIELDS = {
    "accepted",
    "crop_ready_manifest_rows",
    "empty_scene_inventory_rows",
    "unique_accepted_sha256",
    "rejected_records",
    "accepted_by_teacher_material",
    "unique_object_groups",
    "unique_capture_sessions",
}
LINEAGE_OUTPUT_FIELDS = {
    "csv_sha256",
    "jsonl_sha256",
    "empty_scene_csv_sha256",
    "empty_scene_jsonl_sha256",
    "rejections_sha256",
}
REJECTION_REPORT_FIELDS = {
    "accepted",
    "crop_ready_manifest_rows",
    "empty_scene_inventory_rows",
    "queue_rows",
    "rejected_records",
    "reason_counts",
    "rejections",
}
QUALITY_REJECTION_FIELDS = {
    "sha256",
    "queue_line",
    "label_line",
    "reasons",
    "teacher_training_usable",
    "teacher_quality_reason",
}


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value: object, *, pretty: bool = True) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2 if pretty else None,
            sort_keys=True,
            separators=None if pretty else (",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"JSON contains duplicate key: {key}")
        value[key] = item
    return value


def _load_json_bytes(content: bytes, *, description: str) -> object:
    try:
        return json.loads(
            content.decode("utf-8-sig"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not valid UTF-8 JSON") from error


def _load_jsonl_bytes(content: bytes, *, description: str) -> list[dict[str, object]]:
    try:
        rendered = content.decode("utf-8-sig")
    except UnicodeError as error:
        raise ValueError(f"{description} is not valid UTF-8 JSONL") from error
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(rendered.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{description} has invalid JSON at line {line_number}"
            ) from error
        if type(value) is not dict:
            raise ValueError(f"{description} line {line_number} must be an object")
        row = dict(value)
        if "_input_line" in row:
            raise ValueError(
                f"{description} line {line_number} uses reserved _input_line"
            )
        row["_input_line"] = line_number
        rows.append(row)
    return rows


def _require_sha(value: object, *, description: str) -> str:
    if type(value) is not str or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{description} must be a lowercase SHA-256")
    return value


def _require_count(value: object, *, description: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{description} must be a non-negative integer")
    return value


def _require_finite_number(value: object, *, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{description} must be a finite number")
    return normalized


def _timestamp(value: object, *, description: str) -> datetime:
    if type(value) is not str:
        raise ValueError(f"{description} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{description} is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{description} must include an explicit UTC offset")
    return parsed.astimezone(timezone.utc)


def _stable_regular_file(path: Path, *, description: str) -> tuple[Path, bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} must be a regular non-symlink file")
    _reject_symlink_components(path, description=description)
    resolved = path.resolve(strict=True)
    return resolved, _stable_bytes(resolved, description=description)


def _stable_directory(path: Path, *, description: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{description} must be a regular non-symlink directory")
    _reject_symlink_components(path, description=description)
    return path.resolve(strict=True)


def _known_audit_shas(content: bytes) -> set[str]:
    value = _load_json_bytes(content, description="known audit")
    if type(value) is not dict:
        raise ValueError("known audit must be a SHA-keyed object")
    result: set[str] = set()
    for raw_sha, row in value.items():
        sha = _require_sha(raw_sha, description="known audit key")
        if sha in result:
            raise ValueError("known audit contains duplicate SHA")
        if type(row) is not dict or not row:
            raise ValueError("known audit values must be non-empty objects")
        if row.get("split") not in KNOWN_AUDIT_SPLITS:
            raise ValueError("known audit split is invalid")
        result.add(sha)
    return result


def _validate_rejection_report(
    value: object, *, queue_rows: int
) -> list[dict[str, object]]:
    if type(value) is not dict or set(value) != REJECTION_REPORT_FIELDS:
        raise ValueError("teacher rejection report schema is not exact")
    for field in (
        "accepted",
        "crop_ready_manifest_rows",
        "empty_scene_inventory_rows",
        "queue_rows",
        "rejected_records",
    ):
        _require_count(value.get(field), description=f"teacher rejections.{field}")
    if value.get("queue_rows") != queue_rows:
        raise ValueError("teacher rejection queue_rows mismatch")
    rejections = value.get("rejections")
    if type(rejections) is not list or not all(type(row) is dict for row in rejections):
        raise ValueError("teacher rejections must be an array of objects")
    if value.get("rejected_records") != len(rejections):
        raise ValueError("teacher rejected_records mismatch")
    normalized_reasons: list[str] = []
    seen_quality_shas: set[str] = set()
    for index, row in enumerate(rejections):
        reasons = row.get("reasons")
        if (
            type(reasons) is not list
            or not reasons
            or any(type(reason) is not str or not reason for reason in reasons)
            or reasons != sorted(set(reasons))
        ):
            raise ValueError(f"teacher rejection {index} reasons are not canonical")
        normalized_reasons.extend(reasons)
        for field in ("queue_line", "label_line"):
            if field in row and (type(row[field]) is not int or row[field] <= 0):
                raise ValueError(
                    f"teacher rejection {index} {field} must be a positive integer"
                )
        if "teacher_training_usable" in row:
            sha = _require_sha(
                row.get("sha256"), description=f"teacher rejection {index} SHA"
            )
            if sha in seen_quality_shas:
                raise ValueError("teacher rejections contain duplicate decision SHA")
            seen_quality_shas.add(sha)
    expected_counts = dict(sorted(Counter(normalized_reasons).items()))
    if value.get("reason_counts") != expected_counts:
        raise ValueError("teacher rejection reason_counts mismatch")
    expected_order = sorted(
        rejections,
        key=lambda row: (
            str(row.get("sha256") or ""),
            row.get("queue_line", -1),
            row.get("label_line", -1),
            tuple(row["reasons"]),
        ),
    )
    if rejections != expected_order:
        raise ValueError("teacher rejections are not in canonical order")
    return rejections


def _validate_lineage(
    value: object,
    *,
    input_contents: Mapping[str, bytes],
    teacher_output_contents: Mapping[str, bytes],
    rejection_report: Mapping[str, object],
) -> float:
    if type(value) is not dict or set(value) != LINEAGE_FIELDS:
        raise ValueError("teacher lineage top-level schema is not exact")
    if value.get("builder") != "scripts/build_operational_teacher_manifest.py":
        raise ValueError("teacher lineage builder mismatch")
    if value.get("portable") is not False:
        raise ValueError("teacher lineage portable flag mismatch")
    if value.get("local_only_contains_absolute_paths") is not True:
        raise ValueError("teacher lineage local-only flag mismatch")

    policy = value.get("policy")
    if type(policy) is not dict or set(policy) != LINEAGE_POLICY_FIELDS:
        raise ValueError("teacher lineage policy schema is not exact")
    expected_policy = {
        "allowed_roles": ["calibration", "train"],
        "blind_test_eligible": False,
        "ground_truth_authority": "vlm_teacher_pseudo_label_train_only",
        "teacher_label_schema_version": TEACHER_LABEL_SCHEMA_VERSION,
        "negative_source_role": "train",
        "negative_source_is_training_crop": False,
        "negative_source_requires_runtime_proposal_mining": True,
        "role": "train",
        "operational_capture_cutoff_kst": OPERATIONAL_CAPTURE_CUTOFF_KST.isoformat(),
        "operational_capture_cutoff_utc": (
            OPERATIONAL_CAPTURE_CUTOFF_UTC.isoformat().replace("+00:00", "Z")
        ),
        "blur_filter_enabled": False,
        "deployed_prediction_filter_enabled": False,
        "minimum_image_width": MINIMUM_IMAGE_WIDTH,
        "minimum_image_height": MINIMUM_IMAGE_HEIGHT,
        "extreme_exposure_fraction": EXTREME_EXPOSURE_FRACTION,
        "underexposed_luma_max": UNDEREXPOSED_LUMA_MAX,
        "overexposed_luma_min": OVEREXPOSED_LUMA_MIN,
    }
    for field, expected in expected_policy.items():
        if policy.get(field) != expected:
            raise ValueError(f"teacher lineage policy.{field} mismatch")
    if type(policy.get("fold")) is not str or not policy["fold"].strip():
        raise ValueError("teacher lineage policy.fold is invalid")
    minimum_confidence = _require_finite_number(
        policy.get("minimum_confidence"),
        description="teacher lineage policy.minimum_confidence",
    )
    if not 0 <= minimum_confidence <= 1:
        raise ValueError("teacher lineage minimum_confidence is outside 0..1")
    for field in (
        "burst_gap_seconds",
        "minimum_bbox_area_ratio",
        "maximum_bbox_area_ratio",
        "extreme_exposure_fraction",
        "underexposed_luma_max",
        "overexposed_luma_min",
    ):
        _require_finite_number(
            policy.get(field), description=f"teacher lineage policy.{field}"
        )
    if policy["burst_gap_seconds"] < 0:
        raise ValueError("teacher lineage burst_gap_seconds is negative")
    if not (
        0
        < policy["minimum_bbox_area_ratio"]
        < policy["maximum_bbox_area_ratio"]
        <= 1
    ):
        raise ValueError("teacher lineage bbox area ratios are invalid")

    inputs = value.get("inputs")
    if type(inputs) is not dict or set(inputs) != LINEAGE_INPUT_FIELDS:
        raise ValueError("teacher lineage inputs schema is not exact")
    for name, content in input_contents.items():
        field = f"{name}_sha256"
        expected = _require_sha(
            inputs.get(field), description=f"teacher lineage inputs.{field}"
        )
        if expected != _sha256_bytes(content):
            raise ValueError(f"teacher lineage input binding mismatch: {name}")
    for field in ("provider_a_name", "provider_b_name"):
        if type(inputs.get(field)) is not str or not inputs[field].strip():
            raise ValueError(f"teacher lineage inputs.{field} is invalid")
    if inputs["provider_a_name"].strip().casefold() == inputs[
        "provider_b_name"
    ].strip().casefold():
        raise ValueError("teacher lineage provider names must be distinct")

    counts = value.get("counts")
    if type(counts) is not dict or set(counts) != LINEAGE_COUNT_FIELDS:
        raise ValueError("teacher lineage counts schema is not exact")
    for field in LINEAGE_COUNT_FIELDS - {"accepted_by_teacher_material"}:
        _require_count(counts.get(field), description=f"teacher lineage counts.{field}")
    if type(counts.get("accepted_by_teacher_material")) is not dict:
        raise ValueError("teacher lineage accepted_by_teacher_material is invalid")
    if any(
        type(material) is not str
        or not material
        or type(count) is not int
        or count <= 0
        for material, count in counts["accepted_by_teacher_material"].items()
    ):
        raise ValueError("teacher lineage accepted material counts are invalid")
    for field in (
        "accepted",
        "crop_ready_manifest_rows",
        "empty_scene_inventory_rows",
        "rejected_records",
    ):
        if counts[field] != rejection_report[field]:
            raise ValueError(f"teacher lineage/rejections count mismatch: {field}")

    outputs = value.get("output_digests")
    if type(outputs) is not dict or set(outputs) != LINEAGE_OUTPUT_FIELDS:
        raise ValueError("teacher lineage output digest schema is not exact")
    output_binding_names = {
        "csv": "csv_sha256",
        "jsonl": "jsonl_sha256",
        "empty_scene_csv": "empty_scene_csv_sha256",
        "empty_scene_jsonl": "empty_scene_jsonl_sha256",
        "rejections": "rejections_sha256",
    }
    for name, field in output_binding_names.items():
        expected = _require_sha(
            outputs.get(field), description=f"teacher lineage outputs.{field}"
        )
        if expected != _sha256_bytes(teacher_output_contents[name]):
            raise ValueError(f"teacher lineage output binding mismatch: {name}")
    return minimum_confidence


def _csv_output_shas(
    content: bytes, *, fieldnames: Sequence[str], description: str
) -> list[str]:
    try:
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
        rows = list(reader)
    except (UnicodeError, csv.Error) as error:
        raise ValueError(f"{description} is not valid UTF-8 CSV") from error
    if reader.fieldnames != list(fieldnames):
        raise ValueError(f"{description} header schema is not exact")
    shas: list[str] = []
    for index, row in enumerate(rows, 2):
        if set(row) != set(fieldnames) or None in row:
            raise ValueError(f"{description} row {index} schema is not exact")
        shas.append(
            _require_sha(
                row.get("source_sha256"),
                description=f"{description} row {index} source SHA",
            )
        )
    if shas != sorted(shas) or len(shas) != len(set(shas)):
        raise ValueError(f"{description} source SHAs are not unique and sorted")
    return shas


def _jsonl_output_shas(
    content: bytes, *, fieldnames: Sequence[str], description: str
) -> list[str]:
    rows = _load_jsonl_bytes(content, description=description)
    shas: list[str] = []
    for row in rows:
        line = row["_input_line"]
        if set(row) - {"_input_line"} != set(fieldnames):
            raise ValueError(f"{description} line {line} schema is not exact")
        shas.append(
            _require_sha(
                row.get("source_sha256"),
                description=f"{description} line {line} source SHA",
            )
        )
    if shas != sorted(shas) or len(shas) != len(set(shas)):
        raise ValueError(f"{description} source SHAs are not unique and sorted")
    return shas


def _validate_teacher_acceptance_outputs(
    *,
    teacher_output_contents: Mapping[str, bytes],
    rejection_report: Mapping[str, object],
    rejections: Sequence[Mapping[str, object]],
    queue_rows: Sequence[Mapping[str, object]],
    label_rows: Sequence[Mapping[str, object]],
    minimum_confidence: float,
    selected_shas: set[str],
) -> None:
    accepted_csv = _csv_output_shas(
        teacher_output_contents["csv"],
        fieldnames=MANIFEST_FIELDS,
        description="teacher accepted CSV",
    )
    accepted_jsonl = _jsonl_output_shas(
        teacher_output_contents["jsonl"],
        fieldnames=MANIFEST_FIELDS,
        description="teacher accepted JSONL",
    )
    empty_csv = _csv_output_shas(
        teacher_output_contents["empty_scene_csv"],
        fieldnames=EMPTY_SCENE_INVENTORY_FIELDS,
        description="teacher empty-scene CSV",
    )
    empty_jsonl = _jsonl_output_shas(
        teacher_output_contents["empty_scene_jsonl"],
        fieldnames=EMPTY_SCENE_INVENTORY_FIELDS,
        description="teacher empty-scene JSONL",
    )
    if accepted_csv != accepted_jsonl or empty_csv != empty_jsonl:
        raise ValueError("teacher CSV/JSONL accepted source bindings differ")
    if len(accepted_jsonl) != rejection_report["crop_ready_manifest_rows"]:
        raise ValueError("teacher crop-ready output count mismatch")
    if len(empty_jsonl) != rejection_report["empty_scene_inventory_rows"]:
        raise ValueError("teacher empty-scene output count mismatch")
    all_accepted = [*accepted_jsonl, *empty_jsonl]
    if len(all_accepted) != rejection_report["accepted"]:
        raise ValueError("teacher total accepted output count mismatch")
    if len(all_accepted) != len(set(all_accepted)):
        raise ValueError("teacher accepted outputs contain duplicate source SHA")
    if selected_shas.intersection(all_accepted):
        raise ValueError("quality-excluded source is also present in accepted outputs")

    queue_by_sha = {
        _require_sha(row.get("sha256"), description="teacher queue partition SHA"): row
        for row in queue_rows
    }
    label_by_sha = {
        _require_sha(row.get("sha256"), description="teacher label partition SHA"): row
        for row in label_rows
    }
    rejected_shas = [
        _require_sha(
            row.get("sha256"), description="teacher rejection partition SHA"
        )
        for row in rejections
    ]
    if len(rejected_shas) != len(set(rejected_shas)):
        raise ValueError("teacher rejections contain duplicate source SHA")
    rejection_by_sha = dict(zip(rejected_shas, rejections, strict=True))
    accepted_set = set(all_accepted)
    queue_set = set(queue_by_sha)
    rejected_queue_set = set(rejected_shas).intersection(queue_set)
    if accepted_set.intersection(rejected_queue_set):
        raise ValueError("teacher accepted and rejected source sets overlap")
    if not accepted_set <= queue_set:
        raise ValueError("teacher accepted output contains a source outside the queue")
    if accepted_set.union(rejected_queue_set) != queue_set:
        raise ValueError("teacher accepted/rejected outputs do not partition the queue")
    for sha, queue in queue_by_sha.items():
        label = label_by_sha.get(sha)
        if label is None:
            continue
        decision, decision_reasons = _teacher_consensus(label)
        if decision is None or decision_reasons:
            continue
        rejection = rejection_by_sha.get(sha)
        if rejection is None:
            if decision.get("training_usable") is False:
                raise ValueError(
                    "unusable teacher consensus is missing its decision rejection"
                )
            continue
        if (
            set(rejection) != QUALITY_REJECTION_FIELDS
            or rejection.get("teacher_training_usable")
            is not decision.get("training_usable")
            or rejection.get("teacher_quality_reason")
            != decision.get("quality_reason")
        ):
            raise ValueError("teacher rejection does not match its exact consensus")
    for sha in accepted_set:
        queue = queue_by_sha[sha]
        label = label_by_sha.get(sha)
        if label is None:
            raise ValueError("teacher accepted source has no unique label")
        if label.get("input_image_sha256") != sha or label.get(
            "image_ref"
        ) != queue.get("image_ref"):
            raise ValueError("teacher accepted label source binding mismatch")
        decision, decision_reasons = _teacher_consensus(label)
        if (
            decision is None
            or decision_reasons
            or decision.get("training_usable") is not True
            or decision.get("quality_reason") != "usable"
            or decision.get("minimum_confidence", -1) < minimum_confidence
        ):
            raise ValueError("teacher accepted source does not have a usable consensus")


def _index_unique_rows(
    rows: Sequence[dict[str, object]], *, description: str
) -> tuple[dict[str, dict[str, object]], dict[int, dict[str, object]]]:
    by_sha: dict[str, dict[str, object]] = {}
    by_line: dict[int, dict[str, object]] = {}
    for row in rows:
        line = row.get("_input_line")
        if type(line) is not int or line <= 0 or line in by_line:
            raise ValueError(f"{description} has invalid line identity")
        sha = _require_sha(row.get("sha256"), description=f"{description} line {line} SHA")
        if sha in by_sha:
            raise ValueError(f"{description} contains duplicate SHA")
        by_sha[sha] = row
        by_line[line] = row
    return by_sha, by_line


def _inventory_by_sha(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for index, row in enumerate(rows):
        sha = _require_sha(row.get("sha256"), description=f"capture inventory row {index} SHA")
        if sha in result:
            raise ValueError("capture inventory contains duplicate SHA")
        result[sha] = row
    return result


def _quality_rows(
    *,
    rejections: Sequence[dict[str, object]],
    queue_rows: Sequence[dict[str, object]],
    label_rows: Sequence[dict[str, object]],
    inventory_rows: Sequence[dict[str, object]],
    known_shas: set[str],
    minimum_confidence: float,
    image_root: Path,
) -> tuple[list[dict[str, str]], list[tuple[Path, bytes, str, str]]]:
    queue_by_sha, queue_by_line = _index_unique_rows(
        queue_rows, description="teacher queue"
    )
    label_by_sha, label_by_line = _index_unique_rows(
        label_rows, description="teacher labels"
    )
    inventory_by_sha = _inventory_by_sha(inventory_rows)
    selected: list[dict[str, str]] = []
    source_bindings: list[tuple[Path, bytes, str, str]] = []
    selected_shas: set[str] = set()

    for index, rejection in enumerate(rejections):
        if "teacher_training_usable" not in rejection:
            continue
        if set(rejection) != QUALITY_REJECTION_FIELDS:
            raise ValueError(f"quality rejection {index} schema is not exact")
        if type(rejection.get("teacher_training_usable")) is not bool:
            raise ValueError(
                f"quality rejection {index} teacher_training_usable is not boolean"
            )
        teacher_training_usable = rejection["teacher_training_usable"]
        reason = rejection.get("teacher_quality_reason")
        if type(reason) is not str or reason not in TEACHER_DECISION_QUALITY_REASONS:
            raise ValueError(f"teacher rejection {index} has unsupported quality reason")
        if teacher_training_usable is False:
            if reason not in TEACHER_QUALITY_REASONS:
                raise ValueError(
                    f"quality rejection {index} has unsupported teacher reason"
                )
            if rejection.get("reasons") != [f"training_unusable_{reason}"]:
                raise ValueError(
                    f"quality rejection {index} has additional rejection reasons"
                )
        elif reason != "usable" or any(
            str(value).startswith("training_unusable_")
            for value in rejection["reasons"]
        ):
            raise ValueError(
                f"teacher rejection {index} usable decision is inconsistent"
            )
        sha = _require_sha(
            rejection.get("sha256"), description=f"quality rejection {index} SHA"
        )
        if sha in selected_shas:
            raise ValueError("quality rejections contain duplicate selected SHA")
        if sha in known_shas:
            raise ValueError("quality rejection SHA is already in known audit")
        queue_line = rejection.get("queue_line")
        label_line = rejection.get("label_line")
        if type(queue_line) is not int or type(label_line) is not int:
            raise ValueError("quality rejection line bindings must be integers")
        queue = queue_by_line.get(queue_line)
        label = label_by_line.get(label_line)
        if queue is None or queue_by_sha.get(sha) is not queue or queue.get("sha256") != sha:
            raise ValueError("quality rejection queue line/SHA binding mismatch")
        if label is None or label_by_sha.get(sha) is not label or label.get("sha256") != sha:
            raise ValueError("quality rejection label line/SHA binding mismatch")
        if set(queue) - {"_input_line"} != {
            "sha256",
            "timestamp",
            "image_ref",
            "decision",
        }:
            raise ValueError("quality teacher queue row schema is not exact")
        if queue.get("decision") != "teacher_required":
            raise ValueError("quality teacher queue decision mismatch")
        if label.get("input_image_sha256") != sha or label.get("image_ref") != queue.get(
            "image_ref"
        ):
            raise ValueError("quality teacher label source binding mismatch")
        decision, decision_reasons = _teacher_consensus(label)
        if decision is None or decision_reasons:
            raise ValueError("quality teacher consensus is not exact and valid")
        if (
            decision.get("training_usable") is not teacher_training_usable
            or decision.get("quality_reason") != reason
        ):
            raise ValueError("quality teacher decision does not match rejection")
        if teacher_training_usable is True:
            continue
        if decision.get("minimum_confidence", -1) < minimum_confidence:
            raise ValueError("quality teacher decision is below minimum confidence")

        inventory = inventory_by_sha.get(sha)
        if inventory is None:
            raise ValueError("quality source is absent from capture inventory")
        for field in ("sha256", "timestamp", "image_ref", "decision"):
            if inventory.get(field) != queue.get(field):
                raise ValueError(f"quality queue/inventory {field} mismatch")
        queue_timestamp = _timestamp(
            queue.get("timestamp"), description="quality queue timestamp"
        )
        inventory_timestamp = _timestamp(
            inventory.get("timestamp"), description="quality inventory timestamp"
        )
        if queue_timestamp != inventory_timestamp:
            raise ValueError("quality queue/inventory timestamp mismatch")
        if queue_timestamp < OPERATIONAL_CAPTURE_CUTOFF_UTC:
            raise ValueError("quality source is before the operational cutoff")

        image_ref = queue.get("image_ref")
        if type(image_ref) is not str:
            raise ValueError("quality source image_ref is invalid")
        source = _resolve_source(image_root, image_ref, row_number=queue_line)
        source_content = _stable_bytes(
            source, description=f"quality source {sha}"
        )
        if _sha256_bytes(source_content) != sha:
            raise ValueError("quality source bytes do not match teacher SHA")
        canonical_reason = QUALITY_EXCLUSION_REASON_ALIASES.get(reason, reason)
        if canonical_reason not in QUALITY_EXCLUSION_REASONS:
            raise RuntimeError("quality reason does not map to the producer contract")
        selected.append({"path": image_ref, "reason": reason})
        source_bindings.append((source, source_content, sha, canonical_reason))
        selected_shas.add(sha)

    if not selected:
        raise ValueError("no eligible post-cutoff teacher quality exclusions")
    selected.sort(key=lambda row: row["path"])
    return selected, source_bindings


def _source_csv_bytes(rows: Sequence[Mapping[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=("path", "reason"), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _marker_bytes(files: Mapping[str, bytes]) -> bytes:
    return "".join(
        f"{_sha256_bytes(content)}  {name}\n" for name, content in sorted(files.items())
    ).encode("ascii")


def _publish_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a complete directory without replacing a peer."""

    if os.name == "nt":
        # Windows rename fails when the destination already exists.
        os.rename(source, destination)
        return
    if not sys.platform.startswith("linux"):
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory publication is unsupported",
            os.fspath(destination),
        )
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(
            errno.ENOTSUP,
            "renameat2 is required for atomic no-replace publication",
            os.fspath(destination),
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,  # AT_FDCWD
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            os.fspath(destination),
        )


def assemble_operational_quality_exclusions(
    *,
    teacher_output_dir: Path,
    teacher_queue: Path,
    teacher_labels: Path,
    capture_inventory: Path,
    known_audit: Path,
    provider_a_manifest: Path,
    provider_a_model: Path,
    provider_a_spec: Path,
    provider_b_manifest: Path,
    provider_b_model: Path,
    provider_b_spec: Path,
    image_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    input_paths = {
        "teacher_queue": teacher_queue,
        "teacher_labels": teacher_labels,
        "capture_inventory": capture_inventory,
        "known_audit": known_audit,
        "provider_a_manifest": provider_a_manifest,
        "provider_a_model": provider_a_model,
        "provider_a_spec": provider_a_spec,
        "provider_b_manifest": provider_b_manifest,
        "provider_b_model": provider_b_model,
        "provider_b_spec": provider_b_spec,
    }
    resolved_inputs: dict[str, Path] = {}
    input_contents: dict[str, bytes] = {}
    for name, path in input_paths.items():
        resolved, content = _stable_regular_file(path, description=name)
        resolved_inputs[name] = resolved
        input_contents[name] = content

    teacher_output_dir = _stable_directory(
        teacher_output_dir, description="teacher output directory"
    )
    expected_teacher_names = set(ARTIFACT_NAMES.values())
    if {path.name for path in teacher_output_dir.iterdir()} != expected_teacher_names:
        raise ValueError("teacher output directory file set is not exact")
    resolved_teacher_outputs: dict[str, Path] = {}
    teacher_output_contents: dict[str, bytes] = {}
    for name, filename in ARTIFACT_NAMES.items():
        resolved, content = _stable_regular_file(
            teacher_output_dir / filename, description=f"teacher output {name}"
        )
        resolved_teacher_outputs[name] = resolved
        teacher_output_contents[name] = content

    image_root = _stable_directory(image_root, description="image root")
    normalized_output = Path(os.path.abspath(output_dir))
    if normalized_output.exists() or normalized_output.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable output: {normalized_output}")
    output_parent = _stable_directory(
        normalized_output.parent, description="output parent"
    )
    normalized_output = output_parent / normalized_output.name
    if normalized_output.is_relative_to(teacher_output_dir):
        raise ValueError("output directory must not be inside teacher output authority")

    queue_rows = _load_jsonl_bytes(
        input_contents["teacher_queue"], description="teacher queue"
    )
    label_rows = _load_jsonl_bytes(
        input_contents["teacher_labels"], description="teacher labels"
    )
    inventory_value = _load_json_bytes(
        input_contents["capture_inventory"], description="capture inventory"
    )
    if type(inventory_value) is not list or not all(
        type(row) is dict for row in inventory_value
    ):
        raise ValueError("capture inventory must be an array of objects")
    inventory_rows = [dict(row) for row in inventory_value]
    known_shas = _known_audit_shas(input_contents["known_audit"])
    rejection_value = _load_json_bytes(
        teacher_output_contents["rejections"], description="teacher rejections"
    )
    rejections = _validate_rejection_report(
        rejection_value, queue_rows=len(queue_rows)
    )
    lineage_value = _load_json_bytes(
        teacher_output_contents["lineage"], description="teacher lineage"
    )
    minimum_confidence = _validate_lineage(
        lineage_value,
        input_contents=input_contents,
        teacher_output_contents=teacher_output_contents,
        rejection_report=rejection_value,
    )
    quality_rows, source_bindings = _quality_rows(
        rejections=rejections,
        queue_rows=queue_rows,
        label_rows=label_rows,
        inventory_rows=inventory_rows,
        known_shas=known_shas,
        minimum_confidence=minimum_confidence,
        image_root=image_root,
    )
    _validate_teacher_acceptance_outputs(
        teacher_output_contents=teacher_output_contents,
        rejection_report=rejection_value,
        rejections=rejections,
        queue_rows=queue_rows,
        label_rows=label_rows,
        minimum_confidence=minimum_confidence,
        selected_shas={sha for _, _, sha, _ in source_bindings},
    )

    repo_root = Path(__file__).resolve().parents[1]
    code_paths = {
        "assembler": Path(__file__).resolve(),
        "quality_producer": repo_root / "scripts" / "build_v4_quality_exclusion_manifest.py",
        "teacher_builder": repo_root / "scripts" / "build_operational_teacher_manifest.py",
        "teacher_contract": repo_root / "scripts" / "operational_teacher_contract.py",
    }
    resolved_code: dict[str, Path] = {}
    code_contents: dict[str, bytes] = {}
    for name, path in code_paths.items():
        resolved, content = _stable_regular_file(path, description=f"code {name}")
        resolved_code[name] = resolved
        code_contents[name] = content

    staging = Path(
        tempfile.mkdtemp(prefix=f".{normalized_output.name}.", dir=output_parent)
    )
    try:
        source_csv = staging / "source-list.csv"
        source_csv.write_bytes(_source_csv_bytes(quality_rows))
        manifest_path = staging / ASSEMBLY_FILES["manifest"]
        manifest = build_quality_exclusion_manifest(
            source_list=source_csv,
            image_root=image_root,
            output_path=manifest_path,
        )
        source_csv.unlink()
        manifest_content = _stable_bytes(
            manifest_path, description="assembled quality manifest"
        )
        if _load_json_bytes(
            manifest_content, description="assembled quality manifest"
        ) != manifest:
            raise RuntimeError("assembled quality manifest bytes do not match result")
        expected_entries = sorted(
            (
                {"source_sha256": sha, "reason": canonical_reason}
                for _, _, sha, canonical_reason in source_bindings
            ),
            key=lambda entry: entry["source_sha256"],
        )
        if manifest.get("entries") != expected_entries:
            raise RuntimeError(
                "assembled manifest does not exactly match teacher quality decisions"
            )

        receipt = {
            "schema_version": 1,
            "assembly_schema": ASSEMBLY_SCHEMA,
            "artifact_role": ASSEMBLY_ROLE,
            "status": ASSEMBLY_STATUS,
            "quality_exclusion_contract": QUALITY_EXCLUSION_CONTRACT,
            "operational_capture_cutoff_kst": (
                OPERATIONAL_CAPTURE_CUTOFF_KST.isoformat()
            ),
            "teacher_label_schema_version": TEACHER_LABEL_SCHEMA_VERSION,
            "selected_source_count": manifest["excluded_source_count"],
            "reason_counts": manifest["reason_counts"],
            "quality_manifest_sha256": _sha256_bytes(manifest_content),
            "quality_source_list_sha256": manifest["source_list_sha256"],
            "input_sha256": {
                **{
                    name: _sha256_bytes(content)
                    for name, content in sorted(input_contents.items())
                },
                **{
                    f"teacher_output_{name}": _sha256_bytes(content)
                    for name, content in sorted(teacher_output_contents.items())
                },
            },
            "observed_code_sha256": {
                name: _sha256_bytes(content)
                for name, content in sorted(code_contents.items())
            },
            "scope": {
                "teacher_subjective_quality_only": True,
                "objective_queue_rejections_recoverable": False,
                "paths_or_private_ids_exported": False,
                "trusted_policy_pinned": False,
                "executed_code_cryptographically_attested": False,
            },
            "authority": {field: False for field in AUTHORITY_FIELDS},
        }
        receipt_content = _json_bytes(receipt)
        receipt_path = staging / ASSEMBLY_FILES["receipt"]
        receipt_path.write_bytes(receipt_content)
        marker_content = _marker_bytes(
            {
                ASSEMBLY_FILES["manifest"]: manifest_content,
                ASSEMBLY_FILES["receipt"]: receipt_content,
            }
        )
        marker_path = staging / ASSEMBLY_FILES["marker"]
        marker_path.write_bytes(marker_content)

        for name, path in resolved_inputs.items():
            if _stable_bytes(path, description=f"final input rehash {name}") != input_contents[
                name
            ]:
                raise RuntimeError(f"assembler input changed before publish: {name}")
        for name, path in resolved_teacher_outputs.items():
            if _stable_bytes(
                path, description=f"final teacher output rehash {name}"
            ) != teacher_output_contents[name]:
                raise RuntimeError(
                    f"teacher output changed before assembly publish: {name}"
                )
        if {path.name for path in teacher_output_dir.iterdir()} != expected_teacher_names:
            raise RuntimeError("teacher output directory changed before assembly publish")
        for name, path in resolved_code.items():
            if _stable_bytes(path, description=f"final code rehash {name}") != code_contents[
                name
            ]:
                raise RuntimeError(f"assembler code changed before publish: {name}")
        for source, expected_content, expected_sha, _ in source_bindings:
            final_content = _stable_bytes(
                source, description="final quality source rehash"
            )
            if final_content != expected_content or _sha256_bytes(final_content) != expected_sha:
                raise RuntimeError("quality source changed before assembly publish")
        if set(path.name for path in staging.iterdir()) != set(ASSEMBLY_FILES.values()):
            raise RuntimeError("assembly staging file set is not exact")
        if _stable_bytes(
            manifest_path, description="final assembled quality manifest rehash"
        ) != manifest_content:
            raise RuntimeError("assembled quality manifest changed before publish")
        if _stable_bytes(
            receipt_path, description="final assembly receipt rehash"
        ) != receipt_content:
            raise RuntimeError("assembly receipt changed before publish")
        if _stable_bytes(
            marker_path, description="final assembly marker rehash"
        ) != marker_content:
            raise RuntimeError("assembly marker changed before publish")
        try:
            _publish_directory_no_replace(staging, normalized_output)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite immutable output: {normalized_output}"
            ) from error
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-output-dir", required=True, type=Path)
    parser.add_argument("--teacher-queue", required=True, type=Path)
    parser.add_argument("--teacher-labels", required=True, type=Path)
    parser.add_argument("--capture-inventory", required=True, type=Path)
    parser.add_argument("--known-audit", required=True, type=Path)
    for prefix in ("a", "b"):
        parser.add_argument(f"--provider-{prefix}-manifest", required=True, type=Path)
        parser.add_argument(f"--provider-{prefix}-model", required=True, type=Path)
        parser.add_argument(f"--provider-{prefix}-spec", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = assemble_operational_quality_exclusions(
        teacher_output_dir=args.teacher_output_dir,
        teacher_queue=args.teacher_queue,
        teacher_labels=args.teacher_labels,
        capture_inventory=args.capture_inventory,
        known_audit=args.known_audit,
        provider_a_manifest=args.provider_a_manifest,
        provider_a_model=args.provider_a_model,
        provider_a_spec=args.provider_a_spec,
        provider_b_manifest=args.provider_b_manifest,
        provider_b_model=args.provider_b_model,
        provider_b_spec=args.provider_b_spec,
        image_root=args.image_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
