"""Shared, policy-independent operational quality assembly contract.

QX3 and candidate authority use this same validator.  Keeping it separate from
candidate's approved-policy pin prevents a QX3/policy/source SHA dependency cycle.
This module only validates evidence; it grants no training or deployment authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence
from zoneinfo import ZoneInfo

QUALITY_CONTRACT = "v4_capture_quality_exclusions.sha256_reason_only.v1"
QUALITY_ROLE = (
    "v4_capture_quality_exclusion_manifest_selection_only_"
    "not_ground_truth_or_authority"
)
QUALITY_REASONS = {
    "severe_frame_crop",
    "person_occlusion_or_dominance",
    "excessive_background_or_multi_object",
    "unreadable_boundary",
    "too_low_resolution",
    "extreme_exposure",
}
QUALITY_ASSEMBLY_SCHEMA = "operational_quality_exclusion_assembly.v1"
QUALITY_ASSEMBLY_ROLE = (
    "operational_quality_exclusion_assembly_selection_only_"
    "not_ground_truth_or_authority"
)
QUALITY_ASSEMBLY_STATUS = "operational_quality_exclusions_assembled"
QUALITY_ASSEMBLY_MODE = "objective_and_subjective_quality"
QUALITY_ASSEMBLY_TEACHER_SCHEMA = "operational_teacher_label.v3"
QUALITY_ASSEMBLY_FILES = {
    "manifest": "operational_quality_exclusions.json",
    "receipt": "operational_quality_exclusion_assembly.json",
    "marker": "assembly.sha256",
}
QUALITY_ASSEMBLY_INPUT_SHA_FIELDS = {
    "teacher_queue",
    "teacher_labels",
    "capture_inventory",
    "known_audit",
    "provider_a_manifest",
    "provider_a_model",
    "provider_a_spec",
    "provider_b_manifest",
    "provider_b_model",
    "provider_b_spec",
    "teacher_output_csv",
    "teacher_output_jsonl",
    "teacher_output_empty_scene_csv",
    "teacher_output_empty_scene_jsonl",
    "teacher_output_rejections",
    "teacher_output_lineage",
    "objective_prepare_capture_index",
    "objective_prepare_capture_inventory",
    "objective_prepare_teacher_queue",
    "objective_prepare_objective_rejections",
    "objective_prepare_summary",
    "objective_prepare_objective_receipt",
}
QUALITY_ASSEMBLY_CODE_SHA_FIELDS = {
    "assembler",
    "quality_producer",
    "teacher_builder",
    "teacher_contract",
    "objective_queue_preparer",
}
QUALITY_ASSEMBLY_SCOPE_FIELDS = {
    "teacher_subjective_quality_included",
    "objective_queue_quality_included",
    "objective_prepare_bundle_validated",
    "subjective_quality_source_count",
    "objective_quality_source_count",
    "paths_or_private_ids_exported",
    "trusted_policy_pinned",
    "executed_code_cryptographically_attested",
}
QUALITY_ASSEMBLY_RECEIPT_FIELDS = {
    "schema_version",
    "assembly_schema",
    "artifact_role",
    "status",
    "assembly_mode",
    "quality_exclusion_contract",
    "operational_capture_cutoff_kst",
    "teacher_label_schema_version",
    "selected_source_count",
    "reason_counts",
    "quality_manifest_sha256",
    "quality_source_list_sha256",
    "input_sha256",
    "observed_code_sha256",
    "scope",
    "authority",
}
KST = ZoneInfo("Asia/Seoul")
OPERATIONAL_CUTOFF = datetime(2026, 8, 1, 0, 0, 0, tzinfo=KST)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FALSE_AUTHORITY_FIELDS = (
    "selection", "ground_truth", "replay", "training", "calibration",
    "blind_test", "deployment",
)
def _reject_duplicate_keys(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _canonical_compact_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _require_bool(value: object, expected: bool, field: str) -> None:
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{field} must be {expected!r}")


def _exact_json_equal(left: object, right: object) -> bool:
    """Compare JSON values without Python's bool/int equality aliasing."""

    if type(left) is not type(right):
        return False
    if type(left) is dict:
        assert type(right) is dict
        if set(left) != set(right):
            return False
        return all(_exact_json_equal(left[key], right[key]) for key in left)
    if type(left) is list:
        assert type(right) is list
        return len(left) == len(right) and all(
            _exact_json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _reject_symlink_components(path: Path, description: str) -> Path:
    absolute = Path(os.path.abspath(path))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{description} path contains a symlink: {cursor}")
    return absolute


def _regular_file(path: Path, description: str) -> Path:
    absolute = _reject_symlink_components(path, description)
    if absolute.is_symlink() or not absolute.is_file():
        raise ValueError(f"{description} must be a regular non-symlink file: {path}")
    return absolute


def _stable_bytes(path: Path, description: str) -> bytes:
    resolved = _regular_file(path, description)
    before = resolved.stat()
    with resolved.open("rb") as handle:
        opened_before = os.fstat(handle.fileno())
        content = handle.read()
        opened_after = os.fstat(handle.fileno())
    after = resolved.stat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if not (
        identity(before) == identity(opened_before) == identity(opened_after) == identity(after)
    ):
        raise RuntimeError(f"{description} changed while being read: {resolved}")
    return content


def _load_json(path: Path, description: str) -> tuple[dict[str, object], bytes]:
    content = _stable_bytes(path, description)
    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {description}: {error}") from error
    if type(value) is not dict:
        raise ValueError(f"{description} root must be an object")
    return value, content


def _canonical_quality_entries(entries: Sequence[Mapping[str, str]]) -> bytes:
    return (
        json.dumps(
            list(entries), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _validate_quality_manifest(
    value: Mapping[str, object], *, assembly_bundle: _QualityAssemblyBundle | None = None,
) -> dict[str, str]:
    expected_fields = {
        "schema_version", "artifact_role", "quality_exclusion_contract", "status",
        "excluded_source_count", "max_excluded_sources", "reason_counts",
        "source_list_sha256", "entries", "authority",
    }
    if set(value) != expected_fields:
        raise ValueError("quality exclusion top-level schema mismatch")
    if type(value.get("schema_version")) is not int or value.get("schema_version") != 1:
        raise ValueError("quality exclusion schema_version must be integer 1")
    if value.get("artifact_role") != QUALITY_ROLE:
        raise ValueError("quality exclusion artifact_role mismatch")
    if value.get("quality_exclusion_contract") != QUALITY_CONTRACT:
        raise ValueError("quality exclusion contract mismatch")
    if value.get("status") != "quality_exclusions_ready":
        raise ValueError("quality exclusion status mismatch")
    authority = value.get("authority")
    if type(authority) is not dict or set(authority) != set(FALSE_AUTHORITY_FIELDS):
        raise ValueError("quality exclusion authority schema mismatch")
    for field in FALSE_AUTHORITY_FIELDS:
        _require_bool(authority.get(field), False, f"quality authority.{field}")
    entries = value.get("entries")
    if type(entries) is not list:
        raise ValueError("quality exclusions entries must be a list")
    if not entries:
        if not isinstance(assembly_bundle, _QualityAssemblyBundle):
            raise ValueError("empty quality exclusions require a validated full assembly bundle")
        if assembly_bundle.manifest_content != _canonical_json(value):
            raise ValueError("empty quality exclusions differ from the validated assembly manifest")
        _rehash_operational_quality_assembly(assembly_bundle)
    if len(entries) > 100:
        raise ValueError("quality exclusions may contain at most 100 entries")
    parsed: dict[str, str] = {}
    normalized: list[dict[str, str]] = []
    for index, entry in enumerate(entries):
        if type(entry) is not dict or set(entry) != {"source_sha256", "reason"}:
            raise ValueError(f"quality exclusion entry {index} must be SHA/reason only")
        source_sha = _require_sha256(entry.get("source_sha256"), "quality source")
        reason = entry.get("reason")
        if type(reason) is not str or reason not in QUALITY_REASONS:
            raise ValueError("dent/crush/object condition is not a capture-quality exclusion")
        if source_sha in parsed:
            raise ValueError("duplicate quality exclusion SHA")
        parsed[source_sha] = reason
        normalized.append({"source_sha256": source_sha, "reason": reason})
    normalized.sort(key=lambda row: row["source_sha256"])
    if type(value.get("excluded_source_count")) is not int or value.get(
        "excluded_source_count"
    ) != len(normalized):
        raise ValueError("quality excluded_source_count mismatch")
    if type(value.get("max_excluded_sources")) is not int or value.get(
        "max_excluded_sources"
    ) != 100:
        raise ValueError("quality max_excluded_sources must be exactly 100")
    reason_counts = dict(sorted(Counter(row["reason"] for row in normalized).items()))
    if not _exact_json_equal(value.get("reason_counts"), reason_counts):
        raise ValueError("quality reason_counts mismatch")
    expected_sha = _sha256_bytes(_canonical_quality_entries(normalized))
    if value.get("source_list_sha256") != expected_sha:
        raise ValueError("quality source_list_sha256 mismatch")
    return parsed


@dataclass(frozen=True)
class _QualityAssemblyBundle:
    root: Path
    manifest_path: Path
    receipt_path: Path
    marker_path: Path
    manifest_content: bytes
    receipt_content: bytes
    marker_content: bytes
    observed_code_sha256: dict[str, str]
    validator_path: Path
    validator_content: bytes


def _quality_assembly_validator_path() -> Path:
    return _regular_file(Path(__file__), "quality assembly validator")


def _quality_assembly_code_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    return {
        "assembler": root / "scripts" / "assemble_operational_quality_exclusions.py",
        "quality_producer": root / "scripts" / "build_v4_quality_exclusion_manifest.py",
        "teacher_builder": root / "scripts" / "build_operational_teacher_manifest.py",
        "teacher_contract": root / "scripts" / "operational_teacher_contract.py",
        "objective_queue_preparer": root
        / "scripts"
        / "prepare_operational_capture_queue.py",
    }


def _quality_assembly_code_hashes() -> dict[str, str]:
    return {
        name: _sha256_bytes(
            _stable_bytes(path, f"quality assembly observed code {name}")
        )
        for name, path in sorted(_quality_assembly_code_paths().items())
    }


def _quality_assembly_marker_bytes(
    *, manifest_content: bytes, receipt_content: bytes
) -> bytes:
    contents = {
        QUALITY_ASSEMBLY_FILES["manifest"]: manifest_content,
        QUALITY_ASSEMBLY_FILES["receipt"]: receipt_content,
    }
    return "".join(
        f"{_sha256_bytes(content)}  {name}\n"
        for name, content in sorted(contents.items())
    ).encode("ascii")


def _validate_operational_quality_assembly(
    *,
    receipt_path: Path,
    quality_path: Path,
    quality_value: Mapping[str, object],
    quality_content: bytes,
    output_dir: Path,
) -> _QualityAssemblyBundle:
    """Validate the immutable full objective+subjective quality bundle."""

    validator_path = _quality_assembly_validator_path()
    validator_content = _stable_bytes(validator_path, "quality assembly validator")
    resolved_receipt = _regular_file(
        receipt_path, "quality exclusion assembly receipt"
    )
    root = _reject_symlink_components(
        resolved_receipt.parent, "quality exclusion assembly directory"
    )
    if root.is_symlink() or not root.is_dir():
        raise ValueError("quality exclusion assembly parent must be a directory")
    expected_names = set(QUALITY_ASSEMBLY_FILES.values())
    if {entry.name for entry in root.iterdir()} != expected_names:
        raise ValueError("quality exclusion assembly directory file set mismatch")
    expected_receipt = root / QUALITY_ASSEMBLY_FILES["receipt"]
    expected_manifest = root / QUALITY_ASSEMBLY_FILES["manifest"]
    expected_marker = root / QUALITY_ASSEMBLY_FILES["marker"]
    resolved_manifest = _regular_file(
        quality_path, "quality exclusion assembly manifest"
    )
    if resolved_receipt != expected_receipt:
        raise ValueError("quality exclusion assembly receipt basename mismatch")
    if resolved_manifest != expected_manifest:
        raise ValueError(
            "quality exclusions must be the manifest beside the assembly receipt"
        )
    marker_path = _regular_file(expected_marker, "quality exclusion assembly marker")
    normalized_output = Path(os.path.abspath(output_dir))
    if normalized_output.is_relative_to(root):
        raise ValueError("output directory must not be inside quality assembly evidence")

    receipt, receipt_content = _load_json(
        resolved_receipt, "quality exclusion assembly receipt"
    )
    marker_content = _stable_bytes(marker_path, "quality exclusion assembly marker")
    if quality_content != _canonical_json(quality_value):
        raise ValueError("quality exclusion assembly manifest is not canonical JSON")
    if receipt_content != _canonical_json(receipt):
        raise ValueError("quality exclusion assembly receipt is not canonical JSON")
    if set(receipt) != QUALITY_ASSEMBLY_RECEIPT_FIELDS:
        raise ValueError("quality exclusion assembly receipt schema mismatch")
    if type(receipt.get("schema_version")) is not int or receipt.get(
        "schema_version"
    ) != 1:
        raise ValueError("quality exclusion assembly schema_version mismatch")
    exact_fields = {
        "assembly_schema": QUALITY_ASSEMBLY_SCHEMA,
        "artifact_role": QUALITY_ASSEMBLY_ROLE,
        "status": QUALITY_ASSEMBLY_STATUS,
        "assembly_mode": QUALITY_ASSEMBLY_MODE,
        "quality_exclusion_contract": QUALITY_CONTRACT,
        "operational_capture_cutoff_kst": OPERATIONAL_CUTOFF.isoformat(),
        "teacher_label_schema_version": QUALITY_ASSEMBLY_TEACHER_SCHEMA,
    }
    for field, expected in exact_fields.items():
        if receipt.get(field) != expected:
            raise ValueError(f"quality exclusion assembly {field} mismatch")

    selected_count = receipt.get("selected_source_count")
    if type(selected_count) is not int or selected_count < 0:
        raise ValueError("quality exclusion assembly selected_source_count mismatch")
    if selected_count != quality_value.get("excluded_source_count"):
        raise ValueError("quality exclusion assembly selected count mismatch")
    if not _exact_json_equal(
        receipt.get("reason_counts"), quality_value.get("reason_counts")
    ):
        raise ValueError("quality exclusion assembly reason counts mismatch")
    if receipt.get("quality_manifest_sha256") != _sha256_bytes(quality_content):
        raise ValueError("quality exclusion assembly manifest SHA mismatch")
    if receipt.get("quality_source_list_sha256") != quality_value.get(
        "source_list_sha256"
    ):
        raise ValueError("quality exclusion assembly source-list SHA mismatch")

    scope = receipt.get("scope")
    if type(scope) is not dict or set(scope) != QUALITY_ASSEMBLY_SCOPE_FIELDS:
        raise ValueError("quality exclusion assembly scope schema mismatch")
    for field in (
        "teacher_subjective_quality_included",
        "objective_queue_quality_included",
    ):
        if type(scope.get(field)) is not bool:
            raise ValueError(f"quality exclusion assembly scope.{field} must be boolean")
    _require_bool(
        scope.get("objective_prepare_bundle_validated"),
        True,
        "quality exclusion assembly scope.objective_prepare_bundle_validated",
    )
    for field in (
        "paths_or_private_ids_exported",
        "trusted_policy_pinned",
        "executed_code_cryptographically_attested",
    ):
        _require_bool(
            scope.get(field), False, f"quality exclusion assembly scope.{field}"
        )
    subjective_count = scope.get("subjective_quality_source_count")
    objective_count = scope.get("objective_quality_source_count")
    if (
        type(subjective_count) is not int
        or subjective_count < 0
        or type(objective_count) is not int
        or objective_count < 0
        or subjective_count + objective_count != selected_count
    ):
        raise ValueError("quality exclusion assembly scope counts mismatch")
    if scope.get("teacher_subjective_quality_included") is not (
        subjective_count > 0
    ) or scope.get("objective_queue_quality_included") is not (objective_count > 0):
        raise ValueError("quality exclusion assembly scope inclusion flags mismatch")

    authority = receipt.get("authority")
    if type(authority) is not dict or set(authority) != set(FALSE_AUTHORITY_FIELDS):
        raise ValueError("quality exclusion assembly authority schema mismatch")
    for field in FALSE_AUTHORITY_FIELDS:
        _require_bool(
            authority.get(field), False, f"quality exclusion assembly authority.{field}"
        )

    input_sha256 = receipt.get("input_sha256")
    if type(input_sha256) is not dict or set(input_sha256) != (
        QUALITY_ASSEMBLY_INPUT_SHA_FIELDS
    ):
        raise ValueError("quality exclusion assembly input SHA schema mismatch")
    for field, digest in input_sha256.items():
        _require_sha256(digest, f"quality exclusion assembly input_sha256.{field}")

    observed_code = receipt.get("observed_code_sha256")
    if type(observed_code) is not dict or set(observed_code) != (
        QUALITY_ASSEMBLY_CODE_SHA_FIELDS
    ):
        raise ValueError("quality exclusion assembly observed-code schema mismatch")
    normalized_observed = {
        str(field): _require_sha256(
            digest, f"quality exclusion assembly observed_code_sha256.{field}"
        )
        for field, digest in observed_code.items()
    }
    if normalized_observed != _quality_assembly_code_hashes():
        raise ValueError("quality exclusion assembly observed code is stale")

    expected_marker_content = _quality_assembly_marker_bytes(
        manifest_content=quality_content, receipt_content=receipt_content
    )
    if marker_content != expected_marker_content:
        raise ValueError("quality exclusion assembly marker mismatch")
    if _stable_bytes(validator_path, "quality assembly validator rehash") != validator_content:
        raise RuntimeError("quality assembly validator changed during validation")
    bundle = _QualityAssemblyBundle(
        root=root,
        manifest_path=resolved_manifest,
        receipt_path=resolved_receipt,
        marker_path=marker_path,
        manifest_content=quality_content,
        receipt_content=receipt_content,
        marker_content=marker_content,
        observed_code_sha256=normalized_observed,
        validator_path=validator_path,
        validator_content=validator_content,
    )
    # Zero is a real result only within this complete, revalidated receipt and
    # marker. A naked empty manifest remains invalid at the standalone parser.
    _validate_quality_manifest(quality_value, assembly_bundle=bundle)
    return bundle


def _rehash_operational_quality_assembly(bundle: _QualityAssemblyBundle) -> None:
    if _stable_bytes(bundle.validator_path, "quality assembly validator final rehash") != (
        bundle.validator_content
    ):
        raise RuntimeError("quality assembly validator changed")
    if {entry.name for entry in bundle.root.iterdir()} != set(
        QUALITY_ASSEMBLY_FILES.values()
    ):
        raise RuntimeError("quality exclusion assembly directory changed")
    for path, expected, description in (
        (bundle.manifest_path, bundle.manifest_content, "manifest"),
        (bundle.receipt_path, bundle.receipt_content, "receipt"),
        (bundle.marker_path, bundle.marker_content, "marker"),
    ):
        if _stable_bytes(path, f"quality exclusion assembly {description} final rehash") != (
            expected
        ):
            raise RuntimeError(f"quality exclusion assembly {description} changed")
    if _quality_assembly_code_hashes() != bundle.observed_code_sha256:
        raise RuntimeError("quality exclusion assembly observed code changed")
