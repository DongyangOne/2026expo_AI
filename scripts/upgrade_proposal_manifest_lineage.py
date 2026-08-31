"""Upgrade legacy proposal-crop manifests to the strict lineage schema.

This tool is intentionally conservative.  It hashes both the source image and
the emitted crop, refuses any train/validation identity leakage, and never
labels an upgraded legacy manifest as eligible for a blind deployment gate.

The upgrader writes equivalent CSV and JSONL manifests together with lineage
and rejection reports.  All artifacts are rendered before the first write, so
``--dry-run`` performs the complete validation without touching the filesystem.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import io
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np


ROLE_BY_SPLIT = {
    "training": "train",
    "validation": "model_validation",
}
CANONICAL_FIELDS = (
    "sample_id",
    "role",
    "split_role",
    "fold",
    "source_sha256",
    "image_sha256",
    "content_identity",
    "object_group",
    "capture_session",
    "origin",
    "selection_reason",
)
REQUIRED_LEGACY_FIELDS = {
    "filepath",
    "split",
    "source_id",
    "source_path_b64",
    "material",
    "category",
    "dent",
    "label",
    "foreign_material",
    "source_object_count",
    "crop_object_count",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[/\\]")
DECLARED_SOURCE_HASH_FIELDS = ("source_sha256", "source_image_sha256")
DECLARED_IMAGE_HASH_FIELDS = ("image_sha256", "crop_sha256", "content_sha256")
MATERIAL_CLASS_NAMES = (
    "can",
    "pet",
    "paper",
    "plastic",
    "styrofoam",
    "vinyl",
    "glass",
    "battery",
    "fluorescent",
    "background",
)
VALIDATOR_ARTIFACT_ROLE = "v4_development_candidates_not_blind_or_deployment_authority"
VALIDATOR_MANIFEST_SCHEMA = "proposal_verifier.v4.bgfix.v1"
VALIDATOR_BACKGROUND_POLICY = "strict-zero-intersection"
VALIDATOR_BACKGROUND_MARGIN = 0.10
VALIDATOR_BINDING_FIELDS = (
    "input_manifest_sha256",
    "dataset_info_sha256",
    "detector_model_sha256",
    "inference_spec_sha256",
    "validated_manifest_sha256",
)


class UpgradeRejected(ValueError):
    """Raised when one or more rows make a safe upgrade impossible."""

    def __init__(self, rejections: Sequence[Mapping[str, object]]) -> None:
        self.rejections = [dict(item) for item in rejections]
        summary = "; ".join(str(item.get("error", "rejected")) for item in self.rejections[:3])
        if len(self.rejections) > 3:
            summary += f"; and {len(self.rejections) - 3} more"
        super().__init__(summary)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sha256_file(path: Path) -> str:
    """Hash one regular file and reject a concurrent metadata change."""

    before = path.stat()
    digest = _sha256_file(path)
    after = path.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise ValueError(f"file changed while hashing: {path}")
    return digest


def _required_sha256(value: object, *, field: str) -> str:
    digest = str(value or "").strip().lower()
    if not SHA256_RE.fullmatch(digest):
        raise ValueError(f"{field} must be a 64-character lowercase SHA-256")
    return digest


def _load_validator_report(
    path: Path,
    *,
    expected_report_sha256: str,
    validated_manifest_sha256: str,
    validated_manifest_rows: int,
) -> tuple[dict[str, object], str]:
    """Load the validator attestation and bind it to the supplied CSV bytes."""

    if not path.is_file():
        raise FileNotFoundError(f"validator report does not exist: {path}")
    raw = path.read_bytes()
    actual_report_sha = _sha256_bytes(raw)
    if actual_report_sha != _required_sha256(
        expected_report_sha256, field="validator_report_sha256"
    ):
        raise ValueError("validator report SHA-256 differs from the trusted pin")
    try:
        report = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid validator report JSON: {path}") from error
    if not isinstance(report, dict):
        raise ValueError("validator report must be a JSON object")
    expected_contract = {
        "schema_version": 1,
        "artifact_role": VALIDATOR_ARTIFACT_ROLE,
        "ready_for_lineage_upgrade": True,
        "blind_test_eligible": False,
        "production_deployment_authorized": False,
    }
    for field, expected in expected_contract.items():
        if report.get(field) != expected:
            raise ValueError(f"validator report has invalid {field}")
    if report.get("rows") != validated_manifest_rows:
        raise ValueError("validator report row count differs from validated manifest")
    bindings = report.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("validator report bindings must be an object")
    normalized_bindings = {
        field: _required_sha256(bindings.get(field), field=f"validator.bindings.{field}")
        for field in VALIDATOR_BINDING_FIELDS
    }
    if normalized_bindings["validated_manifest_sha256"] != validated_manifest_sha256:
        raise ValueError("validator report does not bind the supplied validated manifest")
    contract = report.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("validator report contract must be an object")
    expected_semantics = {
        "manifest_schema_version": VALIDATOR_MANIFEST_SCHEMA,
        "background_policy": VALIDATOR_BACKGROUND_POLICY,
        "background_gt_margin": VALIDATOR_BACKGROUND_MARGIN,
        "source_object_count_semantics": "complete_source_frame",
        "crop_object_count_semantics": "final_padded_verifier_crop",
        "explicit_label_file_required": True,
        "visual_judge_still_required": True,
    }
    for field, expected in expected_semantics.items():
        if contract.get(field) != expected:
            raise ValueError(f"validator report contract has invalid {field}")
    provenance = contract.get("proposal_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("validator proposal provenance must be an object")
    for field in (
        "detector_artifact_bytes_bound",
        "inference_spec_bytes_bound",
        "dataset_info_bytes_bound",
        "source_bbox_crop_bytes_recomputed",
    ):
        if provenance.get(field) is not True:
            raise ValueError(f"validator proposal provenance has invalid {field}")
    if provenance.get("production_or_blind_authority") is not False:
        raise ValueError("validator proposal provenance grants forbidden authority")
    normalized = dict(report)
    normalized["bindings"] = normalized_bindings
    return normalized, actual_report_sha


def _perceptual_hash(path: Path) -> int:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot decode crop for pHash quarantine: {path}")
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(np.float32(resized))[:8, :8].reshape(-1)
    median = float(np.median(coefficients[1:]))
    bits = coefficients > median
    bits[0] = False
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def _phash_bucket_keys(value: int, threshold: int) -> tuple[tuple[int, int], ...]:
    widths = [64 // (threshold + 1)] * (threshold + 1)
    for index in range(64 % (threshold + 1)):
        widths[index] += 1
    keys = []
    offset = 0
    for index, width in enumerate(widths):
        keys.append((index, (value >> offset) & ((1 << width) - 1)))
        offset += width
    return tuple(keys)


def _clean_row(row: Mapping[str, object], *, location: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_key, raw_value in row.items():
        if raw_key is None:
            raise ValueError(f"{location}: unnamed extra CSV column")
        key = str(raw_key).strip()
        if not key:
            raise ValueError(f"{location}: empty column name")
        if key in result:
            raise ValueError(f"{location}: duplicate column {key!r}")
        result[key] = "" if raw_value is None else str(raw_value).strip()
    return result


def _read_csv(path: Path) -> tuple[list[dict[str, str]], bytes]:
    if not path.is_file():
        raise FileNotFoundError(f"input manifest does not exist: {path}")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"input manifest is not UTF-8: {path}") from error
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""))
        fields = [str(field).strip() for field in (reader.fieldnames or [])]
        if not fields:
            raise ValueError(f"input CSV has no header: {path}")
        if any(not field for field in fields) or len(fields) != len(set(fields)):
            raise ValueError(f"input CSV has empty or duplicate headers: {path}")
        missing = sorted(REQUIRED_LEGACY_FIELDS - set(fields))
        if missing:
            raise ValueError(f"input CSV is missing required columns {missing}: {path}")
        rows = [
            _clean_row(row, location=f"{path}:{line}")
            for line, row in enumerate(reader, start=2)
        ]
    except csv.Error as error:
        raise ValueError(f"cannot parse input CSV {path}: {error}") from error
    if not rows:
        raise ValueError(f"input CSV is empty: {path}")
    return rows, raw


def _decode_source_path(value: str) -> str:
    """Decode one URL-safe base64 filesystem path with strict validation."""
    encoded = value.strip()
    if not encoded:
        raise ValueError("source_path_b64 is empty")
    try:
        ascii_value = encoded.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("source_path_b64 is not ASCII") from error
    if len(ascii_value) % 4 == 1:
        raise ValueError("source_path_b64 has invalid length")
    ascii_value += b"=" * ((-len(ascii_value)) % 4)
    try:
        decoded = base64.b64decode(ascii_value, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("source_path_b64 is malformed") from error
    if not decoded or b"\x00" in decoded:
        raise ValueError("source_path_b64 decodes to an empty or NUL-containing path")
    decoded_path = os.fsdecode(decoded)
    if not decoded_path.strip():
        raise ValueError("source_path_b64 decodes to an empty path")
    return decoded_path


def _encode_source_path(path: Path) -> str:
    return base64.urlsafe_b64encode(os.fsencode(str(path))).decode("ascii")


def _portable_normalized(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    if len(normalized) > 1:
        normalized = normalized.rstrip("/")
    return normalized


def _portable_key(value: str) -> str:
    normalized = _portable_normalized(value)
    if WINDOWS_DRIVE_RE.match(normalized) or value.startswith(("\\\\", "//")):
        return normalized.casefold()
    return normalized


def parse_path_remap(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("path remap must use FROM=TO")
    source, destination = value.split("=", 1)
    source = source.strip()
    destination = destination.strip()
    if not source or not destination:
        raise argparse.ArgumentTypeError("path remap FROM and TO must be non-empty")
    return source, destination


def _apply_path_remaps(value: str, remaps: Sequence[tuple[str, str]]) -> str:
    original_normalized = _portable_normalized(value)
    original_key = _portable_key(value)
    matches: list[tuple[int, str, str, str]] = []
    for source, destination in remaps:
        source_normalized = _portable_normalized(source)
        source_key = _portable_key(source)
        if original_key == source_key or original_key.startswith(source_key + "/"):
            matches.append((len(source_key), source_normalized, source_key, destination))
    if not matches:
        return value
    _, source_normalized, _, destination = max(matches, key=lambda item: item[0])
    suffix = original_normalized[len(source_normalized) :].lstrip("/")
    if suffix:
        return destination.rstrip("/\\") + os.sep + suffix.replace("/", os.sep)
    return destination


def _portable_is_absolute(value: str) -> bool:
    return (
        Path(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or PurePosixPath(value.replace("\\", "/")).is_absolute()
    )


def _resolve_existing_path(
    raw_path: str,
    *,
    manifest_dir: Path,
    remaps: Sequence[tuple[str, str]],
    kind: str,
) -> Path:
    remapped = _apply_path_remaps(raw_path, remaps)
    candidate = Path(remapped)
    if not _portable_is_absolute(remapped):
        candidate = manifest_dir / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"{kind} file does not exist: {candidate}")
    return candidate


def _declared_hash(
    row: Mapping[str, str], fields: Iterable[str], *, location: str
) -> str:
    values = {
        row.get(field, "").strip().lower()
        for field in fields
        if row.get(field, "").strip()
    }
    if len(values) > 1:
        raise ValueError(f"{location}: conflicting declared SHA-256 fields")
    if not values:
        return ""
    value = next(iter(values))
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"{location}: declared SHA-256 must be 64 lowercase hex characters")
    return value


def _verify_hash(
    row: Mapping[str, str],
    fields: Iterable[str],
    actual: str,
    *,
    location: str,
    kind: str,
) -> None:
    declared = _declared_hash(row, fields, location=location)
    if declared and declared != actual:
        raise ValueError(f"{location}: declared {kind} SHA-256 does not match file content")


def _load_group_map(path: Path | None) -> tuple[dict[str, dict[str, str]], str | None]:
    if path is None:
        return {}, None
    if not path.is_file():
        raise FileNotFoundError(f"group map does not exist: {path}")
    raw = path.read_bytes()
    try:
        parsed = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid group map JSON: {path}") from error
    if isinstance(parsed, dict) and "groups" in parsed:
        parsed = parsed["groups"]
    if not isinstance(parsed, dict):
        raise ValueError("group map must be an object keyed by source SHA or decoded path")
    result: dict[str, dict[str, str]] = {}
    for raw_key, raw_value in parsed.items():
        key = str(raw_key).strip()
        if not key or not isinstance(raw_value, dict):
            raise ValueError("every group map entry must be an object with a non-empty key")
        object_group = str(raw_value.get("object_group", "")).strip()
        capture_session = str(raw_value.get("capture_session", "")).strip()
        if not object_group or not capture_session:
            raise ValueError(
                f"group map entry {key!r} requires object_group and capture_session"
            )
        normalized_key = key.lower() if SHA256_RE.fullmatch(key.lower()) else _portable_key(key)
        entry = {
            "object_group": object_group,
            "capture_session": capture_session,
        }
        previous = result.get(normalized_key)
        if previous is not None and previous != entry:
            raise ValueError(f"conflicting group map entries for {key!r}")
        result[normalized_key] = entry
    return result, _sha256_bytes(raw)


def _mapped_group(
    group_map: Mapping[str, Mapping[str, str]],
    *,
    source_sha256: str,
    decoded_path: str,
    resolved_path: Path,
) -> tuple[str, str] | None:
    candidates = (
        source_sha256,
        _portable_key(decoded_path),
        _portable_key(str(resolved_path)),
    )
    matches = {
        (
            str(group_map[key]["object_group"]),
            str(group_map[key]["capture_session"]),
        )
        for key in candidates
        if key in group_map
    }
    if len(matches) > 1:
        raise ValueError("group map has conflicting SHA/path matches for one source")
    return next(iter(matches)) if matches else None


def _sample_id(source_sha256: str, image_sha256: str, object_group: str) -> str:
    payload = json.dumps(
        {
            "image_sha256": image_sha256,
            "object_group": object_group,
            "source_sha256": source_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "proposal_" + _sha256_bytes(payload)[:24]


def _set_canonical(row: dict[str, str], field: str, value: str) -> None:
    """Set a strict field while retaining a changed legacy value."""
    previous = row.get(field, "")
    if field in row and previous != value:
        legacy_name = f"legacy_{field}"
        if legacy_name in row and row[legacy_name] != previous:
            raise ValueError(f"cannot preserve original {field}: {legacy_name} already conflicts")
        row[legacy_name] = previous
    row[field] = value


def _normalize_role(row: Mapping[str, str], *, location: str) -> tuple[str, str]:
    split = row.get("split", "").strip().lower()
    if split not in ROLE_BY_SPLIT:
        raise ValueError(
            f"{location}: unexpected split {split!r}; only training/validation can be upgraded"
        )
    role = ROLE_BY_SPLIT[split]
    for field in ("role", "split_role"):
        explicit = row.get(field, "").strip().lower()
        if explicit and explicit != role:
            raise ValueError(
                f"{location}: unexpected {field}={explicit!r}; expected {role!r} from split"
            )
    return split, role


def _validate_core_labels(row: Mapping[str, str], *, location: str) -> None:
    source_id = row.get("source_id", "").strip()
    if not source_id:
        raise ValueError(f"{location}: source_id is empty")
    try:
        material = int(row.get("material", ""))
    except ValueError as error:
        raise ValueError(f"{location}: material must be an integer") from error
    if material not in range(len(MATERIAL_CLASS_NAMES)):
        raise ValueError(f"{location}: material must be 0..9")
    category = row.get("category", "").strip().lower()
    if category != MATERIAL_CLASS_NAMES[material]:
        raise ValueError(
            f"{location}: category {category!r} does not match material {material} "
            f"({MATERIAL_CLASS_NAMES[material]!r})"
        )
    for field in ("dent", "label", "foreign_material"):
        try:
            value = int(row.get(field, ""))
        except ValueError as error:
            raise ValueError(f"{location}: {field} must be an integer") from error
        if value not in {-1, 0, 1}:
            raise ValueError(f"{location}: {field} must be -1, 0, or 1")
    try:
        source_object_count = int(row.get("source_object_count", ""))
    except ValueError as error:
        raise ValueError(f"{location}: source_object_count must be an integer") from error
    if source_object_count not in {0, 1}:
        raise ValueError(f"{location}: source_object_count must be zero or one")
    try:
        crop_object_count = int(row.get("crop_object_count", ""))
    except ValueError as error:
        raise ValueError(f"{location}: crop_object_count must be an integer") from error
    expected_crop_object_count = 0 if material == len(MATERIAL_CLASS_NAMES) - 1 else 1
    if crop_object_count != expected_crop_object_count:
        raise ValueError(
            f"{location}: crop_object_count={crop_object_count} conflicts with "
            f"material={material}; expected {expected_crop_object_count}"
        )
    if crop_object_count > source_object_count:
        raise ValueError(
            f"{location}: crop_object_count exceeds source_object_count"
        )


def _source_reference_bbox(
    row: Mapping[str, str], *, location: str
) -> dict[str, str]:
    """Preserve or derive the source-reference bbox expected by strict audits."""
    source_fields = (
        "source_bbox_x",
        "source_bbox_y",
        "source_bbox_w",
        "source_bbox_h",
    )
    crop_fields = ("crop_x1", "crop_y1", "crop_x2", "crop_y2")
    present_source = [bool(row.get(field, "").strip()) for field in source_fields]
    present_crop = [bool(row.get(field, "").strip()) for field in crop_fields]
    if any(present_source) and not all(present_source):
        raise ValueError(f"{location}: source bbox fields are incomplete")
    if any(present_crop) and not all(present_crop):
        raise ValueError(f"{location}: crop bbox fields are incomplete")
    if all(present_source):
        try:
            x, y, width, height = (float(row[field]) for field in source_fields)
        except ValueError as error:
            raise ValueError(f"{location}: source bbox fields must be numeric") from error
    elif all(present_crop):
        try:
            x1, y1, x2, y2 = (float(row[field]) for field in crop_fields)
        except ValueError as error:
            raise ValueError(f"{location}: crop bbox fields must be numeric") from error
        x, y, width, height = x1, y1, x2 - x1, y2 - y1
    else:
        return {}
    if width <= 0 or height <= 0:
        raise ValueError(f"{location}: source-reference bbox must have positive size")

    def render(value: float) -> str:
        return str(int(value)) if value.is_integer() else format(value, ".12g")

    return dict(zip(source_fields, (render(x), render(y), render(width), render(height))))


def _normalize_row(
    row: Mapping[str, str],
    *,
    manifest_path: Path,
    line: int,
    remaps: Sequence[tuple[str, str]],
    group_map: Mapping[str, Mapping[str, str]],
    origin: str,
) -> dict[str, str]:
    location = f"{manifest_path}:{line}"
    split, role = _normalize_role(row, location=location)
    _validate_core_labels(row, location=location)
    decoded_source = _decode_source_path(row.get("source_path_b64", ""))
    source_path = _resolve_existing_path(
        decoded_source,
        manifest_dir=manifest_path.parent,
        remaps=remaps,
        kind="source",
    )
    crop_path = _resolve_existing_path(
        row.get("filepath", ""),
        manifest_dir=manifest_path.parent,
        remaps=remaps,
        kind="crop",
    )
    source_sha = _stable_sha256_file(source_path)
    image_sha = _stable_sha256_file(crop_path)
    _verify_hash(
        row,
        DECLARED_SOURCE_HASH_FIELDS,
        source_sha,
        location=location,
        kind="source",
    )
    _verify_hash(
        row,
        DECLARED_IMAGE_HASH_FIELDS,
        image_sha,
        location=location,
        kind="crop",
    )

    trusted = _mapped_group(
        group_map,
        source_sha256=source_sha,
        decoded_path=decoded_source,
        resolved_path=source_path,
    )
    if trusted is None:
        object_group = f"source_sha256:{source_sha}"
        capture_session = f"source_sha256:{source_sha}"
        selection_reason = "source_sha_group_fallback"
    else:
        object_group, capture_session = trusted
        selection_reason = "trusted_group_map"

    normalized = dict(row)
    for field, value in _source_reference_bbox(row, location=location).items():
        if not normalized.get(field, "").strip():
            normalized[field] = value
    canonical = {
        "filepath": crop_path.as_posix(),
        "source_path_b64": _encode_source_path(source_path),
        "split": split,
        "sample_id": _sample_id(source_sha, image_sha, object_group),
        "role": role,
        "split_role": role,
        "fold": row.get("fold", "").strip() or role,
        "source_sha256": source_sha,
        "image_sha256": image_sha,
        "content_identity": f"sha256:{image_sha}",
        "object_group": object_group,
        "capture_session": capture_session,
        "origin": origin,
        "selection_reason": selection_reason,
    }
    for field, value in canonical.items():
        _set_canonical(normalized, field, value)
    return normalized


def _verify_row_files_unchanged(rows: Sequence[Mapping[str, str]]) -> None:
    """Re-hash source and crop files before publishing their identities."""

    source_cache: dict[str, str] = {}
    crop_cache: dict[str, str] = {}
    for row in rows:
        sample_id = row["sample_id"]
        decoded_source = _decode_source_path(row["source_path_b64"])
        source_path = Path(decoded_source).resolve()
        crop_path = Path(row["filepath"]).resolve()
        source_key = str(source_path)
        crop_key = str(crop_path)
        actual_source = source_cache.get(source_key)
        if actual_source is None:
            actual_source = _stable_sha256_file(source_path)
            source_cache[source_key] = actual_source
        actual_crop = crop_cache.get(crop_key)
        if actual_crop is None:
            actual_crop = _stable_sha256_file(crop_path)
            crop_cache[crop_key] = actual_crop
        if actual_source != row["source_sha256"]:
            raise ValueError(f"source file changed during upgrade for {sample_id}")
        if actual_crop != row["image_sha256"]:
            raise ValueError(f"crop file changed during upgrade for {sample_id}")


def _partition_leakage(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    errors: list[dict[str, object]] = []
    for field in ("source_sha256", "image_sha256", "object_group", "capture_session"):
        partitions: dict[str, set[tuple[str, str]]] = defaultdict(set)
        samples: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            value = row[field].lower() if field.endswith("sha256") else row[field]
            partitions[value].add((row["role"], row["fold"]))
            samples[value].add(row["sample_id"])
        for value in sorted(partitions):
            values = partitions[value]
            roles = {role for role, _ in values}
            folds_by_role: dict[str, set[str]] = defaultdict(set)
            for role, fold in values:
                folds_by_role[role].add(fold)
            if len(roles) > 1:
                errors.append(
                    {
                        "code": "cross_role_leakage",
                        "field": field,
                        "identity": value,
                        "partitions": [list(item) for item in sorted(values)],
                        "sample_ids": sorted(samples[value]),
                        "error": f"{field} crosses mutually exclusive roles",
                    }
                )
            elif any(len(folds) > 1 for folds in folds_by_role.values()):
                errors.append(
                    {
                        "code": "cross_fold_leakage",
                        "field": field,
                        "identity": value,
                        "partitions": [list(item) for item in sorted(values)],
                        "sample_ids": sorted(samples[value]),
                        "error": f"{field} crosses folds within one role",
                    }
                )
    return errors


def _deduplicate(rows: Sequence[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    unique: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = row["sample_id"]
        previous = unique.get(sample_id)
        if previous is None:
            unique[sample_id] = row
            continue
        if previous != row:
            differing = sorted(
                key
                for key in set(previous) | set(row)
                if previous.get(key, "") != row.get(key, "")
            )
            raise ValueError(
                f"duplicate conflicting sample {sample_id}: fields {', '.join(differing)}"
            )
    ordered = sorted(
        unique.values(),
        key=lambda item: (item["role"], item["fold"], item["sample_id"]),
    )
    return ordered, len(rows) - len(ordered)


def _quarantine_validation_near_train(
    rows: Sequence[dict[str, str]], threshold: int
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    """Drop only validation crops visually near a training crop.

    Training data is never removed or relabeled.  This is a conservative,
    deterministic fallback for legacy datasets that have no trustworthy
    physical-object/sequence mapping.  A later strict combined audit must still
    fail on any cross-partition near duplicate left by other manifests.
    """
    if not 0 <= threshold <= 7:
        raise ValueError("near-pHash quarantine distance must be between 0 and 7")

    train_by_hash: dict[int, list[dict[str, str]]] = defaultdict(list)
    buckets: dict[tuple[int, int], set[int]] = defaultdict(set)
    for row in sorted(
        (item for item in rows if item["role"] == "train"),
        key=lambda item: item["sample_id"],
    ):
        phash = _perceptual_hash(Path(row["filepath"]))
        train_by_hash[phash].append(row)
        for key in _phash_bucket_keys(phash, threshold):
            buckets[key].add(phash)

    quarantined: list[dict[str, object]] = []
    quarantined_ids: set[str] = set()
    for row in sorted(
        (item for item in rows if item["role"] == "model_validation"),
        key=lambda item: item["sample_id"],
    ):
        phash = _perceptual_hash(Path(row["filepath"]))
        candidates: set[int] = set()
        for key in _phash_bucket_keys(phash, threshold):
            candidates.update(buckets.get(key, ()))
        matches = sorted(
            (
                (int((phash ^ candidate).bit_count()), candidate)
                for candidate in candidates
                if (phash ^ candidate).bit_count() <= threshold
            ),
            key=lambda item: (item[0], item[1]),
        )
        if not matches:
            continue
        minimum_distance = matches[0][0]
        matching_train_rows = sorted(
            (
                candidate_row
                for distance, candidate_hash in matches
                if distance == minimum_distance
                for candidate_row in train_by_hash[candidate_hash]
            ),
            key=lambda item: item["sample_id"],
        )
        quarantined_ids.add(row["sample_id"])
        quarantined.append(
            {
                "sample_id": row["sample_id"],
                "source_sha256": row["source_sha256"],
                "image_sha256": row["image_sha256"],
                "material": int(row["material"]),
                "category": row["category"],
                "phash": f"{phash:016x}",
                "minimum_distance": minimum_distance,
                "matching_train_sample_ids": [
                    item["sample_id"] for item in matching_train_rows[:8]
                ],
            }
        )

    kept = [row for row in rows if row["sample_id"] not in quarantined_ids]
    quarantined.sort(key=lambda item: str(item["sample_id"]))
    return kept, quarantined


def _fieldnames(rows: Sequence[Mapping[str, str]]) -> list[str]:
    strict = [
        "filepath",
        "split",
        "source_id",
        "material",
        "category",
        "dent",
        "label",
        "foreign_material",
        "source_object_count",
        *CANONICAL_FIELDS,
    ]
    extras = sorted({key for row in rows for key in row} - set(strict))
    return [*strict, *extras]


def _render_csv(rows: Sequence[Mapping[str, str]], fields: Sequence[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return output.getvalue().encode("utf-8")


def _render_jsonl(rows: Sequence[Mapping[str, str]], fields: Sequence[str]) -> bytes:
    return "".join(
        json.dumps(
            {field: row.get(field, "") for field in fields},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    ).encode("utf-8")


def _render_json(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _stage_exclusive(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _publish_exclusive_bundle(artifacts: Sequence[tuple[Path, bytes]]) -> None:
    targets = [path.resolve(strict=False) for path, _ in artifacts]
    existing = [path for path in targets if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing artifact: {existing[0]}")
    staged: list[Path] = []
    published: list[Path] = []
    try:
        staged = [_stage_exclusive(path, content) for path, content in artifacts]
        for temporary, target in zip(staged, targets, strict=True):
            os.link(temporary, target)
            published.append(target)
    except BaseException:
        for target in reversed(published):
            target.unlink(missing_ok=True)
        raise
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)


def _artifact_paths_unique(paths: Sequence[Path]) -> None:
    resolved = [path.resolve(strict=False) for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError("output CSV, JSONL, lineage, and rejections paths must be different")


def _validate_path_remaps(remaps: Sequence[tuple[str, str]]) -> None:
    destinations: dict[str, str] = {}
    for source, destination in remaps:
        if not source.strip() or not destination.strip():
            raise ValueError("path remap FROM and TO must be non-empty")
        key = _portable_key(source)
        previous = destinations.get(key)
        if previous is not None and _portable_key(previous) != _portable_key(destination):
            raise ValueError(f"conflicting path remaps for {source!r}")
        destinations[key] = destination


def upgrade_proposal_manifests(
    *,
    inputs: Sequence[Path],
    validator_report_paths: Sequence[Path],
    validator_report_sha256s: Sequence[str],
    output_csv: Path,
    output_jsonl: Path,
    lineage_path: Path,
    rejections_path: Path,
    path_remaps: Sequence[tuple[str, str]] = (),
    group_map_path: Path | None = None,
    group_map_sha256: str | None = None,
    origin: str = "legacy_proposal_manifest_upgrade",
    quarantine_validation_near_phash_distance: int | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, object]:
    """Upgrade legacy CSVs or raise :class:`UpgradeRejected` without manifests."""
    if not inputs:
        raise ValueError("at least one input manifest is required")
    if len(validator_report_paths) != len(inputs) or len(validator_report_sha256s) != len(inputs):
        raise ValueError("every input manifest requires one validator report and trusted SHA pin")
    if not origin.strip():
        raise ValueError("origin must be non-empty")
    if (
        quarantine_validation_near_phash_distance is not None
        and not 0 <= quarantine_validation_near_phash_distance <= 7
    ):
        raise ValueError(
            "quarantine_validation_near_phash_distance must be between 0 and 7"
        )
    _validate_path_remaps(path_remaps)
    if overwrite:
        raise ValueError("overwrite is forbidden for immutable lineage artifacts")
    if group_map_path is None:
        if group_map_sha256 is not None:
            raise ValueError("group_map_sha256 requires group_map_path")
        if quarantine_validation_near_phash_distance is None:
            raise ValueError(
                "pHash quarantine is required when no trusted group map is supplied"
            )
    elif group_map_sha256 is None:
        raise ValueError("group_map_sha256 is required with group_map_path")
    outputs = (output_csv, output_jsonl, lineage_path, rejections_path)
    _artifact_paths_unique(outputs)
    if output_csv.suffix.lower() != ".csv":
        raise ValueError("output_csv must use the .csv extension")
    if output_jsonl.suffix.lower() not in {".jsonl", ".ndjson"}:
        raise ValueError("output_jsonl must use the .jsonl or .ndjson extension")
    if not dry_run:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite existing artifacts: {existing}")

    group_map, group_map_sha = _load_group_map(group_map_path)
    if group_map_path is not None and group_map_sha != _required_sha256(
        group_map_sha256, field="group_map_sha256"
    ):
        raise ValueError("group map SHA-256 differs from the trusted pin")
    loaded: list[
        tuple[Path, list[dict[str, str]], bytes, Path, dict[str, object], str]
    ] = []
    for raw_path, raw_report_path, report_pin in zip(
        inputs, validator_report_paths, validator_report_sha256s, strict=True
    ):
        path = Path(raw_path).resolve()
        rows, raw = _read_csv(path)
        report_path = Path(raw_report_path).resolve()
        report, report_sha = _load_validator_report(
            report_path,
            expected_report_sha256=report_pin,
            validated_manifest_sha256=_sha256_bytes(raw),
            validated_manifest_rows=len(rows),
        )
        loaded.append((path, rows, raw, report_path, report, report_sha))
    loaded.sort(key=lambda item: (_sha256_bytes(item[2]), item[0].as_posix()))

    normalized: list[dict[str, str]] = []
    rejections: list[dict[str, object]] = []
    input_entries: list[dict[str, object]] = []
    validator_entries: list[dict[str, object]] = []
    for path, rows, raw, report_path, validator_report, validator_report_sha in loaded:
        input_entries.append(
            {
                "path": path.as_posix(),
                "sha256": _sha256_bytes(raw),
                "rows": len(rows),
            }
        )
        validator_entries.append(
            {
                "path": report_path.as_posix(),
                "sha256": validator_report_sha,
                "artifact_role": validator_report["artifact_role"],
                "ready_for_lineage_upgrade": True,
                "bindings": validator_report["bindings"],
            }
        )
        for line, row in enumerate(rows, start=2):
            try:
                normalized.append(
                    _normalize_row(
                        row,
                        manifest_path=path,
                        line=line,
                        remaps=path_remaps,
                        group_map=group_map,
                        origin=origin.strip(),
                    )
                )
            except (OSError, UnicodeError, ValueError) as error:
                rejections.append(
                    {
                        "code": "invalid_row",
                        "input": path.as_posix(),
                        "line": line,
                        "error": str(error),
                    }
                )

    quarantined_validation: list[dict[str, object]] = []
    if (
        not rejections
        and quarantine_validation_near_phash_distance is None
        and any(row["selection_reason"] == "source_sha_group_fallback" for row in normalized)
    ):
        rejections.append(
            {
                "code": "untrusted_group_fallback",
                "error": (
                    "trusted group map does not cover every source and pHash "
                    "quarantine is disabled"
                ),
            }
        )
    if not rejections:
        rejections.extend(_partition_leakage(normalized))
        try:
            rows, duplicates_removed = _deduplicate(normalized)
        except ValueError as error:
            rejections.append({"code": "duplicate_conflict", "error": str(error)})
            rows = []
            duplicates_removed = 0
        if (
            not rejections
            and quarantine_validation_near_phash_distance is not None
        ):
            try:
                rows, quarantined_validation = _quarantine_validation_near_train(
                    rows, quarantine_validation_near_phash_distance
                )
            except (OSError, ValueError, cv2.error) as error:
                rejections.append(
                    {
                        "code": "near_phash_quarantine_failed",
                        "error": str(error),
                    }
                )
                rows = []
    else:
        rows = []
        duplicates_removed = 0

    rejection_report: dict[str, object] = {
        "schema_version": 1,
        "builder": "scripts/upgrade_proposal_manifest_lineage.py",
        "blind_test_eligible": False,
        "inputs": input_entries,
        "rejected_count": len(rejections),
        "quarantined_validation_count": len(quarantined_validation),
        "quarantined_validation": quarantined_validation,
        "rejections": sorted(
            rejections,
            key=lambda item: (
                str(item.get("input", "")),
                int(item.get("line", 0)),
                str(item.get("code", "")),
                str(item.get("field", "")),
                str(item.get("identity", "")),
            ),
        ),
    }
    rejection_bytes = _render_json(rejection_report)
    if rejections:
        if not dry_run:
            _publish_exclusive_bundle(((rejections_path, rejection_bytes),))
        raise UpgradeRejected(rejection_report["rejections"])

    _verify_row_files_unchanged(rows)
    fields = _fieldnames(rows)
    csv_bytes = _render_csv(rows, fields)
    jsonl_bytes = _render_jsonl(rows, fields)
    role_counts = Counter(row["role"] for row in rows)
    reason_counts = Counter(row["selection_reason"] for row in rows)
    lineage: dict[str, object] = {
        "schema_version": 1,
        "builder": "scripts/upgrade_proposal_manifest_lineage.py",
        "blind_test_eligible": False,
        "blind_test_ineligibility_reason": (
            "legacy proposal manifests lack an independently collected blind holdout"
        ),
        "inputs": input_entries,
        "validator_reports": validator_entries,
        "path_remaps": [
            {"from": source, "to": destination} for source, destination in path_remaps
        ],
        "group_map": (
            None
            if group_map_path is None
            else {
                "path": Path(group_map_path).resolve().as_posix(),
                "sha256": group_map_sha,
                "entries": len(group_map),
            }
        ),
        "grouping_policy": {
            "trusted_mapping_reason": "trusted_group_map",
            "fallback_reason": "source_sha_group_fallback",
            "fallback_is_conservative": False,
            "fallback_requires_phash_quarantine": group_map_path is None,
        },
        "duplicates_removed": duplicates_removed,
        "near_phash_quarantine": {
            "enabled": quarantine_validation_near_phash_distance is not None,
            "distance": quarantine_validation_near_phash_distance,
            "policy": "drop_model_validation_near_train_only",
            "training_rows_removed": 0,
            "validation_rows_removed": len(quarantined_validation),
            "removed_by_category": dict(
                sorted(
                    Counter(
                        str(item["category"]) for item in quarantined_validation
                    ).items()
                )
            ),
        },
        "rows": len(rows),
        "role_counts": dict(sorted(role_counts.items())),
        "selection_reason_counts": dict(sorted(reason_counts.items())),
        "unique_sources": len({row["source_sha256"] for row in rows}),
        "unique_images": len({row["image_sha256"] for row in rows}),
        "unique_object_groups": len({row["object_group"] for row in rows}),
        "outputs": {
            "csv": {
                "path": output_csv.resolve(strict=False).as_posix(),
                "sha256": _sha256_bytes(csv_bytes),
            },
            "jsonl": {
                "path": output_jsonl.resolve(strict=False).as_posix(),
                "sha256": _sha256_bytes(jsonl_bytes),
            },
            "rejections": {
                "path": rejections_path.resolve(strict=False).as_posix(),
                "sha256": _sha256_bytes(rejection_bytes),
            },
        },
        "dry_run": dry_run,
    }
    lineage_bytes = _render_json(lineage)
    if not dry_run:
        _verify_row_files_unchanged(rows)
        artifacts = (
            (output_csv, csv_bytes),
            (output_jsonl, jsonl_bytes),
            (rejections_path, rejection_bytes),
            (lineage_path, lineage_bytes),
        )
        _publish_exclusive_bundle(artifacts)
    return lineage


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Upgrade legacy proposal crop CSVs to strict, leakage-safe lineage manifests."
        )
    )
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--validator-report", action="append", required=True, type=Path)
    parser.add_argument(
        "--validator-report-sha256", action="append", required=True
    )
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--lineage-json", required=True, type=Path)
    parser.add_argument("--rejections-json", required=True, type=Path)
    parser.add_argument(
        "--path-remap",
        action="append",
        default=[],
        type=parse_path_remap,
        metavar="FROM=TO",
    )
    parser.add_argument("--group-map", type=Path)
    parser.add_argument("--group-map-sha256")
    parser.add_argument("--origin", default="legacy_proposal_manifest_upgrade")
    parser.add_argument(
        "--quarantine-validation-near-phash-distance",
        type=int,
        metavar="0..7",
        help=(
            "원본 object/sequence ID가 없을 때 train crop과 이 거리 이내인 "
            "model_validation crop만 결정적으로 제외합니다."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        lineage = upgrade_proposal_manifests(
            inputs=args.input,
            validator_report_paths=args.validator_report,
            validator_report_sha256s=args.validator_report_sha256,
            output_csv=args.output_csv,
            output_jsonl=args.output_jsonl,
            lineage_path=args.lineage_json,
            rejections_path=args.rejections_json,
            path_remaps=args.path_remap,
            group_map_path=args.group_map,
            group_map_sha256=args.group_map_sha256,
            origin=args.origin,
            quarantine_validation_near_phash_distance=(
                args.quarantine_validation_near_phash_distance
            ),
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    except UpgradeRejected as error:
        print(
            json.dumps(
                {"ok": False, "rejected_count": len(error.rejections), "rejections": error.rejections},
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ),
            flush=True,
        )
        return 2
    print(json.dumps(lineage, ensure_ascii=False, sort_keys=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
