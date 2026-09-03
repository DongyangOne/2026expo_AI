"""Inventory Pi request captures and create a privacy-safe VLM teacher queue.

Existing audited train/validation assignments are retained by SHA-256.  New
images are never assigned the deployed prediction as ground truth; they are
sent to a separate teacher queue without client_id values.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

KST = timezone(timedelta(hours=9))
OPERATIONAL_CAPTURE_CUTOFF_KST = datetime(2026, 8, 1, 0, 0, 0, tzinfo=KST)
MINIMUM_IMAGE_WIDTH = 160
MINIMUM_IMAGE_HEIGHT = 120
EXTREME_EXPOSURE_FRACTION = 0.995
UNDEREXPOSED_LUMA_MAX = 5
OVEREXPOSED_LUMA_MIN = 250
OBJECTIVE_EVIDENCE_SCHEMA = "operational_objective_quality_evidence.v1"
OBJECTIVE_EVIDENCE_ROLE = (
    "operational_objective_capture_quality_local_evidence_"
    "not_ground_truth_or_authority"
)
OBJECTIVE_REJECTIONS_FILE = "objective_quality_rejections.jsonl"
OBJECTIVE_RECEIPT_FILE = "objective_quality_evidence.json"
OBJECTIVE_CAPTURE_INDEX_FILE = "capture_metadata_index.jsonl"
OBJECTIVE_RAW_REASONS = frozenset(
    {
        "image_unreadable",
        "image_resolution_below_minimum",
        "image_extreme_underexposure",
        "image_extreme_overexposure",
    }
)
OBJECTIVE_REASON_PRIORITY = (
    ("image_unreadable", "objective_unreadable"),
    ("image_resolution_below_minimum", "resolution_too_low"),
    ("image_extreme_underexposure", "extreme_exposure"),
    ("image_extreme_overexposure", "extreme_exposure"),
)
INTEGRITY_REASONS = frozenset(
    {
        "image_missing",
        "invalid_image_sha256",
        "image_sha256_mismatch",
    }
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
AUTHORITY_FIELDS = (
    "selection",
    "ground_truth",
    "replay",
    "training",
    "calibration",
    "blind_test",
    "deployment",
)
OUTPUT_FILES = {
    "capture_index": OBJECTIVE_CAPTURE_INDEX_FILE,
    "capture_inventory": "capture_inventory.json",
    "teacher_queue": "teacher_queue.jsonl",
    "objective_rejections": OBJECTIVE_REJECTIONS_FILE,
    "summary": "queue_summary.json",
    "objective_receipt": OBJECTIVE_RECEIPT_FILE,
}


def _datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO-8601 string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include an explicit UTC offset")
    return parsed


def _valid_sha256(value: object) -> str | None:
    normalized = str(value or "").strip().lower()
    if not SHA256_RE.fullmatch(normalized):
        return None
    return normalized


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


def _load_json(content: bytes, *, description: str) -> object:
    try:
        return json.loads(
            content.decode("utf-8-sig"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not valid UTF-8 JSON") from error


def _load_known_audit_bytes(content: bytes) -> dict[str, dict[str, object]]:
    value = _load_json(content, description="known audit")
    if type(value) is not dict:
        raise ValueError("known audit must be a SHA-keyed JSON object")
    normalized: dict[str, dict[str, object]] = {}
    for raw_key, row in value.items():
        key = _valid_sha256(raw_key)
        if key is None or type(raw_key) is not str or raw_key != key:
            raise ValueError("known audit contains an invalid SHA-256 key")
        if key in normalized:
            raise ValueError("known audit contains a case-insensitive duplicate SHA")
        if type(row) is not dict or not row:
            raise ValueError("known audit values must be non-empty objects")
        if row.get("split") not in {"train", "validation", "protected_validation"}:
            raise ValueError("known audit split is invalid")
        normalized[key] = dict(row)
    return normalized


def _reject_symlink_components(path: Path, *, description: str) -> None:
    absolute = Path(os.path.abspath(path))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{description} path contains a symlink component")


def _stable_regular_bytes(path: Path, *, description: str) -> tuple[Path, bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} must be a regular non-symlink file")
    _reject_symlink_components(path, description=description)
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    with resolved.open("rb") as handle:
        opened_before = os.fstat(handle.fileno())
        content = handle.read()
        opened_after = os.fstat(handle.fileno())
    after = resolved.stat()
    identity = lambda value: (  # noqa: E731 - exact file identity tuple
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if not (
        identity(before)
        == identity(opened_before)
        == identity(opened_after)
        == identity(after)
    ):
        raise RuntimeError(f"{description} changed while being read")
    return resolved, content


def _publish_directory_no_replace(source: Path, destination: Path) -> None:
    if os.name == "nt":
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
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            os.fspath(destination),
        )


def _objective_quality_reason(reasons: list[str]) -> str | None:
    reason_set = set(reasons)
    if (
        not reason_set
        or reason_set.intersection(INTEGRITY_REASONS)
        or not reason_set <= OBJECTIVE_RAW_REASONS
    ):
        return None
    for raw_reason, quality_reason in OBJECTIVE_REASON_PRIORITY:
        if raw_reason in reason_set:
            return quality_reason
    return None


def _resolve_capture_image(captures_root: Path, value: object) -> tuple[Path, str]:
    """Resolve an untrusted metadata path without allowing it to escape root."""
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or "\\" in value
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError("image path must be a non-empty relative string")
    relative = PurePosixPath(value)
    if re.match(r"^[A-Za-z]:", value) or relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ) or relative.as_posix() != value:
        raise ValueError("image path must stay relative to the capture root")
    candidate = captures_root
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink():
            raise ValueError("image path contains a symlink component")
    resolved = candidate.resolve(strict=False)
    try:
        portable = resolved.relative_to(captures_root)
    except ValueError as error:
        raise ValueError("image path escapes the capture root") from error
    return resolved, portable.as_posix()


def _capture_relative_ref(captures_root: Path, path: Path) -> str:
    resolved = path.resolve(strict=True)
    try:
        portable = resolved.relative_to(captures_root).as_posix()
    except ValueError as error:
        raise ValueError("capture metadata escapes the capture root") from error
    if not portable or PurePosixPath(portable).is_absolute() or any(
        part in {"", ".", ".."} for part in PurePosixPath(portable).parts
    ) or "\\" in portable or "\n" in portable or "\r" in portable:
        raise ValueError("capture metadata reference is not normalized")
    return portable


def _image_quality_assessment(
    path: Path, declared_sha256: object
) -> tuple[list[str], Path | None, bytes | None]:
    """Return only objective, conservative capture-quality failures.

    Blur and deployed-model predictions are deliberately excluded from this
    gate: blur thresholds are camera/domain dependent, and deployed output can
    never become a data-selection authority for its own retraining set.
    """
    reasons = []
    if path.is_symlink() or not path.is_file():
        return ["image_missing"], None, None
    try:
        resolved, content = _stable_regular_bytes(path, description="capture image")
    except (OSError, RuntimeError, ValueError):
        return ["image_missing"], None, None

    declared = _valid_sha256(declared_sha256)
    if (
        declared is None
        or type(declared_sha256) is not str
        or declared_sha256 != declared
    ):
        reasons.append("invalid_image_sha256")
    elif _sha256_bytes(content) != declared:
        reasons.append("image_sha256_mismatch")

    try:
        encoded = np.frombuffer(content, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except cv2.error:
        image = None
    if image is None or image.ndim != 3:
        reasons.append("image_unreadable")
        return sorted(set(reasons)), resolved, content

    height, width = image.shape[:2]
    if width < MINIMUM_IMAGE_WIDTH or height < MINIMUM_IMAGE_HEIGHT:
        reasons.append("image_resolution_below_minimum")

    luma = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels = float(luma.size)
    if (
        pixels
        and np.count_nonzero(luma <= UNDEREXPOSED_LUMA_MAX) / pixels
        >= EXTREME_EXPOSURE_FRACTION
    ):
        reasons.append("image_extreme_underexposure")
    if (
        pixels
        and np.count_nonzero(luma >= OVEREXPOSED_LUMA_MIN) / pixels
        >= EXTREME_EXPOSURE_FRACTION
    ):
        reasons.append("image_extreme_overexposure")
    return sorted(set(reasons)), resolved, content


def _image_quality_reasons(path: Path, declared_sha256: object) -> list[str]:
    return _image_quality_assessment(path, declared_sha256)[0]


def _nearest_shadow(
    metadata: dict, shadows_by_client: dict[str, list[dict]], max_seconds: float = 30
) -> dict | None:
    client_id = metadata.get("request", {}).get("client_id")
    timestamp = _datetime(metadata["timestamp"])
    candidates = shadows_by_client.get(client_id, [])
    if not candidates:
        return None
    best = min(candidates, key=lambda row: abs((_datetime(row["timestamp"]) - timestamp).total_seconds()))
    delta = abs((_datetime(best["timestamp"]) - timestamp).total_seconds())
    return best if delta <= max_seconds else None


def _quality_policy() -> dict[str, object]:
    return {
        "minimum_width": MINIMUM_IMAGE_WIDTH,
        "minimum_height": MINIMUM_IMAGE_HEIGHT,
        "extreme_exposure_fraction": EXTREME_EXPOSURE_FRACTION,
        "underexposed_luma_max": UNDEREXPOSED_LUMA_MAX,
        "overexposed_luma_min": OVEREXPOSED_LUMA_MIN,
        "blur_filter_enabled": False,
        "deployed_prediction_filter_enabled": False,
        "objective_reason_priority": [
            {"raw_reason": raw, "quality_reason": quality}
            for raw, quality in OBJECTIVE_REASON_PRIORITY
        ],
    }


def _jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(_json_bytes(row, pretty=False) for row in rows)


def prepare_queue(
    *,
    captures_dir: Path,
    shadow_log: Path,
    known_audit: Path,
    output_dir: Path,
    start_kst: datetime,
) -> dict:
    if start_kst.tzinfo is None or start_kst.utcoffset() is None:
        raise ValueError("start_kst must include an explicit UTC offset")
    start_kst = start_kst.astimezone(KST)
    if start_kst < OPERATIONAL_CAPTURE_CUTOFF_KST:
        raise ValueError(
            "start_kst cannot be earlier than the fixed operational capture cutoff "
            f"{OPERATIONAL_CAPTURE_CUTOFF_KST.isoformat()}"
        )
    captures_dir_arg = Path(os.path.abspath(captures_dir))
    if captures_dir_arg.is_symlink() or not captures_dir_arg.is_dir():
        raise NotADirectoryError(f"captures_dir is not a directory: {captures_dir}")
    _reject_symlink_components(captures_dir_arg, description="captures_dir")
    captures_root = captures_dir_arg.resolve(strict=True)

    normalized_output = Path(os.path.abspath(output_dir))
    if normalized_output.exists() or normalized_output.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite immutable output: {normalized_output}"
        )
    output_parent = normalized_output.parent
    if output_parent.is_symlink() or not output_parent.is_dir():
        raise ValueError("output parent must be an existing non-symlink directory")
    _reject_symlink_components(output_parent, description="output parent")
    output_parent = output_parent.resolve(strict=True)
    normalized_output = output_parent / normalized_output.name
    if normalized_output.is_relative_to(captures_root):
        raise ValueError("output directory must not be inside captures_dir")

    known_path, known_content = _stable_regular_bytes(
        known_audit, description="known audit"
    )
    known = _load_known_audit_bytes(known_content)
    shadow_present = shadow_log.exists() or shadow_log.is_symlink()
    shadow_path: Path | None = None
    shadow_content: bytes | None = None
    if shadow_present:
        shadow_path, shadow_content = _stable_regular_bytes(
            shadow_log, description="shadow log"
        )

    initial_metadata_paths = tuple(sorted(captures_root.rglob("*.json")))
    latest_by_sha: dict[str, tuple[dict, datetime, Path, str]] = {}
    objective_rows_by_sha: dict[str, dict[str, object]] = {}
    metadata_bindings: dict[Path, bytes] = {}
    source_bindings: dict[Path, bytes] = {}
    metadata_snapshots: list[tuple[Path, str, bytes]] = []
    for metadata_path in initial_metadata_paths:
        resolved_metadata, metadata_content = _stable_regular_bytes(
            metadata_path, description="capture metadata snapshot"
        )
        metadata_ref = _capture_relative_ref(captures_root, resolved_metadata)
        metadata_bindings[resolved_metadata] = metadata_content
        metadata_snapshots.append(
            (resolved_metadata, metadata_ref, metadata_content)
        )
    metadata_snapshots.sort(key=lambda value: value[1])
    capture_index = [
        {
            "schema_version": 1,
            "metadata_ref": metadata_ref,
            "metadata_sha256": _sha256_bytes(metadata_content),
        }
        for _, metadata_ref, metadata_content in metadata_snapshots
    ]
    rows_after_cutoff = 0
    rejected_capture_rows = 0
    rejection_counts = Counter()
    for resolved_metadata, metadata_ref, metadata_content in metadata_snapshots:
        try:
            metadata = _load_json(metadata_content, description="capture metadata")
            if type(metadata) is not dict:
                raise ValueError("capture metadata must be an object")
            captured_at = _datetime(metadata["timestamp"])
        except (OSError, KeyError, TypeError, ValueError):
            rejected_capture_rows += 1
            rejection_counts["capture_timestamp_missing_invalid_or_naive"] += 1
            continue
        captured_at_kst = captured_at.astimezone(KST)
        if captured_at_kst < start_kst:
            continue
        rows_after_cutoff += 1
        try:
            image_metadata = metadata["image"]
            sha256 = _valid_sha256(image_metadata["sha256"])
            source_image_path, image_ref = _resolve_capture_image(
                captures_root, image_metadata["path"]
            )
        except (KeyError, TypeError, ValueError):
            sha256 = None
            source_image_path = Path()
            image_ref = ""
            rejected_capture_rows += 1
            rejection_counts["image_path_invalid_or_outside_capture_root"] += 1
            continue
        quality_reasons, resolved_source, source_content = _image_quality_assessment(
            source_image_path, image_metadata["sha256"]
        )
        if resolved_source is not None and source_content is not None:
            source_bindings[resolved_source] = source_content
        if quality_reasons:
            rejected_capture_rows += 1
            rejection_counts.update(quality_reasons)
            quality_reason = _objective_quality_reason(quality_reasons)
            if quality_reason is not None:
                assert sha256 is not None
                if sha256 in objective_rows_by_sha:
                    raise ValueError(
                        "duplicate objective-quality source SHA is ambiguous"
                    )
                metadata_sha = _sha256_bytes(metadata_content)
                objective_rows_by_sha[sha256] = {
                    "schema_version": 1,
                    "source_sha256": sha256,
                    "capture_timestamp_utc": captured_at.astimezone(
                        timezone.utc
                    ).isoformat().replace("+00:00", "Z"),
                    "metadata_ref": metadata_ref,
                    "metadata_sha256": metadata_sha,
                    "image_ref": image_ref,
                    "raw_reasons": quality_reasons,
                    "quality_reason": quality_reason,
                }
            continue
        assert sha256 is not None
        previous = latest_by_sha.get(sha256)
        if previous is None or captured_at > previous[1]:
            latest_by_sha[sha256] = (
                metadata,
                captured_at,
                source_image_path,
                image_ref,
            )

    shadows_by_client: dict[str, list[dict]] = defaultdict(list)
    if shadow_content is not None:
        try:
            shadow_text = shadow_content.decode("utf-8-sig")
        except UnicodeError as error:
            raise ValueError("shadow log is not valid UTF-8") from error
        for line_number, line in enumerate(shadow_text.splitlines(), 1):
            if line.strip():
                try:
                    row = json.loads(
                        line, object_pairs_hook=_reject_duplicate_keys
                    )
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"shadow log has invalid JSON at line {line_number}"
                    ) from error
                if type(row) is not dict:
                    raise ValueError(
                        f"shadow log line {line_number} must be an object"
                    )
                try:
                    _datetime(row["timestamp"])
                except (KeyError, TypeError, ValueError):
                    continue
                shadows_by_client[row.get("client_id")].append(row)

    inventory = []
    teacher_queue = []
    decisions = Counter()
    for sha256, (metadata, captured_at, source_image_path, image_ref) in sorted(
        latest_by_sha.items(), key=lambda item: item[1][1]
    ):
        is_known = sha256 in known
        known_row = known[sha256] if is_known else None
        result = metadata.get("result", {})
        classification = result.get("classification") or {}
        shadow = _nearest_shadow(metadata, shadows_by_client)
        verifier = ((shadow or {}).get("verifier") or {}).get("material") or {}
        if is_known:
            decision = (
                "known_train" if known_row.get("split") == "train" else "protected_validation"
            )
            expected = known_row.get("label")
        else:
            decision = "teacher_required"
            expected = None
        decisions[decision] += 1
        row = {
            "sha256": sha256,
            "timestamp": metadata["timestamp"],
            "image_ref": image_ref,
            "decision": decision,
            "known_label": expected,
            "deployed": {
                "status": result.get("status"),
                "class_name": classification.get("class_name"),
                "confidence": classification.get("confidence"),
                "bbox": result.get("bbox"),
            },
            "verifier": {
                "class_name": verifier.get("class_name"),
                "confidence": verifier.get("confidence"),
                "agreement": (shadow or {}).get("material_agreement"),
            },
        }
        inventory.append(row)
        if decision == "teacher_required":
            # Portable teacher input contains only the immutable image identity,
            # capture time and a capture-root-relative reference.  Private IDs,
            # deployed predictions and verifier predictions remain diagnostic
            # inventory only and can never steer the teacher.
            teacher_queue.append(
                {
                    "sha256": sha256,
                    "timestamp": metadata["timestamp"],
                    "image_ref": image_ref,
                    "decision": "teacher_required",
                }
            )

    suppressed_known = sorted(set(objective_rows_by_sha).intersection(known))
    for sha256 in suppressed_known:
        del objective_rows_by_sha[sha256]
    objective_rows = [objective_rows_by_sha[sha] for sha in sorted(objective_rows_by_sha)]
    objective_reason_counts = dict(
        sorted(Counter(row["quality_reason"] for row in objective_rows).items())
    )
    quality_policy = _quality_policy()
    summary = {
        "operational_capture_cutoff_kst": OPERATIONAL_CAPTURE_CUTOFF_KST.isoformat(),
        "start_kst": start_kst.isoformat(),
        "capture_rows_after_cutoff": rows_after_cutoff,
        "capture_rows_rejected": rejected_capture_rows,
        "capture_rejection_counts": dict(sorted(rejection_counts.items())),
        "capture_metadata_index_rows": len(capture_index),
        "unique_images": len(latest_by_sha),
        "decisions": dict(decisions),
        "teacher_queue": len(teacher_queue),
        "objective_quality_rejections": len(objective_rows),
        "objective_quality_reason_counts": objective_reason_counts,
        "objective_known_audit_suppressed": len(suppressed_known),
        "objective_evidence_schema": OBJECTIVE_EVIDENCE_SCHEMA,
        "client_ids_exported": False,
        "quality_policy": quality_policy,
    }
    output_contents = {
        "capture_index": _jsonl_bytes(capture_index),
        "capture_inventory": _json_bytes(inventory),
        "teacher_queue": _jsonl_bytes(teacher_queue),
        "objective_rejections": _jsonl_bytes(objective_rows),
        "summary": _json_bytes(summary),
    }
    source_binding_value = [
        {
            "source_sha256": row["source_sha256"],
            "metadata_sha256": row["metadata_sha256"],
            "quality_reason": row["quality_reason"],
        }
        for row in objective_rows
    ]
    receipt = {
        "schema_version": 1,
        "evidence_schema": OBJECTIVE_EVIDENCE_SCHEMA,
        "artifact_role": OBJECTIVE_EVIDENCE_ROLE,
        "status": "objective_quality_evidence_prepared",
        "local_only": True,
        "operational_capture_cutoff_kst": (
            OPERATIONAL_CAPTURE_CUTOFF_KST.isoformat()
        ),
        "start_kst": start_kst.isoformat(),
        "quality_policy": quality_policy,
        "counts": {
            "capture_metadata_index_rows": len(capture_index),
            "objective_quality_rejections": len(objective_rows),
            "objective_known_audit_suppressed": len(suppressed_known),
            "objective_quality_reason_counts": objective_reason_counts,
        },
        "input_digests": {
            "known_audit_sha256": _sha256_bytes(known_content),
            "shadow_log_present": shadow_present,
            "shadow_log_sha256": (
                _sha256_bytes(shadow_content)
                if shadow_content is not None
                else None
            ),
            "objective_source_bindings_sha256": _sha256_bytes(
                _json_bytes(source_binding_value, pretty=False)
            ),
        },
        "output_digests": {
            f"{name}_sha256": _sha256_bytes(content)
            for name, content in sorted(output_contents.items())
        },
        "privacy": {
            "objective_evidence_structured_client_id_fields_exported": False,
            "objective_evidence_structured_device_id_fields_exported": False,
            "objective_evidence_prediction_outputs_exported": False,
            "objective_evidence_absolute_paths_exported": False,
            "objective_evidence_untrusted_relative_local_refs_present": True,
            "objective_evidence_relative_refs_may_contain_identifiers": True,
        },
        "authority": {field: False for field in AUTHORITY_FIELDS},
    }
    output_contents["objective_receipt"] = _json_bytes(receipt)

    staging = Path(
        tempfile.mkdtemp(prefix=f".{normalized_output.name}.", dir=output_parent)
    )
    try:
        for name, filename in OUTPUT_FILES.items():
            (staging / filename).write_bytes(output_contents[name])

        if _stable_regular_bytes(known_path, description="known audit final rehash")[
            1
        ] != known_content:
            raise RuntimeError("known audit changed before queue publication")
        if shadow_path is not None and shadow_content is not None:
            if _stable_regular_bytes(
                shadow_path, description="shadow log final rehash"
            )[1] != shadow_content:
                raise RuntimeError("shadow log changed before queue publication")
        elif shadow_log.exists() or shadow_log.is_symlink():
            raise RuntimeError("shadow log appeared before queue publication")
        for path, content in metadata_bindings.items():
            if _stable_regular_bytes(
                path, description="capture metadata final rehash"
            )[1] != content:
                raise RuntimeError("capture metadata changed before queue publication")
        for path, content in source_bindings.items():
            if _stable_regular_bytes(
                path, description="capture image final rehash"
            )[1] != content:
                raise RuntimeError("capture image changed before queue publication")
        if tuple(sorted(captures_root.rglob("*.json"))) != initial_metadata_paths:
            raise RuntimeError("capture metadata set changed before queue publication")
        if {path.name for path in staging.iterdir()} != set(OUTPUT_FILES.values()):
            raise RuntimeError("queue staging file set is not exact")
        for name, filename in OUTPUT_FILES.items():
            if _stable_regular_bytes(
                staging / filename, description=f"queue output {name}"
            )[1] != output_contents[name]:
                raise RuntimeError(f"queue output changed before publish: {name}")
        try:
            _publish_directory_no_replace(staging, normalized_output)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite immutable output: {normalized_output}"
            ) from error
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures-dir", required=True, type=Path)
    parser.add_argument("--shadow-log", required=True, type=Path)
    parser.add_argument("--known-audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-kst", default="2026-08-01T00:00:00+09:00")
    args = parser.parse_args()
    start = _datetime(args.start_kst).astimezone(KST)
    prepare_queue(
        captures_dir=args.captures_dir,
        shadow_log=args.shadow_log,
        known_audit=args.known_audit,
        output_dir=args.output_dir,
        start_kst=start,
    )


if __name__ == "__main__":
    main()
