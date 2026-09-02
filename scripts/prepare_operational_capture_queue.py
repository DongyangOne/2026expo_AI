"""Inventory Pi request captures and create a privacy-safe VLM teacher queue.

Existing audited train/validation assignments are retained by SHA-256.  New
images are never assigned the deployed prediction as ground truth; they are
sent to a separate teacher queue without client_id values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

try:
    from scripts.operational_teacher_contract import load_known_audit
except ModuleNotFoundError:
    from operational_teacher_contract import load_known_audit  # type: ignore


KST = timezone(timedelta(hours=9))
OPERATIONAL_CAPTURE_CUTOFF_KST = datetime(2026, 8, 1, 0, 0, 0, tzinfo=KST)
MINIMUM_IMAGE_WIDTH = 160
MINIMUM_IMAGE_HEIGHT = 120
EXTREME_EXPOSURE_FRACTION = 0.995
UNDEREXPOSED_LUMA_MAX = 5
OVEREXPOSED_LUMA_MIN = 250


def _datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO-8601 string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include an explicit UTC offset")
    return parsed


def _valid_sha256(value: object) -> str | None:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64:
        return None
    try:
        int(normalized, 16)
    except ValueError:
        return None
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_capture_image(captures_root: Path, value: object) -> tuple[Path, str]:
    """Resolve an untrusted metadata path without allowing it to escape root."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("image path must be a non-empty relative string")
    relative = Path(value.strip())
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise ValueError("image path must stay relative to the capture root")
    resolved = (captures_root / relative).resolve(strict=False)
    try:
        portable = resolved.relative_to(captures_root)
    except ValueError as error:
        raise ValueError("image path escapes the capture root") from error
    return resolved, portable.as_posix()


def _image_quality_reasons(path: Path, declared_sha256: object) -> list[str]:
    """Return only objective, conservative capture-quality failures.

    Blur and deployed-model predictions are deliberately excluded from this
    gate: blur thresholds are camera/domain dependent, and deployed output can
    never become a data-selection authority for its own retraining set.
    """
    reasons = []
    if not path.is_file():
        return ["image_missing"]

    declared = _valid_sha256(declared_sha256)
    if declared is None:
        reasons.append("invalid_image_sha256")
    elif _sha256_file(path) != declared:
        reasons.append("image_sha256_mismatch")

    try:
        encoded = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except (OSError, cv2.error):
        image = None
    if image is None or image.ndim != 3:
        reasons.append("image_unreadable")
        return sorted(set(reasons))

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
    return sorted(set(reasons))


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
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    captures_root = captures_dir.resolve(strict=True)
    if not captures_root.is_dir():
        raise NotADirectoryError(f"captures_dir is not a directory: {captures_dir}")
    latest_by_sha: dict[str, tuple[dict, Path, datetime, Path, str]] = {}
    rows_after_cutoff = 0
    rejected_capture_rows = 0
    rejection_counts = Counter()
    for metadata_path in sorted(captures_dir.rglob("*.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            captured_at = _datetime(metadata["timestamp"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
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
        quality_reasons = _image_quality_reasons(source_image_path, sha256)
        if quality_reasons:
            rejected_capture_rows += 1
            rejection_counts.update(quality_reasons)
            continue
        assert sha256 is not None
        previous = latest_by_sha.get(sha256)
        if previous is None or captured_at > previous[2]:
            latest_by_sha[sha256] = (
                metadata,
                metadata_path,
                captured_at,
                source_image_path,
                image_ref,
            )

    shadows_by_client: dict[str, list[dict]] = defaultdict(list)
    if shadow_log.is_file():
        for line in shadow_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                try:
                    _datetime(row["timestamp"])
                except (KeyError, TypeError, ValueError):
                    continue
                shadows_by_client[row.get("client_id")].append(row)
    known = load_known_audit(known_audit)

    inventory = []
    teacher_queue = []
    decisions = Counter()
    for sha256, (metadata, metadata_path, captured_at, source_image_path, image_ref) in sorted(
        latest_by_sha.items(), key=lambda item: item[1][2]
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

    (output_dir / "capture_inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "teacher_queue.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in teacher_queue),
        encoding="utf-8",
    )
    summary = {
        "operational_capture_cutoff_kst": OPERATIONAL_CAPTURE_CUTOFF_KST.isoformat(),
        "start_kst": start_kst.isoformat(),
        "capture_rows_after_cutoff": rows_after_cutoff,
        "capture_rows_rejected": rejected_capture_rows,
        "capture_rejection_counts": dict(sorted(rejection_counts.items())),
        "unique_images": len(latest_by_sha),
        "decisions": dict(decisions),
        "teacher_queue": len(teacher_queue),
        "client_ids_exported": False,
        "quality_policy": {
            "minimum_width": MINIMUM_IMAGE_WIDTH,
            "minimum_height": MINIMUM_IMAGE_HEIGHT,
            "extreme_exposure_fraction": EXTREME_EXPOSURE_FRACTION,
            "underexposed_luma_max": UNDEREXPOSED_LUMA_MAX,
            "overexposed_luma_min": OVEREXPOSED_LUMA_MIN,
            "blur_filter_enabled": False,
            "deployed_prediction_filter_enabled": False,
        },
    }
    (output_dir / "queue_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
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
