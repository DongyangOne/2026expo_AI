"""Build an immutable SHA-only capture-quality exclusion manifest for v4.

The adjudication source is a strict CSV with exactly ``path,reason`` columns.
Paths are resolved beneath ``--image-root`` only to hash the source bytes; no
path, filename, timestamp, or other identifying value is copied to the output.
This manifest is an exclusion policy input, never label or model authority.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath


QUALITY_EXCLUSION_CONTRACT = "v4_capture_quality_exclusions.sha256_reason_only.v1"
QUALITY_EXCLUSION_ROLE = (
    "v4_capture_quality_exclusion_manifest_selection_only_"
    "not_ground_truth_or_authority"
)
QUALITY_EXCLUSION_REASONS = (
    "severe_frame_crop",
    "person_occlusion_or_dominance",
    "excessive_background_or_multi_object",
    "unreadable_boundary",
    "too_low_resolution",
    "extreme_exposure",
)
QUALITY_EXCLUSION_REASON_ALIASES = {
    "clutter_or_multiple_objects": "excessive_background_or_multi_object",
    "boundary_unreadable": "unreadable_boundary",
    "objective_unreadable": "unreadable_boundary",
    "resolution_too_low": "too_low_resolution",
}
QUALITY_EXCLUSION_INPUT_REASONS = (
    *QUALITY_EXCLUSION_REASONS,
    *QUALITY_EXCLUSION_REASON_ALIASES,
)
OPERATIONAL_CUTOFF_REASON = "captured_before_2026_08_01"
OBJECT_CONDITION_REASONS = frozenset({"dent", "crush", "object_dented"})
QUALITY_EXCLUSION_MAX_SOURCES = 100
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _canonical_entries_bytes(entries: list[dict[str, str]]) -> bytes:
    return (
        json.dumps(
            entries,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _stable_bytes(path: Path, *, description: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} must be a regular non-symlink file: {path}")
    before = path.stat()
    with path.open("rb") as handle:
        opened_before = os.fstat(handle.fileno())
        content = handle.read()
        opened_after = os.fstat(handle.fileno())
    after = path.stat()
    identity = lambda value: (  # noqa: E731 - compact exact identity tuple
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
        raise RuntimeError(f"{description} changed while being read: {path}")
    return content


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_quality_reason(reason: str, *, description: str) -> str:
    if reason == OPERATIONAL_CUTOFF_REASON:
        raise ValueError(
            f"{description} uses {OPERATIONAL_CUTOFF_REASON}; the operational "
            "cutoff is enforced from captured_at, not as a quality reason"
        )
    if reason in OBJECT_CONDITION_REASONS:
        raise ValueError(
            f"{description} uses a dent/crush/object condition, "
            "not a capture-quality reason"
        )
    canonical = QUALITY_EXCLUSION_REASON_ALIASES.get(reason, reason)
    if canonical not in QUALITY_EXCLUSION_REASONS:
        raise ValueError(f"{description} has unknown reason")
    return canonical


def _reject_symlink_components(path: Path, *, description: str) -> None:
    absolute = Path(os.path.abspath(path))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{description} path must not contain symlink components")


def _manifest_value(entries: list[dict[str, str]]) -> dict[str, object]:
    """Serialize the contract; this helper does not validate or publish evidence.

    Empty values are used only inside the full operational assembler after its
    objective and subjective coverage checks. The standalone producer still
    requires a nonempty adjudication source list.
    """
    if any(entry.get("reason") not in QUALITY_EXCLUSION_REASONS for entry in entries):
        raise ValueError("quality exclusion manifest entries must use canonical reasons")
    entries = sorted(entries, key=lambda item: item["source_sha256"])
    if len(entries) > QUALITY_EXCLUSION_MAX_SOURCES:
        raise ValueError(
            "quality exclusion manifest exceeds max_excluded_sources="
            f"{QUALITY_EXCLUSION_MAX_SOURCES}"
        )
    reason_counts = Counter(item["reason"] for item in entries)
    authority = {
        "selection": False,
        "ground_truth": False,
        "replay": False,
        "training": False,
        "calibration": False,
        "blind_test": False,
        "deployment": False,
    }
    return {
        "schema_version": 1,
        "artifact_role": QUALITY_EXCLUSION_ROLE,
        "quality_exclusion_contract": QUALITY_EXCLUSION_CONTRACT,
        "status": "quality_exclusions_ready",
        "excluded_source_count": len(entries),
        "max_excluded_sources": QUALITY_EXCLUSION_MAX_SOURCES,
        "reason_counts": dict(sorted(reason_counts.items())),
        "source_list_sha256": _sha256_bytes(_canonical_entries_bytes(entries)),
        "entries": entries,
        "authority": authority,
    }


def _publish_exclusive(path: Path, content: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable output: {path}")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("output parent must be an existing non-symlink directory")
    cursor = Path(parent.anchor)
    for part in parent.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("output parent path must not contain symlink components")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite immutable output: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_source(image_root: Path, raw: str, *, row_number: int) -> Path:
    if not raw or "\n" in raw or "\r" in raw or "\\" in raw:
        raise ValueError(f"source row {row_number} has an invalid relative path")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"source row {row_number} path must be normalized and relative")
    candidate = image_root
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink():
            raise ValueError(
                f"source row {row_number} path must not contain symlink components"
            )
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(image_root)
    except ValueError as error:
        raise ValueError(f"source row {row_number} escapes image root") from error
    return resolved


def build_quality_exclusion_manifest(
    *, source_list: Path, image_root: Path, output_path: Path
) -> dict[str, object]:
    source_list_arg = source_list
    if source_list_arg.is_symlink() or not source_list_arg.is_file():
        raise ValueError("source_list must be a regular non-symlink file")
    _reject_symlink_components(source_list_arg, description="source_list")
    source_list = source_list_arg.resolve(strict=True)
    image_root_arg = image_root
    if image_root_arg.is_symlink() or not image_root_arg.is_dir():
        raise ValueError("image_root must be a regular non-symlink directory")
    _reject_symlink_components(image_root_arg, description="image_root")
    image_root = image_root_arg.resolve(strict=True)
    source_content = _stable_bytes(source_list, description="quality exclusion source list")
    try:
        decoded = source_content.decode("utf-8-sig")
        reader = csv.DictReader(decoded.splitlines())
        rows = list(reader)
    except (UnicodeError, csv.Error) as error:
        raise ValueError("quality exclusion source list is not valid UTF-8 CSV") from error
    if reader.fieldnames != ["path", "reason"]:
        raise ValueError("quality exclusion source list must have exact path,reason header")

    entries: list[dict[str, str]] = []
    seen_paths: set[Path] = set()
    seen_hashes: set[str] = set()
    source_bindings: list[tuple[Path, str]] = []
    for row_number, row in enumerate(rows, start=2):
        if set(row) != {"path", "reason"} or None in row:
            raise ValueError(f"quality exclusion source row {row_number} is malformed")
        reason = _canonical_quality_reason(
            row["reason"], description=f"quality exclusion source row {row_number}"
        )
        source = _resolve_source(image_root, row["path"], row_number=row_number)
        if source in seen_paths:
            raise ValueError(f"quality exclusion source row {row_number} duplicates a path")
        content = _stable_bytes(source, description=f"quality exclusion source row {row_number}")
        source_sha256 = _sha256_bytes(content)
        if not SHA256_RE.fullmatch(source_sha256) or source_sha256 in seen_hashes:
            raise ValueError(f"quality exclusion source row {row_number} duplicates source bytes")
        seen_paths.add(source)
        seen_hashes.add(source_sha256)
        source_bindings.append((source, source_sha256))
        entries.append({"source_sha256": source_sha256, "reason": reason})

    if not entries:
        raise ValueError(
            "quality exclusion manifest must contain at least one source; "
            "zero exclusions require the full operational quality assembler"
        )
    manifest = _manifest_value(entries)

    # Rehash all authority inputs immediately before the immutable publication.
    if _stable_bytes(
        source_list, description="quality exclusion source list final rehash"
    ) != source_content:
        raise RuntimeError("quality exclusion source list changed during build")
    for source, expected_sha in source_bindings:
        if _sha256_bytes(
            _stable_bytes(source, description="quality exclusion source final rehash")
        ) != expected_sha:
            raise RuntimeError(f"quality exclusion source changed during build: {source}")
    normalized_output = Path(os.path.abspath(output_path))
    _publish_exclusive(normalized_output, _json_bytes(manifest))
    return manifest


def build_single_quality_exclusion_manifest(
    *, source_path: Path, reason: str, output_path: Path
) -> dict[str, object]:
    reason = _canonical_quality_reason(
        reason, description="single-source quality exclusion reason"
    )
    source_arg = source_path
    if source_arg.is_symlink() or not source_arg.is_file():
        raise ValueError("single source must be a regular non-symlink file")
    _reject_symlink_components(source_arg, description="single source")
    source = source_arg.resolve(strict=True)
    content = _stable_bytes(source, description="single quality exclusion source")
    entry = {"source_sha256": _sha256_bytes(content), "reason": reason}
    manifest = _manifest_value([entry])
    if _stable_bytes(
        source, description="single quality exclusion source final rehash"
    ) != content:
        raise RuntimeError("single quality exclusion source changed during build")
    _publish_exclusive(Path(os.path.abspath(output_path)), _json_bytes(manifest))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-list", type=Path)
    source.add_argument("--source", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--reason", choices=QUALITY_EXCLUSION_INPUT_REASONS)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.source_list is not None:
        if args.image_root is None or args.reason is not None:
            parser.error("--source-list requires --image-root and forbids --reason")
    elif args.reason is None or args.image_root is not None:
        parser.error("--source requires --reason and forbids --image-root")
    return args


def main() -> None:
    args = parse_args()
    if args.source_list is not None:
        build_quality_exclusion_manifest(
            source_list=args.source_list,
            image_root=args.image_root,
            output_path=args.output,
        )
    else:
        build_single_quality_exclusion_manifest(
            source_path=args.source,
            reason=args.reason,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
