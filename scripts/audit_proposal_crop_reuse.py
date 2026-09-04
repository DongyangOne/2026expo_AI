"""Verify existing proposal crop bytes against pinned metadata, without new inference.

This is a reuse-candidate inventory, not the formal protected snapshot or training
approval. Original source paths/hashes are metadata claims; originals are not read.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import stat
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

METADATA_LIMIT = 32 * 1024**2
CROP_LIMIT = 64 * 1024**2
AUTHORITY = {
    "original_sources_rehashed": False, "formal_protected_coverage": False,
    "training_authorized": False, "blind_test_authorized": False,
    "deployment_authorized": False, "crop_transform_recomputed": False,
    "detector_inference_executed": False,
}
FIELDS = {"filepath", "source_sha256", "source_path_b64", "split", "image_sha256", "crop_bytes",
          "detector_model_sha256", "inference_spec_sha256", "source_width", "source_height",
          "crop_x1", "crop_y1", "crop_x2", "crop_y2"}


class ReuseAuditError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise ReuseAuditError(message)


def sha256_value(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "invalid SHA256")
    return value


def checked_path(value, *, exists=True):
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts, "absolute non-traversing path required")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink path forbidden")
    return path.resolve(strict=exists)


def read_file(path, limit, *, keep=False):
    """Bounded stable read; crop pixels are never decoded or copied to output."""
    path = checked_path(path)
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit, "file size/type guard")
    digest, chunks, total = hashlib.sha256(), [], 0
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(1024**2), b""):
            total += len(chunk)
            require(total <= limit, "file grew beyond read limit")
            digest.update(chunk)
            if keep:
                chunks.append(chunk)
        consumed = os.fstat(handle.fileno())
    after = checked_path(path).stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    require(len({identity(s) for s in (before, opened, consumed, after)}) == 1
            and total == before.st_size and before.st_ctime_ns == after.st_ctime_ns
            and opened.st_ctime_ns == consumed.st_ctime_ns, "file changed during read")
    return digest.hexdigest(), total, b"".join(chunks) if keep else None


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def source_path(value):
    require(type(value) is str and value.startswith("/") and "\\" not in value and "\x00" not in value,
            "source metadata requires an absolute POSIX path")
    path = PurePosixPath(value)
    require(".." not in path.parts and path.as_posix() == value and value != "/", "noncanonical source path")
    return value


def decode_source(value):
    require(type(value) is str, "missing encoded source path")
    try:
        raw = base64.b64decode(value, altchars=b"-_", validate=True)
        return source_path(raw.decode("utf-8", "surrogateescape"))
    except (ValueError, UnicodeError) as error:
        raise ReuseAuditError("invalid encoded source path") from error


def integer(value, *, positive=False):
    require(type(value) is str and re.fullmatch(r"0|[1-9][0-9]*", value) is not None, "invalid CSV integer")
    result = int(value)
    require(not positive or result > 0, "positive CSV integer required")
    return result


def crop_path(root, value, split):
    require(type(value) is str and value and "\\" not in value and "\x00" not in value,
            "invalid crop filepath")
    relative = PurePosixPath(value)
    require(not relative.is_absolute() and ":" not in value and ".." not in relative.parts
            and relative.as_posix() == value and relative.parts[0] == split,
            "crop filepath escapes or changes split")
    path = checked_path(root.joinpath(*relative.parts))
    require(path.is_relative_to(root), "crop escapes root")
    return path


def audit_reuse(*, manifest: Path, manifest_sha256: str, selection: Path, selection_sha256: str,
                crop_root: Path, model_sha256: str, spec_sha256: str, output: Path) -> dict:
    manifest_sha256, selection_sha256, model_sha256, spec_sha256 = map(
        sha256_value, (manifest_sha256, selection_sha256, model_sha256, spec_sha256))
    manifest, selection, crop_root = map(checked_path, (manifest, selection, crop_root))
    output = checked_path(output, exists=False)
    require(manifest != selection and crop_root.is_dir(), "distinct metadata and crop directory required")
    protected = (manifest.parent, selection.parent, crop_root, checked_path(Path(__file__).absolute()).parent)
    require(not output.exists() and not any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected),
            "output must be fresh and disjoint from metadata/code/crop roots")
    pins = {}
    blobs = []
    for path, expected in ((manifest, manifest_sha256), (selection, selection_sha256)):
        digest, size, blob = read_file(path, METADATA_LIMIT, keep=True)
        require(digest == expected, "metadata SHA256 mismatch")
        pins[path] = (digest, size, METADATA_LIMIT)
        blobs.append(blob)
    module = checked_path(Path(__file__).absolute())
    digest, size, _ = read_file(module, METADATA_LIMIT)
    pins[module] = (digest, size, METADATA_LIMIT)

    def invalid_constant(_):
        raise ReuseAuditError("nonfinite JSON constant")
    inventory = json.loads(blobs[1], object_pairs_hook=unique_object, parse_constant=invalid_constant)
    selected = inventory.get("selected_sources") if type(inventory) is dict else None
    require(type(selected) is list and bool(selected), "nonempty selected_sources required")
    by_sha, selected_paths = {}, set()
    for row in selected:
        require(type(row) is dict, "invalid selected source")
        digest, path = sha256_value(row.get("source_sha256")), source_path(row.get("path"))
        require(digest not in by_sha and path not in selected_paths, "duplicate selected source SHA/path")
        require(row.get("split") in ("training", "validation") and type(row.get("explicit_empty_label")) is bool
                and type(row.get("selection_cohort")) is str and bool(row["selection_cohort"]),
                "invalid selected source split/metadata")
        by_sha[digest] = row
        selected_paths.add(path)
        logical_parent = Path(path).parent
        if logical_parent.is_absolute():
            require(not output.is_relative_to(logical_parent), "output overlaps declared original source directory")
    reader = csv.DictReader(io.StringIO(blobs[0].decode("utf-8-sig"), newline=""))
    headers = reader.fieldnames
    require(headers is not None and len(headers) == len(set(headers)) and FIELDS <= set(headers),
            "missing or duplicate manifest headers")
    records, seen_sources, seen_paths, seen_files = [], set(), set(), set()
    crop_sha_groups = defaultdict(list)
    for row in reader:
        require(None not in row and all(v is not None for v in row.values()), "malformed CSV row")
        source_sha = sha256_value(row["source_sha256"])
        path, split = decode_source(row["source_path_b64"]), row["split"]
        require(source_sha in by_sha, "manifest source missing from selection")
        chosen = by_sha[source_sha]
        require(path == chosen["path"] and split == chosen["split"], "manifest source path/split differs from selection")
        require(source_sha not in seen_sources and path not in seen_paths, "duplicate manifest source")
        require(row["detector_model_sha256"] == model_sha256 and row["inference_spec_sha256"] == spec_sha256,
                "manifest detector/spec pin mismatch")
        actual_path = crop_path(crop_root, row["filepath"], split)
        require(actual_path not in seen_files, "duplicate crop filepath")
        expected_size = integer(row["crop_bytes"], positive=True)
        require(expected_size <= CROP_LIMIT, "declared crop exceeds size limit")
        width, height = (integer(row[key], positive=True) for key in ("source_width", "source_height"))
        x1, y1, x2, y2 = (integer(row[key]) for key in ("crop_x1", "crop_y1", "crop_x2", "crop_y2"))
        require(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, "invalid declared crop geometry")
        crop_sha = sha256_value(row["image_sha256"])
        actual_sha, size, _ = read_file(actual_path, CROP_LIMIT)
        require((actual_sha, size) == (crop_sha, expected_size), "crop size/SHA256 mismatch")
        pins[actual_path] = (actual_sha, size, CROP_LIMIT)
        seen_sources.add(source_sha)
        seen_paths.add(path)
        seen_files.add(actual_path)
        crop_sha_groups[crop_sha].append(len(records))
        records.append({"source_path": path, "source_sha256": source_sha, "split": split,
                        "crop_path": str(actual_path), "crop_sha256": actual_sha, "crop_bytes": size,
                        "declared_crop_xyxy": [x1, y1, x2, y2], "declared_source_size_wh": [width, height],
                        "selection_cohort": chosen["selection_cohort"],
                        "selection_explicit_empty_label": chosen["explicit_empty_label"]})
    require(bool(records), "empty crop manifest")
    missing = [{"source_path": row["path"], "source_sha256": digest, "split": row["split"],
                "selection_explicit_empty_label": row["explicit_empty_label"],
                "selection_cohort": row["selection_cohort"], "reason": "no_manifest_crop"}
               for digest, row in sorted(by_sha.items()) if digest not in seen_sources]
    duplicates = [{"crop_sha256": digest, "record_indices": indices}
                  for digest, indices in sorted(crop_sha_groups.items()) if len(indices) > 1]
    report = {"schema": "proposal_crop_reuse_audit.v1", "status": "reuse_candidates_verified",
              "scope": "existing crop bytes and metadata join only; not a formal protected snapshot",
              "selection_sources": len(selected), "verified_crop_rows": len(records), "missing_sources": len(missing),
              "verified_crop_bytes": sum(row["crop_bytes"] for row in records),
              "counts_by_split": dict(sorted(Counter(row["split"] for row in records).items())),
              "bindings": {"manifest_path": str(manifest), "manifest_sha256": manifest_sha256,
                           "selection_path": str(selection), "selection_sha256": selection_sha256,
                           "crop_root": str(crop_root), "declared_model_sha256": model_sha256,
                           "declared_spec_sha256": spec_sha256, "audit_code_sha256": pins[module][0]},
              "records": records, "missing_selection_sources": missing, "duplicate_crop_sha_groups": duplicates,
              "model_and_spec_files_rehashed": False, "crop_pixels_decoded": False, **AUTHORITY}
    payload = (json.dumps(report, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()

    def recheck():
        for path, (expected_sha, expected_size, limit) in pins.items():
            actual_sha, size, _ = read_file(path, limit)
            require((actual_sha, size) == (expected_sha, expected_size), "audit input changed before publication completed")

    recheck()
    output.mkdir(parents=True, exist_ok=False)
    identity = (output.stat().st_dev, output.stat().st_ino)
    report_path = output / "report.json"
    try:
        with report_path.open("xb") as handle:
            handle.write(payload)
        recheck()
        published_output = checked_path(output).stat()
        require((published_output.st_dev, published_output.st_ino) == identity, "output ownership changed")
        actual, size, _ = read_file(report_path, max(METADATA_LIMIT, len(payload)))
        require((actual, size) == (hashlib.sha256(payload).hexdigest(), len(payload)), "published report changed")
    except BaseException:
        checked_path(output)
        if (output.stat().st_dev, output.stat().st_ino) == identity:
            if report_path.is_file() and not report_path.is_symlink() and report_path.read_bytes() == payload:
                report_path.unlink()  # Only this run's exact failed publication.
            with (output / "failed.json").open("x", encoding="utf-8") as handle:
                json.dump({"status": "failed", "partial_outputs_not_authoritative": True, **AUTHORITY}, handle)
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "selection", "crop-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("manifest-sha256", "selection-sha256", "model-sha256", "spec-sha256"):
        parser.add_argument("--" + name, required=True, type=sha256_value)
    report = audit_reuse(**vars(parser.parse_args(argv)))
    print(json.dumps({key: report[key] for key in ("status", "selection_sources", "verified_crop_rows", "missing_sources")}
                     | AUTHORITY, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, **AUTHORITY}), flush=True)
        raise SystemExit(1) from None
