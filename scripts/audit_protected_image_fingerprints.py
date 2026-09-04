"""Read and fingerprint every protected image; snapshots grant no training authority."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
from pathlib import Path

try:
    from scripts import audit_aihub_original_annotations as original
except ModuleNotFoundError:
    import audit_aihub_original_annotations as original

ROLES = {"qx3", "capture", "known_audit"}
SHA = re.compile(r"[0-9a-f]{64}")
AUTHORITY = {"training_authorized": False, "deployment_authorized": False,
             "blind_test_authorized": False, "selection_authorized": False}


def _sha(value: object) -> str:
    if type(value) is not str or SHA.fullmatch(value) is None:
        raise ValueError("expected lowercase SHA256")
    return value


def _path(value: object, roots: tuple[Path, ...] | None = None, *, exists=True) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError("path must be textual")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("absolute non-traversing path required")
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError("symlink path component forbidden")
    resolved = path.resolve(strict=exists)
    if roots is not None and not any(resolved.is_relative_to(root) for root in roots):
        raise ValueError("path outside allowed roots")
    return resolved


def _encode(path: Path) -> str:
    return base64.urlsafe_b64encode(os.fsencode(path)).decode("ascii")


class _Budget:
    def __init__(self, maximum: int, max_file: int):
        self.maximum, self.max_file, self.used = maximum, max_file, 0

    def limit(self, path: Path) -> int:
        remaining = min(self.max_file, self.maximum - self.used)
        if remaining <= 0 or path.stat().st_size > remaining:
            raise ValueError("protected fingerprint read budget exceeded")
        return remaining

    def read(self, path: Path) -> bytes:
        _path(path)
        value = original.read_stable(path, self.limit(path))
        self.used += len(value)
        _path(path)
        return value


def _json(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def audit_fingerprints(*, inventory: Path, inventory_sha256: str, allowed_roots: list[Path],
                       output: Path, max_read_bytes: int = 1024**3,
                       max_file_bytes: int = 64 * 1024**2) -> dict:
    for number in (max_read_bytes, max_file_bytes):
        if type(number) is not int or number <= 0:
            raise ValueError("positive exact integer byte limits required")
    if not allowed_roots:
        raise ValueError("at least one allowed root is required")
    roots = tuple(_path(root) for root in allowed_roots)
    if any(not root.is_dir() for root in roots):
        raise ValueError("allowed root must be a directory")
    inventory = _path(inventory)
    output = _path(output, exists=False)
    if output.exists():
        raise FileExistsError("protected fingerprint output must be fresh")
    expected_inventory_sha = _sha(inventory_sha256)
    budget = _Budget(max_read_bytes, max_file_bytes)
    content = budget.read(inventory)
    if hashlib.sha256(content).hexdigest() != expected_inventory_sha:
        raise ValueError("inventory SHA mismatch")
    def invalid_constant(_):
        raise ValueError("invalid JSON constant")
    data = json.loads(content, object_pairs_hook=original.unique_object, parse_constant=invalid_constant)
    if type(data) is not dict or set(data) != {"records", "metadata_bindings"}:
        raise ValueError("invalid protected inventory schema")
    if type(data["records"]) is not list or not data["records"]:
        raise ValueError("protected records must be a nonempty list")
    if type(data["metadata_bindings"]) is not list or not data["metadata_bindings"]:
        raise ValueError("metadata bindings must be a nonempty list")
    records, metadata, seen_shas, seen_paths = [], [], set(), set()
    for row in data["records"]:
        if type(row) is not dict or set(row) != {"sha256", "path", "roles"}:
            raise ValueError("invalid protected record fields")
        sha = _sha(row["sha256"])
        roles = row["roles"]
        if (type(roles) is not list or not roles or any(type(role) is not str or role not in ROLES for role in roles)
                or len(set(roles)) != len(roles)):
            raise ValueError("invalid or duplicate protected roles")
        path = _path(row["path"], roots)
        if sha in seen_shas or path in seen_paths:
            raise ValueError("protected SHA/path union must be unique")
        seen_shas.add(sha)
        seen_paths.add(path)
        records.append((sha, path, sorted(roles)))
    metadata_paths = set()
    for row in data["metadata_bindings"]:
        if type(row) is not dict or set(row) != {"path", "sha256"}:
            raise ValueError("invalid metadata binding fields")
        path, sha = _path(row["path"]), _sha(row["sha256"])
        if path in metadata_paths:
            raise ValueError("duplicate metadata path")
        metadata_paths.add(path)
        if hashlib.sha256(budget.read(path)).hexdigest() != sha:
            raise ValueError("metadata SHA mismatch")
        metadata.append((path, sha))
    # Pin the exact implementation used for pixel decode and the pHash convention.
    code_paths = (_path(Path(__file__).absolute()), _path(Path(original.__file__).absolute()))
    code = [(path, hashlib.sha256(budget.read(path)).hexdigest()) for path in code_paths]
    results = []
    for sha, path, roles in records:
        _path(path, roots)
        shape, actual_sha, size, phash = original.image_evidence(path, budget.limit(path))
        budget.used += size
        _path(path, roots)
        if actual_sha != sha:
            raise ValueError("protected source SHA mismatch")
        if (len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape)
                or type(phash) is not str or re.fullmatch(r"[0-9a-f]{16}", phash) is None):
            raise ValueError("invalid protected image evidence")
        results.append({"source_sha256": sha, "source_path_b64": _encode(path), "roles": roles,
                        "image_height": shape[0], "image_width": shape[1],
                        "source_bytes": size, "source_phash64": phash})
        if len(results) % 100 == 0:
            print(json.dumps({"verified": len(results), "expected": len(records)}), flush=True)
    def recheck():
        for path, sha in [(inventory, expected_inventory_sha), *metadata, *code,
                          *((path, sha) for sha, path, _ in records)]:
            _path(path, roots if path in seen_paths else None)
            if hashlib.sha256(budget.read(path)).hexdigest() != sha:
                raise ValueError("protected input changed during audit")
    recheck()
    if len(results) != len(seen_shas) or {row["source_sha256"] for row in results} != seen_shas:
        raise ValueError("protected source coverage mismatch")
    report = {"schema": "protected_image_fingerprint_snapshot.v1", "status": "snapshot_complete",
              "inventory_sha256": expected_inventory_sha, "expected_sources": len(records),
              "verified_sources": len(results), "missing_sources": 0, "snapshot_only": True,
              "consumer_must_rehash_sources": True,
              "coverage_scope": "all records in pinned inventory; metadata union completeness is not inferred",
              "perceptual_hash_convention": "audit_aihub_original_annotations.image_evidence grayscale32 DCT8 median_no_dc",
              "metadata_bindings": [{"path_b64": _encode(path), "sha256": sha} for path, sha in metadata],
              "code_sha256": {path.name: sha for path, sha in code},
              "read_bytes_before_publication": budget.used, "records": results, **AUTHORITY}
    report_bytes = _json(report)
    output.mkdir(parents=True, exist_ok=False)
    identity = (output.stat().st_dev, output.stat().st_ino)
    try:
        with (output / "report.json").open("xb") as handle:
            handle.write(report_bytes)
        recheck()
        if budget.read(output / "report.json") != report_bytes:
            raise ValueError("protected snapshot changed during publication")
    except BaseException:
        # Preserve evidence but make any partially published snapshot unusable.
        _path(output)
        if (output.stat().st_dev, output.stat().st_ino) == identity:
            report_path = output / "report.json"
            if report_path.is_file() and not report_path.is_symlink() and report_path.read_bytes() == report_bytes:
                report_path.unlink()  # Only this run's exact failed publication.
            with (output / "failed.json").open("xb") as handle:
                handle.write(_json({"status": "failed", **AUTHORITY}))
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--allowed-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-read-gib", type=float, default=1)
    parser.add_argument("--max-file-mib", type=float, default=64)
    args = parser.parse_args()
    if any(not math.isfinite(n) or n <= 0 for n in (args.max_read_gib, args.max_file_mib)):
        parser.error("positive finite read limits required")
    result = audit_fingerprints(inventory=args.inventory, inventory_sha256=args.inventory_sha256,
        allowed_roots=args.allowed_root, output=args.output, max_read_bytes=int(args.max_read_gib * 1024**3),
        max_file_bytes=int(args.max_file_mib * 1024**2))
    print(json.dumps({key: result[key] for key in ("status", "verified_sources", "training_authorized")}), flush=True)


if __name__ == "__main__":
    main()
