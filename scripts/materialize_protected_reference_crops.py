"""Create protection-only ROIs from pinned historical references, never targets.

Known-audit bbox takes priority; a historical deployed bbox is used only when
the former is absent. Neither source is asserted to be ground truth or a fresh
YOLO observation. Sources without either reference remain explicitly listed.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import platform
from collections import Counter
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

try:
    from scripts import audit_proposal_crop_reuse as files
    from scripts import verifier_preprocessing_contract as crop_contract
except ModuleNotFoundError:
    import audit_proposal_crop_reuse as files
    import verifier_preprocessing_contract as crop_contract

AUTHORITY = {"training_authorized": False, "blind_test_authorized": False,
             "deployment_authorized": False, "formal_protected_coverage": False,
             "label_authority": False, "selection_authorized": False,
             "semantic_truth_established": False, "runtime_detector_executed": False,
             "state_targets_emitted": False}
CROP_CONFIG = {"size": 320, "padding": 0.08, "letterbox_fill": 114, "jpeg_quality": 92}


class ReferenceError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise ReferenceError(message)


def code_paths():
    return {path.name: files.checked_path(path) for path in
            (Path(__file__).absolute(), Path(files.__file__).absolute(), Path(crop_contract.__file__).absolute())}


def render(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode()


def parse_json(content):
    def bad(_): raise ReferenceError("nonfinite JSON constant")
    return json.loads(content, object_pairs_hook=files.unique_object, parse_constant=bad)


def decode_path(value):
    require(type(value) is str, "missing source path")
    return files.checked_path(Path(os.fsdecode(base64.b64decode(value, altchars=b"-_", validate=True))))


def source_roles(value):
    require(type(value) is list and value and all(type(role) is str and role in {"qx3", "capture", "known_audit"} for role in value)
            and len(set(value)) == len(value), "invalid protected roles")
    return sorted(value)


def capture_path(root, value):
    require(type(value) is str and value and "\\" not in value and ":" not in value and "\x00" not in value,
            "invalid capture image_ref")
    relative = PurePosixPath(value)
    require(not relative.is_absolute() and ".." not in relative.parts and relative.as_posix() == value
            and len(relative.parts) > 0, "noncanonical capture image_ref")
    result = files.checked_path(root.joinpath(*relative.parts))
    require(result.is_relative_to(root), "capture reference escapes root")
    return result


def select_reference(known, captured, known_sha, capture_sha):
    """Do not silently fall back from a malformed declared known bbox."""
    known_bbox = known.get("bbox") if known is not None else None
    if known_bbox is not None:
        kind = known.get("bbox_source")
        require(type(kind) is str and bool(kind) and len(kind) <= 128, "known bbox lacks provenance")
        return {"kind": "known_audit_reference", "bbox_source": kind, "bbox_xyxy": known_bbox,
                "metadata_sha256": known_sha, "field": "bbox"}
    if captured is not None:
        deployed = captured.get("deployed")
        require(type(deployed) is dict, "invalid deployed reference metadata")
        if deployed.get("bbox") is not None:
            return {"kind": "historical_deployed_reference", "bbox_source": "historical_deployed_bbox",
                    "bbox_xyxy": deployed["bbox"], "metadata_sha256": capture_sha, "field": "deployed.bbox"}
    return None


def materialize(*, known_audit: Path, known_audit_sha256: str, capture_inventory: Path,
                capture_inventory_sha256: str, capture_root: Path, protected_report: Path,
                protected_report_sha256: str, code_pins: dict[str, str], output: Path,
                expected_sources: int | None = None, expected_crops: int | None = None,
                expected_missing: int | None = None):
    for count in (expected_sources, expected_crops, expected_missing):
        require(count is None or (type(count) is int and count >= 0), "expected counts must be nonnegative integers")
    metadata = [(files.checked_path(p), files.sha256_value(digest)) for p, digest in
                ((known_audit, known_audit_sha256), (capture_inventory, capture_inventory_sha256),
                 (protected_report, protected_report_sha256))]
    known_audit, capture_inventory, protected_report = [p for p, _ in metadata]
    require(len({p for p, _ in metadata}) == 3, "metadata inputs must be distinct")
    root, output = files.checked_path(capture_root), files.checked_path(output, exists=False)
    require(root.is_dir(), "capture root must be a directory")
    code = code_paths()
    require(type(code_pins) is dict and set(code_pins) == set(code), "exact producer/helper code pins required")
    protected_roots = {root, *(p.parent for p, _ in metadata), *(p.parent for p in code.values())}
    require(not output.exists() and not any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected_roots),
            "output must be fresh and disjoint from input trees")
    pins = {}
    def pin(path, digest, limit=files.METADATA_LIMIT, keep=False):
        actual, size, content = files.read_file(path, limit, keep=keep)
        require(actual == digest, "input SHA256 mismatch")
        require(path not in pins or pins[path] == (digest, size, limit), "conflicting input binding")
        pins[path] = (digest, size, limit)
        return content
    for name, path in code.items(): pin(path, files.sha256_value(code_pins[name]))
    known, captures, snapshot = [parse_json(pin(p, digest, keep=True)) for p, digest in metadata]
    require(type(known) is dict and type(captures) is list, "invalid known/capture metadata schema")
    for digest, row in known.items():
        files.sha256_value(digest)
        require(type(row) is dict, "invalid known reference row")
        if "sha256" in row: require(row["sha256"] == digest, "known row SHA mismatch")
    captured = {}
    for row in captures:
        require(type(row) is dict, "invalid capture record")
        digest = files.sha256_value(row.get("sha256"))
        require(digest not in captured, "duplicate capture source SHA")
        captured[digest] = row
    require(type(snapshot) is dict and snapshot.get("schema") == "protected_image_fingerprint_snapshot.v1"
            and snapshot.get("status") == "snapshot_complete" and snapshot.get("snapshot_only") is True
            and snapshot.get("consumer_must_rehash_sources") is True and type(snapshot.get("records")) is list,
            "invalid fingerprint snapshot")
    for key in ("training_authorized", "deployment_authorized", "blind_test_authorized", "selection_authorized"):
        require(snapshot.get(key) is False, "fingerprint snapshot cannot grant authority")
    count = len(snapshot["records"])
    require(count > 0 and all(type(snapshot.get(k)) is int and snapshot[k] == count for k in ("expected_sources", "verified_sources"))
            and type(snapshot.get("missing_sources")) is int and snapshot["missing_sources"] == 0, "incomplete fingerprint snapshot")
    protected, seen_paths = {}, set()
    for row in snapshot["records"]:
        require(type(row) is dict, "invalid fingerprint record")
        digest = files.sha256_value(row.get("source_sha256"))
        require(digest not in protected, "duplicate fingerprint source SHA")
        protected[digest] = row
        source_roles(row.get("roles"))
    raw = {digest for digest, row in protected.items() if set(row["roles"]) & {"capture", "known_audit"}}
    require(raw and raw == set(known) | set(captured), "reference metadata does not exactly cover raw protected source union")
    require(expected_sources is None or len(raw) == expected_sources, "expected source count mismatch")
    planned = []
    for digest in sorted(raw):
        row = protected[digest]
        path, role_list = decode_path(row.get("source_path_b64")), source_roles(row["roles"])
        require(path not in seen_paths, "duplicate raw source path")
        seen_paths.add(path)
        expected_roles = ({"known_audit"} if digest in known else set()) | ({"capture"} if digest in captured else set())
        require(set(role_list) - {"qx3"} == expected_roles, "metadata and protected roles differ")
        require(not output.is_relative_to(path.parent) and not path.is_relative_to(output), "output overlaps raw source tree")
        require(all(type(row.get(k)) is int and row[k] > 0 for k in ("source_bytes", "image_width", "image_height"))
                and row["source_bytes"] <= files.CROP_LIMIT and row["image_width"] * row["image_height"] <= 16_000_000,
                "invalid source dimensions/byte limit")
        if digest in captured:
            require(capture_path(root, captured[digest].get("image_ref")) == path, "capture path differs from fingerprint source")
        reference = select_reference(known.get(digest), captured.get(digest), known_audit_sha256, capture_inventory_sha256)
        if reference is not None:
            bbox = reference["bbox_xyxy"]
            require(type(bbox) is list and len(bbox) == 4
                    and all(type(value) in (int, float) and math.isfinite(value) for value in bbox),
                    "reference bbox must be one finite numeric XYXY array")
            bounds = crop_contract.padded_clipped_bbox(bbox, width=row["image_width"], height=row["image_height"], padding=.08)
        else: bounds = None
        planned.append((digest, path, role_list, row, reference, bounds))
    crop_count = sum(reference is not None for _, _, _, _, reference, _ in planned)
    require(expected_crops is None or crop_count == expected_crops, "expected crop count mismatch")
    require(expected_missing is None or len(raw) - crop_count == expected_missing, "expected missing count mismatch")
    require(not any((p.parent / "failed.json").exists() for p in pins), "input failure marker")
    output.mkdir(parents=True, exist_ok=False)
    owned = (output.stat().st_dev, output.stat().st_ino)
    crop_pins, results, published = {}, [], None
    report_path = output / "report.json"
    def recheck():
        current = files.checked_path(output).stat()
        require((current.st_dev, current.st_ino) == owned, "output ownership changed")
        for path, (digest, size, limit) in {**pins, **crop_pins}.items():
            actual, current_size, _ = files.read_file(path, limit)
            require((actual, current_size) == (digest, size), "input or crop changed during materialization")
        require(not any((p.parent / "failed.json").exists() for p in pins), "input failure marker appeared")
    try:
        for digest, path, role_list, row, reference, bounds in planned:
            content = pin(path, digest, files.CROP_LIMIT, keep=True)
            require(len(content) == row["source_bytes"], "source bytes differ from fingerprint snapshot")
            pixels = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
            require(pixels is not None and pixels.shape[:2] == (row["image_height"], row["image_width"]), "source pixel decode/shape mismatch")
            item = {"source_sha256": digest, "source_path_b64": row["source_path_b64"],
                    "source_bytes": len(content), "image_width": row["image_width"], "image_height": row["image_height"],
                    "roles": role_list, "status": "reference_roi_generated" if reference is not None else "missing_reference",
                    "reference": reference, "crop": None, "object_absence_established": False}
            if reference is not None:
                crop, actual_bounds = crop_contract.crop_and_letterbox_bgr(pixels, reference["bbox_xyxy"], padding=.08, size=320, fill=114)
                require(actual_bounds == bounds, "crop geometry mismatch")
                ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                require(ok, "ROI JPEG encoding failed")
                path_out = output / "crops" / (digest + ".jpg")
                path_out.parent.mkdir(exist_ok=True)
                files.checked_path(path_out, exists=False)
                with path_out.open("xb") as handle: handle.write(encoded.tobytes())
                actual, size, _ = files.read_file(path_out, files.CROP_LIMIT)
                require(actual == hashlib.sha256(encoded.tobytes()).hexdigest(), "published crop bytes differ")
                crop_pins[path_out] = (actual, size, files.CROP_LIMIT)
                item["crop"] = {"path": path_out.relative_to(output).as_posix(), "sha256": actual, "bytes": size,
                                "width": 320, "height": 320, "bounds_xyxy": list(bounds), "provenance": reference["kind"]}
            results.append(item)
        recheck()
        kinds = Counter(row["reference"]["kind"] for row in results if row["reference"] is not None)
        report = {"schema": "protected_reference_roi.v1", "status": "reference_materialization_complete",
                  "artifact_role": "protected_reference_roi_not_ground_truth_yolo_or_formal_inventory",
                  "raw_source_count": len(results), "reference_roi_count": crop_count,
                  "missing_reference_count": len(results) - crop_count, "reference_counts": dict(sorted(kinds.items())),
                  "records": results, "crop_configuration": CROP_CONFIG,
                  "bindings": {"known_audit_sha256": known_audit_sha256, "capture_inventory_sha256": capture_inventory_sha256,
                               "protected_report_sha256": protected_report_sha256, "capture_root": str(root),
                               "code_sha256": dict(sorted(code_pins.items())),
                               "input_files": [{"path_b64": base64.urlsafe_b64encode(os.fsencode(p)).decode(), "sha256": digest}
                                               for p, (digest, _, _) in pins.items()]},
                  "runtime": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
                              "opencv_build_sha256": hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest()}, **AUTHORITY}
        published = render(report)
        with report_path.open("xb") as handle: handle.write(published)
        recheck()
        require(files.read_file(report_path, files.METADATA_LIMIT, keep=True)[2] == published, "published report changed")
        return report
    except BaseException:
        current = files.checked_path(output).stat()
        if (current.st_dev, current.st_ino) == owned:
            if published is not None and report_path.is_file() and not report_path.is_symlink() and report_path.read_bytes() == published:
                report_path.unlink()
            with (output / "failed.json").open("xb") as handle:
                handle.write(render({"status": "failed", "partial_outputs_not_authoritative": True, **AUTHORITY}))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("known-audit", "capture-inventory", "capture-root", "protected-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("known-audit-sha256", "capture-inventory-sha256", "protected-report-sha256"):
        parser.add_argument("--" + name, type=files.sha256_value, required=True)
    for name in ("expected-sources", "expected-crops", "expected-missing"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--code-pin", action="append", required=True)
    args = vars(parser.parse_args(argv))
    pins = {}
    for value in args.pop("code_pin"):
        name, separator, digest = value.partition("=")
        require(separator and name not in pins, "malformed or duplicate code pin")
        pins[name] = files.sha256_value(digest)
    report = materialize(**args, code_pins=pins)
    print(json.dumps({k: report[k] for k in ("status", "raw_source_count", "reference_roi_count", "missing_reference_count")} | AUTHORITY), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, **AUTHORITY}), flush=True)
        raise SystemExit(1) from None
