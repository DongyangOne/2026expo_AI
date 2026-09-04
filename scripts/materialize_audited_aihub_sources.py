"""Materialize pinned AIHub original annotations into a non-authoritative snapshot."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

try:
    from scripts import audit_aihub_original_annotations as audit
except ModuleNotFoundError:
    import audit_aihub_original_annotations as audit

AUTHORITY = {"training_authorized": False, "deployment_authorized": False,
             "blind_test_authorized": False}
POLICY = {"name": "audited_original_materialization_quality_v1", "min_original_side": 320,
          "bbox_area_ratio": [0.04, 0.80], "resized_gray_brightness": [18, 238],
          "min_resized_laplacian_variance": 20, "resize_max_long_side": 640,
          "resize_upscale": False, "resize_rounding": "python_round",
          "resize_interpolation": "INTER_AREA", "jpeg_quality": 90,
          "yolo_coordinate_decimal_places": 8}
UNKNOWN = {"dent": -1, "label": -1, "foreign_material": -1}


def _sha(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("invalid SHA256")
    return value


def _path(value, root=None, *, exists=True):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("absolute non-traversing path required")
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlink path forbidden")
    resolved = path.resolve(strict=exists)
    if root is not None and not resolved.is_relative_to(root):
        raise ValueError("path outside dataset root")
    return resolved


def _decode(value, root):
    if type(value) is not str:
        raise ValueError("invalid encoded source path")
    raw = base64.b64decode(value, altchars=b"-_", validate=True)
    if base64.urlsafe_b64encode(raw).decode("ascii") != value:
        raise ValueError("noncanonical encoded source path")
    return _path(os.fsdecode(raw), root)


def _json(data):
    return (json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def _parse(raw):
    def invalid(_):
        raise ValueError("invalid JSON constant")
    return json.loads(raw, object_pairs_hook=audit.unique_object, parse_constant=invalid)


def _digest(path):
    path = _path(path)
    before = path.stat()
    if not path.is_file():
        raise ValueError("regular file required")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        consumed = os.fstat(handle.fileno())
    after = _path(path).stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if (not path.is_file() or len({identity(s) for s in (before, opened, consumed, after)}) != 1
            or before.st_ctime_ns != after.st_ctime_ns or opened.st_ctime_ns != consumed.st_ctime_ns):
        raise ValueError("file changed during hashing")
    return digest.hexdigest()


def _read(path, expected, limit):
    raw = audit.read_stable(_path(path), limit)
    if hashlib.sha256(raw).hexdigest() != _sha(expected):
        raise ValueError("pinned input SHA256 mismatch")
    _path(path)
    return raw


def _validated_row(row, root):
    if not isinstance(row, dict) or re.fullmatch(r"[0-9a-f]{20}", str(row.get("source_id"))) is None:
        raise ValueError("invalid source identity")
    if row.get("split") not in ("training", "validation"):
        raise ValueError("invalid official split")
    cid = row.get("class_id")
    if type(cid) is not int or cid not in range(9) or row.get("class_name") != audit.CLASS_NAMES[cid]:
        raise ValueError("invalid original class")
    if row.get("status", "verified_pair") != "verified_pair":
        raise ValueError("unverified cohort row")
    conditions = row.get("conditions")
    if (not isinstance(conditions, dict) or set(conditions) != set(UNKNOWN)
            or any(type(v) is not int or v != -1 for v in conditions.values())):
        raise ValueError("state targets must remain unknown")
    if type(row.get("annotation_dent")) is not int or row["annotation_dent"] not in (-1, 0, 1):
        raise ValueError("invalid annotation reference state")
    if any(type(row.get(k)) is not int or row[k] <= 0 for k in ("image_width", "image_height")):
        raise ValueError("invalid original dimensions")
    bbox = row.get("bbox_xywh")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("invalid original bbox")
    audit.strict_bbox([bbox], row["image_width"], row["image_height"])
    for key in ("source_sha256", "label_sha256"):
        _sha(row.get(key))
    return _decode(row["source_path_b64"], root), _decode(row["label_path_b64"], root)


def _prepare(row, source, label):
    raw = _read(source, row["source_sha256"], 64 * 1024**2)
    label_raw = _read(label, row["label_sha256"], 1024**2)
    pixels = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if pixels is None or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("undecodable original image")
    h, w = pixels.shape[:2]
    if (h, w) != (row["image_height"], row["image_width"]):
        raise ValueError("original image dimensions changed")
    manifest = {"source_id": row["source_id"], "split": row["split"],
                "category": row["class_name"], "material": str(row["class_id"]),
                "source_object_count": "1", "source_width": str(w), "source_height": str(h)}
    manifest.update({f"source_bbox_{k}": str(v) for k, v in zip("xywh", row["bbox_xywh"])})
    verified = audit.validate_pair(manifest, _parse(label_raw), (h, w), source, label)
    if any(verified[k] != row[k] for k in ("class_id", "class_name", "bbox_xywh", "annotation_dent", "conditions")):
        raise ValueError("original annotation/cohort disagreement")
    scale = min(1.0, 640 / max(h, w))
    resized = pixels if scale == 1 else cv2.resize(
        pixels, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    x, y, bw, bh = verified["bbox_xywh"]
    measures = {"original_min_side": min(h, w), "bbox_area_ratio": bw * bh / (w * h),
                "resized_gray_brightness": float(gray.mean()),
                "resized_laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var())}
    reason = ("resolution_too_small" if min(h, w) < 320 else
              "bbox_area_outside_range" if not 0.04 <= measures["bbox_area_ratio"] <= 0.80 else
              "exposure_outside_range" if not 18 <= measures["resized_gray_brightness"] <= 238 else
              "too_blurry" if measures["resized_laplacian_variance"] < 20 else None)
    if reason:
        return None, None, measures, reason, resized.shape[:2]
    ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("JPEG encoding failed")
    normalized = ((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h)
    label_text = f"{verified['class_id']} " + " ".join(f"{v:.8f}" for v in normalized) + "\n"
    return encoded.tobytes(), label_text.encode(), measures, None, resized.shape[:2]


def materialize_sources(*, cohort: Path, cohort_sha256: str, dataset_root: Path, output: Path,
                        min_free_gib=300, max_output_gib=30, max_sources=0):
    if (type(max_sources) is not int or max_sources < 0
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in (min_free_gib, max_output_gib))
            or min_free_gib < 0 or max_output_gib <= 0):
        raise ValueError("invalid resource limits")
    root, cohort = _path(dataset_root), _path(cohort)
    output = _path(output, exists=False)
    if not root.is_dir() or output.exists():
        raise ValueError("dataset root must exist and output must be new")
    data = _parse(_read(cohort, cohort_sha256, 1024**3))
    if (not isinstance(data, dict) or data.get("schema") != "aihub_original_cohort_v1"
            or data.get("status") != "cohort_planned" or data.get("training_authorized") is not False
            or data.get("deployment_authorized") is not False):
        raise ValueError("invalid non-authoritative cohort contract")
    rows, metadata = data.get("records"), data.get("metadata_bindings")
    if not isinstance(rows, list) or not rows or not isinstance(metadata, list) or not metadata:
        raise ValueError("cohort records and metadata bindings required")
    bindings = {cohort: _sha(cohort_sha256)}
    protected = [root, cohort.parent]
    for item in metadata:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("invalid metadata binding")
        path, digest = _path(item["path"]), _sha(item["sha256"])
        if path in bindings:
            raise ValueError("duplicate metadata binding")
        bindings[path] = digest
        protected.append(path.parent)
    code_sha = {}
    for module_path in (Path(__file__).resolve(), Path(audit.__file__).resolve()):
        digest = _digest(module_path)
        if module_path in bindings and bindings[module_path] != digest:
            raise ValueError("pinned code metadata SHA256 mismatch")
        bindings[module_path] = digest
        code_sha[module_path.name] = digest
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError("output overlaps immutable input evidence")
    selected, ids, hashes, paths = [], set(), set(), set()
    for index, row in enumerate(rows):
        source, label = _validated_row(row, root)
        if row["source_id"] in ids or row["source_sha256"] in hashes or source in paths:
            raise ValueError("duplicate cohort source")
        ids.add(row["source_id"])
        hashes.add(row["source_sha256"])
        paths.add(source)
        if not max_sources or index < max_sources:
            selected.append((row, source, label))
    for path, sha in bindings.items():
        if _digest(path) != sha:
            raise ValueError("metadata or code changed")
    nearest = output.parent
    while not nearest.exists():
        nearest = nearest.parent
    if shutil.disk_usage(nearest).free < min_free_gib * 1024**3:
        raise ValueError("insufficient free disk reserve")
    output.mkdir(parents=True, exist_ok=False)
    output_identity = (output.stat().st_dev, output.stat().st_ino)
    produced, bytes_written, counts, exclusions = {}, 0, Counter(), Counter()
    ready_path = output / "snapshot_ready.json"

    def write(relative, raw):
        nonlocal bytes_written
        if bytes_written + len(raw) > max_output_gib * 1024**3:
            raise ValueError("output byte budget exceeded")
        if shutil.disk_usage(output).free - len(raw) < min_free_gib * 1024**3:
            raise ValueError("insufficient free disk reserve")
        dest = _path(output / relative, exists=False)
        if not dest.is_relative_to(output):
            raise ValueError("output path escaped")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as handle:
            handle.write(raw)
        produced[dest] = hashlib.sha256(raw).hexdigest()
        bytes_written += len(raw)

    def recheck():
        for path, sha in bindings.items():
            if _digest(path) != sha:
                raise ValueError("pinned metadata or code changed")
        for row, source, label in selected:
            if _digest(source) != row["source_sha256"] or _digest(label) != row["label_sha256"]:
                raise ValueError("original image or annotation changed")
        for path, sha in produced.items():
            if _digest(path) != sha:
                raise ValueError("materialized output changed")

    try:
        lineage, rejected = [], []
        for index, (row, source, label) in enumerate(selected, 1):
            encoded, yolo, measures, reason, shape = _prepare(row, source, label)
            if _digest(source) != row["source_sha256"] or _digest(label) != row["label_sha256"]:
                raise ValueError("original pair changed after decode")
            if reason:
                exclusions[reason] += 1
                rejected.append({"source_id": row["source_id"], "source_sha256": row["source_sha256"],
                                 "label_sha256": row["label_sha256"], "reason": reason, "quality": measures})
            else:
                split = "train" if row["split"] == "training" else "val"
                stem = f"{row['class_name']}_{row['source_id']}"
                image_ref, label_ref = f"images/{split}/{stem}.jpg", f"labels/{split}/{stem}.txt"
                write(image_ref, encoded)
                write(label_ref, yolo)
                lineage.append({**row, "conditions": dict(UNKNOWN), "image_ref": image_ref,
                                "label_ref": label_ref, "image_sha256": hashlib.sha256(encoded).hexdigest(),
                                "yolo_label_sha256": hashlib.sha256(yolo).hexdigest(),
                                "materialized_height": shape[0], "materialized_width": shape[1],
                                "quality": measures, "annotation_authority": "original_aihub_json",
                                **AUTHORITY})
                counts[f"{row['split']}/{row['class_name']}"] += 1
            if index % 100 == 0:
                print(json.dumps({"verified": index, "expected": len(selected), "materialized": sum(counts.values())}), flush=True)
        write("lineage.jsonl", b"".join(_json(r) for r in lineage))
        write("excluded.jsonl", b"".join(_json(r) for r in rejected))
        yaml = f"path: {json.dumps(output.as_posix(), ensure_ascii=True)}\ntrain: images/train\nval: images/val\nnames:\n"
        yaml += "".join(f"  {i}: {name}\n" for i, name in enumerate(audit.CLASS_NAMES))
        write("dataset.yaml", yaml.encode())
        recheck()
        report = {"schema": "audited_aihub_source_snapshot_v1", "status": "snapshot_complete",
                  "snapshot_only": True, "cohort_sha256": cohort_sha256, "code_sha256": code_sha,
                  "full_cohort": data.get("full_cohort") is True and not max_sources,
                  "pending_checks": data.get("pending_checks", ["source_coverage_and_leakage_not_proven"]),
                  "cohort_records": len(rows), "requested_max_sources": max_sources,
                  "verified_sources": len(selected), "unprocessed_sources": len(rows) - len(selected),
                  "materialized_sources": len(lineage), "quality_excluded_sources": len(rejected),
                  "counts": dict(counts), "exclusions": dict(exclusions), "quality_policy": POLICY,
                  "bytes_written_before_report": bytes_written, "consumer_must_rehash_sources": True,
                  "metadata_bindings": metadata, "lineage_sha256": produced[output / "lineage.jsonl"],
                  "excluded_sha256": produced[output / "excluded.jsonl"], **AUTHORITY}
        write("report.json", _json(report))
        recheck()
        write("snapshot_ready.json", _json({"status": "snapshot_complete", "snapshot_only": True,
              "report_sha256": produced[output / "report.json"], **AUTHORITY}))
        recheck()
        return report
    except BaseException as error:
        current = _path(output).stat()
        if (current.st_dev, current.st_ino) != output_identity:
            raise ValueError("output ownership changed") from error
        if ready_path.is_file() and not ready_path.is_symlink():
            ready_path.unlink()
        failure = {"status": "failed", "exception_type": type(error).__name__,
                   "partial_outputs_are_not_authoritative": True, **AUTHORITY}
        with (output / "failed.json").open("xb") as handle:
            handle.write(_json(failure))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--cohort-sha256", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-free-gib", type=float, default=300)
    parser.add_argument("--max-output-gib", type=float, default=30)
    parser.add_argument("--max-sources", type=int, default=0)
    args = parser.parse_args()
    try:
        report = materialize_sources(**vars(args))
    except Exception as error:
        print(json.dumps({"status": "failed", "exception_type": type(error).__name__, **AUTHORITY}), flush=True)
        raise SystemExit(1) from None
    print(json.dumps({"status": report["status"], "verified_sources": report["verified_sources"],
                      "materialized_sources": report["materialized_sources"], **AUTHORITY}), flush=True)


if __name__ == "__main__":
    main()
