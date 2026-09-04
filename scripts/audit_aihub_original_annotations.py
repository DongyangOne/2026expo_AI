"""Bind AIHub originals to annotation bytes; never grant training/deployment authority.

The deterministic pilot checks the existing manifest against actual source pixels
and original JSON, not against a detector prediction. Unknown state labels remain
masked. Reports contain only technical evidence, not photographer metadata.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import heapq
import json
import math
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

CLASS_NAMES = ["can", "pet", "paper", "plastic", "styrofoam", "vinyl", "glass", "battery", "fluorescent"]
ALIASES = [("철캔", "알루미늄캔", "알류미늄캔", "금속캔"), ("페트병", "무색단일", "유색단일"),
           ("종이",), ("플라스틱",), ("스티로폼",), ("비닐",), ("유리병",), ("건전지",), ("형광등",)]
DENT_MAP = {"원형": 0, "찌그러짐": 1, "완전압착": 1}


class AnnotationError(ValueError):
    pass


def resolve_material(*values: str) -> int:
    if not values or any(not isinstance(v, str) for v in values):
        raise AnnotationError("invalid material fields")
    text = " ".join(values)
    tokens = set(re.findall(r"[a-zA-Z]+", text.lower()))
    matches = {i for i, aliases in enumerate(ALIASES) if any(a in text for a in aliases)}
    matches.update(i for i, name in enumerate(CLASS_NAMES) if name in tokens)
    if tokens & {"pe", "pp", "ps"}:
        matches.add(3)
    if len(matches) != 1:
        raise AnnotationError("unknown or conflicting material")
    return matches.pop()


def number(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise AnnotationError("non-finite or non-numeric coordinate")
    return float(value)


def strict_bbox(points, width: int, height: int) -> list[float]:
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise AnnotationError("invalid image dimensions")
    if not isinstance(points, list) or not points or any(not isinstance(p, (list, tuple)) for p in points):
        raise AnnotationError("invalid points")
    if len(points) == 1 and len(points[0]) == 4:
        x, y, w, h = map(number, points[0])
    elif len(points) >= 3 and all(len(p) == 2 for p in points):
        vertices = [tuple(map(number, p)) for p in points]
        if len(set(vertices)) < 3:
            raise AnnotationError("degenerate polygon")
        area2 = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(vertices, vertices[1:] + vertices[:1]))
        if area2 == 0:
            raise AnnotationError("zero-area polygon")
        xs, ys = zip(*vertices)
        x, y = min(xs), min(ys)
        w, h = max(xs) - x, max(ys) - y
    else:
        raise AnnotationError("malformed or multiple boxes")
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
        raise AnnotationError("bbox outside image or empty")
    return [x, y, w, h]


def annotation_location(source: Path, split: str) -> Path:
    mapping = {"training": ("Training", "TS_", "TL_"), "validation": ("Validation", "VS_", "VL_")}
    if split not in mapping:
        raise AnnotationError("invalid official split")
    official, sp, lp = mapping[split]
    if (source.parent.parent.name != "01.원천데이터" or source.parent.parent.parent.name != official
            or not source.parent.name.startswith(sp + "2.직접촬영_")):
        raise AnnotationError("source path/split disagreement")
    folder = re.sub(r"_\d+$", "", source.parent.name)
    return source.parent.parent.parent / "02.라벨링데이터" / (lp + folder[3:]) / (source.stem + ".json")


def validate_pair(row: dict, payload: dict, image_shape: tuple[int, int], source: Path, label: Path) -> dict:
    expected = annotation_location(source, row.get("split"))
    if label.name != expected.name or not label.is_relative_to(expected.parent):
        raise AnnotationError("annotation path mismatch")
    official = source.parent.parent.parent.name
    sid = hashlib.sha1(f"{official}/{expected.parent.name}/{label.name}".encode("utf-8", "surrogateescape")).hexdigest()[:20]
    if row.get("source_id") != sid:
        raise AnnotationError("source_id mismatch")
    if not isinstance(payload, dict) or not isinstance(payload.get("IMAGE_INFO"), dict):
        raise AnnotationError("invalid IMAGE_INFO")
    info, annotations = payload["IMAGE_INFO"], payload.get("ANNOTATION_INFO")
    if not isinstance(annotations, list) or len(annotations) != 1 or not isinstance(annotations[0], dict):
        raise AnnotationError("annotation must contain exactly one object")
    h, w = image_shape
    if (type(info.get("IMAGE_WIDTH")) is not int or type(info.get("IMAGE_HEIGHT")) is not int
            or (info["IMAGE_HEIGHT"], info["IMAGE_WIDTH"]) != (h, w)):
        raise AnnotationError("annotation dimensions mismatch")
    if info.get("FILE_NAME") != source.name:
        raise AnnotationError("annotation filename mismatch")
    if row.get("source_object_count") != "1" or row.get("source_width") != str(w) or row.get("source_height") != str(h):
        raise AnnotationError("manifest dimensions/object count mismatch")
    ann = annotations[0]
    material = resolve_material(ann.get("CLASS", ""), ann.get("DETAILS", ""))
    if (material != resolve_material(source.parent.name) or row.get("category") != CLASS_NAMES[material]
            or row.get("material") != str(material)):
        raise AnnotationError("material annotation/folder/manifest mismatch")
    points = ann.get("POINTS")
    bbox = strict_bbox(points, w, h)
    shape = ann.get("SHAPE_TYPE")
    if shape is not None and shape not in ("BOX", "POLYGON"):
        raise AnnotationError("unsupported shape type")
    if shape == "BOX" and len(points) != 1 or shape == "POLYGON" and len(points) < 3:
        raise AnnotationError("shape/points mismatch")
    try:
        recorded = [float(row[f"source_bbox_{axis}"]) for axis in ("x", "y", "w", "h")]
    except (KeyError, TypeError, ValueError) as exc:
        raise AnnotationError("invalid manifest bbox") from exc
    if any(not math.isfinite(a) or abs(a - b) > 1e-6 for a, b in zip(recorded, bbox)):
        raise AnnotationError("manifest bbox mismatch")
    return {"class_id": material, "class_name": CLASS_NAMES[material], "bbox_xywh": bbox,
            "annotation_dent": DENT_MAP.get(ann.get("DAMAGE"), -1) if material in (0, 1) else -1,
            "conditions": {"dent": -1, "label": -1, "foreign_material": -1}}


def read_stable(path: Path, limit: int) -> bytes:
    before = path.stat()
    if not path.is_file() or before.st_size > limit:
        raise AnnotationError("file size guard")
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        data = handle.read(limit + 1)
        consumed = os.fstat(handle.fileno())
    after = path.stat()
    # Windows path.stat/fstat can expose different ctime semantics. Compare ctime
    # within each API, and file identity/size/mtime across all observations.
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if (len(data) > limit or len(data) != before.st_size
            or len({identity(s) for s in (before, opened, consumed, after)}) != 1
            or before.st_ctime_ns != after.st_ctime_ns or opened.st_ctime_ns != consumed.st_ctime_ns):
        raise AnnotationError("source file changed during read")
    return data


def read_image(path: Path, max_bytes: int = 64 * 1024**2):
    shape, sha, size, _ = image_evidence(path, max_bytes, perceptual=False)
    return shape, sha, size


def image_evidence(path: Path, max_bytes: int = 64 * 1024**2, *, perceptual=True):
    data = read_stable(path, max_bytes)
    pixels = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if pixels is None or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise AnnotationError("undecodable image")
    phash = None
    if perceptual:
        # Same direct grayscale decode/DCT convention as audit_verifier_dataset.
        gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise AnnotationError("undecodable grayscale image")
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        coefficients = cv2.dct(np.float32(resized))[:8, :8].reshape(-1)
        bits = coefficients > float(np.median(coefficients[1:]))
        bits[0] = False
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        phash = f"{value:016x}"
    return tuple(pixels.shape[:2]), hashlib.sha256(data).hexdigest(), len(data), phash


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AnnotationError("duplicate JSON key")
        result[key] = value
    return result


def require_source_path(path: Path, root: Path):
    if not path.is_absolute() or not path.is_relative_to(root) or not path.resolve(strict=True).is_relative_to(root):
        raise AnnotationError("path outside dataset root")
    current = path
    while current != root:
        if current.is_symlink():
            raise AnnotationError("symlink in dataset source path")
        current = current.parent


def encode_path(path: Path) -> str:
    return base64.urlsafe_b64encode(os.fsencode(path)).decode("ascii")


def select_rows(manifest: Path, per_class_split: int):
    groups, seen, counts = {}, set(), Counter()
    with manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "source_path_b64" not in reader.fieldnames:
            raise AnnotationError("missing source_path_b64 manifest column")
        for row in reader:
            key = row.get("split"), row.get("category")
            sid = row.get("source_id", "")
            if key[0] not in ("training", "validation") or key[1] not in CLASS_NAMES or not re.fullmatch(r"[0-9a-f]{20}", sid):
                raise AnnotationError("invalid manifest identity")
            if sid in seen:
                raise AnnotationError("duplicate manifest source_id")
            seen.add(sid)
            counts[key] += 1
            # Stable selection is determined before any annotation result is read.
            score = int(hashlib.sha256(("original-annotation-v1:" + sid).encode()).hexdigest(), 16)
            heap = groups.setdefault(key, [])
            item = (-score, sid, row)
            if len(heap) < per_class_split:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    if len(groups) != 18:
        raise AnnotationError("manifest missing class/split groups")
    rows = [item[2] for key in sorted(groups) for item in sorted(groups[key], reverse=True)]
    return rows, {f"{a}/{b}": n for (a, b), n in sorted(counts.items())}


def digest_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class-split", type=int, default=2)
    parser.add_argument("--max-read-gib", type=float, default=1)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    args = parser.parse_args()
    if args.per_class_split <= 0 or not math.isfinite(args.max_read_gib) or args.max_read_gib <= 0:
        parser.error("positive sample/read limits required")
    if digest_file(args.manifest) != args.manifest_sha256:
        raise AnnotationError("manifest SHA256 mismatch")
    rows, counts = select_rows(args.manifest, args.per_class_split)
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.dataset_root.resolve(strict=True)
    cache, results, reasons, total_bytes = {}, [], Counter(), 0
    cache_lock, budget_lock = Lock(), Lock()
    cv2.setNumThreads(1)
    started = time.monotonic()

    def inspect_row(row):
        nonlocal total_bytes
        result = {"source_id": row["source_id"], "split": row["split"], "declared_class": row["category"]}
        try:
            source = Path(os.fsdecode(base64.b64decode(row["source_path_b64"], altchars=b"-_", validate=True)))
            require_source_path(source, root)
            expected = annotation_location(source, row["split"])
            with cache_lock:
                if expected.parent not in cache:
                    index = {}
                    for path in expected.parent.rglob("*.json"):
                        index.setdefault(path.name, []).append(path)
                    cache[expected.parent] = index
                matches = cache[expected.parent].get(expected.name, [])
            if len(matches) != 1:
                raise AnnotationError("missing or ambiguous annotation filename")
            label = matches[0]
            require_source_path(label, root)
            anticipated = source.stat().st_size + label.stat().st_size
            with budget_lock:
                if total_bytes + anticipated > args.max_read_gib * 1024**3:
                    raise AnnotationError("total read budget exhausted")
                total_bytes += anticipated
            raw_label = read_stable(label, 1024**2)
            # Reject non-standard JSON constants rather than allowing NaN coercion.
            def invalid_constant(_):
                raise AnnotationError("invalid JSON constant")
            payload = json.loads(raw_label, parse_constant=invalid_constant, object_pairs_hook=unique_object)
            shape, source_sha, size, phash = image_evidence(source)
            result.update(validate_pair(row, payload, shape, source, label))
            result.update(status="verified_pair", source_path_b64=encode_path(source), label_path_b64=encode_path(label),
                          source_sha256=source_sha, label_sha256=hashlib.sha256(raw_label).hexdigest(),
                          source_bytes=size, image_height=shape[0], image_width=shape[1], source_phash64=phash)
        except (AnnotationError, OSError, ValueError, KeyError, cv2.error) as exc:
            # Do not expose annotation payloads or source file names in errors.
            reason = str(exc) if isinstance(exc, AnnotationError) else type(exc).__name__
            result.update(status="quarantined", reason=reason)
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(inspect_row, rows)):
            results.append(result)
            if result["status"] == "quarantined":
                reasons[result["reason"]] += 1
            if (i + 1) % 100 == 0 or i + 1 == len(rows):
                elapsed = time.monotonic() - started
                print(json.dumps({"processed": i + 1, "total": len(rows), "verified": i + 1 - sum(reasons.values()),
                                  "elapsed_seconds": round(elapsed, 2), "remaining_seconds_linear": round(elapsed / (i + 1) * (len(rows) - i - 1))}), flush=True)
    if digest_file(args.manifest) != args.manifest_sha256:
        raise AnnotationError("manifest changed during audit")
    report = {"schema": "aihub_original_annotation_audit_v1", "manifest_sha256": args.manifest_sha256,
              "manifest_counts": counts, "selected": len(rows), "verified": len(rows) - sum(reasons.values()),
              "quarantined": sum(reasons.values()), "reasons": dict(reasons), "read_bytes": total_bytes,
              "sampling": "sha256(original-annotation-v1:source_id), lowest per class and official split",
              "perceptual_hash": "direct-grayscale-imdecode_area32_dct8_median-exclude-dc_64bit",
              "workers": args.workers, "elapsed_seconds": round(time.monotonic() - started, 2),
              "snapshot_only": True, "consumer_must_rehash_source_and_annotation": True,
              "training_authorized": False, "deployment_authorized": False,
              "scope": "source pixels and original annotation binding only; no leakage or commercial approval", "records": results}
    with (args.output / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("records", "manifest_counts")}), flush=True)


if __name__ == "__main__":
    main()
