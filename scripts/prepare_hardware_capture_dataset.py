"""운영 캡처를 안전한 YOLO/검증기 학습 후보 데이터로 변환한다.

입력 캡처는 SHA-256으로 중복 제거하며, audit spec에 기록한 snapshot anchor가
맞지 않으면 즉시 중단한다. 배경 사진은 빈 YOLO 라벨로 포함하고, 혼합 또는
다중 물체 사진은 상태 검증 manifest에는 남기되 재료 검출 학습에서는 제외한다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

try:
    from extract_verifier_crops import CLASS_NAMES, letterbox
except ImportError:  # pragma: no cover - python -m scripts... 지원
    from scripts.extract_verifier_crops import CLASS_NAMES, letterbox


CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}
SPECIAL_LABELS = {"negative", "exclude"}


def load_unique_captures(captures_dir: Path) -> list[dict]:
    """동일 이미지 SHA 중 가장 최근 metadata 하나만 반환한다."""
    latest: dict[str, tuple[dict, Path]] = {}
    for metadata_path in sorted(captures_dir.rglob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        sha256 = metadata["image"]["sha256"]
        previous = latest.get(sha256)
        if previous is None or metadata["timestamp"] > previous[0]["timestamp"]:
            latest[sha256] = (metadata, metadata_path)

    rows = []
    for index, (metadata, metadata_path) in enumerate(
        sorted(latest.values(), key=lambda item: item[0]["timestamp"]), start=1
    ):
        rows.append(
            {
                "index": index,
                "sha256": metadata["image"]["sha256"],
                "metadata": metadata,
                "metadata_path": metadata_path,
                "image_path": captures_dir / metadata["image"]["path"],
            }
        )
    return rows


def resolve_audit(rows: list[dict], spec: dict) -> dict[int, dict]:
    expected_count = int(spec["snapshot"]["unique_count"])
    if len(rows) != expected_count:
        raise ValueError(f"snapshot mismatch: unique={len(rows)}, expected={expected_count}")

    for raw_index, sha_prefix in spec["snapshot"].get("anchors", {}).items():
        index = int(raw_index)
        if not rows[index - 1]["sha256"].startswith(sha_prefix):
            raise ValueError(
                f"snapshot anchor mismatch at #{index}: "
                f"{rows[index - 1]['sha256'][:12]} != {sha_prefix}"
            )

    labels: dict[int, str] = {}
    for label, indices in spec["labels"].items():
        if label not in CLASS_IDS and label not in SPECIAL_LABELS:
            raise ValueError(f"unknown audit label: {label}")
        for index in indices:
            if index in labels:
                raise ValueError(f"duplicate audit label at #{index}")
            labels[index] = label

    missing = sorted(set(range(1, len(rows) + 1)) - set(labels))
    if missing:
        raise ValueError(f"unlabeled capture indices: {missing}")

    source_by_index = {}
    for source, indices in spec.get("sources", {}).items():
        for index in indices:
            source_by_index[index] = source

    group_by_index = {}
    for group_name, indices in spec.get("object_groups", {}).items():
        for index in indices:
            group_by_index[index] = group_name

    validation_groups = set(spec.get("validation_groups", []))
    multi_or_mixed = set(spec.get("not_single_object", []))
    foreign = set(spec.get("foreign_material", []))
    dented = set(spec.get("is_dented", []))
    not_dented = set(spec.get("not_dented", []))
    has_label = set(spec.get("has_label", []))
    no_label = set(spec.get("no_label", []))

    resolved = {}
    for row in rows:
        index = row["index"]
        group = group_by_index.get(index, f"capture_{row['sha256'][:12]}")
        resolved[index] = {
            "label": labels[index],
            "source": source_by_index.get(index, "hardware"),
            "object_group": group,
            "split": "val" if group in validation_groups else "train",
            "is_single_object": index not in multi_or_mixed,
            "has_foreign_material": True if index in foreign else (False if labels[index] != "exclude" else None),
            "is_dented": True if index in dented else (False if index in not_dented else None),
            "has_label": True if index in has_label else (False if index in no_label else None),
        }
    return resolved


def load_candidates(path: Path | None) -> dict[str, list[dict]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {row["sha256"]: row.get("candidates", []) for row in data}


def _box_score(box: list[float], confidence: float, width: int, height: int) -> float:
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 2 or box_h <= 2:
        return -math.inf
    area_ratio = box_w * box_h / (width * height)
    if area_ratio < 0.003 or area_ratio > 0.92:
        return -math.inf
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    center_penalty = math.hypot(cx - 0.5, cy - 0.55)
    area_penalty = abs(math.log(max(area_ratio, 1e-6) / 0.20)) * 0.05
    return float(confidence) - center_penalty * 0.20 - area_penalty


def choose_bbox(
    row: dict,
    audit: dict,
    spec: dict,
    candidates: dict[str, list[dict]],
    image_shape: tuple[int, int, int],
) -> tuple[list[float] | None, str, float | None]:
    index = row["index"]
    override = spec.get("bbox_overrides", {}).get(str(index))
    if override:
        return [float(value) for value in override], "audited_override", 1.0

    label = audit["label"]
    if label not in CLASS_IDS:
        return None, "not_applicable", None

    height, width = image_shape[:2]
    expected = [
        item for item in candidates.get(row["sha256"], [])
        if item.get("class_name") == label
    ]
    ranked = sorted(
        expected,
        key=lambda item: _box_score(item["bbox"], item["confidence"], width, height),
        reverse=True,
    )
    if ranked and _box_score(ranked[0]["bbox"], ranked[0]["confidence"], width, height) > -0.25:
        return ranked[0]["bbox"], "low_conf_expected_class", float(ranked[0]["confidence"])

    result_bbox = row["metadata"].get("result", {}).get("bbox")
    if result_bbox:
        return [float(value) for value in result_bbox], "deployed_bbox_fallback", None
    return None, "missing", None


def _clip_bbox(box: list[float], width: int, height: int) -> list[float] | None:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
    y1, y2 = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return [x1, y1, x2, y2]


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f"failed to encode image: {path}")
    encoded.tofile(path)


def _render_overlays(records: list[dict], output_dir: Path) -> None:
    overlay_dir = output_dir / "audit_overlays"
    cols, rows_per, tile_w, tile_h = 3, 5, 420, 330
    for page, start in enumerate(range(0, len(records), cols * rows_per), start=1):
        canvas = np.full((rows_per * tile_h, cols * tile_w, 3), 245, dtype=np.uint8)
        for position, record in enumerate(records[start : start + cols * rows_per]):
            image = cv2.imdecode(np.fromfile(record["source_image"], dtype=np.uint8), cv2.IMREAD_COLOR)
            box = record.get("bbox")
            if box:
                x1, y1, x2, y2 = (int(value) for value in box)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            height, width = image.shape[:2]
            scale = min(tile_w / width, (tile_h - 45) / height)
            resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))
            x = position % cols * tile_w + (tile_w - resized.shape[1]) // 2
            y = position // cols * tile_h + 40 + (tile_h - 45 - resized.shape[0]) // 2
            canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
            caption = (
                f"#{record['index']:03d} {record['label']} {record['split']} "
                f"{record['bbox_source']}"
            )
            cv2.putText(
                canvas,
                caption,
                (position % cols * tile_w + 5, position // cols * tile_h + 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        _write_image(overlay_dir / f"page_{page:02d}.jpg", canvas)


def prepare_dataset(
    captures_dir: Path,
    audit_spec_path: Path,
    output_dir: Path,
    candidates_path: Path | None = None,
    crop_size: int = 320,
    render_overlays: bool = False,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    spec = json.loads(audit_spec_path.read_text(encoding="utf-8"))
    rows = load_unique_captures(captures_dir)
    resolved = resolve_audit(rows, spec)
    candidates = load_candidates(candidates_path)

    detector_sources = set(spec.get("detector_sources", ["hardware"]))
    verifier_rows = []
    records = []
    skipped_bbox = []
    counts = Counter()

    for row in rows:
        audit = resolved[row["index"]]
        image = cv2.imdecode(np.fromfile(row["image_path"], dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unreadable image: {row['image_path']}")
        height, width = image.shape[:2]
        bbox, bbox_source, bbox_confidence = choose_bbox(
            row, audit, spec, candidates, image.shape
        )
        if bbox is not None:
            bbox = _clip_bbox(bbox, width, height)

        label = audit["label"]
        include_detector = (
            audit["source"] in detector_sources
            and audit["is_single_object"]
            and label != "exclude"
            and (label == "negative" or bbox is not None)
        )
        if label in CLASS_IDS and audit["source"] in detector_sources and audit["is_single_object"] and bbox is None:
            skipped_bbox.append(row["index"])

        if include_detector:
            split = audit["split"]
            stem = f"capture_{row['sha256'][:16]}"
            destination_image = output_dir / "yolo" / "images" / split / f"{stem}.jpg"
            destination_label = output_dir / "yolo" / "labels" / split / f"{stem}.txt"
            _write_image(destination_image, image)
            destination_label.parent.mkdir(parents=True, exist_ok=True)
            if label == "negative":
                destination_label.write_text("", encoding="utf-8")
            else:
                x1, y1, x2, y2 = bbox
                normalized = (
                    (x1 + x2) / 2 / width,
                    (y1 + y2) / 2 / height,
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                )
                destination_label.write_text(
                    f"{CLASS_IDS[label]} " + " ".join(f"{value:.8f}" for value in normalized) + "\n",
                    encoding="utf-8",
                )
            counts[("detector", split, label)] += 1

        if label in CLASS_IDS and bbox is not None:
            x1, y1, x2, y2 = bbox
            padding_x, padding_y = (x2 - x1) * 0.08, (y2 - y1) * 0.08
            left, top = max(0, int(x1 - padding_x)), max(0, int(y1 - padding_y))
            right, bottom = min(width, int(x2 + padding_x)), min(height, int(y2 + padding_y))
            crop = letterbox(image[top:bottom, left:right], crop_size)
            relative_crop = Path("verifier") / audit["split"] / label / f"capture_{row['sha256'][:16]}.jpg"
            _write_image(output_dir / relative_crop, crop)
            verifier_rows.append(
                {
                    "filepath": relative_crop.as_posix(),
                    "split": "validation" if audit["split"] == "val" else "training",
                    "source_id": row["sha256"],
                    "material": CLASS_IDS[label],
                    "category": label,
                    "dent": -1 if audit["is_dented"] is None else int(audit["is_dented"]),
                    "label": -1 if audit["has_label"] is None else int(audit["has_label"]),
                    "foreign_material": -1 if audit["has_foreign_material"] is None else int(audit["has_foreign_material"]),
                    "label_proxy": -1,
                    "raw_dirtiness": "hardware_capture",
                    "source_object_count": 1 if audit["is_single_object"] else 2,
                }
            )

        records.append(
            {
                "index": row["index"],
                "sha256": row["sha256"],
                **audit,
                "bbox": bbox,
                "bbox_source": bbox_source,
                "bbox_confidence": bbox_confidence,
                "included_in_detector": include_detector,
                "source_image": str(row["image_path"]),
            }
        )

    yaml_lines = [
        f"path: {str((output_dir / 'yolo').resolve()).replace(chr(92), '/')}",
        "train: images/train",
        "val: images/val",
        "names:",
    ] + [f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)]
    (output_dir / "yolo" / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    manifest_path = output_dir / "verifier" / "hardware_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "filepath", "split", "source_id", "material", "category", "dent", "label",
            "foreign_material", "label_proxy", "raw_dirtiness", "source_object_count",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(verifier_rows)

    resolved_path = output_dir / "resolved_audit_by_sha.json"
    resolved_path.write_text(
        json.dumps({record["sha256"]: record for record in records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = {
        "snapshot_unique_images": len(rows),
        "snapshot_digest": hashlib.sha256("\n".join(row["sha256"] for row in rows).encode()).hexdigest(),
        "detector_counts": {
            f"{split}/{label}": count for (_, split, label), count in sorted(counts.items())
        },
        "detector_total": sum(counts.values()),
        "verifier_total": len(verifier_rows),
        "skipped_missing_bbox_indices": skipped_bbox,
        "audit_spec": str(audit_spec_path.resolve()),
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if render_overlays:
        _render_overlays(records, output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures-dir", type=Path, required=True)
    parser.add_argument("--audit-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--render-overlays", action="store_true")
    args = parser.parse_args()
    report = prepare_dataset(
        args.captures_dir,
        args.audit_spec,
        args.output_dir,
        args.candidates,
        args.crop_size,
        args.render_overlays,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
