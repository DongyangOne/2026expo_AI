"""단일 객체 원본에서 균형 잡힌 9종 YOLO 재학습 데이터셋을 생성한다.

검증기 manifest의 원본 이미지 경로와 bbox를 사용하므로 누적된 변환 폴더를 다시
샘플링하지 않는다. 클래스별 촬영 조건을 round-robin하고, 실제 이미지를 열어 bbox,
노출, 초점과 중복을 확인한 뒤 긴 변 640px JPEG와 YOLO 라벨을 저장한다.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = (
    "can", "pet", "paper", "plastic", "styrofoam",
    "vinyl", "glass", "battery", "fluorescent",
)
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}
RARE_CLASSES = {"battery", "fluorescent"}


@dataclass
class Prepared:
    row: dict[str, str]
    encoded: bytes | None = None
    visual_hash: str | None = None
    label: str | None = None
    reason: str | None = None


def decode_source_path(value: str) -> Path:
    padding = "=" * (-len(value) % 4)
    return Path(base64.urlsafe_b64decode(value + padding).decode("utf-8"))


def _stable_score(row: dict[str, str], seed: int) -> str:
    key = f"{seed}|{row.get('source_id', '')}|{row.get('filepath', '')}"
    return hashlib.sha256(key.encode()).hexdigest()


def _area_bin(row: dict[str, str]) -> str:
    width = max(1.0, float(row["source_width"]))
    height = max(1.0, float(row["source_height"]))
    area = float(row["source_bbox_w"]) * float(row["source_bbox_h"]) / (width * height)
    if area < 0.15:
        return "small"
    if area < 0.45:
        return "medium"
    return "large"


def _stratum(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        _area_bin(row),
        row.get("dent", "-1"),
        row.get("raw_dirtiness", "unknown"),
    )


def diverse_order(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    """촬영 조건이 큰 그룹에 묻히지 않게 strata를 순환하며 반환한다."""
    groups: dict[tuple[str, str, str], deque[dict[str, str]]] = {}
    for key, values in _group_by_stratum(rows).items():
        values.sort(key=lambda row: _stable_score(row, seed))
        groups[key] = deque(values)
    keys = sorted(groups, key=lambda key: hashlib.sha256(f"{seed}|{key}".encode()).hexdigest())
    ordered: list[dict[str, str]] = []
    while keys:
        next_keys = []
        for key in keys:
            queue = groups[key]
            if queue:
                ordered.append(queue.popleft())
            if queue:
                next_keys.append(key)
        keys = next_keys
    return ordered


def _group_by_stratum(rows: list[dict[str, str]]):
    grouped = defaultdict(list)
    for row in rows:
        grouped[_stratum(row)].append(row)
    return grouped


def read_candidates(manifest: Path, seed: int) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    with manifest.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            category = row.get("category", "")
            if row.get("split", "").lower() != "training" or category not in CLASS_IDS:
                continue
            if int(row.get("source_object_count", "0")) != 1:
                continue
            identity = (row.get("source_id", ""), row.get("source_path_b64", ""))
            if not all(identity) or identity in seen:
                continue
            seen.add(identity)
            grouped[category].append(row)
    return {category: diverse_order(rows, seed) for category, rows in grouped.items()}


def _resize_long_side(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, size / max(height, width))
    if scale == 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _difference_hash(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tiny = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = tiny[:, 1:] > tiny[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def prepare_row(
    row: dict[str, str],
    size: int,
    min_area_ratio: float,
    max_area_ratio: float,
    min_focus: float,
    min_brightness: float,
    max_brightness: float,
) -> Prepared:
    try:
        path = decode_source_path(row["source_path_b64"])
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            return Prepared(row, reason="image_missing")
        height, width = image.shape[:2]
        if min(height, width) < 320:
            return Prepared(row, reason="resolution_too_small")

        x = float(row["source_bbox_x"])
        y = float(row["source_bbox_y"])
        box_w = float(row["source_bbox_w"])
        box_h = float(row["source_bbox_h"])
        if box_w <= 1 or box_h <= 1 or x < 0 or y < 0:
            return Prepared(row, reason="invalid_bbox")
        if x + box_w > width + 2 or y + box_h > height + 2:
            return Prepared(row, reason="bbox_outside_image")
        area_ratio = box_w * box_h / (width * height)
        if not min_area_ratio <= area_ratio <= max_area_ratio:
            return Prepared(row, reason="bbox_area_outside_range")

        resized = _resize_long_side(image, size)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        if not min_brightness <= brightness <= max_brightness:
            return Prepared(row, reason="exposure_outside_range")
        focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if focus < min_focus:
            return Prepared(row, reason="too_blurry")

        ok, encoded = cv2.imencode(
            ".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        if not ok:
            return Prepared(row, reason="jpeg_encode_failed")
        class_id = CLASS_IDS[row["category"]]
        cx = (x + box_w / 2) / width
        cy = (y + box_h / 2) / height
        normalized_w = box_w / width
        normalized_h = box_h / height
        values = (cx, cy, normalized_w, normalized_h)
        if not all(math.isfinite(value) and 0 < value <= 1 for value in values):
            return Prepared(row, reason="invalid_normalized_bbox")
        label = f"{class_id} {cx:.8f} {cy:.8f} {normalized_w:.8f} {normalized_h:.8f}\n"
        return Prepared(
            row,
            encoded=encoded.tobytes(),
            visual_hash=_difference_hash(resized),
            label=label,
        )
    except Exception as error:  # 개별 손상 파일이 전체 선별을 중단하지 않게 한다.
        return Prepared(row, reason=f"exception:{type(error).__name__}")


def _write_prepared(item: Prepared, output_dir: Path) -> tuple[str, str]:
    category = item.row["category"]
    source_id = item.row["source_id"]
    stem = f"{category}_{source_id[:24]}"
    image_path = output_dir / "images" / "train" / f"{stem}.jpg"
    label_path = output_dir / "labels" / "train" / f"{stem}.txt"
    image_path.write_bytes(item.encoded or b"")
    label_path.write_text(item.label or "", encoding="utf-8")
    return stem, str(decode_source_path(item.row["source_path_b64"]))


def _merge_hardware(hardware_dir: Path, output_dir: Path, repeats: int) -> int:
    images = hardware_dir / "images" / "train"
    labels = hardware_dir / "labels" / "train"
    if not images.is_dir() or not labels.is_dir():
        raise FileNotFoundError("hardware YOLO directory must contain images/train and labels/train")
    copied = 0
    for image in sorted(images.iterdir()):
        if not image.is_file():
            continue
        label = labels / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"missing hardware label: {label}")
        for repeat in range(repeats):
            stem = f"hardware_r{repeat + 1}_{image.stem}"
            shutil.copy2(image, output_dir / "images" / "train" / f"{stem}{image.suffix.lower()}")
            shutil.copy2(label, output_dir / "labels" / "train" / f"{stem}.txt")
            copied += 1
    return copied


def build_dataset(args) -> dict:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)

    candidates = read_candidates(Path(args.manifest), args.seed)
    selected_counts = Counter()
    rejected_counts = Counter()
    selected_rows = []
    visual_hashes: dict[str, set[str]] = defaultdict(set)

    for category in CLASS_NAMES:
        rows = candidates.get(category, [])
        target = len(rows) if category in RARE_CLASSES else min(args.per_class, len(rows))
        if target == 0:
            raise RuntimeError(f"no training candidates for class: {category}")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for start in range(0, len(rows), args.workers * 8):
                if selected_counts[category] >= target:
                    break
                chunk = rows[start:start + args.workers * 8]
                prepared = executor.map(
                    lambda row: prepare_row(
                        row, args.imgsz, args.min_area_ratio, args.max_area_ratio,
                        args.min_focus, args.min_brightness, args.max_brightness,
                    ),
                    chunk,
                )
                for item in prepared:
                    if selected_counts[category] >= target:
                        break
                    if item.reason:
                        rejected_counts[item.reason] += 1
                        continue
                    if item.visual_hash in visual_hashes[category]:
                        rejected_counts["duplicate_visual_hash"] += 1
                        continue
                    visual_hashes[category].add(item.visual_hash or "")
                    stem, source_path = _write_prepared(item, output_dir)
                    selected_counts[category] += 1
                    selected_rows.append(
                        {
                            "stem": stem,
                            "category": category,
                            "class_id": CLASS_IDS[category],
                            "source_id": item.row["source_id"],
                            "source_path": source_path,
                            "area_bin": _area_bin(item.row),
                            "dent": item.row.get("dent", "-1"),
                            "raw_dirtiness": item.row.get("raw_dirtiness", ""),
                        }
                    )
                    if len(selected_rows) % 1000 == 0:
                        free_gb = shutil.disk_usage(output_dir).free / 1024 ** 3
                        if free_gb < args.min_free_gb:
                            raise RuntimeError(
                                f"free space {free_gb:.1f}GB is below {args.min_free_gb:.1f}GB"
                            )
                        print(
                            f"selected={len(selected_rows):,} class={category} "
                            f"free={free_gb:.1f}GB",
                            flush=True,
                        )

    hardware_count = 0
    if args.hardware_yolo_dir:
        hardware_count = _merge_hardware(
            Path(args.hardware_yolo_dir), output_dir, args.hardware_repeats
        )

    manifest_path = output_dir / "selected_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        fields = [
            "stem", "category", "class_id", "source_id", "source_path",
            "area_bin", "dent", "raw_dirtiness",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected_rows)

    yaml_lines = [
        f"path: {output_dir.as_posix()}",
        "train: images/train",
        f"val: {args.validation_images}",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    (output_dir / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    summary = {
        "manifest": str(args.manifest),
        "selected_original": sum(selected_counts.values()),
        "selected_by_class": dict(selected_counts),
        "hardware_repeated": hardware_count,
        "hardware_repeats": args.hardware_repeats if args.hardware_yolo_dir else 0,
        "rejected": dict(rejected_counts),
        "selection": {
            "single_object_only": True,
            "per_class": args.per_class,
            "rare_classes_use_all": sorted(RARE_CLASSES),
            "imgsz": args.imgsz,
            "min_area_ratio": args.min_area_ratio,
            "max_area_ratio": args.max_area_ratio,
            "min_focus": args.min_focus,
            "brightness": [args.min_brightness, args.max_brightness],
            "seed": args.seed,
        },
    }
    (output_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hardware-yolo-dir")
    parser.add_argument("--hardware-repeats", type=int, default=5)
    parser.add_argument("--validation-images", required=True)
    parser.add_argument("--per-class", type=int, default=10_000)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--min-area-ratio", type=float, default=0.025)
    parser.add_argument("--max-area-ratio", type=float, default=0.95)
    parser.add_argument("--min-focus", type=float, default=12.0)
    parser.add_argument("--min-brightness", type=float, default=12.0)
    parser.add_argument("--max-brightness", type=float, default=245.0)
    parser.add_argument("--min-free-gb", type=float, default=500.0)
    args = parser.parse_args()
    if args.per_class < 1 or args.workers < 1 or args.hardware_repeats < 1:
        parser.error("per-class, workers and hardware-repeats must be positive")
    if not 0 < args.min_area_ratio < args.max_area_ratio <= 1:
        parser.error("bbox area ratio bounds are invalid")
    return args


if __name__ == "__main__":
    build_dataset(parse_args())
