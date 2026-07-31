"""9종 품목 + 상태 멀티태스크 검증기용 객체 crop 데이터 생성.

AI Hub 원본의 공식 Training/Validation 분리를 유지하고 직접촬영 데이터만 사용한다.
기존 JSON만으로 라벨과 실제 외부 이물질을 구분할 수 없으므로 `label`과
`foreign_material`은 -1(학습 마스킹)로 둔다. `DIRTINESS`에서 만든 값은
검수 참고용 `label_proxy`에만 기록한다.

실행 예시 (NAS Docker):
  python /app/extract_verifier_crops.py \
    --dataset-dir /app/ai_dataset/학습용_데이터 \
    --output-dir /app/crops_verifier_v1 \
    --size 320 --workers 2 --max-per-folder 10000 --val-max-per-folder 2000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = [
    "can", "pet", "paper", "plastic", "styrofoam",
    "vinyl", "glass", "battery", "fluorescent",
]

_CATEGORY_KEYS = (
    (("철캔", "알루미늄캔", "금속캔"), 0),
    (("페트병", "무색단일", "유색단일"), 1),
    (("종이",), 2),
    (("플라스틱", ".PE", ".PP", ".PS"), 3),
    (("스티로폼",), 4),
    (("비닐",), 5),
    (("유리병",), 6),
    (("건전지",), 7),
    (("형광등",), 8),
)

DENT_CLASSES = {0, 1}  # can, pet
LABEL_CLASSES = {1, 3}  # pet, plastic
DENT_MAP = {"원형": 0, "찌그러짐": 1, "완전압착": 1}
LABEL_PROXY_MAP = {"오염없음": 0, "이물질(외부)": 1}


def make_source_key(split_name: str, label_dir_name: str, filename: str) -> str:
    """파일시스템의 surrogateescape 문자가 섞인 이름도 안정적으로 해시한다."""
    value = f"{split_name}/{label_dir_name}/{filename}"
    return hashlib.sha1(value.encode("utf-8", errors="surrogateescape")).hexdigest()[:20]


def category_id(*values: str) -> int | None:
    text = " ".join(values)
    for keys, class_id in _CATEGORY_KEYS:
        if any(key in text for key in keys):
            return class_id
    return None


def points_to_bbox(points):
    if not points or not isinstance(points[0], (list, tuple)):
        return None
    if len(points[0]) == 4:
        return [float(value) for value in points[0]]
    if len(points[0]) == 2:
        xs = [float(point[0]) for point in points if len(point) >= 2]
        ys = [float(point[1]) for point in points if len(point) >= 2]
        if xs and ys:
            return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
    return None


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = size / max(height, width)
    new_h = max(1, int(height * scale))
    new_w = max(1, int(width * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def _imread_unicode(path: str):
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def _init_worker():
    cv2.setNumThreads(1)


def _stride_sample(items: list[Path], cap: int) -> list[Path]:
    if cap > 0 and len(items) > cap:
        step = len(items) / cap
        return [items[int(index * step)] for index in range(cap)]
    return items


def _source_map(source_base: Path) -> dict[str, list[Path]]:
    mapping: dict[str, list[Path]] = {}
    for directory in source_base.iterdir():
        if directory.is_dir():
            base = re.sub(r"_\d+$", "", directory.name)
            mapping.setdefault(base, []).append(directory)
    return mapping


def _find_image(stem: str, directories: list[Path]) -> Path | None:
    for directory in directories:
        for extension in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            candidate = directory / f"{stem}{extension}"
            if candidate.exists():
                return candidate
    return None


def collect_tasks(
    dataset_root: Path,
    output_dir: Path,
    split_name: str,
    cap_per_folder: int,
    size: int,
    padding: float,
):
    split_dir = dataset_root / "01-1.정식개방데이터" / split_name
    label_base = split_dir / "02.라벨링데이터"
    source_base = split_dir / "01.원천데이터"
    sources = _source_map(source_base)
    tasks = []

    for label_dir in sorted(path for path in label_base.iterdir() if path.is_dir()):
        if not (label_dir.name.startswith("TL_2.") or label_dir.name.startswith("VL_2.")):
            continue
        folder_class = category_id(label_dir.name)
        if folder_class is None:
            print(f"[WARN] 품목 매핑 실패: {label_dir.name}", flush=True)
            continue
        source_prefix = ("TS_" if label_dir.name.startswith("TL_") else "VS_") + label_dir.name[3:]
        source_dirs = sources.get(source_prefix, [])
        if not source_dirs:
            print(f"[WARN] 원천 폴더 없음: {source_prefix}", flush=True)
            continue

        json_files = _stride_sample(sorted(label_dir.rglob("*.json")), cap_per_folder)
        paired = 0
        for json_path in json_files:
            image_path = _find_image(json_path.stem, source_dirs)
            if image_path is None:
                continue
            source_key = make_source_key(split_name, label_dir.name, json_path.name)
            tasks.append(
                (
                    str(image_path), str(json_path), split_name.lower(), folder_class,
                    source_key, str(output_dir), size, padding,
                )
            )
            paired += 1
        print(f"  {split_name} {label_dir.name}: {paired:,}쌍", flush=True)
    return tasks


def worker(task):
    image_path, json_path, split, folder_class, source_key, output_dir, size, padding = task
    image = _imread_unicode(image_path)
    if image is None:
        return []
    try:
        with open(json_path, encoding="utf-8") as file:
            annotations = json.load(file).get("ANNOTATION_INFO", [])
    except Exception:
        return []

    height, width = image.shape[:2]
    rows = []
    for ann_index, annotation in enumerate(annotations):
        bbox = points_to_bbox(annotation.get("POINTS", []))
        if bbox is None:
            continue
        material = category_id(annotation.get("CLASS", ""), annotation.get("DETAILS", ""))
        material = folder_class if material is None else material

        x, y, box_w, box_h = bbox
        x1 = max(0, int(x - box_w * padding))
        y1 = max(0, int(y - box_h * padding))
        x2 = min(width, int(x + box_w * (1 + padding)))
        y2 = min(height, int(y + box_h * (1 + padding)))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = letterbox(image[y1:y2, x1:x2], size)
        class_name = CLASS_NAMES[material]
        filename = f"{source_key}_{ann_index:02d}.jpg"
        relative_path = Path(split) / class_name / filename
        absolute_path = Path(output_dir) / relative_path
        ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            continue
        encoded.tofile(absolute_path)

        dent = DENT_MAP.get(annotation.get("DAMAGE", ""), -1) if material in DENT_CLASSES else -1
        dirtiness = annotation.get("DIRTINESS", "")
        label_proxy = LABEL_PROXY_MAP.get(dirtiness, -1) if material in LABEL_CLASSES else -1
        rows.append(
            (
                relative_path.as_posix(), split, source_key, material, class_name,
                dent, -1, -1, label_proxy, dirtiness,
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-per-folder", type=int, default=10000)
    parser.add_argument("--val-max-per-folder", type=int, default=2000)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "manifest.csv"
    if manifest_path.exists():
        raise SystemExit(f"[ERROR] 기존 manifest가 있습니다: {manifest_path}")

    for split in ("training", "validation"):
        for class_name in CLASS_NAMES:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    tasks = []
    tasks.extend(
        collect_tasks(
            dataset_root, output_dir, "Training", args.max_per_folder,
            args.size, args.padding,
        )
    )
    tasks.extend(
        collect_tasks(
            dataset_root, output_dir, "Validation", args.val_max_per_folder,
            args.size, args.padding,
        )
    )
    if not tasks:
        raise SystemExit("[ERROR] 이미지/JSON 페어를 찾지 못했습니다.")

    rows = []
    stats = Counter()
    with Pool(args.workers, initializer=_init_worker) as pool:
        for index, result_rows in enumerate(pool.imap_unordered(worker, tasks, chunksize=32), 1):
            rows.extend(result_rows)
            for row in result_rows:
                stats[(row[1], row[4])] += 1
            if index % 5000 == 0:
                print(f"  진행 {index:,}/{len(tasks):,} JSON", flush=True)

    with open(manifest_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "filepath", "split", "source_id", "material", "category",
                "dent", "label", "foreign_material", "label_proxy", "raw_dirtiness",
            ]
        )
        writer.writerows(rows)

    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "classes": CLASS_NAMES,
                "input_size": args.size,
                "padding": args.padding,
                "label_policy": "label and foreign_material are masked until human review",
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n완료: {len(rows):,} crops → {manifest_path}", flush=True)
    for (split, class_name), count in sorted(stats.items()):
        print(f"  {split:10s} {class_name:12s}: {count:,}", flush=True)


if __name__ == "__main__":
    main()
