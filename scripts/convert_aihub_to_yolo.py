"""
AI Hub JSON 라벨 → YOLO 포맷 변환 스크립트

AI Hub 실제 폴더 구조:
  Training/01.원천데이터/TS_카테고리/*.jpg     ← 학습 이미지
  Training/02.라벨링데이터/TL_카테고리/*.json  ← 학습 라벨
  Validation/01.원천데이터/VS_카테고리/*.jpg   ← 검증 이미지
  Validation/02.라벨링데이터/VL_카테고리/*.json← 검증 라벨

JSON 구조:
  IMAGE_INFO.FILE_NAME          = 이미지 파일명 (JSON과 동일 stem)
  IMAGE_INFO.IMAGE_WIDTH/HEIGHT = 이미지 크기
  ANNOTATION_INFO[].CLASS       = 카테고리 (예: "플라스틱")
  ANNOTATION_INFO[].DETAILS     = 세부 카테고리 (예: "PP")
  ANNOTATION_INFO[].POINTS[0]   = [x, y, w, h] COCO 포맷

※ AI Hub이 이미 Training/Validation 분리 제공 → 랜덤 분할 없이 그대로 사용

사용법:
  python scripts/convert_aihub_to_yolo.py \
    --dataset_dir /app/ai_dataset/학습용_데이터 \
    --output_dir  /app/yolo_dataset_9class
"""

import argparse
import json
import shutil
from pathlib import Path

from tqdm import tqdm

# ─────────────────────────────────────────
# 카테고리 매핑 (9클래스)
# ─────────────────────────────────────────
CATEGORY_MAP = {
    # can (0)
    "철캔":       0,
    "알루미늄캔": 0,
    "금속캔":     0,
    # pet (1)
    "페트병":     1,
    "무색단일":   1,
    "유색단일":   1,
    # paper (2)
    "종이":       2,
    # plastic (3)
    "플라스틱":   3,
    "PE":         3,
    "PP":         3,
    "PS":         3,
    # styrofoam (4)
    "스티로폼":   4,
    # vinyl (5)
    "비닐":       5,
    # glass (6)
    "유리병":     6,
    # battery (7) → 거부
    "건전지":     7,
    # fluorescent (8) → 거부
    "형광등":     8,
}

CLASS_NAMES = ["can", "pet", "paper", "plastic", "styrofoam", "vinyl", "glass", "battery", "fluorescent"]


def find_category_id(class_str: str, details_str: str = "") -> int:
    combined = f"{class_str} {details_str}"
    for key, class_id in CATEGORY_MAP.items():
        if key in combined:
            return class_id
    return 3  # 기본값: plastic


def convert_bbox_to_yolo(points, img_w, img_h):
    """COCO bbox [x, y, w, h] → YOLO [cx, cy, w, h] 정규화"""
    x, y, w, h = points
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    return cx, cy, nw, nh


def process_label_file(json_path: Path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    img_info = data.get("IMAGE_INFO", {})
    img_w = img_info.get("IMAGE_WIDTH", 0)
    img_h = img_info.get("IMAGE_HEIGHT", 0)

    if not img_w or not img_h:
        return []

    labels = []
    for ann in data.get("ANNOTATION_INFO", []):
        pts = ann.get("POINTS", [])
        if not pts or len(pts[0]) != 4:
            continue
        class_id = find_category_id(ann.get("CLASS", ""), ann.get("DETAILS", ""))
        cx, cy, nw, nh = convert_bbox_to_yolo(pts[0], img_w, img_h)
        labels.append((class_id, cx, cy, nw, nh))

    return labels


def build_src_dir_map(src_base: Path) -> dict:
    """
    원천데이터 폴더를 prefix 기준으로 매핑.
    TS_2.직접촬영_01.금속캔_001.철캔_1  →  key: TS_2.직접촬영_01.금속캔_001.철캔
    TS_1.영상추출_02.종이_001.종이       →  key: TS_1.영상추출_02.종이_001.종이  (그대로)
    """
    import re
    mapping = {}
    if not src_base.exists():
        return mapping
    for d in src_base.iterdir():
        if not d.is_dir():
            continue
        # 말미 _숫자 제거
        base = re.sub(r'_\d+$', '', d.name)
        mapping.setdefault(base, []).append(d)
    return mapping


def find_pairs_in_split(split_dir: Path) -> list:
    """
    Training/Validation 폴더에서 이미지-라벨 쌍 수집.
    직접촬영 _1/_2/_3 분할 폴더도 처리.
    """
    pairs  = []
    src_base = split_dir / "01.원천데이터"
    src_map  = build_src_dir_map(src_base)
    cache    = {}  # prefix → [Path, ...]

    for json_file in split_dir.rglob("*.json"):
        if "라벨링데이터" not in str(json_file):
            continue

        label_folder = json_file.parent.name

        if label_folder.startswith("TL_"):
            src_prefix = "TS_" + label_folder[3:]
        elif label_folder.startswith("VL_"):
            src_prefix = "VS_" + label_folder[3:]
        else:
            continue

        if src_prefix not in cache:
            cache[src_prefix] = src_map.get(src_prefix, [])

        found = False
        for src_dir in cache[src_prefix]:
            for ext in [".jpg", ".jpeg", ".png"]:
                img_path = src_dir / (json_file.stem + ext)
                if img_path.exists():
                    pairs.append((img_path, json_file))
                    found = True
                    break
            if found:
                break

    return pairs


def save_split(pairs: list, output_dir: Path, split: str) -> dict:
    """단일 split(train 또는 val) 저장"""
    img_out = output_dir / split / "images"
    lbl_out = output_dir / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    stats = {c: 0 for c in CLASS_NAMES}

    print(f"\n[{split}] {len(pairs)}장 처리 중...")
    for i, (img_path, json_path) in enumerate(tqdm(pairs)):
        stem    = f"{split}_{i:07d}"
        new_img = img_out / (stem + img_path.suffix)
        new_lbl = lbl_out / (stem + ".txt")

        shutil.copy2(img_path, new_img)

        yolo_labels = process_label_file(json_path)
        with open(new_lbl, "w") as f:
            for class_id, cx, cy, nw, nh in yolo_labels:
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                stats[CLASS_NAMES[class_id]] += 1

    return stats


def write_yaml(output_dir: Path):
    import yaml
    content = {
        "path":  str(output_dir.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "nc":    9,
        "names": CLASS_NAMES,
    }
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(content, f, default_flow_style=False, allow_unicode=True)
    print(f"\ndataset.yaml 생성: {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="학습용_데이터 경로")
    parser.add_argument("--output_dir",  default="/app/yolo_dataset_9class")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)

    train_dir = dataset_dir / "01-1.정식개방데이터" / "Training"
    val_dir   = dataset_dir / "01-1.정식개방데이터" / "Validation"

    print("=== Training 쌍 수집 중... ===")
    train_pairs = find_pairs_in_split(train_dir)
    print(f"Train: {len(train_pairs)}쌍")

    print("=== Validation 쌍 수집 중... ===")
    val_pairs = find_pairs_in_split(val_dir)
    print(f"Val: {len(val_pairs)}쌍")

    if not train_pairs:
        print("[ERROR] Training 쌍이 없습니다. 경로를 확인하세요.")
        exit(1)

    train_stats = save_split(train_pairs, output_dir, "train")
    val_stats   = save_split(val_pairs,   output_dir, "val")
    write_yaml(output_dir)

    print("\n=== 변환 완료 ===")
    for split, stats in [("train", train_stats), ("val", val_stats)]:
        total = sum(stats.values())
        print(f"  {split}: {total}개 라벨")
        for cls, cnt in stats.items():
            print(f"    {cls:12s}: {cnt:,}")
