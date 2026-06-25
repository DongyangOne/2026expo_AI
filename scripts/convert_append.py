"""
누락된 데이터 추가 변환 스크립트
- 기존 yolo_dataset_9class에 직접촬영(TL_2.*) 데이터 추가
- 종이 영상추출(TL_1.영상추출_02.종이*) 데이터 추가
- 기존 파일 번호에 이어서 저장

사용법:
  python scripts/convert_append.py \
    --dataset_dir /app/ai_dataset/학습용_데이터 \
    --output_dir  /app/yolo_dataset_9class
"""

import argparse
import json
import re
import shutil
from pathlib import Path

from tqdm import tqdm

CATEGORY_MAP = {
    "철캔": 0, "알루미늄캔": 0, "금속캔": 0,
    "페트병": 1, "무색단일": 1, "유색단일": 1,
    "종이": 2,
    "플라스틱": 3, "PE": 3, "PP": 3, "PS": 3,
    "스티로폼": 4,
    "비닐": 5,
    "유리병": 6,
    "건전지": 7,
    "형광등": 8,
}
CLASS_NAMES = ["can", "pet", "paper", "plastic", "styrofoam", "vinyl", "glass", "battery", "fluorescent"]


def find_category_id(class_str, details_str=""):
    combined = f"{class_str} {details_str}"
    for key, cid in CATEGORY_MAP.items():
        if key in combined:
            return cid
    return 3


def convert_bbox_to_yolo(points, img_w, img_h):
    x, y, w, h = points
    cx = max(0.0, min(1.0, (x + w / 2) / img_w))
    cy = max(0.0, min(1.0, (y + h / 2) / img_h))
    nw = max(0.0, min(1.0, w / img_w))
    nh = max(0.0, min(1.0, h / img_h))
    return cx, cy, nw, nh


def process_label_file(json_path):
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
        cid = find_category_id(ann.get("CLASS", ""), ann.get("DETAILS", ""))
        cx, cy, nw, nh = convert_bbox_to_yolo(pts[0], img_w, img_h)
        labels.append((cid, cx, cy, nw, nh))
    return labels


def build_src_dir_map(src_base):
    mapping = {}
    if not src_base.exists():
        return mapping
    for d in src_base.iterdir():
        if not d.is_dir():
            continue
        base = re.sub(r'_\d+$', '', d.name)
        mapping.setdefault(base, []).append(d)
    return mapping


def find_missing_train_pairs(dataset_dir: Path) -> list:
    """
    기존에 누락된 Training 데이터만 수집:
    - 직접촬영: TL_2.직접촬영_* 전체
    - 종이 영상추출: TL_1.영상추출_02.종이*
    """
    pairs = []
    train_dir  = dataset_dir / "01-1.정식개방데이터" / "Training"
    src_base   = train_dir / "01.원천데이터"
    label_base = train_dir / "02.라벨링데이터"

    src_map = build_src_dir_map(src_base)
    cache   = {}

    for label_dir in sorted(label_base.iterdir()):
        if not label_dir.is_dir():
            continue
        name = label_dir.name

        # 직접촬영 전체 OR 종이 영상추출만 처리
        is_jikjup  = name.startswith("TL_2.")
        is_paper   = name.startswith("TL_1.") and "종이" in name

        if not (is_jikjup or is_paper):
            continue

        src_prefix = "TS_" + name[3:]

        if src_prefix not in cache:
            cache[src_prefix] = src_map.get(src_prefix, [])

        print(f"  처리 중: {name} → 후보폴더 {len(cache[src_prefix])}개")

        for json_file in tqdm(sorted(label_dir.rglob("*.json")), desc=name, leave=False):
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


def count_existing(output_dir: Path, split: str) -> int:
    img_dir = output_dir / split / "images"
    if not img_dir.exists():
        return 0
    return len(list(img_dir.iterdir()))


def append_split(pairs, output_dir, split, start_idx):
    img_out = output_dir / split / "images"
    lbl_out = output_dir / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    stats = {c: 0 for c in CLASS_NAMES}

    print(f"\n[{split}] {len(pairs)}장 추가 중... (시작 인덱스: {start_idx})")
    for i, (img_path, json_path) in enumerate(tqdm(pairs)):
        idx  = start_idx + i
        stem = f"{split}_{idx:07d}"
        shutil.copy2(img_path, img_out / (stem + img_path.suffix))
        yolo_labels = process_label_file(json_path)
        with open(lbl_out / (stem + ".txt"), "w") as f:
            for cid, cx, cy, nw, nh in yolo_labels:
                f.write(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                stats[CLASS_NAMES[cid]] += 1

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir",  default="/app/yolo_dataset_9class")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)

    # 기존 파일 수 확인
    existing_train = count_existing(output_dir, "train")
    print(f"기존 train 파일 수: {existing_train:,}")

    # 누락 데이터 수집 (직접촬영 + 종이 영상추출)
    print("\n=== 누락 Training 쌍 수집 중... ===")
    missing_pairs = find_missing_train_pairs(dataset_dir)
    print(f"추가할 쌍: {len(missing_pairs):,}")

    if not missing_pairs:
        print("[완료] 추가할 데이터 없음")
        exit(0)

    stats = append_split(missing_pairs, output_dir, "train", existing_train)

    total_train = count_existing(output_dir, "train")
    print(f"\n=== 추가 완료 ===")
    print(f"  train 총 파일 수: {total_train:,}")
    print(f"  추가된 라벨:")
    for cls, cnt in stats.items():
        if cnt > 0:
            print(f"    {cls:12s}: {cnt:,}")
