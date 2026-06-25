"""
convert_remainder.py — convert_v2 의 '여집합'만 추가 변환 (증분).

convert_v2 는 폴더당 --max_per_folder(=15000) 장을 **균등 stride** 로 솎아냈다.
이 스크립트는 동일한 stride 로직을 재현해 **그때 뽑히지 않은 나머지** 이미지만
골라, 이미 만들어진 v2 데이터셋 폴더에 그대로 append 한다.

목적: 이미 640px 로 리사이즈/변환된 train 361,597장은 건너뛰고,
상한에 잘려 누락된 약 44만장만 마저 변환 → v1 과 동일한 '전체 데이터'를
리사이즈된 형태로 확보. (= 전량 학습용)

안전장치:
  - stride 픽을 convert_v2 와 100% 동일하게 재현 → 그 여집합만 처리 (중복 0).
  - 출력 파일명 prefix 를 train_r 로 분리 → 기존 train_0000000.. 과 절대 충돌 없음.
  - --dry_run 으로 변환 없이 폴더별 잔여 수만 먼저 검증 가능.
  - val 은 기본 건너뜀 (기존 47,826장 유지). --include_val 로 확장 가능.

실행 (NAS Docker, convert_v2 와 동일 마운트):
  # 1) 먼저 dry-run 으로 잔여 수 확인 (수 분)
  docker run --rm -v /share/Container:/app ultralytics/ultralytics:latest \
    python /app/convert_remainder.py \
      --dataset_dir /app/ai_dataset/학습용_데이터 \
      --output_dir  /app/yolo_dataset_9class_v2 \
      --cap 15000 --dry_run

  # 2) 실제 변환 (detached, 로그는 docker logs convert_remainder)
  docker run -d --name convert_remainder -v /share/Container:/app ultralytics/ultralytics:latest \
    python /app/convert_remainder.py \
      --dataset_dir /app/ai_dataset/학습용_데이터 \
      --output_dir  /app/yolo_dataset_9class_v2 \
      --imgsz 640 --cap 15000 --workers 12
"""

import argparse
import json
import re
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────
# 9클래스 매핑 (convert_v2 와 동일)
# ─────────────────────────────────────────
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


def find_category_id(class_str: str, details_str: str = "") -> int:
    combined = f"{class_str} {details_str}"
    for key, cid in CATEGORY_MAP.items():
        if key in combined:
            return cid
    return 3  # 기본 plastic


# ─────────────────────────────────────────
# bbox 추출 (convert_v2 와 동일)
# ─────────────────────────────────────────
def points_to_bbox(points):
    if not points:
        return None
    p0 = points[0]
    if not isinstance(p0, (list, tuple)):
        return None
    if len(p0) == 4:                      # BOX
        return [float(v) for v in p0]
    if len(p0) == 2:                      # POLYGON
        xs = [float(p[0]) for p in points if len(p) >= 2]
        ys = [float(p[1]) for p in points if len(p) >= 2]
        if not xs or not ys:
            return None
        x, y = min(xs), min(ys)
        return [x, y, max(xs) - x, max(ys) - y]
    return None


def to_yolo(bbox, w, h):
    x, y, bw, bh = bbox
    cx = max(0.0, min(1.0, (x + bw / 2) / w))
    cy = max(0.0, min(1.0, (y + bh / 2) / h))
    nw = max(0.0, min(1.0, bw / w))
    nh = max(0.0, min(1.0, bh / h))
    return cx, cy, nw, nh


def labels_from_json(json_path: str, w: int, h: int):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for ann in data.get("ANNOTATION_INFO", []):
        bbox = points_to_bbox(ann.get("POINTS", []))
        if bbox is None:
            continue
        cid = find_category_id(ann.get("CLASS", ""), ann.get("DETAILS", ""))
        cx, cy, nw, nh = to_yolo(bbox, w, h)
        if nw <= 0 or nh <= 0:
            continue
        out.append((cid, cx, cy, nw, nh))
    return out


# ─────────────────────────────────────────
# 워커 (convert_v2 와 동일)
# ─────────────────────────────────────────
def _init_worker():
    cv2.setNumThreads(1)


def _imread_unicode(path: str):
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def worker(task):
    img_path, json_path, out_img, out_lbl, imgsz = task
    img = _imread_unicode(img_path)
    if img is None:
        return ("fail", None)

    h, w = img.shape[:2]
    if not w or not h:
        return ("fail", None)

    labels = labels_from_json(json_path, w, h)
    if not labels:
        return ("empty", None)

    scale = imgsz / float(max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return ("fail", None)
    enc.tofile(out_img)

    with open(out_lbl, "w") as f:
        for cid, cx, cy, nw, nh in labels:
            f.write(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    return ("ok", Counter(l[0] for l in labels))


# ─────────────────────────────────────────
# 페어 수집 — stride '여집합'
# ─────────────────────────────────────────
def build_src_dir_map(src_base: Path) -> dict:
    mapping = {}
    if not src_base.exists():
        return mapping
    for d in src_base.iterdir():
        if d.is_dir():
            base = re.sub(r"_\d+$", "", d.name)
            mapping.setdefault(base, []).append(d)
    return mapping


def _stride_remainder(items, cap):
    """
    convert_v2._stride_sample 이 뽑은 인덱스의 **여집합** 반환.
    stride 공식(step = n/cap, picked = {int(i*step)})을 그대로 재현.
    cap<=0 이거나 n<=cap 이면 (이미 전량 사용됐으므로) 잔여 없음 → [].
    """
    n = len(items)
    if not cap or n <= cap:
        return []
    step = n / cap
    picked = {int(i * step) for i in range(cap)}
    return [items[k] for k in range(n) if k not in picked]


def collect_remainder_pairs(split_dir: Path, cap: int) -> list:
    src_base = split_dir / "01.원천데이터"
    label_base = split_dir / "02.라벨링데이터"
    src_map = build_src_dir_map(src_base)
    cache = {}

    pairs = []
    if not label_base.exists():
        print(f"[WARN] 라벨 폴더 없음: {label_base}", flush=True)
        return pairs

    for label_dir in sorted(label_base.iterdir()):
        if not label_dir.is_dir():
            continue
        name = label_dir.name
        if name.startswith("TL_"):
            src_prefix = "TS_" + name[3:]
        elif name.startswith("VL_"):
            src_prefix = "VS_" + name[3:]
        else:
            continue
        if src_prefix not in cache:
            cache[src_prefix] = src_map.get(src_prefix, [])

        jsons_all = sorted(str(p) for p in label_dir.rglob("*.json"))
        jsons_rem = _stride_remainder(jsons_all, cap)

        folder_pairs = []
        for jf in jsons_rem:
            stem = Path(jf).stem
            for src_dir in cache[src_prefix]:
                hit = None
                for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                    cand = src_dir / (stem + ext)
                    if cand.exists():
                        hit = str(cand)
                        break
                if hit:
                    folder_pairs.append((hit, jf))
                    break
        print(f"  {name}: 전체 {len(jsons_all)} − 픽 {min(len(jsons_all), cap) if cap else len(jsons_all)} → 잔여 {len(folder_pairs)}쌍", flush=True)
        pairs.extend(folder_pairs)

    return pairs


def process_remainder(pairs, output_dir: Path, split: str, imgsz: int, workers: int, start_idx: int):
    img_out = output_dir / split / "images"
    lbl_out = output_dir / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    # prefix 'r' 로 기존 {split}_NNNNNNN 과 네임스페이스 분리 (충돌 0)
    tasks = []
    for i, (img_path, json_path) in enumerate(pairs):
        stem = f"{split}_r{start_idx + i:07d}"
        tasks.append((img_path, json_path, str(img_out / (stem + ".jpg")), str(lbl_out / (stem + ".txt")), imgsz))

    print(f"\n[{split}] 잔여 {len(tasks)}장 변환 시작 (workers={workers}, prefix={split}_r)...", flush=True)
    stats = Counter()
    fails = empties = done = 0
    with Pool(processes=workers, initializer=_init_worker) as pool:
        for status, cnt in pool.imap_unordered(worker, tasks, chunksize=64):
            done += 1
            if status == "ok":
                stats.update(cnt)
            elif status == "empty":
                empties += 1
            else:
                fails += 1
            if done % 5000 == 0:
                print(f"    {done}/{len(tasks)} (실패 {fails} / 라벨없음 {empties})", flush=True)

    print(f"[{split}] 완료: {done}장 (실패 {fails}, 라벨없음 스킵 {empties})", flush=True)
    return stats


def count_existing(output_dir: Path, split: str) -> int:
    d = output_dir / split / "images"
    if not d.exists():
        return 0
    return sum(1 for _ in d.iterdir())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--output_dir", required=True, help="기존 v2 데이터셋 폴더 (여기에 append)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--cap", type=int, default=15000, help="convert_v2 에서 쓴 --max_per_folder 와 동일해야 함")
    ap.add_argument("--val_cap", type=int, default=2000, help="--include_val 시 convert_v2 의 --val_max_per_folder 와 동일")
    ap.add_argument("--include_val", action="store_true", help="val 잔여도 추가 (기본: 제외, 47k 유지)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry_run", action="store_true", help="변환 없이 폴더별 잔여 수만 출력")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    base = dataset_dir / "01-1.정식개방데이터"

    if not (output_dir / "train" / "images").exists():
        print(f"[ERROR] 기존 v2 폴더 없음: {output_dir}/train/images — output_dir 확인", flush=True)
        raise SystemExit(1)

    print("=== Training 잔여(여집합) 수집 ===", flush=True)
    train_rem = collect_remainder_pairs(base / "Training", args.cap)
    print(f"Train 잔여 합계: {len(train_rem):,}쌍", flush=True)

    val_rem = []
    if args.include_val:
        print("\n=== Validation 잔여(여집합) 수집 ===", flush=True)
        val_rem = collect_remainder_pairs(base / "Validation", args.val_cap)
        print(f"Val 잔여 합계: {len(val_rem):,}쌍", flush=True)

    existing_train = count_existing(output_dir, "train")
    existing_val = count_existing(output_dir, "val")
    print(f"\n기존 train {existing_train:,}장 + 잔여 {len(train_rem):,} = 최종 {existing_train + len(train_rem):,}장 예상", flush=True)
    if args.include_val:
        print(f"기존 val   {existing_val:,}장 + 잔여 {len(val_rem):,} = 최종 {existing_val + len(val_rem):,}장 예상", flush=True)

    if args.dry_run:
        print("\n[DRY-RUN] 변환 없이 종료. 수치 확인 후 --dry_run 빼고 재실행하세요.", flush=True)
        raise SystemExit(0)

    if not train_rem and not val_rem:
        print("\n[완료] 추가할 잔여 데이터 없음.", flush=True)
        raise SystemExit(0)

    # start_idx: 기존 파일과 prefix(_r)가 다르므로 0부터 시작해도 충돌 없음
    train_stats = process_remainder(train_rem, output_dir, "train", args.imgsz, args.workers, start_idx=0)
    val_stats = Counter()
    if args.include_val:
        val_stats = process_remainder(val_rem, output_dir, "val", args.imgsz, args.workers, start_idx=0)

    print("\n=== 추가된 잔여 라벨 (클래스별) ===", flush=True)
    for split, st in [("train", train_stats), ("val", val_stats)]:
        if not st:
            continue
        print(f"  [{split}] 총 {sum(st.values()):,}")
        for cid, name in enumerate(CLASS_NAMES):
            print(f"    {name:12s}: {st.get(cid, 0):,}")

    print(f"\n최종 train 파일 수: {count_existing(output_dir, 'train'):,}", flush=True)
