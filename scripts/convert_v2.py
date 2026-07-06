"""
AI Hub JSON 라벨 → YOLO 변환 (v2 — 버그수정 + 속도최적화)

v1 대비 개선:
  1. POLYGON 라벨 복원: 꼭짓점 min/max → bbox. (v1은 len!=4 조건으로 폴리곤 20~50%를 통째로 누락)
  2. 이미지 리사이즈 저장: 긴 변 IMGSZ(기본 640) → 4MB가 ~50KB로. 학습 I/O 수십 배 감소 (속도의 핵심)
  3. 멀티프로세싱: 모든 코어 활용 (변환 자체도 빠르게)
  4. 폴더별 상한 샘플링(--max_per_folder): 균등 stride로 솎아냄 → 클래스 균형 + 추가 속도
  5. AI Hub Training/Validation 분리 유지, 직접촬영 _1/_2/_3 분할 폴더 처리

실행 (NAS Docker, 로컬 디스크라 빠름):
  docker run --rm -v /share/Container:/app ultralytics/ultralytics:latest \
    python /app/convert_v2.py \
      --dataset_dir /app/ai_dataset/학습용_데이터 \
      --output_dir  /app/yolo_dataset_9class_v2 \
      --imgsz 640 --max_per_folder 15000 --val_max_per_folder 2000 --workers 16
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
# 9클래스 매핑
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
# bbox 추출 (BOX + POLYGON 모두)
# ─────────────────────────────────────────
def points_to_bbox(points):
    """
    POINTS → [x, y, w, h] (픽셀).
      BOX     : [[x, y, w, h]]           → 그대로
      POLYGON : [[x1,y1],[x2,y2],...]    → 꼭짓점 min/max로 외접 박스
    실패 시 None.
    """
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
    """실제 이미지 크기(w,h)로 정규화. JSON 치수 오류 방어."""
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
# 워커 (이미지 1장 처리)
# ─────────────────────────────────────────
def _init_worker():
    # 프로세스 병렬이므로 OpenCV 내부 스레딩은 끔 (16워커 × N스레드 경합 방지)
    cv2.setNumThreads(1)


def _imread_unicode(path: str):
    """한글 경로 안전 읽기."""
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
        # 이 데이터셋은 모든 이미지에 객체가 있음 → 라벨 0개 = 파싱 실패.
        # 저장하면 "객체 있는데 빈 배경"으로 학습되므로 스킵.
        return ("empty", None)

    # 긴 변을 imgsz로 축소 (확대는 안 함)
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
# 페어 수집 (폴더별 상한 샘플링 포함)
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


def _stride_sample(items, cap):
    """균등 stride 샘플링 (앞쪽 편향 방지). cap<=0 이면 전체."""
    if cap and len(items) > cap:
        step = len(items) / cap
        return [items[int(i * step)] for i in range(cap)]
    return items


def collect_pairs(split_dir: Path, max_per_folder: int) -> list:
    src_base = split_dir / "01.원천데이터"
    label_base = split_dir / "02.라벨링데이터"
    src_map = build_src_dir_map(src_base)
    cache = {}

    pairs = []
    if not label_base.exists():
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

        jsons = sorted(str(p) for p in label_dir.rglob("*.json"))
        jsons = _stride_sample(jsons, max_per_folder)

        folder_pairs = []
        for jf in jsons:
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
        print(f"  {name}: {len(folder_pairs)}쌍 (라벨 {len(jsons)})", flush=True)
        pairs.extend(folder_pairs)

    return pairs


def process_split(pairs, output_dir: Path, split: str, imgsz: int, workers: int):
    img_out = output_dir / split / "images"
    lbl_out = output_dir / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    tasks = []
    for i, (img_path, json_path) in enumerate(pairs):
        stem = f"{split}_{i:07d}"
        tasks.append((img_path, json_path, str(img_out / (stem + ".jpg")), str(lbl_out / (stem + ".txt")), imgsz))

    print(f"\n[{split}] {len(tasks)}장 변환 시작 (workers={workers})...", flush=True)
    stats = Counter()
    fails = 0
    empties = 0
    done = 0
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


def write_yaml(output_dir: Path):
    import yaml
    content = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": 9,
        "names": CLASS_NAMES,
    }
    with open(output_dir / "dataset.yaml", "w") as f:
        yaml.dump(content, f, default_flow_style=False, allow_unicode=True)
    print(f"\ndataset.yaml 생성: {output_dir / 'dataset.yaml'}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--output_dir", default="/app/yolo_dataset_9class_v2")
    ap.add_argument("--imgsz", type=int, default=640, help="긴 변 리사이즈 목표")
    ap.add_argument("--max_per_folder", type=int, default=0, help="train 폴더별 최대 장수 (0=전체)")
    ap.add_argument("--val_max_per_folder", type=int, default=2000, help="val 폴더별 최대 장수")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    base = dataset_dir / "01-1.정식개방데이터"

    # 기존 출력과 섞이면 이미지-라벨 불일치 위험 → 사전 경고
    for split in ("train", "val"):
        d = output_dir / split / "images"
        if d.exists() and any(d.iterdir()):
            print(f"[WARN] {d} 에 기존 파일 존재 — 새 출력 폴더 사용을 권장합니다.", flush=True)

    print("=== Training 페어 수집 ===", flush=True)
    train_pairs = collect_pairs(base / "Training", args.max_per_folder)
    print(f"Train 합계: {len(train_pairs)}쌍", flush=True)

    print("\n=== Validation 페어 수집 ===", flush=True)
    val_pairs = collect_pairs(base / "Validation", args.val_max_per_folder)
    print(f"Val 합계: {len(val_pairs)}쌍", flush=True)

    if not train_pairs:
        print("[ERROR] Train 페어 없음 — 경로 확인", flush=True)
        raise SystemExit(1)

    train_stats = process_split(train_pairs, output_dir, "train", args.imgsz, args.workers)
    val_stats = process_split(val_pairs, output_dir, "val", args.imgsz, args.workers)
    write_yaml(output_dir)

    print("\n=== 변환 완료 (클래스별 라벨 수) ===", flush=True)
    for split, st in [("train", train_stats), ("val", val_stats)]:
        print(f"  [{split}] 총 {sum(st.values()):,}")
        for cid, name in enumerate(CLASS_NAMES):
            print(f"    {name:12s}: {st.get(cid, 0):,}")
