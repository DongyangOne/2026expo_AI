"""
상태 분류기용 crop 추출 (페트병 라벨+찌그러짐 / 캔 찌그러짐)

원본 JSON bbox로 객체를 잘라 224px letterbox 저장하고, DAMAGE→dent / DIRTINESS→label
라벨을 manifest.csv 한 줄로 기록한다. 멀티헤드 분류기(공유 백본 + dent/label 헤드) 학습 입력.

- 소스: 직접촬영(키오스크 환경 유사). 영상추출은 컨베이어라 일단 제외.
- dent  : 원형=0 / 찌그러짐·완전압착=1. (손상/파손 등은 페트·캔에 거의 없고 모호 → 스킵)
- label : (페트병만) 오염없음=0 / 이물질(외부)=1. 내부·전체는 -1(라벨헤드 학습 제외, dent는 유효)
- 640px 변환본이 아닌 원본을 쓰는 이유: 변환본은 파일명이 리네임되어 JSON 속성과 매칭 불가 + crop 화질.

실행 (NAS Docker):
  docker run -d --name extract_crops -v /share/Container:/app ultralytics/ultralytics:latest \
    python /app/extract_crops.py \
      --dataset_dir /app/ai_dataset/학습용_데이터 \
      --output_dir  /app/crops_state_v1 \
      --workers 12
"""

import argparse
import csv
import json
import re
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

# 라벨 폴더(TL_) → category. 원천(TS_)은 prefix 치환 + _\d+ 분할 폴더로 탐색.
TARGETS = {
    "TL_2.직접촬영_03.페트병_001.무색단일": "pet",
    "TL_2.직접촬영_03.페트병_002.유색단일": "pet",
    "TL_2.직접촬영_01.금속캔_001.철캔": "can",
    "TL_2.직접촬영_01.금속캔_002.알루미늄캔": "can",
    "TL_2.직접촬영_04.플라스틱_001.PE": "plastic",
    "TL_2.직접촬영_04.플라스틱_002.PP": "plastic",
    "TL_2.직접촬영_04.플라스틱_003.PS": "plastic",
}
# 헤드별 학습 대상: dent=찌그러짐(페트·캔), label=라벨떼기(페트·플라스틱)
DENT_CATS = {"pet", "can"}
LABEL_CATS = {"pet", "plastic"}
DENT_MAP = {"원형": 0, "찌그러짐": 1, "완전압착": 1}
LABEL_MAP = {"오염없음": 0, "이물질(외부)": 1}  # 그 외(내부/전체) → -1
PAD = 0.12   # bbox 주변 맥락 패딩
SIZE = 224


def points_to_bbox(points):
    """POINTS → [x,y,w,h]. BOX는 그대로, POLYGON은 외접 박스. (convert_v2와 동일)"""
    if not points:
        return None
    p0 = points[0]
    if not isinstance(p0, (list, tuple)):
        return None
    if len(p0) == 4:
        return [float(v) for v in p0]
    if len(p0) == 2:
        xs = [float(p[0]) for p in points if len(p) >= 2]
        ys = [float(p[1]) for p in points if len(p) >= 2]
        if not xs or not ys:
            return None
        x, y = min(xs), min(ys)
        return [x, y, max(xs) - x, max(ys) - y]
    return None


def imread_unicode(path):
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def letterbox(img, size=SIZE):
    """비율 유지 resize + 회색(114) 패딩으로 정사각 224. 왜곡 없이 분류기 입력."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, int(h * s)), max(1, int(w * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = r
    return canvas


def _init_worker():
    cv2.setNumThreads(1)


def worker(task):
    img_path, json_path, cat, out_path = task
    img = imread_unicode(img_path)
    if img is None:
        return None
    H, W = img.shape[:2]
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        ann = data["ANNOTATION_INFO"][0]   # 직접촬영은 객체 1개
    except Exception:
        return None

    bbox = points_to_bbox(ann.get("POINTS", []))
    if bbox is None:
        return None
    # 헤드별 라벨: 대상 카테고리만 유효값, 그 외/모호값은 -1(학습 마스킹)
    dent_raw = DENT_MAP.get(ann.get("DAMAGE", ""))
    label_raw = LABEL_MAP.get(ann.get("DIRTINESS", ""), -1)
    dent = dent_raw if (cat in DENT_CATS and dent_raw is not None) else -1
    label = label_raw if cat in LABEL_CATS else -1
    if dent == -1 and label == -1:   # 두 헤드 모두 학습 라벨 없음 → 스킵
        return None

    x, y, w, h = bbox
    x -= w * PAD
    y -= h * PAD
    w *= (1 + 2 * PAD)
    h *= (1 + 2 * PAD)
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(x + w)), min(H, int(y + h))
    if x1 <= x0 or y1 <= y0:
        return None

    crop = letterbox(img[y0:y1, x0:x1])
    ok, enc = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return None
    enc.tofile(out_path)
    return (cat, dent, label, Path(out_path).name)


def collect(train_dir: Path, output_dir: Path, active=None):
    src_base = train_dir / "01.원천데이터"
    label_base = train_dir / "02.라벨링데이터"

    src_map = {}
    for d in src_base.iterdir():
        if d.is_dir():
            base = re.sub(r"_\d+$", "", d.name)
            src_map.setdefault(base, []).append(d)

    tasks, counters = [], Counter()
    for tl_name, cat in TARGETS.items():
        if active and cat not in active:
            continue
        label_dir = label_base / tl_name
        if not label_dir.exists():
            print(f"[WARN] 라벨 폴더 없음: {tl_name}", flush=True)
            continue
        src_prefix = "TS_" + tl_name[3:]
        src_dirs = src_map.get(src_prefix, [])
        if not src_dirs:
            print(f"[WARN] 원천 폴더 없음: {src_prefix}", flush=True)
            continue

        n = 0
        for jf in sorted(label_dir.rglob("*.json")):
            stem = jf.stem
            hit = None
            for sd in src_dirs:
                for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                    cand = sd / (stem + ext)
                    if cand.exists():
                        hit = str(cand)
                        break
                if hit:
                    break
            if hit:
                idx = counters[cat]
                counters[cat] += 1
                out = output_dir / cat / f"{cat}_{idx:07d}.jpg"
                tasks.append((hit, str(jf), cat, str(out)))
                n += 1
        print(f"  {tl_name}: {n}쌍", flush=True)
    return tasks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--output_dir", default="/app/crops_state_v1")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cats", default="", help="처리할 category만 (쉼표구분, 예: plastic). 빈값=전체")
    ap.add_argument("--append", action="store_true", help="manifest 이어쓰기 (기존 페트/캔 유지)")
    args = ap.parse_args()

    train_dir = Path(args.dataset_dir) / "01-1.정식개방데이터" / "Training"
    output_dir = Path(args.output_dir)
    active = {c.strip() for c in args.cats.split(",") if c.strip()} or None
    for cat in set(TARGETS.values()):
        if active and cat not in active:
            continue
        (output_dir / cat).mkdir(parents=True, exist_ok=True)

    print(f"=== crop 페어 수집 (cats={active or '전체'}) ===", flush=True)
    tasks = collect(train_dir, output_dir, active)
    print(f"합계: {len(tasks)} crops 예정", flush=True)
    if not tasks:
        raise SystemExit("[ERROR] 페어 없음 — 경로 확인")

    rows, stats = [], Counter()
    done = 0
    with Pool(args.workers, initializer=_init_worker) as pool:
        for r in pool.imap_unordered(worker, tasks, chunksize=64):
            done += 1
            if done % 10000 == 0:
                print(f"    {done}/{len(tasks)}", flush=True)
            if r is None:
                stats["skipped"] += 1
                continue
            cat, dent, label, name = r
            rows.append((f"{cat}/{name}", cat, dent, label))
            if dent != -1:
                stats[f"{cat}_dent={dent}"] += 1
            if label != -1:
                stats[f"{cat}_label={label}"] += 1

    mpath = output_dir / "manifest.csv"
    need_header = not (args.append and mpath.exists() and mpath.stat().st_size > 0)
    with open(mpath, "a" if args.append else "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(["filepath", "category", "dent", "label"])
        w.writerows(rows)

    print(f"\n=== 완료: {len(rows)} crops (스킵 {stats['skipped']}) ===", flush=True)
    for k in sorted(stats):
        if k != "skipped":
            print(f"  {k:20s}: {stats[k]:,}", flush=True)
    print(f"\nmanifest: {output_dir / 'manifest.csv'}", flush=True)
