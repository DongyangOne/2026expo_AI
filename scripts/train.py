"""
YOLO26n 학습 스크립트
사용법: python scripts/train.py

학습 완료 후 weights/best.pt 에 모델 저장됨
"""

import os
import shutil
from pathlib import Path

from ultralytics import YOLO

DATASET_YAML  = "data/yolo/dataset.yaml"
WEIGHTS_DIR   = Path("weights")
RUNS_DIR      = Path("runs/train")
MODEL_BASE    = "yolo26n.pt"  # YOLO26n — 라즈베리파이 최적 (2026 최신)

# 9개 클래스
CLASS_NAMES = ["can", "pet", "paper", "plastic", "styrofoam", "vinyl", "glass", "battery", "fluorescent"]

# ─────────────────────────────────────────
# 학습 하이퍼파라미터
# RTX 2000 Ada 16GB → batch=32
# ─────────────────────────────────────────
TRAIN_CONFIG = {
    "data":        DATASET_YAML,
    "epochs":      100,
    "imgsz":       640,
    "batch":       32,
    "patience":    20,
    "optimizer":   "AdamW",
    "lr0":         0.001,
    "lrf":         0.01,
    "mosaic":      1.0,
    "mixup":       0.1,
    "flipud":      0.2,
    "fliplr":      0.5,
    "degrees":     10.0,
    "translate":   0.1,
    "scale":       0.5,
    "hsv_h":       0.015,
    "hsv_s":       0.7,
    "hsv_v":       0.4,
    "project":     str(RUNS_DIR),
    "name":        "waste_cls",
    "exist_ok":    True,
    "pretrained":  True,
    "verbose":     True,
    "device":      0,  # GPU 0번
}


def train():
    if not Path(DATASET_YAML).exists():
        print(f"[ERROR] dataset.yaml 없음: {DATASET_YAML}")
        print("먼저 실행: python scripts/convert_aihub_to_yolo.py")
        return

    WEIGHTS_DIR.mkdir(exist_ok=True)

    print("=" * 50)
    print("YOLO26n 학습 시작")
    print(f"  데이터셋: {DATASET_YAML}")
    print(f"  기본 모델: {MODEL_BASE}")
    print(f"  클래스 수: {len(CLASS_NAMES)}")
    print(f"  클래스: {CLASS_NAMES}")
    print(f"  에폭: {TRAIN_CONFIG['epochs']}")
    print("=" * 50)

    model = YOLO(MODEL_BASE)
    results = model.train(**TRAIN_CONFIG)

    # best.pt → weights/ 복사
    best_pt = Path(RUNS_DIR) / "waste_cls" / "weights" / "best.pt"
    if best_pt.exists():
        dest = WEIGHTS_DIR / "best.pt"
        shutil.copy2(best_pt, dest)
        print(f"\n모델 저장 완료: {dest}")
    else:
        print("[WARNING] best.pt를 찾을 수 없습니다.")

    # 성능 요약
    print("\n=== 학습 결과 ===")
    print(f"  mAP50:    {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    print(f"  Precision: {results.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
    print(f"  Recall:    {results.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")

    # 라즈베리파이용 NCNN 내보내기
    export_ncnn()


def export_ncnn():
    """라즈베리파이 ARM 최적화 NCNN 포맷으로 내보내기"""
    best_pt = WEIGHTS_DIR / "best.pt"
    if not best_pt.exists():
        return

    print("\nNCNN 내보내기 중...")
    model = YOLO(str(best_pt))
    model.export(
        format="ncnn",
        imgsz=640,
    )

    # ncnn 폴더 weights/ 로 이동
    ncnn_src = best_pt.parent / "best_ncnn_model"
    if ncnn_src.exists():
        ncnn_dest = WEIGHTS_DIR / "best_ncnn_model"
        if ncnn_dest.exists():
            shutil.rmtree(ncnn_dest)
        shutil.move(str(ncnn_src), ncnn_dest)
        print(f"NCNN 저장 완료: {ncnn_dest}")


if __name__ == "__main__":
    train()
