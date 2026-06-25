#!/bin/bash
# 학습 컨테이너 내부 진입점.
# QNAP 호스트 드라이버 라이브러리를 LD 경로에 주입(이게 없으면 torch가 GPU를 못 봄).
set -e
export LD_LIBRARY_PATH=/qnap/nvidia/lib:/qnap/cuda/lib64:$LD_LIBRARY_PATH

echo "[train_cmd] GPU 어설션 시작"
python3 /app/gpu_assert.py

echo "[train_cmd] 학습 시작: yolo26m, v2 full dataset (~804k train / 47.8k val)"
exec yolo train \
  model=/app/yolo26m.pt \
  data=/app/yolo_dataset_9class_v2/dataset.yaml \
  epochs=50 patience=10 batch=28 imgsz=640 \
  workers=8 cache=False \
  project=/app/runs name=trash_v2_full device=0
