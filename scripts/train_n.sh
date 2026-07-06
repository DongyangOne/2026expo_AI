#!/bin/bash
# YOLO26n 학습
# docker run 시 반드시:
#   -v /share/CACHEDEV1_DATA/.qpkg/NVIDIA_GPU_DRV/usr/lib:/opt/nvidia/lib
#   --shm-size=8g
#   --gpus all
export LD_LIBRARY_PATH=/opt/nvidia/lib:$LD_LIBRARY_PATH

echo "[train_n] GPU 확인"
python /app/gpu_check.py

echo "[train_n] YOLO26n 학습 시작 (v2 full, 640px)"
exec yolo train \
  model=yolo26n.pt \
  data=/app/yolo_dataset_9class_v2/dataset.yaml \
  epochs=100 patience=15 batch=64 imgsz=640 \
  workers=8 cache=False device=0 \
  project=/app/runs name=trash_n
