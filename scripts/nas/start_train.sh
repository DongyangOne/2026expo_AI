#!/bin/bash
# 학습 컨테이너 기동 (root 전제 — train_watch.sh가 sudo로 호출).
# GPU 연결: QNAP NVIDIA QPKG 라이브러리 직접 마운트 + 디바이스 노드 + privileged.
D=/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker

echo "[start_train] $(date '+%F %T') 메모리 정리 (drop_caches + compact)"
sync
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || echo "[start_train] drop_caches 실패(권한?)"
[ -f /proc/sys/vm/compact_memory ] && echo 1 > /proc/sys/vm/compact_memory 2>/dev/null

echo "[start_train] 기존 train_yolo 제거(있으면)"
$D rm -f train_yolo 2>/dev/null

echo "[start_train] 학습 컨테이너 기동"
$D run -d \
  --name train_yolo \
  --privileged \
  --ipc=host \
  --restart=no \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e LD_LIBRARY_PATH=/qnap/nvidia/lib:/qnap/cuda/lib64 \
  -v /share/CACHEDEV1_DATA/.qpkg/NVIDIA_GPU_DRV/usr/lib:/qnap/nvidia/lib:ro \
  -v /share/CACHEDEV1_DATA/.qpkg/NVIDIA_GPU_DRV/cuda-12.9/lib64:/qnap/cuda/lib64:ro \
  --device /dev/nvidia0 --device /dev/nvidiactl \
  --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  --device /dev/nvidia-modeset \
  --device /dev/nvidia-caps/nvidia-cap1 \
  --device /dev/nvidia-caps/nvidia-cap2 \
  -v /share/Container:/app \
  ultralytics/ultralytics:latest \
  bash /app/train_cmd.sh

sleep 5
$D ps --filter name=train_yolo --format '[start_train] 상태: {{.Names}} {{.Status}}'
