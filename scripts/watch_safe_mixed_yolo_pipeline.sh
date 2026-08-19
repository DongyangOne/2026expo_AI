#!/bin/sh
# Safe two-stage YOLO adaptation.  This script never changes production weights.
set -eu

DOCKER=/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker
ROOT=/share/Container
WORK=$ROOT/yolo_mixed_replay_v2_20260819
METRICS=$WORK/metrics
RUNS=$ROOT/runs
BASE_MODEL=$RUNS/trash_v2_full-2/weights/best.pt
BASE_DATA=$ROOT/yolo_dataset_9class_v2/dataset.yaml
COMMERCIAL_LIST=$ROOT/yolo_commercial_single_v1_20260813/train_balanced.txt
HARDWARE_DATA=$ROOT/hardware_capture_prep_20260803/dataset_v2/yolo
STAGE_A_NAME=trash_mixed_replay_stage_a_20260819
STAGE_B_NAME=trash_mixed_replay_stage_b_20260819
STAGE_A_CONTAINER=train_yolo_mixed_stage_a_20260819
STAGE_B_CONTAINER=train_yolo_mixed_stage_b_20260819

gpu_args() {
  printf '%s\n' \
    --privileged --shm-size 8g \
    --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
    --device /dev/nvidia-uvm-tools --device /dev/nvidia-modeset \
    --device /dev/nvidia-caps/nvidia-cap1 --device /dev/nvidia-caps/nvidia-cap2
}

wait_container() {
  container=$1
  while [ "$($DOCKER inspect "$container" --format '{{.State.Running}}' 2>/dev/null || echo missing)" = "true" ]; do
    sleep 60
  done
  exit_code=$($DOCKER inspect "$container" --format '{{.State.ExitCode}}' 2>/dev/null || echo missing)
  if [ "$exit_code" != "0" ]; then
    echo "$container failed: exit=$exit_code"
    $DOCKER logs --tail 200 "$container" 2>&1 || true
    exit 12
  fi
}

run_validation() {
  model=$1
  output=$2
  name=$3
  [ -s "$output" ] && return 0
  if [ -e "$RUNS/$name" ]; then
    echo "validation run exists without report: $RUNS/$name"
    exit 13
  fi
  # shellcheck disable=SC2046
  $DOCKER run --rm $(gpu_args) \
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v /share/CACHEDEV1_DATA/.qpkg/NVIDIA_GPU_DRV/usr/lib:/qnap/nvidia/lib:ro \
    -v /share/CACHEDEV1_DATA/.qpkg/NVIDIA_GPU_DRV/cuda-12.9/lib64:/qnap/cuda/lib64:ro \
    -v "$ROOT":/app \
    ultralytics/ultralytics:latest bash -lc \
    "export LD_LIBRARY_PATH=/qnap/nvidia/lib:/qnap/cuda/lib64:\${LD_LIBRARY_PATH:-}; python3 /app/gpu_assert.py; python3 /app/expo_ai_commercial_20260813/scripts/evaluate_yolo_validation.py --model '$model' --data /app/yolo_dataset_9class_v2/dataset.yaml --output '$output' --project /app/runs --name '$name' --device 0 --batch 28 --workers 6 --imgsz 640"
}

run_hardware() {
  model=$1
  output=$2
  [ -s "$output" ] && return 0
  # shellcheck disable=SC2046
  $DOCKER run --rm $(gpu_args) \
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v /share/CACHEDEV1_DATA/.qpkg/NVIDIA_GPU_DRV/usr/lib:/qnap/nvidia/lib:ro \
    -v /share/CACHEDEV1_DATA/.qpkg/NVIDIA_GPU_DRV/cuda-12.9/lib64:/qnap/cuda/lib64:ro \
    -v "$ROOT":/app \
    ultralytics/ultralytics:latest bash -lc \
    "export LD_LIBRARY_PATH=/qnap/nvidia/lib:/qnap/cuda/lib64:\${LD_LIBRARY_PATH:-}; python3 /app/gpu_assert.py; python3 /app/expo_ai_commercial_20260813/scripts/evaluate_hardware_detector.py --model '$model' --dataset-dir /app/hardware_capture_prep_20260803/dataset_v2/yolo --output '$output' --thresholds 0.25 0.55 --device 0 --batch 28 --imgsz 640"
}

run_gate() {
  candidate_validation=$1
  candidate_hardware=$2
  output=$3
  $DOCKER run --rm -v "$ROOT":/app ultralytics/ultralytics:latest \
    python3 /app/expo_ai_commercial_20260813/scripts/check_yolo_candidate_gate.py \
      --baseline-validation /app/yolo_mixed_replay_v2_20260819/metrics/baseline_original.json \
      --candidate-validation "$candidate_validation" \
      --baseline-hardware /app/yolo_mixed_replay_v2_20260819/metrics/baseline_hardware.json \
      --candidate-hardware "$candidate_hardware" \
      --output "$output" \
      --max-original-drop 0.01 --min-hardware-gain 0.05
}

mkdir -p "$WORK"
for required in "$BASE_MODEL" "$BASE_DATA" "$COMMERCIAL_LIST" "$HARDWARE_DATA/dataset.yaml"; do
  if [ ! -e "$required" ]; then
    echo "required input missing: $required"
    exit 10
  fi
done

if [ ! -s "$WORK/mixed_replay_summary.json" ]; then
  echo "preparing deterministic mixed replay list"
  $DOCKER run --rm -v "$ROOT":/app ultralytics/ultralytics:latest \
    python3 /app/expo_ai_commercial_20260813/scripts/prepare_mixed_replay_yolo_list.py \
      --base-dataset-dir /app/yolo_dataset_9class_v2 \
      --commercial-list /app/yolo_commercial_single_v1_20260813/train_balanced.txt \
      --output-dir /app/yolo_mixed_replay_v2_20260819 \
      --validation-images /app/yolo_dataset_9class_v2/val/images \
      --target-per-class 20000 --rare-target-per-class 5000 \
      --trusted-negative-dir /app/hardware_capture_prep_20260803/dataset_v2/yolo \
      --trusted-negative-repeats 100 --seed 20260819
fi
mkdir -p "$METRICS"

echo "evaluating unchanged production baseline"
run_validation /app/runs/trash_v2_full-2/weights/best.pt \
  /app/yolo_mixed_replay_v2_20260819/metrics/baseline_original.json \
  eval_mixed_baseline_original_20260819
run_hardware /app/runs/trash_v2_full-2/weights/best.pt \
  /app/yolo_mixed_replay_v2_20260819/metrics/baseline_hardware.json

if [ ! -s "$RUNS/$STAGE_A_NAME/weights/best.pt" ]; then
  if $DOCKER inspect "$STAGE_A_CONTAINER" >/dev/null 2>&1 || [ -e "$RUNS/$STAGE_A_NAME" ]; then
    echo "stage A target already exists but has no best checkpoint"
    exit 14
  fi
  echo "starting stage A: frozen backbone/head adaptation"
  # shellcheck disable=SC2046
  $DOCKER run -d --name "$STAGE_A_CONTAINER" $(gpu_args) \
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v /share/CACHEDEV1_DATA/.qpkg/NVIDIA_GPU_DRV/usr/lib:/qnap/nvidia/lib:ro \
    -v /share/CACHEDEV1_DATA/.qpkg/NVIDIA_GPU_DRV/cuda-12.9/lib64:/qnap/cuda/lib64:ro \
    -v "$ROOT":/app \
    ultralytics/ultralytics:latest bash -lc \
    'export LD_LIBRARY_PATH=/qnap/nvidia/lib:/qnap/cuda/lib64:${LD_LIBRARY_PATH:-}; python3 /app/gpu_assert.py; exec yolo detect train model=/app/runs/trash_v2_full-2/weights/best.pt data=/app/yolo_mixed_replay_v2_20260819/dataset_mixed.yaml epochs=5 patience=5 imgsz=640 batch=28 device=0 workers=6 cache=False project=/app/runs name=trash_mixed_replay_stage_a_20260819 exist_ok=False pretrained=True freeze=20 optimizer=AdamW lr0=0.0001 lrf=0.2 weight_decay=0.0005 warmup_epochs=0.5 warmup_bias_lr=0.01 cos_lr=True amp=True seed=20260819 deterministic=True mosaic=0.0 mixup=0.0 copy_paste=0.0 close_mosaic=0 degrees=4.0 translate=0.05 scale=0.25 shear=0.0 perspective=0.0 flipud=0.0 fliplr=0.5 hsv_h=0.01 hsv_s=0.30 hsv_v=0.20 save_period=1'
  wait_container "$STAGE_A_CONTAINER"
fi

run_validation /app/runs/$STAGE_A_NAME/weights/best.pt \
  /app/yolo_mixed_replay_v2_20260819/metrics/stage_a_original.json \
  eval_mixed_stage_a_original_20260819
run_hardware /app/runs/$STAGE_A_NAME/weights/best.pt \
  /app/yolo_mixed_replay_v2_20260819/metrics/stage_a_hardware.json

if run_gate \
  /app/yolo_mixed_replay_v2_20260819/metrics/stage_a_original.json \
  /app/yolo_mixed_replay_v2_20260819/metrics/stage_a_hardware.json \
  /app/yolo_mixed_replay_v2_20260819/metrics/stage_a_gate.json; then
  printf '%s\n' "/app/runs/$STAGE_A_NAME/weights/best.pt" > "$WORK/selected_candidate.txt"
  echo "stage A passed every gate; candidate recorded without deployment"
  exit 0
fi

base_ok=$($DOCKER run --rm -v "$ROOT":/app ultralytics/ultralytics:latest python3 -c \
  'import json; d=json.load(open("/app/yolo_mixed_replay_v2_20260819/metrics/stage_a_gate.json")); print(int(d["checks"]["base_map50_95_preserved"] and d["checks"]["base_recall_preserved"]))')
if [ "$base_ok" != "1" ]; then
  echo "stage A failed original-distribution preservation; refusing further fine-tuning"
  printf '%s\n' "stage_a_base_regression" > "$WORK/rejected.txt"
  exit 20
fi

if [ ! -s "$RUNS/$STAGE_B_NAME/weights/best.pt" ]; then
  if $DOCKER inspect "$STAGE_B_CONTAINER" >/dev/null 2>&1 || [ -e "$RUNS/$STAGE_B_NAME" ]; then
    echo "stage B target already exists but has no best checkpoint"
    exit 15
  fi
  echo "stage A preserved baseline but missed hardware gate; starting low-LR stage B"
  # shellcheck disable=SC2046
  $DOCKER run -d --name "$STAGE_B_CONTAINER" $(gpu_args) \
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v /share/CACHEDEV1_DATA/.qpkg/NVIDIA_GPU_DRV/usr/lib:/qnap/nvidia/lib:ro \
    -v /share/CACHEDEV1_DATA/.qpkg/NVIDIA_GPU_DRV/cuda-12.9/lib64:/qnap/cuda/lib64:ro \
    -v "$ROOT":/app \
    ultralytics/ultralytics:latest bash -lc \
    'export LD_LIBRARY_PATH=/qnap/nvidia/lib:/qnap/cuda/lib64:${LD_LIBRARY_PATH:-}; python3 /app/gpu_assert.py; exec yolo detect train model=/app/runs/trash_mixed_replay_stage_a_20260819/weights/best.pt data=/app/yolo_mixed_replay_v2_20260819/dataset_mixed.yaml epochs=3 patience=3 imgsz=640 batch=28 device=0 workers=6 cache=False project=/app/runs name=trash_mixed_replay_stage_b_20260819 exist_ok=False pretrained=True freeze=10 optimizer=AdamW lr0=0.00003 lrf=0.2 weight_decay=0.0005 warmup_epochs=0.25 warmup_bias_lr=0.005 cos_lr=True amp=True seed=20260819 deterministic=True mosaic=0.0 mixup=0.0 copy_paste=0.0 close_mosaic=0 degrees=3.0 translate=0.04 scale=0.20 shear=0.0 perspective=0.0 flipud=0.0 fliplr=0.5 hsv_h=0.01 hsv_s=0.25 hsv_v=0.18 save_period=1'
  wait_container "$STAGE_B_CONTAINER"
fi

run_validation /app/runs/$STAGE_B_NAME/weights/best.pt \
  /app/yolo_mixed_replay_v2_20260819/metrics/stage_b_original.json \
  eval_mixed_stage_b_original_20260819
run_hardware /app/runs/$STAGE_B_NAME/weights/best.pt \
  /app/yolo_mixed_replay_v2_20260819/metrics/stage_b_hardware.json

if run_gate \
  /app/yolo_mixed_replay_v2_20260819/metrics/stage_b_original.json \
  /app/yolo_mixed_replay_v2_20260819/metrics/stage_b_hardware.json \
  /app/yolo_mixed_replay_v2_20260819/metrics/stage_b_gate.json; then
  printf '%s\n' "/app/runs/$STAGE_B_NAME/weights/best.pt" > "$WORK/selected_candidate.txt"
  echo "stage B passed every gate; candidate recorded without deployment"
  exit 0
fi

printf '%s\n' "no_candidate_passed_all_gates" > "$WORK/rejected.txt"
echo "no candidate passed every gate; production remains unchanged"
exit 21
