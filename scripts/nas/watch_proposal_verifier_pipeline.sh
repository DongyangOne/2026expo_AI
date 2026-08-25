#!/bin/sh
set -eu

# Build and train a 9-material + background proposal verifier on QNAP.
# This pipeline never changes the deployed Pi model.  It stops at a candidate
# artifact so production promotion remains gated by end-to-end hardware tests.

DOCKER_BIN=/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker
ROOT=/share/Container
SOURCE=$ROOT/expo_ai_proposal_20260825
WORK=$ROOT/proposal_verifier_bg_v1_20260825
CONTROL=$ROOT/proposal_verifier_bg_v1_20260825_control
RUN=$ROOT/runs/proposal_verifier_bg_v1_20260825
PREP_CONTAINER=prepare_proposal_verifier_bg_v1_20260825
TRAIN_CONTAINER=train_proposal_verifier_bg_v1_20260825
ULTRALYTICS_IMAGE=ultralytics/ultralytics:latest
TRAIN_IMAGE=expo-verifier-train:20260731
LOG=$CONTROL/pipeline.log

mkdir -p "$CONTROL"
exec >>"$LOG" 2>&1

date
printf '%s\n' "proposal verifier pipeline started"

if [ -e "$CONTROL/candidate_ready.txt" ] || [ -e "$CONTROL/failed.txt" ]; then
  printf '%s\n' "terminal marker already exists; refusing to restart"
  exit 2
fi
if [ -d "$RUN" ] && [ "$(find "$RUN" -mindepth 1 -maxdepth 1 2>/dev/null | head -1)" ]; then
  printf '%s\n' "training output is nonempty: $RUN"
  exit 2
fi

wait_container() {
  container_name=$1
  while [ "$($DOCKER_BIN inspect -f '{{.State.Running}}' "$container_name" 2>/dev/null || true)" = "true" ]; do
    date
    $DOCKER_BIN logs --tail 8 "$container_name" 2>&1 || true
    sleep 20
  done
  exit_code=$($DOCKER_BIN inspect -f '{{.State.ExitCode}}' "$container_name")
  $DOCKER_BIN logs --tail 80 "$container_name" 2>&1 || true
  if [ "$exit_code" != "0" ]; then
    printf '%s\n' "$container_name exit=$exit_code" > "$CONTROL/failed.txt"
    exit "$exit_code"
  fi
}

$DOCKER_BIN run --rm --gpus all "$ULTRALYTICS_IMAGE" \
  python3 -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

if [ ! -s "$WORK/manifest.csv" ]; then
  $DOCKER_BIN run -d --name "$PREP_CONTAINER" --gpus all \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "$ROOT:/app" \
    "$ULTRALYTICS_IMAGE" \
    python3 /app/expo_ai_proposal_20260825/scripts/prepare_proposal_verifier_dataset.py \
      --model /app/runs/trash_v2_full-2/weights/best.pt \
      --data /app/yolo_mixed_replay_v2_20260819/dataset_mixed.yaml \
      --dataset-dir /app/yolo_mixed_replay_v2_20260819 \
      --output-dir /app/proposal_verifier_bg_v1_20260825 \
      --device 0 --batch 12 --imgsz 640 --conf 0.05 \
      --positive-iou 0.50 --negative-iou 0.10 \
      --crop-size 320 --padding 0.08 \
      --max-per-class 10000 --val-max-per-class 2000 \
      --max-background 20000 --val-max-background 4000 \
      --seed 20260825 --min-free-gb 500 --max-output-gb 30
  wait_container "$PREP_CONTAINER"
fi

test -s "$WORK/manifest.csv"
test -s "$WORK/dataset_info.json"

$DOCKER_BIN run -d --name "$TRAIN_CONTAINER" --gpus all \
  -v "$ROOT:/app" \
  "$TRAIN_IMAGE" \
  python3 /app/expo_ai_proposal_20260825/scripts/train_verifier.py \
    --manifest /app/proposal_verifier_bg_v1_20260825/manifest.csv \
    --output-dir /app/runs/proposal_verifier_bg_v1_20260825 \
    --include-background \
    --init-checkpoint /app/runs/verifier_curated_v7_hard100_mnv3_20260803/best_verifier.pt \
    --backbone mobilenet_v3_small --size 320 \
    --epochs 50 --patience 10 --batch 192 --workers 6 --lr 0.0003 \
    --camera-augmentation --material-weight 2.0 \
    --selection-material-weight 2.0 \
    --selection-material-target paper \
    --selection-material-target styrofoam \
    --selection-material-target plastic \
    --selection-material-target vinyl \
    --selection-material-target background \
    --selection-material-target-weight 2.0
wait_container "$TRAIN_CONTAINER"

test -s "$RUN/best_verifier.pt"
test -s "$RUN/verifier.onnx"
test -s "$RUN/verifier_metadata.json"
printf '%s\n' "$RUN/verifier.onnx" > "$CONTROL/candidate_ready.txt"
date
printf '%s\n' "candidate ready; production model unchanged"
