#!/bin/sh
set -eu

# Fine-tune the rejected proposal/background verifier with clean runtime-top1
# crops and hardware-only hard samples.  The fixed hardware holdout is used
# only after training.  This pipeline never replaces a production model.

DOCKER_BIN=/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker
ROOT=/share/Container
CODE=$ROOT/expo_ai_proposal_hwadapt_v2_20260826
BASE_RUN=$ROOT/runs/proposal_verifier_bg_v1_20260825
MANIFESTS=$ROOT/proposal_verifier_bg_hwadapt_v2_20260826_manifests
HARDWARE=$ROOT/hardware_proposal_runtime_clean_20260826
HARDWARE_GT=$ROOT/hardware_capture_prep_20260803/dataset_v2
CONTROL=$ROOT/proposal_verifier_bg_hwadapt_v2_20260826_control
RUN=$ROOT/runs/proposal_verifier_bg_hwadapt_v2_20260826
TRAIN_IMAGE=expo-verifier-train:20260731
TRAIN_NAME=train_proposal_verifier_bg_hwadapt_v2_20260826
EVAL_NAME=evaluate_proposal_verifier_bg_hwadapt_v2_20260826
LOG=$CONTROL/pipeline.log

mkdir -p "$CONTROL"
exec >>"$LOG" 2>&1

date
printf '%s\n' "clean hardware-adaptation pipeline started"

if [ -e "$CONTROL/offline_candidate_ready.txt" ] || \
   [ -e "$CONTROL/rejected.txt" ] || \
   [ -e "$CONTROL/failed.txt" ]; then
  printf '%s\n' "terminal marker already exists; refusing to restart"
  exit 2
fi
if [ -d "$RUN" ] && [ "$(find "$RUN" -mindepth 1 -maxdepth 1 2>/dev/null | head -1)" ]; then
  printf '%s\n' "training output is nonempty: $RUN"
  exit 2
fi
if $DOCKER_BIN inspect "$TRAIN_NAME" >/dev/null 2>&1 || \
   $DOCKER_BIN inspect "$EVAL_NAME" >/dev/null 2>&1; then
  printf '%s\n' "pipeline container name already exists"
  exit 2
fi

for artifact in \
  "$CODE/scripts/train_verifier.py" \
  "$CODE/scripts/evaluate_hybrid_policy.py" \
  "$BASE_RUN/best_verifier.pt" \
  "$MANIFESTS/base_clean_runtime_top1_aihub_only.csv" \
  "$HARDWARE/hardware_background_internal_split.csv" \
  "$HARDWARE/hardware_positive_training.csv" \
  "$HARDWARE/hardware_styrofoam_training.csv" \
  "$HARDWARE/hardware_background_training_internal.csv" \
  "$HARDWARE_GT/verifier/hardware_material_single_training_only.csv" \
  "$CONTROL/baseline_hardware_ncnn.json"; do
  if [ ! -s "$artifact" ]; then
    printf '%s\n' "missing artifact: $artifact" > "$CONTROL/failed.txt"
    exit 126
  fi
done

# The gate evaluates the exported verifier with ONNX Runtime.  Check this
# before the multi-hour training run so a stale training image fails fast.
if ! $DOCKER_BIN run --rm "$TRAIN_IMAGE" \
  python3 -c 'import onnxruntime as ort; print(ort.__version__)'; then
  printf '%s\n' "training image missing onnxruntime; rebuild Dockerfile.training" > "$CONTROL/failed.txt"
  exit 124
fi

if ! $DOCKER_BIN run --rm --gpus all ultralytics/ultralytics:latest \
  python3 -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'; then
  printf '%s\n' "GPU preflight failed" > "$CONTROL/failed.txt"
  exit 125
fi

if ! $DOCKER_BIN run -d --name "$TRAIN_NAME" --gpus all --shm-size 8g \
  -v "$ROOT:/app" \
  "$TRAIN_IMAGE" \
  python3 /app/expo_ai_proposal_hwadapt_v2_20260826/scripts/train_verifier.py \
    --manifest /app/proposal_verifier_bg_hwadapt_v2_20260826_manifests/base_clean_runtime_top1_aihub_only.csv \
    --manifest /app/hardware_proposal_runtime_clean_20260826/hardware_background_internal_split.csv \
    --oversample-spec /app/hardware_proposal_runtime_clean_20260826/hardware_positive_training.csv=100 \
    --oversample-spec /app/hardware_proposal_runtime_clean_20260826/hardware_styrofoam_training.csv=150 \
    --oversample-spec /app/hardware_capture_prep_20260803/dataset_v2/verifier/hardware_material_single_training_only.csv=50 \
    --oversample-spec /app/hardware_proposal_runtime_clean_20260826/hardware_background_training_internal.csv=200 \
    --output-dir /app/runs/proposal_verifier_bg_hwadapt_v2_20260826 \
    --include-background \
    --init-checkpoint /app/runs/proposal_verifier_bg_v1_20260825/best_verifier.pt \
    --backbone mobilenet_v3_small --size 320 \
    --epochs 20 --patience 5 --batch 192 --workers 6 --lr 0.0001 \
    --backbone-lr 0.00003 --head-lr 0.0003 \
    --label-smoothing 0.05 \
    --class-weight-mode effective-number --class-weight-beta 0.999 \
    --camera-augmentation --material-weight 2.0 \
    --selection-material-weight 2.0 \
    --selection-material-target paper \
    --selection-material-target styrofoam \
    --selection-material-target plastic \
    --selection-material-target vinyl \
    --selection-material-target-weight 2.0; then
  printf '%s\n' "training container failed to start" > "$CONTROL/failed.txt"
  exit 125
fi

while [ "$($DOCKER_BIN inspect -f '{{.State.Running}}' "$TRAIN_NAME" 2>/dev/null || true)" = "true" ]; do
  date
  $DOCKER_BIN logs --tail 8 "$TRAIN_NAME" 2>&1 || true
  sleep 30
done

TRAIN_EXIT=$($DOCKER_BIN inspect -f '{{.State.ExitCode}}' "$TRAIN_NAME" 2>/dev/null || echo 127)
$DOCKER_BIN logs --tail 120 "$TRAIN_NAME" 2>&1 || true
if [ "$TRAIN_EXIT" != "0" ]; then
  printf '%s\n' "$TRAIN_NAME exit=$TRAIN_EXIT" > "$CONTROL/failed.txt"
  exit "$TRAIN_EXIT"
fi

for artifact in "$RUN/best_verifier.pt" "$RUN/verifier.onnx" "$RUN/verifier_metadata.json"; do
  if [ ! -s "$artifact" ]; then
    printf '%s\n' "missing trained artifact: $artifact" > "$CONTROL/failed.txt"
    exit 126
  fi
done

set +e
$DOCKER_BIN run --name "$EVAL_NAME" --cpus 4 --memory 4g \
  -v "$ROOT:/app" \
  "$TRAIN_IMAGE" \
  python3 /app/expo_ai_proposal_hwadapt_v2_20260826/scripts/evaluate_hybrid_policy.py \
    --baseline-report /app/proposal_verifier_bg_hwadapt_v2_20260826_control/baseline_hardware_ncnn.json \
    --baseline-threshold 0.25 \
    --raw-image-root /app/hardware_capture_prep_20260803/dataset_v2/yolo/images/val \
    --verifier-model /app/runs/proposal_verifier_bg_hwadapt_v2_20260826/verifier.onnx \
    --allow-background-veto \
    --confidence-thresholds 0.90 \
    --margin-thresholds 0.30 \
    --max-wrong-to-wrong 0 \
    --output /app/runs/proposal_verifier_bg_hwadapt_v2_20260826/hybrid_hardware_gate.json
EVAL_RUN_EXIT=$?
set -e

EVAL_EXIT=$($DOCKER_BIN inspect -f '{{.State.ExitCode}}' "$EVAL_NAME" 2>/dev/null || echo 127)
$DOCKER_BIN logs --tail 160 "$EVAL_NAME" 2>&1 || true
if [ "$EVAL_RUN_EXIT" != "0" ] || [ "$EVAL_EXIT" != "0" ]; then
  printf '%s\n' "$EVAL_NAME run_exit=$EVAL_RUN_EXIT container_exit=$EVAL_EXIT" > "$CONTROL/failed.txt"
  exit 125
fi

set +e
$DOCKER_BIN run --rm -v "$ROOT:/app" "$TRAIN_IMAGE" python3 -c \
  'import json, pathlib; p=pathlib.Path("/app/runs/proposal_verifier_bg_hwadapt_v2_20260826/hybrid_hardware_gate.json"); raise SystemExit(0 if json.loads(p.read_text(encoding="utf-8"))["deployment_gate"]["passed"] else 3)'
GATE_EXIT=$?
set -e
if [ "$GATE_EXIT" = "0" ]; then
  {
    printf 'onnx=%s\n' "$RUN/verifier.onnx"
    printf 'metadata=%s\n' "$RUN/verifier_metadata.json"
    printf 'gate_report=%s\n' "$RUN/hybrid_hardware_gate.json"
    printf '%s\n' "scope=offline gate only; runtime integration and independent end-to-end hardware validation required"
  } > "$CONTROL/offline_candidate_ready.txt"
  printf '%s\n' "offline hardware gate passed; production model unchanged"
elif [ "$GATE_EXIT" = "3" ]; then
  printf '%s\n' "$RUN/hybrid_hardware_gate.json" > "$CONTROL/rejected.txt"
  printf '%s\n' "hardware gate rejected candidate; production model unchanged"
else
  printf '%s\n' "gate report validation failed exit=$GATE_EXIT" > "$CONTROL/failed.txt"
  exit 125
fi
date
