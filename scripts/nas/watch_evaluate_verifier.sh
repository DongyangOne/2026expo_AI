#!/bin/sh
# 검증기 학습 종료 후 고정 하드웨어 holdout에서 기존/후보 ONNX를 자동 비교한다.

set -eu

DOCKER=/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker
TRAIN_NAME=${1:?training container name is required}
IMAGE=${2:?evaluation image is required}
MANIFEST=${3:?hardware manifest is required}
BASELINE=${4:?baseline ONNX is required}
CANDIDATE_DIR=${5:?candidate output directory is required}
EVALUATOR=${6:?evaluator script is required}
OUTPUT=${7:?comparison output is required}
EVAL_NAME=${8:-evaluate-verifier-holdout}

container_path() {
  case "$1" in
    /share/Container/*) printf '/app/%s' "${1#/share/Container/}" ;;
    *) printf '%s' "$1" ;;
  esac
}

echo "[evaluate-watch] $(date '+%F %T') waiting for $TRAIN_NAME"
while [ "$($DOCKER inspect -f '{{.State.Running}}' "$TRAIN_NAME" 2>/dev/null)" = "true" ]; do
  sleep 60
done

EXIT_CODE=$($DOCKER inspect -f '{{.State.ExitCode}}' "$TRAIN_NAME" 2>/dev/null || echo 127)
if [ "$EXIT_CODE" != "0" ]; then
  echo "[evaluate-watch] training failed: exit=$EXIT_CODE"
  exit "$EXIT_CODE"
fi

CANDIDATE="$CANDIDATE_DIR/verifier.onnx"
for artifact in "$BASELINE" "$CANDIDATE" "$EVALUATOR" "$MANIFEST"; do
  if [ ! -s "$artifact" ]; then
    echo "[evaluate-watch] missing artifact: $artifact"
    exit 126
  fi
done

CONTAINER_MANIFEST=$(container_path "$MANIFEST")
CONTAINER_BASELINE=$(container_path "$BASELINE")
CONTAINER_CANDIDATE=$(container_path "$CANDIDATE")
CONTAINER_EVALUATOR=$(container_path "$EVALUATOR")
CONTAINER_OUTPUT=$(container_path "$OUTPUT")

$DOCKER rm -f "$EVAL_NAME" >/dev/null 2>&1 || true
echo "[evaluate-watch] $(date '+%F %T') evaluating hardware holdout"
$DOCKER run --rm --name "$EVAL_NAME" --cpus 4 --memory 4g \
  -v /share/Container:/app \
  "$IMAGE" \
  python "$CONTAINER_EVALUATOR" \
    --manifest "$CONTAINER_MANIFEST" \
    --model "baseline=$CONTAINER_BASELINE" \
    --model "candidate=$CONTAINER_CANDIDATE" \
    --split validation \
    --output "$CONTAINER_OUTPUT"
echo "[evaluate-watch] $(date '+%F %T') comparison=$OUTPUT"
