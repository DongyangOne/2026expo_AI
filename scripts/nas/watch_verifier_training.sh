#!/bin/sh
# QNAP verifier 학습이 끝날 때까지 감시하고 NVIDIA 모듈을 정리한 뒤 Ollama를 복구한다.

set -u

DOCKER=/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker
KERNEL_DIR=/share/CACHEDEV1_DATA/.qpkg/NvKernelDriver/kernel-open
TRAIN_NAME=${1:?training container name is required}
OLLAMA_NAMES=${2:-naco-ollama}
OUTPUT_DIR=${3:-}

echo "[watch] $(date '+%F %T') waiting for $TRAIN_NAME"
while [ "$($DOCKER inspect -f '{{.State.Running}}' "$TRAIN_NAME" 2>/dev/null)" = "true" ]; do
  sleep 60
done

EXIT_CODE=$($DOCKER inspect -f '{{.State.ExitCode}}' "$TRAIN_NAME" 2>/dev/null || echo 127)
echo "[watch] $(date '+%F %T') training exit=$EXIT_CODE"
if [ "$EXIT_CODE" = "0" ] && [ -n "$OUTPUT_DIR" ]; then
  for artifact in best_verifier.pt verifier.onnx verifier_metadata.json; do
    if [ ! -s "$OUTPUT_DIR/$artifact" ]; then
      echo "[watch] missing artifact: $OUTPUT_DIR/$artifact"
      EXIT_CODE=126
    fi
  done
fi

echo "[watch] restoring QNAP GPU for $OLLAMA_NAMES"
/sbin/gpuhal_app -r f >/dev/null 2>&1 || true
i=0
while ps | grep -q '[n]vidia-smi'; do
  i=$((i + 1))
  [ "$i" -ge 45 ] && break
  sleep 1
done

for module in nvidia_uvm nvidia_drm nvidia_modeset nvidia; do
  i=0
  while grep -q "^${module} " /proc/modules; do
    rmmod "$module" >/dev/null 2>&1 || true
    i=$((i + 1))
    [ "$i" -ge 30 ] && break
    sleep 1
  done
done

if ! grep -q '^nvidia ' /proc/modules; then
  insmod "$KERNEL_DIR/nvidia.ko" || true
  insmod "$KERNEL_DIR/nvidia-modeset.ko" || true
  insmod "$KERNEL_DIR/nvidia-drm.ko" || true
  insmod "$KERNEL_DIR/nvidia-uvm.ko" || true
fi
/sbin/gpuhal_app -e f 4 0 0 >/dev/null 2>&1 || true
sleep 8
i=0
while ps | grep -q '[n]vidia-smi'; do
  i=$((i + 1))
  [ "$i" -ge 45 ] && break
  sleep 1
done

OLD_IFS=$IFS
IFS=,
for OLLAMA_NAME in $OLLAMA_NAMES; do
  $DOCKER start "$OLLAMA_NAME" >/dev/null 2>&1 || true
done
IFS=$OLD_IFS
sleep 15
IFS=,
for OLLAMA_NAME in $OLLAMA_NAMES; do
  $DOCKER ps -a --filter "name=$OLLAMA_NAME" --format '[watch] ollama={{.Names}} {{.Status}}'
done
IFS=$OLD_IFS
echo "[watch] $(date '+%F %T') done training_exit=$EXIT_CODE"
exit "$EXIT_CODE"
