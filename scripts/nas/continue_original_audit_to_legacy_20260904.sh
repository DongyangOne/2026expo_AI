#!/bin/sh
# Queue exactly one protected-legacy source audit after the pinned original job.
# Metadata cohort planning may run beside it; neither stage trains or deploys.
set -eu
umask 077
ROOT=/share/Container
JOB=$ROOT/operational_refresh_80bf78a_20260904_101000
CONTROL=$JOB/legacy_full_continuation_v1_20260904
CODE=$JOB/legacy_link_code_20260904
OUT=$ROOT/legacy_aihub_link_full_v1_20260904
DOCKER=/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker
PRODUCER=306f871abf56624d3076ed01373905972b6d7e99b30e82dea599e1a6a9b0c0c1
IMAGE=sha256:5e81d79be3ac54fb3dd1e8f65e44c4f2db0d154fb31674d8eef1fc729e3b6980
NAME=audit_legacy_aihub_link_full_v1_20260904
no_links() {
  case "$1" in "$ROOT"/*) ;; *) return 1;; esac
  current=$1
  while [ "$current" != "$ROOT" ]; do
    [ ! -L "$current" ] || return 1
    current=${current%/*}
  done
}
no_links "$CONTROL" && no_links "$CODE" && no_links "$OUT"
[ ! -e "$CONTROL" ] && [ ! -L "$CONTROL" ] && [ ! -e "$OUT" ] && [ ! -L "$OUT" ]
mkdir "$CONTROL"
terminal=0
on_exit() {
  rc=$?
  if [ "$terminal" = 0 ] && [ "$rc" != 0 ]; then
    printf '%s\n' "bridge failed or observation interrupted: exit=$rc; inspect exact container before retry" > "$CONTROL/observation_error.txt"
  fi
}
trap on_exit 0
fail() { printf '%s\n' "$1" > "$CONTROL/failed.txt"; terminal=1; exit 1; }
pinned() { [ -f "$1" ] && no_links "$1" && [ "$(sha256sum "$1" | awk '{print $1}')" = "$2" ]; }
check_inputs() {
  pinned "$CODE/scripts/audit_legacy_aihub_links.py" 24a4fea8bb9c77d39ae5dd3d5de1ce243dbf33ec13e23c8a7eddaed0a17526f6 &&
  pinned "$CODE/scripts/convert_v2.py" 52dee1d692f979c352f32510a74a84ea6e8892c08a47e1d29b8922b31ff95c22 &&
  pinned "$CODE/scripts/convert_remainder.py" 46cd95812d9e3fea0938d57a7e3ba48479dc897ca144a6e536c7e3cab8ae0c01 &&
  pinned "$JOB/protected_fingerprints_v1_20260904/result/report.json" d33cb1105310bbc7da8dff3748d17350a9d5b67351e75d76a789521681c1aa41 &&
  pinned "$ROOT/crops_verifier_single_v3/manifest.csv" c42f6a31382da5e060bfc784f0460ccc37d6a8e198577db3ba00b821968bafe7 &&
  [ ! -e "$JOB/protected_fingerprints_v1_20260904/result/failed.json" ]
}
check_inputs || fail 'pinned code or inputs changed'
"$DOCKER" inspect -f '{{.Id}} {{.State.Status}} {{.State.ExitCode}} {{.State.OOMKilled}} {{.Image}}' "$PRODUCER" > "$CONTROL/producer_before.txt"
read -r before_id before_status before_exit before_oom before_image < "$CONTROL/producer_before.txt"
[ "$before_id" = "$PRODUCER" ] && [ "$before_image" = "$IMAGE" ] || fail 'original producer identity mismatch'
case "$before_status" in
  running) "$DOCKER" wait "$PRODUCER" > "$CONTROL/producer_wait.txt" ;;
  exited) ;;
  *) fail 'original producer is neither running nor exited' ;;
esac
"$DOCKER" inspect -f '{{.Id}} {{.State.Status}} {{.State.ExitCode}} {{.State.OOMKilled}} {{.Image}}' "$PRODUCER" > "$CONTROL/producer_after.txt"
read -r after_id after_status after_exit after_oom after_image < "$CONTROL/producer_after.txt"
[ "$after_id" = "$PRODUCER" ] && [ "$after_image" = "$IMAGE" ] &&
  [ "$after_status" = exited ] && [ "$after_exit" = 0 ] && [ "$after_oom" = false ] || fail 'original producer did not finish successfully'
no_links "$JOB/original_annotation_full_v2_20260904/result/report.json" &&
  [ -f "$JOB/original_annotation_full_v2_20260904/result/report.json" ] &&
  [ ! -e "$JOB/original_annotation_full_v2_20260904/result/failed.json" ] || fail 'original report missing or failed'
sha256sum "$JOB/original_annotation_full_v2_20260904/result/report.json" > "$CONTROL/original_report.sha256"
check_inputs || fail 'pinned inputs changed while waiting'
[ ! -e "$OUT" ] && [ ! -L "$OUT" ] || fail 'immutable output already exists'
mkdir "$OUT"
# UID 0 requires DAC_OVERRIDE to traverse the host user's 0700 output directory.
# All dataset/code mounts remain read-only. No Docker socket or GPU is exposed.
"$DOCKER" run -d --name "$NAME" --runtime runc --network none --read-only \
  --cpus 1 --memory 2g --memory-swap 2g --pids-limit 128 \
  --cap-drop ALL --cap-add DAC_OVERRIDE --security-opt no-new-privileges \
  --tmpfs /tmp:rw,nosuid,nodev,size=64m \
  -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -v "$ROOT:/app:ro" -v "$OUT:/app/legacy_aihub_link_full_v1_20260904:rw" \
  --entrypoint python3 "$IMAGE" \
  /app/operational_refresh_80bf78a_20260904_101000/legacy_link_code_20260904/scripts/audit_legacy_aihub_links.py \
  --protected-report /app/operational_refresh_80bf78a_20260904_101000/protected_fingerprints_v1_20260904/result/report.json \
  --protected-report-sha256 d33cb1105310bbc7da8dff3748d17350a9d5b67351e75d76a789521681c1aa41 \
  --v3-manifest /app/crops_verifier_single_v3/manifest.csv \
  --v3-manifest-sha256 c42f6a31382da5e060bfc784f0460ccc37d6a8e198577db3ba00b821968bafe7 \
  --converter /app/operational_refresh_80bf78a_20260904_101000/legacy_link_code_20260904/scripts/convert_v2.py \
  --converter-sha256 52dee1d692f979c352f32510a74a84ea6e8892c08a47e1d29b8922b31ff95c22 \
  --remainder /app/operational_refresh_80bf78a_20260904_101000/legacy_link_code_20260904/scripts/convert_remainder.py \
  --remainder-sha256 46cd95812d9e3fea0938d57a7e3ba48479dc897ca144a6e536c7e3cab8ae0c01 \
  --dataset-dir /app/ai_dataset/학습용_데이터 \
  --output /app/legacy_aihub_link_full_v1_20260904/result --max-per-kind 0 > "$CONTROL/audit_container_id.txt"
# Dispatch is not completion. The next monitor must inspect this exact ID/report.
audit_id=$(cat "$CONTROL/audit_container_id.txt")
[ "${#audit_id}" -eq 64 ] || exit 75
case "$audit_id" in *[!0-9a-f]*) exit 75;; esac
terminal=1
printf '%s\n' 'protected legacy audit dispatched; no training or deployment' > "$CONTROL/dispatched.txt"
