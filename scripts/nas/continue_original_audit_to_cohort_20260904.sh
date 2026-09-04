#!/bin/sh
# One-shot, metadata-only continuation. Never restarts the producer or trains/deploys.
set -eu
umask 077

ROOT=/share/Container
JOB=$ROOT/operational_refresh_80bf78a_20260904_101000
DOCKER=/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker
PRODUCER=audit_original_annotation_full_v2_20260904
ORIGINAL_REPORT=$JOB/original_annotation_full_v2_20260904/result/report.json
PROTECTED_REPORT=$JOB/protected_fingerprints_v1_20260904/result/report.json
PROTECTED_SHA=d33cb1105310bbc7da8dff3748d17350a9d5b67351e75d76a789521681c1aa41
SELECTED=$ROOT/yolo_commercial_single_v1_20260813/selected_manifest.csv
SELECTED_SHA=2f026c4d914ff3b5c6a8e3bf89280678a847e37a5f22611a481d792e8223012a
AUDITOR=$JOB/audit_aihub_original_annotations_v2_20260904.py
AUDITOR_SHA=d3d82f6ee009a397716da7abde6e9499d54d85febb55a7c1f23ba337f5d70f99

: "${CONTROL:?required}" "${CODE_ROOT:?required}" "${PLANNER_SHA256:?required}"
: "${ORIGINAL_HELPER_SHA256:?required}" "${IMAGE_ID:?required}" "${PLANNER_CONTAINER:?required}"

valid_sha() { [ "${#1}" -eq 64 ] && case "$1" in *[!0-9a-f]*) return 1;; *) return 0;; esac; }
no_links() {
    case "$1" in "$ROOT"/*) ;; *) return 1;; esac
    current=$1
    while [ "$current" != "$ROOT" ]; do
        [ ! -L "$current" ] || return 1
        current=${current%/*}
    done
}
scoped_path() {
    case "$1" in *[!A-Za-z0-9_./-]*|*/../*|*/./*|*//*) return 1;; esac
    case "$1" in "$JOB"/*) ;; "$ROOT"/*) rest=${1#"$ROOT"/}; case "$rest" in */*) return 1;; esac;; *) return 1;; esac
    case "$1" in */|*/..|*/.) return 1;; esac
    no_links "$1"
}
digest() { sha256sum "$1" | awk '{print $1}'; }
pinned() { [ -f "$1" ] && no_links "$1" && [ "$(digest "$1")" = "$2" ]; }
scoped_path "$CONTROL" && scoped_path "$CODE_ROOT" || exit 64
case "$CONTROL/" in "$CODE_ROOT/"*) exit 64;; esac
case "$CODE_ROOT/" in "$CONTROL/"*) exit 64;; esac
valid_sha "$PLANNER_SHA256" && valid_sha "$ORIGINAL_HELPER_SHA256" || exit 64
case "$IMAGE_ID" in sha256:*) valid_sha "${IMAGE_ID#sha256:}" || exit 64;; *) exit 64;; esac
case "$PLANNER_CONTAINER" in cohort_original_20260904_*) ;; *) exit 64;; esac
case "$PLANNER_CONTAINER" in *[!A-Za-z0-9_.-]*) exit 64;; esac
[ -x "$DOCKER" ] || exit 64
# The directory itself is the exclusive lock. It is never removed/reused.
mkdir "$CONTROL" || exit 73
terminal=0
fail() { printf '%s\n' "$1" > "$CONTROL/failed.txt"; terminal=1; exit 1; }
observation_error() {
    printf '%s\n' "$1" > "$CONTROL/observation_error.txt"
    terminal=1
    exit 75
}
on_exit() {
    rc=$?
    if [ "$terminal" -eq 0 ]; then
        printf '%s\n' "unexpected bridge exit: $rc; no training/deployment authority" > "$CONTROL/failed.txt"
    fi
}
trap on_exit 0
# No external timeout applet is available on this NAS. Hold the direct child at
# a start gate until its Linux process identity is captured, then exec Docker.
# The watchdog only signals that exact child PID/starttime, never a process group.
process_start() {
    [ -r "/proc/$1/stat" ] || return 1
    awk '{ sub(/^.*\) /, ""); print $20 }' "/proc/$1/stat" 2>/dev/null
}
bounded_command() {
    bound_seconds=$1
    shift
    bound_dir=$(mktemp -d "$CONTROL/.bounded.XXXXXX") || return 125
    (
        trap - 0
        gate_wait=0
        while [ ! -f "$bound_dir/start" ]; do
            gate_wait=$((gate_wait + 1))
            [ "$gate_wait" -le "$bound_seconds" ] || exit 125
            sleep 1
        done
        [ "$(cat "$bound_dir/start")" = run ] || exit 125
        exec "$@"
    ) &
    bound_pid=$!
    bound_start=$(process_start "$bound_pid") || bound_start=
    case "$bound_start" in
        ''|*[!0-9]*)
            printf '%s\n' abort > "$bound_dir/start"
            wait "$bound_pid" || true
            return 125 ;;
    esac
    printf '%s %s\n' "$bound_pid" "$bound_start" > "$bound_dir/identity"
    (
        trap - 0
        sleep "$bound_seconds"
        [ ! -f "$bound_dir/done" ] || exit 0
        [ "$(process_start "$bound_pid" || true)" = "$bound_start" ] || exit 0
        : > "$bound_dir/timed_out"
        kill -TERM "$bound_pid" 2>/dev/null || true
        sleep 2
        [ ! -f "$bound_dir/done" ] || exit 0
        [ "$(process_start "$bound_pid" || true)" = "$bound_start" ] || exit 0
        kill -KILL "$bound_pid" 2>/dev/null || true
    ) </dev/null >/dev/null 2>&1 &
    printf '%s\n' run > "$bound_dir/start"
    bound_rc=0
    wait "$bound_pid" || bound_rc=$?
    : > "$bound_dir/done"
    [ ! -f "$bound_dir/timed_out" ] || bound_rc=124
    return "$bound_rc"
}
PLANNER=$CODE_ROOT/scripts/plan_aihub_original_cohort.py
HELPER=$CODE_ROOT/scripts/audit_aihub_original_annotations.py
check_inputs() {
    scoped_path "$CODE_ROOT" && scoped_path "$CONTROL" &&
    [ ! -L "$CODE_ROOT/scripts" ] &&
    pinned "$PLANNER" "$PLANNER_SHA256" && pinned "$HELPER" "$ORIGINAL_HELPER_SHA256" &&
    pinned "$PROTECTED_REPORT" "$PROTECTED_SHA" && pinned "$SELECTED" "$SELECTED_SHA" &&
    pinned "$AUDITOR" "$AUDITOR_SHA" &&
    [ ! -e "$JOB/protected_fingerprints_v1_20260904/result/failed.json" ]
}
check_inputs || fail 'pinned code or input mismatch'
if ! bounded_command 30 "$DOCKER" image inspect --format '{{.Id}}' "$IMAGE_ID" > "$CONTROL/image_id.txt" 2> "$CONTROL/image_inspect.err"; then
    observation_error 'image inspect failed or timed out; no retry performed'
fi
[ "$(cat "$CONTROL/image_id.txt")" = "$IMAGE_ID" ] || fail 'image identity mismatch'
inspect_state() {
    bounded_command 30 "$DOCKER" inspect --format '{{.Id}} {{.State.Status}} {{.State.ExitCode}} {{.State.OOMKilled}} {{.Image}}' "$1" > "$2" 2> "$2.err"
}
inspect_state "$PRODUCER" "$CONTROL/producer_before.txt" || observation_error 'producer inspect failed or timed out'
read -r producer_id producer_status producer_exit producer_oom producer_image < "$CONTROL/producer_before.txt"
valid_sha "$producer_id" && [ "$producer_image" = "$IMAGE_ID" ] || fail 'producer identity or image mismatch'
case "$producer_status" in
    running)
        if ! "$DOCKER" wait "$producer_id" > "$CONTROL/producer_wait.txt" 2> "$CONTROL/producer_wait.err"; then
            observation_error 'producer wait transport failed; producer was not restarted'
        fi ;;
    exited|dead) ;;
    *) observation_error 'producer is not running or terminal; left unchanged';;
esac
inspect_state "$producer_id" "$CONTROL/producer_after.txt" || observation_error 'producer terminal inspect failed or timed out'
read -r after_id after_status after_exit after_oom after_image < "$CONTROL/producer_after.txt"
[ "$after_id" = "$producer_id" ] && [ "$after_image" = "$IMAGE_ID" ] || observation_error 'producer identity or image changed'
case "$after_status" in exited|dead) ;; *) observation_error 'producer terminal state not confirmed';; esac
[ "$after_status" = exited ] && [ "$after_exit" = 0 ] && [ "$after_oom" = false ] || fail 'producer terminated unsuccessfully or OOM'
[ -f "$ORIGINAL_REPORT" ] && no_links "$ORIGINAL_REPORT" || fail 'completed producer report missing or symlinked'
ORIGINAL_SHA=$(digest "$ORIGINAL_REPORT")
valid_sha "$ORIGINAL_SHA" || fail 'invalid original report digest'
printf '%s  %s\n' "$ORIGINAL_SHA" "$ORIGINAL_REPORT" > "$CONTROL/original_report.sha256"
check_inputs || fail 'pinned inputs changed while waiting'

app_control=/app${CONTROL#"$ROOT"}
app_code=/app${CODE_ROOT#"$ROOT"}
app_job=/app${JOB#"$ROOT"}
if ! bounded_command 60 "$DOCKER" create --name "$PLANNER_CONTAINER" --runtime runc \
    --network none --read-only --cpus 2 --memory 3g --memory-swap 3g --pids-limit 128 \
    --cap-drop ALL --security-opt no-new-privileges \
    --tmpfs /tmp:rw,nosuid,nodev,size=64m \
    -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    -v "$ROOT:/app:ro" -v "$CONTROL:$app_control:rw" --workdir "$app_code" \
    --entrypoint python3 "$IMAGE_ID" "$app_code/scripts/plan_aihub_original_cohort.py" \
    --original-report "$app_job/original_annotation_full_v2_20260904/result/report.json" --original-report-sha256 "$ORIGINAL_SHA" \
    --protected-report "$app_job/protected_fingerprints_v1_20260904/result/report.json" --protected-report-sha256 "$PROTECTED_SHA" \
    --selected-manifest /app/yolo_commercial_single_v1_20260813/selected_manifest.csv --selected-manifest-sha256 "$SELECTED_SHA" \
    --original-auditor "$app_job/audit_aihub_original_annotations_v2_20260904.py" --original-auditor-sha256 "$AUDITOR_SHA" \
    --output "$app_control/cohort" > "$CONTROL/planner_id.txt" 2> "$CONTROL/planner_create.err"; then
    observation_error 'planner create failed or timed out; inspect before any retry'
fi
planner_id=$(cat "$CONTROL/planner_id.txt")
valid_sha "$planner_id" || observation_error 'planner identity unavailable after create'
if ! bounded_command 60 "$DOCKER" start "$planner_id" > "$CONTROL/planner_start.txt" 2> "$CONTROL/planner_start.err"; then
    observation_error 'planner start failed or timed out; inspect before any retry'
fi
if ! "$DOCKER" wait "$planner_id" > "$CONTROL/planner_wait.txt" 2> "$CONTROL/planner_wait.err"; then
    observation_error 'planner wait transport failed; no retry performed'
fi
inspect_state "$planner_id" "$CONTROL/planner_after.txt" || observation_error 'planner terminal inspect failed or timed out'
read -r result_id result_status result_exit result_oom result_image < "$CONTROL/planner_after.txt"
[ "$result_id" = "$planner_id" ] && [ "$result_image" = "$IMAGE_ID" ] || observation_error 'planner identity or image changed'
case "$result_status" in exited|dead) ;; *) observation_error 'planner terminal state not confirmed';; esac
[ "$result_status" = exited ] && [ "$result_exit" = 0 ] && [ "$result_oom" = false ] || fail 'planner terminated unsuccessfully or OOM'
bounded_command 30 "$DOCKER" logs "$planner_id" > "$CONTROL/planner.log" 2>&1 || observation_error 'planner log retrieval failed'
check_inputs && pinned "$ORIGINAL_REPORT" "$ORIGINAL_SHA" || fail 'pinned inputs changed during cohort planning'
COHORT=$CONTROL/cohort/cohort.json
[ -f "$COHORT" ] && no_links "$COHORT" && [ ! -e "$CONTROL/cohort/failed.json" ] || fail 'cohort output missing or failed'
COHORT_SHA=$(digest "$COHORT")
valid_sha "$COHORT_SHA" || fail 'invalid cohort digest'
printf '{"status":"cohort_metadata_ready","cohort_sha256":"%s","original_report_sha256":"%s","producer_id":"%s","planner_id":"%s","training_authorized":false,"deployment_authorized":false}\n' \
    "$COHORT_SHA" "$ORIGINAL_SHA" "$producer_id" "$planner_id" > "$CONTROL/cohort_ready.json"
check_inputs && pinned "$ORIGINAL_REPORT" "$ORIGINAL_SHA" && pinned "$COHORT" "$COHORT_SHA" || fail 'publication-boundary input or output mutation; failed marker dominates ready'
terminal=1
printf '%s\n' 'cohort metadata ready; no training, deployment, or service change authorized'
