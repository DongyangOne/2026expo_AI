#!/bin/sh
# Keep the production Ollama container available without resetting the QNAP GPU.
# Create /control/pause before intentional maintenance to suspend recovery.

set -u

SOCKET="${DOCKER_SOCKET:-/var/run/docker.sock}"
CONTAINER="${OLLAMA_CONTAINER:-naco-ollama}"
INTERVAL="${WATCH_INTERVAL_SECONDS:-30}"
START_GRACE="${START_GRACE_SECONDS:-45}"
UNHEALTHY_LIMIT="${UNHEALTHY_LIMIT:-3}"
RUNTIME_CHECK_INTERVAL="${RUNTIME_CHECK_INTERVAL_SECONDS:-60}"
GPU_RESTART_COOLDOWN="${GPU_RESTART_COOLDOWN_SECONDS:-900}"
OLLAMA_URL="${OLLAMA_URL:-http://naco-ollama:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-minicpm-v4.5:8b}"
API="http://localhost"
failures=0
last_state=""
needs_warmup=1
last_runtime_check=0
GPU_RESTART_MARKER="/control/last_gpu_restart_epoch"

log() {
  printf '%s %s\n' "$(date -Iseconds)" "$*"
}

inspect_container() {
  curl --silent --show-error --max-time 10 \
    --unix-socket "$SOCKET" "$API/containers/$CONTAINER/json"
}

post_container_action() {
  action="$1"
  curl --silent --show-error --max-time 40 \
    --unix-socket "$SOCKET" -X POST \
    -o /tmp/docker-response -w '%{http_code}' \
    "$API/containers/$CONTAINER/$action"
}

warm_model() {
  payload="$(printf '{"model":"%s","prompt":"Reply only OK.","stream":false,"keep_alive":"24h","options":{"num_predict":1}}' "$OLLAMA_MODEL")"
  curl --silent --show-error --fail --max-time 180 \
    -H 'Content-Type: application/json' \
    -d "$payload" "$OLLAMA_URL/api/generate" >/dev/null
}

model_runtime_state() {
  if ! payload="$(curl --silent --show-error --fail --max-time 10 \
    "$OLLAMA_URL/api/ps" 2>/dev/null)"; then
    printf '%s\n' unavailable
    return
  fi
  if ! printf '%s' "$payload" | grep -Fq "\"name\":\"$OLLAMA_MODEL\""; then
    printf '%s\n' not_loaded
  elif printf '%s' "$payload" | grep -Eq '"size_vram":[1-9][0-9]*'; then
    printf '%s\n' gpu
  elif printf '%s' "$payload" | grep -Eq '"size_vram":0([,}])'; then
    printf '%s\n' cpu
  else
    printf '%s\n' unknown
  fi
}

restart_for_cpu_fallback() {
  now="$(date +%s)"
  previous=0
  if [ -r "$GPU_RESTART_MARKER" ]; then
    previous="$(cat "$GPU_RESTART_MARKER" 2>/dev/null || printf '0')"
  fi
  case "$previous" in
    ''|*[!0-9]*) previous=0 ;;
  esac
  if [ $((now - previous)) -lt "$GPU_RESTART_COOLDOWN" ]; then
    log "model is on CPU; restart suppressed by ${GPU_RESTART_COOLDOWN}s cooldown"
    return 1
  fi

  printf '%s\n' "$now" >"$GPU_RESTART_MARKER"
  code="$(post_container_action 'restart?t=30' 2>/dev/null || printf '000')"
  log "model is on CPU instead of GPU; bounded restart requested http=$code"
  failures=0
  last_state="restarting"
  needs_warmup=1
  last_runtime_check="$now"
  return 0
}

log "watchdog started container=$CONTAINER interval=${INTERVAL}s"

while :; do
  if [ -e /control/pause ]; then
    failures=0
    if [ "$last_state" != "paused" ]; then
      log "recovery paused by /control/pause"
      last_state="paused"
    fi
    sleep "$INTERVAL"
    continue
  fi

  if ! state="$(inspect_container 2>/dev/null)"; then
    if [ "$last_state" != "missing" ]; then
      log "container inspect failed; refusing to recreate it"
      last_state="missing"
    fi
    sleep "$INTERVAL"
    continue
  fi

  if ! printf '%s' "$state" | grep -q '"Running":true'; then
    code="$(post_container_action start 2>/dev/null || printf '000')"
    log "container was stopped; start requested http=$code"
    last_state="starting"
    failures=0
    needs_warmup=1
    sleep "$START_GRACE"
    continue
  fi

  if printf '%s' "$state" | grep -q '"Health":{"Status":"unhealthy"'; then
    failures=$((failures + 1))
    log "container unhealthy check=$failures/$UNHEALTHY_LIMIT"
    if [ "$failures" -ge "$UNHEALTHY_LIMIT" ]; then
      code="$(post_container_action 'restart?t=30' 2>/dev/null || printf '000')"
      log "container remained unhealthy; restart requested http=$code"
      failures=0
      last_state="restarting"
      needs_warmup=1
      sleep "$START_GRACE"
      continue
    fi
  else
    failures=0
    if [ "$last_state" != "healthy" ]; then
      log "container running"
      last_state="healthy"
    fi
    if [ "$needs_warmup" -eq 1 ]; then
      if warm_model; then
        runtime="$(model_runtime_state)"
        if [ "$runtime" = "gpu" ]; then
          log "model warmed on GPU model=$OLLAMA_MODEL keep_alive=24h"
          needs_warmup=0
          last_runtime_check="$(date +%s)"
        elif [ "$runtime" = "cpu" ]; then
          if restart_for_cpu_fallback; then
            sleep "$START_GRACE"
            continue
          fi
        else
          log "model warmup runtime=$runtime; will retry"
        fi
      else
        log "model warmup failed; will retry"
      fi
    else
      now="$(date +%s)"
      if [ $((now - last_runtime_check)) -ge "$RUNTIME_CHECK_INTERVAL" ]; then
        runtime="$(model_runtime_state)"
        last_runtime_check="$now"
        case "$runtime" in
          gpu) ;;
          not_loaded)
            log "model is not loaded; scheduling warmup"
            needs_warmup=1
            ;;
          cpu)
            if restart_for_cpu_fallback; then
              sleep "$START_GRACE"
              continue
            fi
            ;;
          *) log "model runtime check=$runtime; no destructive recovery attempted" ;;
        esac
      fi
    fi
  fi

  sleep "$INTERVAL"
done
