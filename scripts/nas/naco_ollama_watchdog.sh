#!/bin/sh
# Keep the production Ollama container available without resetting the QNAP GPU.
# Create /control/pause before intentional maintenance to suspend recovery.

set -u

SOCKET="${DOCKER_SOCKET:-/var/run/docker.sock}"
CONTAINER="${OLLAMA_CONTAINER:-naco-ollama}"
INTERVAL="${WATCH_INTERVAL_SECONDS:-30}"
START_GRACE="${START_GRACE_SECONDS:-45}"
UNHEALTHY_LIMIT="${UNHEALTHY_LIMIT:-3}"
OLLAMA_URL="${OLLAMA_URL:-http://naco-ollama:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-minicpm-v4.5:8b}"
API="http://localhost"
failures=0
last_state=""
needs_warmup=1

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
        log "model warmed model=$OLLAMA_MODEL keep_alive=24h"
        needs_warmup=0
      else
        log "model warmup failed; will retry"
      fi
    fi
  fi

  sleep "$INTERVAL"
done
