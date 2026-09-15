#!/usr/bin/env sh
# Provision an API-key gateway for the already-running internal naco-ollama.
# Run on the NAS only.  It never publishes Ollama's own 11434 port.
set -eu

DOCKER_BIN="${DOCKER_BIN:-/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker}"
ROOT="${NACO_GATEWAY_ROOT:-/share/Container/naco_ai/gateway}"
NETWORK="${NACO_OLLAMA_NETWORK:-naco_naco-internal}"
CONTAINER="${NACO_GATEWAY_CONTAINER:-naco-ollama-gateway}"
IMAGE="${NACO_GATEWAY_IMAGE:-nginx:alpine}"
PORT="${NACO_GATEWAY_PORT:-11435}"

# An explicit private bind address is mandatory: never listen on every NAS NIC.
: "${NACO_GATEWAY_BIND_IP:?Set the Pi-reachable private NAS address explicitly}"

if [ ! -x "$DOCKER_BIN" ]; then
  echo "Container Station docker binary unavailable" >&2
  exit 1
fi
if [ "$NACO_GATEWAY_BIND_IP" = "0.0.0.0" ] || [ "$NACO_GATEWAY_BIND_IP" = "::" ]; then
  echo "Refusing public/all-interface gateway binding" >&2
  exit 1
fi
if ! "$DOCKER_BIN" inspect -f '{{.State.Running}}' naco-ollama 2>/dev/null | grep -qx true; then
  echo "naco-ollama is not running; start and verify it before provisioning the gateway" >&2
  exit 1
fi
if ! "$DOCKER_BIN" network inspect "$NETWORK" >/dev/null 2>&1; then
  echo "Expected internal naco network is unavailable: $NETWORK" >&2
  exit 1
fi
if ! "$DOCKER_BIN" image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Gateway image is not present locally: $IMAGE (refusing to pull automatically)" >&2
  exit 1
fi

umask 077
mkdir -p "$ROOT"
KEY_FILE="$ROOT/gateway.env"
TEMPLATE="$ROOT/default.conf.template"
if [ ! -f "$KEY_FILE" ]; then
  if command -v openssl >/dev/null 2>&1; then
    KEY="$(openssl rand -hex 32)"
  else
    KEY="$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  fi
  printf 'NACO_LLM_API_KEY=%s\n' "$KEY" > "$KEY_FILE"
  unset KEY
fi

cat > "$TEMPLATE" <<'EOF'
map_hash_bucket_size 128;

map $http_authorization $naco_llm_authorized {
    default 0;
    "Bearer ${NACO_LLM_API_KEY}" 1;
}

server {
    listen 8080;
    server_name _;
    client_max_body_size 3m;

    location = /healthz {
        add_header Content-Type text/plain;
        return 200 "ok\n";
    }

    location /api/ {
        if ($naco_llm_authorized = 0) { return 401; }
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Request-ID $request_id;
        proxy_pass http://naco-ollama:11434;
    }
}
EOF

"$DOCKER_BIN" rm -f "$CONTAINER" >/dev/null 2>&1 || true
"$DOCKER_BIN" run -d \
  --name "$CONTAINER" \
  --restart unless-stopped \
  --network "$NETWORK" \
  -p "$NACO_GATEWAY_BIND_IP:$PORT:8080" \
  --env-file "$KEY_FILE" \
  -v "$TEMPLATE:/etc/nginx/templates/default.conf.template:ro" \
  "$IMAGE" >/dev/null

"$DOCKER_BIN" inspect -f '{{.State.Running}}' "$CONTAINER" | grep -qx true
echo "NACO_LLM_GATEWAY_READY endpoint=http://$NACO_GATEWAY_BIND_IP:$PORT"
echo "API key stored at $KEY_FILE (not printed); copy it only into the Pi AI server .env"
