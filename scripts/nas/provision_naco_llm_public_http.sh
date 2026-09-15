#!/usr/bin/env sh
# Prepare the public HTTP-01 endpoint for llm.naco.kro.kr.
# Run on the NAS after the IP is reserved for this Docker macvlan container.
set -eu

DOCKER_BIN="${DOCKER_BIN:-/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker}"
STATIC_NETWORK="${NACO_PUBLIC_NETWORK:-qnet-static-bond0-272384}"
INTERNAL_NETWORK="${NACO_OLLAMA_NETWORK:-naco_naco-internal}"
PUBLIC_IP="${NACO_LLM_PUBLIC_IP:-223.194.166.7}"
DOMAIN="${NACO_LLM_DOMAIN:-llm.naco.kro.kr}"
ROOT="${NACO_PUBLIC_GATEWAY_ROOT:-/share/Container/naco_ai/llm_gateway}"
KEY_FILE="${NACO_GATEWAY_KEY_FILE:-/share/Container/naco_ai/gateway/gateway.env}"
CONTAINER="${NACO_PUBLIC_GATEWAY_CONTAINER:-naco-llm-nginx}"
IMAGE="${NACO_PUBLIC_GATEWAY_IMAGE:-nginx:alpine}"

if [ ! -x "$DOCKER_BIN" ]; then echo "Container Station docker unavailable" >&2; exit 1; fi
if "$DOCKER_BIN" network inspect "$STATIC_NETWORK" >/dev/null 2>&1; then :; else
  echo "Static public network unavailable: $STATIC_NETWORK" >&2; exit 1
fi
if "$DOCKER_BIN" network inspect "$INTERNAL_NETWORK" >/dev/null 2>&1; then :; else
  echo "Internal naco network unavailable: $INTERNAL_NETWORK" >&2; exit 1
fi
if "$DOCKER_BIN" image inspect "$IMAGE" >/dev/null 2>&1; then :; else
  echo "Nginx image missing locally: $IMAGE" >&2; exit 1
fi
if [ ! -f "$KEY_FILE" ]; then
  echo "Protected gateway API key file is missing: $KEY_FILE" >&2; exit 1
fi
if ip -4 addr show | grep -Fq "inet $PUBLIC_IP/"; then
  echo "Refusing: $PUBLIC_IP is assigned to the NAS host, not reserved for the container" >&2; exit 1
fi

umask 077
mkdir -p "$ROOT/certbot/www" "$ROOT/certbot/conf" "$ROOT/nginx"
cat > "$ROOT/nginx/default.conf.template" <<EOF
server {
    listen 80;
    server_name $DOMAIN;
    client_max_body_size 3m;

    location /.well-known/acme-challenge/ { root /var/www/certbot; }
    location = /healthz { add_header Content-Type text/plain; return 200 "acme-ready\\n"; }
    location / { return 308 https://\$host\$request_uri; }
}
EOF

"$DOCKER_BIN" run --rm \
  -v "$ROOT/nginx/default.conf.template:/etc/nginx/templates/default.conf.template:ro" \
  "$IMAGE" nginx -t >/dev/null
"$DOCKER_BIN" rm -f "$CONTAINER" >/dev/null 2>&1 || true
"$DOCKER_BIN" run -d --name "$CONTAINER" --restart unless-stopped \
  --network "$STATIC_NETWORK" --ip "$PUBLIC_IP" \
  --env-file "$KEY_FILE" \
  -v "$ROOT/nginx/default.conf.template:/etc/nginx/templates/default.conf.template:ro" \
  -v "$ROOT/certbot/www:/var/www/certbot:ro" \
  -v "$ROOT/certbot/conf:/etc/letsencrypt:ro" \
  "$IMAGE" sh -c 'nginx -t || exit 1; (while :; do sleep 21600; nginx -s reload; done) & exec nginx -g "daemon off;"' >/dev/null
"$DOCKER_BIN" network connect "$INTERNAL_NETWORK" "$CONTAINER"
"$DOCKER_BIN" inspect -f '{{.State.Running}}' "$CONTAINER" | grep -qx true
echo "PUBLIC_HTTP_READY ip=$PUBLIC_IP domain=$DOMAIN"
echo "Create DNS A: $DOMAIN -> $PUBLIC_IP, then run the TLS issuance script."
