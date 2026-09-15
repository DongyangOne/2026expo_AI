#!/usr/bin/env sh
# Issue/renew HTTPS for the already-provisioned Naco LLM Nginx container.
set -eu

DOCKER_BIN="${DOCKER_BIN:-/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker}"
DOMAIN="${NACO_LLM_DOMAIN:-llm.naco.kro.kr}"
ROOT="${NACO_PUBLIC_GATEWAY_ROOT:-/share/Container/naco_ai/llm_gateway}"
CONTAINER="${NACO_PUBLIC_GATEWAY_CONTAINER:-naco-llm-nginx}"
CERTBOT_IMAGE="${NACO_CERTBOT_IMAGE:-certbot/certbot:latest}"
: "${NACO_LETSENCRYPT_EMAIL:?Set a renewal-notification email}"

if ! "$DOCKER_BIN" inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -qx true; then
  echo "Public HTTP gateway is not running" >&2; exit 1
fi
if ! "$DOCKER_BIN" image inspect "$CERTBOT_IMAGE" >/dev/null 2>&1; then
  echo "Certbot image missing locally: $CERTBOT_IMAGE" >&2; exit 1
fi

"$DOCKER_BIN" run --rm \
  -v "$ROOT/certbot/www:/var/www/certbot" \
  -v "$ROOT/certbot/conf:/etc/letsencrypt" \
  "$CERTBOT_IMAGE" certonly --webroot -w /var/www/certbot \
  --email "$NACO_LETSENCRYPT_EMAIL" --agree-tos --non-interactive -d "$DOMAIN"

cat > "$ROOT/nginx/default.conf.template" <<'EOF'
map_hash_bucket_size 128;
map $http_authorization $naco_llm_authorized {
    default 0;
    "Bearer ${NACO_LLM_API_KEY}" 1;
}
server {
    listen 80;
    server_name llm.naco.kro.kr;
    location /.well-known/acme-challenge/ { root /var/www/certbot; }
    location / { return 308 https://$host$request_uri; }
}
server {
    listen 443 ssl;
    server_name llm.naco.kro.kr;
    client_max_body_size 3m;
    ssl_certificate /etc/letsencrypt/live/llm.naco.kro.kr/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/llm.naco.kro.kr/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    location = /healthz { add_header Content-Type text/plain; return 200 "ok\n"; }
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

"$DOCKER_BIN" restart "$CONTAINER" >/dev/null
"$DOCKER_BIN" inspect -f '{{.State.Running}}' "$CONTAINER" | grep -qx true
echo "TLS_READY domain=$DOMAIN"
