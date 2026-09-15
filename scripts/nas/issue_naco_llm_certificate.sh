#!/usr/bin/env sh
# Issue/renew HTTPS for the already-provisioned Naco LLM Nginx container.
set -eu

DOCKER_BIN="${DOCKER_BIN:-/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker}"
DOMAIN="${NACO_LLM_DOMAIN:-llm.naco.kro.kr}"
ROOT="${NACO_PUBLIC_GATEWAY_ROOT:-/share/Container/naco_ai/llm_gateway}"
CONTAINER="${NACO_PUBLIC_GATEWAY_CONTAINER:-naco-llm-nginx}"
CERTBOT_IMAGE="${NACO_CERTBOT_IMAGE:-certbot/certbot:latest}"
NGINX_IMAGE="${NACO_PUBLIC_GATEWAY_IMAGE:-nginx:alpine}"
STATIC_NETWORK="${NACO_PUBLIC_NETWORK:-qnet-static-bond0-272384}"
INTERNAL_NETWORK="${NACO_OLLAMA_NETWORK:-naco_naco-internal}"
PUBLIC_IP="${NACO_LLM_PUBLIC_IP:-223.194.166.7}"
KEY_FILE="${NACO_GATEWAY_KEY_FILE:-/share/Container/naco_ai/gateway/gateway.env}"
RENEWER="${NACO_CERTBOT_CONTAINER:-naco-llm-certbot}"

if ! "$DOCKER_BIN" inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -qx true; then
  echo "Public HTTP gateway is not running" >&2; exit 1
fi
if ! "$DOCKER_BIN" image inspect "$CERTBOT_IMAGE" >/dev/null 2>&1; then
  echo "Certbot image missing locally: $CERTBOT_IMAGE" >&2; exit 1
fi
if ! "$DOCKER_BIN" image inspect "$NGINX_IMAGE" >/dev/null 2>&1; then
  echo "Nginx image missing locally: $NGINX_IMAGE" >&2; exit 1
fi

if [ "${NACO_SKIP_ISSUANCE:-0}" != "1" ]; then
  : "${NACO_ACME_SERVER:?Set the ACME directory URL}"
  : "${NACO_ACME_EAB_KID:?Set the ZeroSSL EAB key id}"
  : "${NACO_ACME_EAB_HMAC_KEY:?Set the ZeroSSL EAB HMAC key}"
  : "${NACO_ACME_EMAIL:?Set a renewal-notification email}"
  "$DOCKER_BIN" run --rm \
    -v "$ROOT/certbot/www:/var/www/certbot" \
    -v "$ROOT/certbot/conf:/etc/letsencrypt" \
    "$CERTBOT_IMAGE" certonly --webroot -w /var/www/certbot \
    --server "$NACO_ACME_SERVER" \
    --eab-kid "$NACO_ACME_EAB_KID" \
    --eab-hmac-key "$NACO_ACME_EAB_HMAC_KEY" \
    --email "$NACO_ACME_EMAIL" --agree-tos --no-eff-email --non-interactive -d "$DOMAIN"
fi

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

"$DOCKER_BIN" rm -f "$CONTAINER" >/dev/null 2>&1 || true
"$DOCKER_BIN" run -d --name "$CONTAINER" --restart unless-stopped \
  --network "$STATIC_NETWORK" --ip "$PUBLIC_IP" \
  --env-file "$KEY_FILE" \
  -v "$ROOT/nginx/default.conf.template:/etc/nginx/templates/default.conf.template:ro" \
  -v "$ROOT/certbot/www:/var/www/certbot:ro" \
  -v "$ROOT/certbot/conf:/etc/letsencrypt:ro" \
  "$NGINX_IMAGE" sh -c 'nginx -t || exit 1; (while :; do sleep 21600; nginx -s reload; done) & exec nginx -g "daemon off;"' >/dev/null
"$DOCKER_BIN" network connect "$INTERNAL_NETWORK" "$CONTAINER"
"$DOCKER_BIN" inspect -f '{{.State.Running}}' "$CONTAINER" | grep -qx true
"$DOCKER_BIN" rm -f "$RENEWER" >/dev/null 2>&1 || true
"$DOCKER_BIN" run -d --name "$RENEWER" --restart unless-stopped \
  --entrypoint /bin/sh \
  -v "$ROOT/certbot/www:/var/www/certbot" \
  -v "$ROOT/certbot/conf:/etc/letsencrypt" \
  "$CERTBOT_IMAGE" sh -c 'trap exit TERM; while :; do certbot renew --non-interactive; sleep 12h & wait ${!}; done;' >/dev/null
echo "TLS_READY domain=$DOMAIN"
