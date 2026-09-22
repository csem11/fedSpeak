#!/bin/bash
# deploy.sh - serve the FedSpeak page at https://fedspeak.csem.ai on the shared
# csem.ai box. Builds on the box, which is arm64, so local Docker is not needed.
#
# Re-runnable: rebuilds the page, replaces the container, and only touches
# Caddy if the site isn't routed yet. Stops before Caddy if DNS isn't ready.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=fedspeak
DOMAIN=fedspeak.csem.ai
HOST=${DEPLOY_HOST:-ubuntu@34.224.114.150}
KEY=${DEPLOY_KEY:-$HOME/.ssh/ria-explorer.pem}
REMOTE_DIR=fedSpeak
BOX_IP=${HOST#*@}
NEIGHBOURS="csem.ai www.csem.ai ria.csem.ai ria-demo.csem.ai meketa.csem.ai treasury-deposit-lab.csem.ai loowit-play.csem.ai etfetch.csem.ai sudoku-solver.csem.ai"
ssh_box() { ssh -i "$KEY" -o ConnectTimeout=10 -o BatchMode=yes "$HOST" "$@"; }
status() { curl -s -o /dev/null -w '%{http_code}' --max-time 20 "https://$1/" || true; }

echo "== 1. rebuild the page from results/"
python3 site/build.py

echo "== 2. copy the page and the deploy files, and nothing else, to the box"
ssh_box "mkdir -p ~/$REMOTE_DIR"
rsync -az --relative -e "ssh -i $KEY -o BatchMode=yes" \
  docs/index.html deploy/Dockerfile deploy/docker-compose.yml deploy/caddy_add_site.sh "$HOST:$REMOTE_DIR/"

echo "== 3. build and start the container"
ssh_box "cd ~/$REMOTE_DIR && (docker network inspect web >/dev/null 2>&1 || docker network create web) \
  && docker compose -f deploy/docker-compose.yml up -d --build && docker image prune -f >/dev/null"
health=starting
for _ in $(seq 1 30); do
  health=$(ssh_box "docker inspect -f '{{.State.Health.Status}}' $NAME" 2>/dev/null || echo missing)
  [ "$health" = healthy ] && break; sleep 3
done
echo "   container health: $health"
[ "$health" = healthy ] || { echo "   container is not healthy; not routing it"; exit 1; }

echo "== 4. route $DOMAIN through Caddy"
if [ "$(dig +short "$DOMAIN" A | tail -1)" != "$BOX_IP" ]; then
  echo "   $DOMAIN does not resolve to $BOX_IP yet."
  echo "   Add a DNS-only (grey cloud) A record in Cloudflare, then re-run this script."
  exit 0
fi
# macOS ships bash 3.2 (no associative arrays), so the baseline goes in a file.
BEFORE=$(mktemp); trap 'rm -f "$BEFORE"' EXIT
echo "   neighbours before:"
for n in $NEIGHBOURS; do c=$(status "$n"); echo "$n $c" >> "$BEFORE"; printf "     %-32s %s\n" "$n" "$c"; done
before_of() { awk -v n="$1" '$1 == n {print $2}' "$BEFORE"; }
ssh_box "bash ~/$REMOTE_DIR/deploy/caddy_add_site.sh $DOMAIN $NAME:80 'fedspeak (fedSpeak repo, github.com/csem11/fedSpeak)'"

echo "== 5. verify"
for _ in $(seq 1 12); do [ "$(status "$DOMAIN")" = 200 ] && break; sleep 5; done   # first request fetches the TLS cert
printf "   %-34s %s\n" "$DOMAIN" "$(status "$DOMAIN")"
changed=0
for n in $NEIGHBOURS; do
  after=$(status "$n"); was=$(before_of "$n"); flag=""; [ "$after" != "$was" ] && { flag="  CHANGED from $was"; changed=1; }
  printf "   %-34s %s%s\n" "$n" "$after" "$flag"
done
ssh_box 'echo "   box: $(df -h / | tail -1 | awk "{print \$4\" disk free\"}"), $(free -m | awk "/Mem:/ {print \$7\" MB memory available\"}")"'
[ $changed = 0 ] && echo "   no neighbour changed status" || { echo "   A NEIGHBOUR CHANGED STATUS: see caddy_add_site.sh for rollback"; exit 1; }
