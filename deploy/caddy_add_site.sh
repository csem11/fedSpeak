#!/bin/bash
# caddy_add_site.sh DOMAIN UPSTREAM LABEL  -  runs ON the shared box.
#
# Adds one site block to the shared Caddy without disturbing the others.
#
# The Caddy image bakes in ~/Fund_Distro_RIA_Explorer/deploy/Caddyfile at build
# time, and the live copy inside the container has since been edited directly,
# so the two can differ. Appending to the host file and reloading applies
# nothing, and copying the host file in can drop live sites. So:
#   1. back up the live config and the host file
#   2. build the new config from the LIVE one plus this block
#   3. validate it inside the container before it replaces anything
#   4. swap it in and reload (graceful, no downtime for other sites)
#   5. append the block to the host file too, so the next image build keeps it
#
# Rollback: docker cp ~/caddy-backups/Caddyfile.live.<ts> deploy-web-1:/etc/caddy/Caddyfile
#           docker exec deploy-web-1 caddy reload --config /etc/caddy/Caddyfile --adapter caddyfile
set -euo pipefail
DOMAIN=$1; UPSTREAM=$2; LABEL=$3
CADDY=${CADDY_CONTAINER:-deploy-web-1}
HOSTFILE=${HOST_CADDYFILE:-$HOME/Fund_Distro_RIA_Explorer/deploy/Caddyfile}
LIVE=/etc/caddy/Caddyfile
BACKUP=$HOME/caddy-backups; mkdir -p "$BACKUP"; TS=$(date +%Y%m%d-%H%M%S)
has_site() { grep -qE "^$(printf '%s' "$DOMAIN" | sed 's/\./\\./g')([ ,{]|$)" "$1"; }

docker exec "$CADDY" cat "$LIVE" > "$BACKUP/Caddyfile.live.$TS"
cp "$HOSTFILE" "$BACKUP/Caddyfile.host.$TS"
echo "   backed up live and host configs to $BACKUP/*.$TS"

BLOCK="
# $LABEL
$DOMAIN {
    encode gzip
    reverse_proxy $UPSTREAM
}"

if has_site "$BACKUP/Caddyfile.live.$TS"; then
  echo "   $DOMAIN is already in the live config; leaving it alone"
else
  NEW="$BACKUP/Caddyfile.new.$TS"
  { cat "$BACKUP/Caddyfile.live.$TS"; printf '%s\n' "$BLOCK"; } > "$NEW"
  docker cp "$NEW" "$CADDY:/etc/caddy/Caddyfile.candidate"
  if ! docker exec "$CADDY" caddy validate --config /etc/caddy/Caddyfile.candidate --adapter caddyfile > "$BACKUP/validate.$TS.log" 2>&1; then
    echo "   new config FAILED validation; live config untouched:"; tail -5 "$BACKUP/validate.$TS.log"
    docker exec "$CADDY" rm -f /etc/caddy/Caddyfile.candidate; exit 1
  fi
  echo "   new config validated"
  docker exec "$CADDY" sh -c "cp /etc/caddy/Caddyfile.candidate $LIVE && rm /etc/caddy/Caddyfile.candidate"
  docker exec "$CADDY" caddy reload --config "$LIVE" --adapter caddyfile
  echo "   live config updated and reloaded"
fi

if ! has_site "$HOSTFILE"; then printf '%s\n' "$BLOCK" >> "$HOSTFILE"; echo "   block appended to the host copy for future image builds"; fi
