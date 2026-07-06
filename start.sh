#!/bin/bash
set -e

# Mirror dashboard-ref-only's startup: create every directory hermes expects
# and seed a default config.yaml if the volume is empty. Without these,
# `hermes dashboard` endpoints that hit logs/, sessions/, cron/, etc. can fail
# with opaque errors even though no auth is actually involved.
mkdir -p /data/.hermes/cron /data/.hermes/sessions /data/.hermes/logs \
         /data/.hermes/memories /data/.hermes/skills /data/.hermes/pairing \
         /data/.hermes/hooks /data/.hermes/image_cache /data/.hermes/audio_cache \
         /data/.hermes/workspace /data/.hermes/skins /data/.hermes/plans \
         /data/.hermes/home

# Stamp the install method as "docker" so hermes treats this as an immutable
# container image, not a pip checkout. hermes's detect_install_method() reads
# $HERMES_HOME/.install_method FIRST (before any .git / pip fallback). Without
# this stamp the template falls through to "pip" — because the Dockerfile strips
# /opt/hermes-agent/.git — and the dashboard's "Update Hermes" button then runs
# a real `hermes update` (PyPI pip-upgrade) INSIDE the running container. That
# upgrade is ephemeral (reverts on the next redeploy) and can desync the Python
# package from the image's pre-built web_dist/ui-tui bundles. Stamping "docker"
# makes that button correctly refuse with "pull a fresh image / redeploy", which
# matches the real upgrade path here (bump HERMES_REF in Railway + redeploy).
# Written unconditionally each boot so it stays correct and self-heals.
printf 'docker\n' > /data/.hermes/.install_method

if [ ! -f /data/.hermes/config.yaml ] && [ -f /opt/hermes-agent/cli-config.yaml.example ]; then
  cp /opt/hermes-agent/cli-config.yaml.example /data/.hermes/config.yaml
fi

[ ! -f /data/.hermes/.env ] && touch /data/.hermes/.env

# Bootstrap OAuth tokens from env var (e.g. xAI Grok SuperGrok).
# Set HERMES_AUTH_JSON_BOOTSTRAP to the contents of a locally-generated
# ~/.hermes/auth.json. Written only once — subsequent token refreshes update
# the file in place on the persistent volume.
if [ ! -f /data/.hermes/auth.json ] && [ -n "${HERMES_AUTH_JSON_BOOTSTRAP}" ]; then
  printf '%s' "${HERMES_AUTH_JSON_BOOTSTRAP}" > /data/.hermes/auth.json
  chmod 600 /data/.hermes/auth.json
fi

# Clear any stale gateway PID file left over from the previous container.
# `hermes gateway` writes /data/.hermes/gateway.pid on start but does not
# remove it on SIGTERM. Since /data is a persistent volume, the file
# survives container restarts and causes every subsequent boot to exit with
# "ERROR gateway.run: PID file race lost to another gateway instance".
# No hermes process can be running at this point (we're pre-exec in a fresh
# container), so removing the file unconditionally is safe.
rm -f /data/.hermes/gateway.pid

# ── Tailscale (userspace networking) ─────────────────────────────────────────
# Railway containers have no /dev/net/tun and no NET_ADMIN capability, so
# tailscaled runs in userspace-networking mode (no kernel TUN device). When
# TS_AUTHKEY is set, join the tailnet and publish the native Hermes dashboard
# (127.0.0.1:9119) over HTTPS so Hermes Desktop can reach
# https://<hostname>.<tailnet>.ts.net without exposing the dashboard publicly.
#
# Node identity/state lives on the persistent /data volume so redeploys reuse
# the same tailnet node instead of registering a new one each boot.
#
# The whole block is best-effort: any tailscale failure logs and continues so
# it can never block `hermes gateway` from starting. Skipped entirely when
# TS_AUTHKEY is unset, preserving the previous public-only behavior.
if [ -n "${TS_AUTHKEY}" ]; then
  mkdir -p /run/tailscale /data/.hermes/tailscale
  TS_HOSTNAME="${TS_HOSTNAME:-hermes-railway}"

  tailscaled \
    --tun=userspace-networking \
    --socket=/run/tailscale/tailscaled.sock \
    --statedir=/data/.hermes/tailscale \
    >/data/.hermes/logs/tailscaled.log 2>&1 &

  # Wait for the daemon socket before `up` (bounded — never hang the boot).
  for _i in $(seq 1 30); do
    [ -S /run/tailscale/tailscaled.sock ] && break
    sleep 0.5
  done

  tailscale up \
    --authkey="${TS_AUTHKEY}" \
    --hostname="${TS_HOSTNAME}" \
    || echo "[tailscale] up failed — continuing without tailnet"

  # Publish the native Hermes dashboard over HTTPS on the tailnet. Requires
  # MagicDNS + HTTPS certs enabled in the tailnet admin console.
  tailscale serve --bg --https=443 http://127.0.0.1:9119 \
    || echo "[tailscale] serve failed — dashboard not published to tailnet"
else
  echo "[tailscale] TS_AUTHKEY not set — skipping tailnet setup"
fi

exec python /app/server.py
