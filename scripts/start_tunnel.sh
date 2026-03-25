#!/usr/bin/env bash
# scripts/start_tunnel.sh
#
# Ensures the SSH tunnel to the Ollama server is running.
# Forwards localhost:11435 → yz@10.130.138.37:11434
#
# Usage:
#   ./scripts/start_tunnel.sh          # start if not already running
#   ./scripts/start_tunnel.sh --kill   # kill the tunnel
#   ./scripts/start_tunnel.sh --status # check status

set -euo pipefail

LOCAL_PORT=11435
REMOTE_HOST="10.130.138.37"
REMOTE_PORT=11434
SSH_USER="yz"
HEALTH_URL="http://localhost:${LOCAL_PORT}/api/tags"

# ── Helpers ───────────────────────────────────────────────────────────────────

is_tunnel_running() {
    # Check if something is already listening on LOCAL_PORT
    lsof -iTCP:${LOCAL_PORT} -sTCP:LISTEN -t &>/dev/null
}

check_health() {
    # Bypass system proxy (Clash/V2Ray) for localhost traffic
    curl -s --noproxy localhost --max-time 4 "${HEALTH_URL}" &>/dev/null
}

kill_tunnel() {
    local pids
    pids=$(lsof -iTCP:${LOCAL_PORT} -sTCP:LISTEN -t 2>/dev/null || true)
    if [[ -z "${pids}" ]]; then
        echo "No tunnel running on port ${LOCAL_PORT}."
    else
        echo "Killing tunnel PIDs: ${pids}"
        kill ${pids}
        echo "Tunnel stopped."
    fi
}

# ── Argument handling ─────────────────────────────────────────────────────────

case "${1:-}" in
    --kill)
        kill_tunnel
        exit 0
        ;;
    --status)
        if is_tunnel_running && check_health; then
            echo "✓ Tunnel is UP  (localhost:${LOCAL_PORT} → ${SSH_USER}@${REMOTE_HOST}:${REMOTE_PORT})"
            # Print available models
            models=$(curl -s --noproxy localhost --max-time 4 "${HEALTH_URL}" \
                     | python3 -c "import sys,json; d=json.load(sys.stdin); [print(' -', m['name']) for m in d.get('models',[])]" 2>/dev/null || echo "  (could not list models)")
            echo "  Available models:"
            echo "${models}"
        else
            echo "✗ Tunnel is DOWN"
        fi
        exit 0
        ;;
esac

# ── Main: start tunnel if needed ──────────────────────────────────────────────

if is_tunnel_running; then
    if check_health; then
        echo "✓ Tunnel already running on localhost:${LOCAL_PORT} — nothing to do."
        exit 0
    else
        echo "⚠ Port ${LOCAL_PORT} is occupied but Ollama health check failed."
        echo "  Killing stale process and restarting …"
        kill_tunnel
        sleep 1
    fi
fi

echo "Starting SSH tunnel: localhost:${LOCAL_PORT} → ${SSH_USER}@${REMOTE_HOST}:${REMOTE_PORT}"
ssh -N -f \
    -o "ExitOnForwardFailure=yes" \
    -o "ServerAliveInterval=30" \
    -o "ServerAliveCountMax=3" \
    -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" \
    "${SSH_USER}@${REMOTE_HOST}"

# Wait up to 6 s for Ollama to respond through the tunnel
echo -n "Waiting for Ollama health check "
for i in $(seq 1 6); do
    if check_health; then
        echo ""
        echo "✓ Tunnel established. Ollama is reachable at localhost:${LOCAL_PORT}"
        echo ""
        # Print available models
        curl -s --noproxy localhost --max-time 4 "${HEALTH_URL}" \
            | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('  Available models:')
for m in d.get('models', []):
    print(f'    - {m[\"name\"]}  ({m[\"details\"][\"parameter_size\"]})')
" 2>/dev/null || true
        exit 0
    fi
    echo -n "."
    sleep 1
done

echo ""
echo "✗ Tunnel started but Ollama health check timed out."
echo "  Check that Ollama is running on ${REMOTE_HOST}:${REMOTE_PORT}"
exit 1
