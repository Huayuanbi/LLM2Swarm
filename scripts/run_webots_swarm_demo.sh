#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORLD_FILE="${REPO_ROOT}/worlds/mavic_2_pro_swarm_demo.wbt"

WEBOTS_HOME_DEFAULT="/Applications/Webots.app/Contents"
WEBOTS_HOME="${WEBOTS_HOME:-${WEBOTS_HOME_DEFAULT}}"
WEBOTS_BIN="${WEBOTS_BIN:-${WEBOTS_HOME}/MacOS/Webots}"
WEBOTS_CONTROLLER_PY="${WEBOTS_HOME}/lib/controller/python"

if [[ ! -x "${WEBOTS_BIN}" ]]; then
  echo "Webots binary not found at ${WEBOTS_BIN}"
  echo "Set WEBOTS_HOME or WEBOTS_BIN before running this script."
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${WEBOTS_CONTROLLER_PY}:${PYTHONPATH:-}"
export SIMULATOR_BACKEND=webots
export WEBOTS_SWARM_DRONES="${WEBOTS_SWARM_DRONES:-drone_1,drone_2,drone_3}"
export WEBOTS_SWARM_DB="${WEBOTS_SWARM_DB:-/tmp/llm2swarm_webots_swarm.sqlite3}"
export WEBOTS_SWARM_WORLD_FILE="${WEBOTS_SWARM_WORLD_FILE:-${WORLD_FILE}}"

DEBUG_UI_PID=""
DEBUG_UI_HOST="${LLM2SWARM_DEBUG_UI_HOST:-127.0.0.1}"
DEBUG_UI_PORT="${LLM2SWARM_DEBUG_UI_PORT:-8765}"
DEBUG_UI_AUTO_START="${LLM2SWARM_DEBUG_UI_AUTO_START:-1}"
DEBUG_UI_OPEN_BROWSER="${LLM2SWARM_DEBUG_UI_OPEN_BROWSER:-1}"
DEBUG_ROOT="${LLM2SWARM_DEBUG_DIR:-/tmp/llm2swarm_debug}"

cleanup() {
  if [[ -n "${DEBUG_UI_PID}" ]] && kill -0 "${DEBUG_UI_PID}" 2>/dev/null; then
    kill "${DEBUG_UI_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

start_debug_ui() {
  if [[ "${LLM2SWARM_DEBUG:-0}" == "0" || "${DEBUG_UI_AUTO_START}" == "0" ]]; then
    return 0
  fi

  if command -v lsof >/dev/null 2>&1; then
    while lsof -iTCP:"${DEBUG_UI_PORT}" -sTCP:LISTEN >/dev/null 2>&1; do
      DEBUG_UI_PORT="$((DEBUG_UI_PORT + 1))"
    done
  fi

  mkdir -p "${DEBUG_ROOT}/${LLM2SWARM_DEBUG_SESSION}"
  local log_file="${DEBUG_ROOT}/${LLM2SWARM_DEBUG_SESSION}/debug_ui.log"

  conda run -n llm2swarm python scripts/debug_gate.py serve \
    --session "${LLM2SWARM_DEBUG_SESSION}" \
    --host "${DEBUG_UI_HOST}" \
    --port "${DEBUG_UI_PORT}" >"${log_file}" 2>&1 &
  DEBUG_UI_PID=$!

  sleep 1
  if kill -0 "${DEBUG_UI_PID}" 2>/dev/null; then
    local panel_url="http://${DEBUG_UI_HOST}:${DEBUG_UI_PORT}"
    echo "  panel url: ${panel_url}"
    echo "  panel log: ${log_file}"
    if [[ "${DEBUG_UI_OPEN_BROWSER}" != "0" ]] && [[ "$(uname -s)" == "Darwin" ]] && command -v open >/dev/null 2>&1; then
      (sleep 1; open "${panel_url}" >/dev/null 2>&1 || true) &
      echo "  browser:   opening debug UI automatically"
    fi
  else
    echo "  panel failed to start; check ${log_file}"
    DEBUG_UI_PID=""
  fi
}

if [[ "${LLM2SWARM_DEBUG:-0}" != "0" && -z "${LLM2SWARM_DEBUG_SESSION:-}" ]]; then
  export LLM2SWARM_DEBUG_SESSION="swarm_$(date +%Y%m%d_%H%M%S)"
fi

if [[ "${LLM2SWARM_DEBUG:-0}" != "0" ]]; then
  echo "LLM2Swarm debug mode is ON"
  echo "  session: ${LLM2SWARM_DEBUG_SESSION}"
  echo "  stages:  ${LLM2SWARM_DEBUG_STAGES:-cloud_request,cloud_response,onboard_request,onboard_response}"
  echo "  inspect: conda run -n llm2swarm python scripts/debug_gate.py show latest"
  echo "  panel:   starting local debug UI automatically"
fi

cd "${REPO_ROOT}"
start_debug_ui
conda run -n llm2swarm python scripts/prepare_webots_swarm_plan.py
conda run -n llm2swarm "${WEBOTS_BIN}" "${WORLD_FILE}"
