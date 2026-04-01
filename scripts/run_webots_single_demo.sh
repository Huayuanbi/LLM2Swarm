#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORLD_FILE="${REPO_ROOT}/worlds/mavic_2_pro_llm_demo.wbt"

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

cd "${REPO_ROOT}"
exec conda run -n llm2swarm "${WEBOTS_BIN}" "${WORLD_FILE}"
