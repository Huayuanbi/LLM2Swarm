"""
config.py — Central configuration for LLM2Swarm.
All tuneable constants live here; nothing is hardcoded elsewhere.
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()  # reads .env if present


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

# ─── Drone fleet ───────────────────────────────────────────────────────────────
DRONE_IDS: list[str] = ["drone_1", "drone_2", "drone_3"]

# ─── Control loop ──────────────────────────────────────────────────────────────
CONTROL_LOOP_INTERVAL: float = 10.0   # seconds between perception/replanning ticks
VLM_TIMEOUT: float           = 20.0   # max seconds to wait for edge VLM response (multimodal ~10s for 4b)
VLM_FALLBACK_DECISION: str   = "continue"  # used if VLM times out or errors
PEER_LOST_TIMEOUT: float     = float(os.getenv("PEER_LOST_TIMEOUT", "35.0"))
CLAIM_LEASE_TTL: float       = float(os.getenv("CLAIM_LEASE_TTL", "90.0"))

# ─── Cloud LLM (Global Planner) ───────────────────────────────────────────────
# To use GPT-4o:   GLOBAL_LLM_BASE_URL="" (or unset),  GLOBAL_LLM_MODEL="gpt-4o"
# To use Ollama:   GLOBAL_LLM_BASE_URL=http://localhost:11435/v1, GLOBAL_LLM_MODEL=qwen3.5:4b
OPENAI_API_KEY: str       = os.getenv("OPENAI_API_KEY", "")
GLOBAL_LLM_BASE_URL: str  = os.getenv("GLOBAL_LLM_BASE_URL", "")   # empty = use OpenAI default
GLOBAL_LLM_API_KEY: str   = os.getenv("GLOBAL_LLM_API_KEY", "") or OPENAI_API_KEY
GLOBAL_LLM_MODEL: str     = os.getenv("GLOBAL_LLM_MODEL", "gpt-4o")
GLOBAL_LLM_TIMEOUT: float = float(os.getenv("GLOBAL_LLM_TIMEOUT", "180.0"))
GLOBAL_LLM_MAX_RETRIES: int = int(os.getenv("GLOBAL_LLM_MAX_RETRIES", "3"))
GLOBAL_LLM_RETRY_BASE_DELAY: float = float(os.getenv("GLOBAL_LLM_RETRY_BASE_DELAY", "2.0"))

# ─── Edge VLM (Local Replanner — Ollama on remote server) ─────────────────────
# If your Mac routes through a proxy (e.g. Clash on :7890), run an SSH tunnel first:
#   ssh -N -L 11434:localhost:11434 yz@10.130.138.37
# then set EDGE_VLM_BASE_URL=http://localhost:11434/v1 in .env
EDGE_VLM_BASE_URL: str = os.getenv("EDGE_VLM_BASE_URL", "http://10.130.138.37:11434/v1")
EDGE_VLM_API_KEY: str  = "ollama"          # Ollama ignores this but OpenAI client requires it
EDGE_VLM_MODEL: str    = "qwen3.5:4b"
EDGE_VLM_TEMPERATURE: float = 0.0

# ─── Onboard Planner (initial task-graph synthesis / future agent adapter) ──
# Defaults to the same local/edge model route, but can later be redirected to a
# stronger LLM or a dedicated agent framework without changing call sites.
ONBOARD_PLANNER_BASE_URL: str = os.getenv("ONBOARD_PLANNER_BASE_URL", EDGE_VLM_BASE_URL)
ONBOARD_PLANNER_API_KEY: str = os.getenv("ONBOARD_PLANNER_API_KEY", EDGE_VLM_API_KEY)
ONBOARD_PLANNER_MODEL: str = os.getenv("ONBOARD_PLANNER_MODEL", EDGE_VLM_MODEL)
ONBOARD_PLANNER_TEMPERATURE: float = float(os.getenv("ONBOARD_PLANNER_TEMPERATURE", "0.0"))
ONBOARD_PLANNER_TIMEOUT: float = float(os.getenv("ONBOARD_PLANNER_TIMEOUT", "90.0"))

# ─── Simulator backend ─────────────────────────────────────────────────────────
# "mock"   — pure-Python simulated physics, no external simulator needed
# "webots" — Webots extern controller via TCP (see controllers/webots_controller.py)
SIMULATOR_BACKEND: str = os.getenv("SIMULATOR_BACKEND", "mock")

# Webots extern controller TCP settings (only used when SIMULATOR_BACKEND="webots")
WEBOTS_HOST: str = "localhost"
WEBOTS_PORT: int = 10020          # base port; drone_N gets port 10020+N

# ─── Mock physics ──────────────────────────────────────────────────────────────
MOCK_MOVE_SPEED: float    = 5.0   # m/s (used by mock controller to advance position)
MOCK_TICK_RATE: float     = 0.05  # seconds between physics ticks in mock (20 Hz)
MOCK_IMAGE_WIDTH: int     = 320
MOCK_IMAGE_HEIGHT: int    = 240

# ─── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ─── Debug gate / human-in-the-loop inspection ────────────────────────────────
DEBUG_MODE: bool = _env_bool("LLM2SWARM_DEBUG", False)
DEBUG_DIR: str = os.getenv("LLM2SWARM_DEBUG_DIR", "/tmp/llm2swarm_debug")
DEBUG_SESSION_ID: str = os.getenv(
    "LLM2SWARM_DEBUG_SESSION",
    f"session_{time.strftime('%Y%m%d_%H%M%S')}",
)
DEBUG_STAGES: str = os.getenv(
    "LLM2SWARM_DEBUG_STAGES",
    "cloud_request,cloud_response,onboard_request,onboard_response",
)
DEBUG_TARGETS: str = os.getenv("LLM2SWARM_DEBUG_TARGETS", "")
DEBUG_POLL_INTERVAL: float = float(os.getenv("LLM2SWARM_DEBUG_POLL_INTERVAL", "0.5"))
DEBUG_VLM_PAUSE_ONCE_PER_ACTOR: bool = _env_bool("LLM2SWARM_DEBUG_VLM_PAUSE_ONCE", True)
