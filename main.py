"""
main.py — LLM2Swarm entry point.

Startup sequence:
  1. Parse CLI args (mission description, drone count, etc.)
  2. Check SSH tunnel / Ollama health.
  3. Initialise SharedMemoryPool.
  4. Call GlobalOperator (GPT-4o) to generate the initial task plan.
     → Falls back to a built-in default plan if the API key is missing/exhausted.
  5. Launch all drone lifecycles concurrently with asyncio.gather.
  6. Handle Ctrl-C for graceful shutdown (drones land before exit).

Usage:
    # Activate env first:
    conda activate llm2swarm

    # Default 3-drone mission:
    python main.py

    # Custom mission:
    python main.py --mission "Drone 1 patrol sector A, Drone 2 standby at base"

    # Override number of drones (uses IDs drone_1 … drone_N):
    python main.py --drones 2

    # Use mock controller (no simulator needed):
    SIMULATOR_BACKEND=mock python main.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import time

# Bypass system proxy for localhost (Clash/V2Ray intercepts it otherwise)
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

from dotenv import load_dotenv
load_dotenv()

import config
from memory.pool import SharedMemoryPool
from operators.global_operator import GlobalOperator
from operators.vlm_agent import VLMAgent
from operators.drone_lifecycle import launch_drone

# ── Logging setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-7s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("main")


# ── Default fallback plan (used when OpenAI API is unavailable) ───────────────

def _default_plan(drone_ids: list[str]) -> dict[str, list[dict]]:
    """
    Hard-coded demonstration plan.  Each drone takes off, flies to a unique
    waypoint, does a short search pattern, then hovers.
    Activated automatically when GPT-4o is unavailable.
    """
    waypoints = [
        (30.0,  0.0, 10.0),
        ( 0.0, 30.0, 12.0),
        (-30.0, 0.0, 15.0),
        (0.0, -30.0, 10.0),
    ]
    plan = {}
    for i, did in enumerate(drone_ids):
        wx, wy, wz = waypoints[i % len(waypoints)]
        plan[did] = [
            {"action": "takeoff",        "params": {"altitude": wz}},
            {"action": "go_to_waypoint", "params": {"x": wx, "y": wy, "z": wz,
                                                     "velocity": 4.0}},
            {"action": "search_pattern", "params": {"center_x": wx, "center_y": wy,
                                                     "radius": 10.0, "altitude": wz}},
            {"action": "hover",          "params": {"duration": 5.0}},
        ]
    return plan


# ── Health checks ─────────────────────────────────────────────────────────────

async def _check_ollama() -> bool:
    """Return True if the Ollama endpoint responds within 4 s."""
    import aiohttp
    url = config.EDGE_VLM_BASE_URL.replace("/v1", "") + "/api/tags"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=4)
            ) as resp:
                return resp.status == 200
    except Exception:
        return False


def _start_tunnel_if_needed() -> None:
    """
    Run scripts/start_tunnel.sh in a subprocess to ensure the SSH tunnel is up.
    Only attempted if EDGE_VLM_BASE_URL points to localhost.
    """
    if "localhost" not in config.EDGE_VLM_BASE_URL:
        return  # direct connection, no tunnel needed

    script = os.path.join(os.path.dirname(__file__), "scripts", "start_tunnel.sh")
    if not os.path.isfile(script):
        logger.warning("Tunnel script not found at %s — skipping.", script)
        return

    logger.info("Ensuring SSH tunnel is up …")
    result = subprocess.run(
        ["bash", script],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        # Print only the ✓/✗ summary line
        for line in result.stdout.splitlines():
            if line.strip():
                logger.info("  %s", line.strip())
    else:
        logger.warning("Tunnel script returned non-zero:\n%s", result.stderr)


# ── Plan acquisition ──────────────────────────────────────────────────────────

async def _get_plan(
    mission: str,
    drone_ids: list[str],
) -> dict[str, list[dict]]:
    """
    Try to get a plan from GPT-4o.  Falls back to the default plan on any error.
    """
    if not config.OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set — using default fallback plan.")
        return _default_plan(drone_ids)

    try:
        logger.info("Calling GlobalOperator (%s) …", config.GLOBAL_LLM_MODEL)
        operator = GlobalOperator(drone_ids=drone_ids)
        global_plan = await operator.plan_mission(mission)
        logger.info("GlobalOperator plan received for: %s", list(global_plan.plan.keys()))
        return global_plan.plan
    except Exception as e:
        logger.warning(
            "GlobalOperator failed (%s) — using default fallback plan.", e
        )
        return _default_plan(drone_ids)


# ── Main ──────────────────────────────────────────────────────────────────────

async def run(mission: str, drone_ids: list[str]) -> None:
    """
    Full swarm lifecycle: plan → launch → run until complete or Ctrl-C.
    """
    t_start = time.monotonic()

    # ── 1. Tunnel ──
    _start_tunnel_if_needed()

    # ── 2. Check Ollama ──
    ollama_ok = await _check_ollama()
    if ollama_ok:
        logger.info("✓ Ollama reachable at %s  (model: %s)",
                    config.EDGE_VLM_BASE_URL, config.EDGE_VLM_MODEL)
    else:
        logger.warning(
            "✗ Ollama NOT reachable at %s — VLM calls will fall back to '%s'.",
            config.EDGE_VLM_BASE_URL,
            config.VLM_FALLBACK_DECISION,
        )

    # ── 3. Memory Pool ──
    memory = SharedMemoryPool(drone_ids)
    logger.info("SharedMemoryPool ready for %d drones.", len(drone_ids))

    # ── 4. Global Plan ──
    plan = await _get_plan(mission, drone_ids)

    # Log the plan concisely
    logger.info("─" * 60)
    logger.info("MISSION: %s", mission)
    logger.info("─" * 60)
    for did, tasks in plan.items():
        actions = " → ".join(t["action"] for t in tasks)
        logger.info("  %-10s  %s", did, actions)
    logger.info("─" * 60)

    # ── 5. Shared VLM agent (one client, all drones share it) ──
    vlm = VLMAgent()

    # ── 6. Launch all drones concurrently ──
    logger.info("Launching %d drone lifecycles …", len(drone_ids))

    async def _run_drone(drone_id: str) -> None:
        tasks = plan.get(drone_id, [])
        if not tasks:
            logger.warning("[%s] No tasks assigned — drone will idle.", drone_id)
        await launch_drone(
            drone_id        = drone_id,
            memory          = memory,
            vlm             = vlm,
            initial_tasks   = tasks,
            stop_when_empty = False,   # production: hover and poll VLM indefinitely
        )

    drone_tasks = [asyncio.create_task(_run_drone(did), name=did) for did in drone_ids]

    try:
        await asyncio.gather(*drone_tasks)
    except asyncio.CancelledError:
        # Ctrl-C propagates here
        pass
    finally:
        # Cancel any still-running drones
        for t in drone_tasks:
            if not t.done():
                t.cancel()
        if drone_tasks:
            await asyncio.gather(*drone_tasks, return_exceptions=True)

    elapsed = time.monotonic() - t_start
    logger.info("─" * 60)
    logger.info("Swarm session ended after %.1f s.", elapsed)

    # Final state report
    all_states = await memory.get_all_states()
    logger.info("Final drone states:")
    for did, state in all_states.items():
        pos = [f"{v:.1f}" for v in state.position]
        logger.info("  %-10s  pos=%s  status=%-12s  obs_count=%d",
                    did, pos, state.status, len(state.observations))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM2Swarm — edge-cloud collaborative multi-drone system"
    )
    parser.add_argument(
        "--mission",
        default=(
            "Drone 1 takes off and patrols a 20 m radius circle centred at (40, 40) "
            "at altitude 12 m. "
            "Drone 2 takes off, flies to waypoint (60, 0, 10) and performs a search "
            "pattern with 15 m radius. "
            "Drone 3 takes off to altitude 8 m and acts as a communication relay, "
            "hovering at (0, 0, 8) for 30 seconds."
        ),
        help="Natural-language mission description sent to the Global Operator.",
    )
    parser.add_argument(
        "--drones",
        type=int,
        default=len(config.DRONE_IDS),
        help=f"Number of drones to launch (default: {len(config.DRONE_IDS)}).",
    )
    args = parser.parse_args()

    drone_ids = [f"drone_{i+1}" for i in range(args.drones)]

    try:
        asyncio.run(run(mission=args.mission, drone_ids=drone_ids))
    except KeyboardInterrupt:
        logger.info("Interrupted by user — shutting down.")


if __name__ == "__main__":
    main()
