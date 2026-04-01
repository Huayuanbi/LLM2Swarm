#!/usr/bin/env python3
"""
Multi-drone Webots controller for LLM2Swarm.

Each Webots robot runs this script in its own controller process. A separate
cloud-side preparation step writes the initial global plan into SQLite before
the robots start, while the controllers themselves only consume their assigned
tasks and share live swarm state.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

import config
from controllers.webots_controller import WebotsController
from memory.sqlite_pool import SQLiteSwarmPool
from operators.drone_lifecycle import DroneLifecycle
from operators.vlm_agent import VLMAgent

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-7s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("webots-multi")


def _parse_drone_id() -> str:
    if len(sys.argv) > 1 and sys.argv[1].strip():
        return sys.argv[1].strip()
    return os.getenv("WEBOTS_DEMO_DRONE_ID", "drone_1")


def _parse_swarm_ids() -> list[str]:
    raw = os.getenv("WEBOTS_SWARM_DRONES", "drone_1,drone_2,drone_3")
    return [item.strip() for item in raw.split(",") if item.strip()]


DRONE_ID = _parse_drone_id()
SWARM_IDS = _parse_swarm_ids()
DB_PATH = os.getenv("WEBOTS_SWARM_DB", "/tmp/llm2swarm_webots_swarm.sqlite3")
PLAN_TIMEOUT = float(os.getenv("WEBOTS_PLAN_TIMEOUT", "180"))
MISSION = os.getenv(
    "WEBOTS_SWARM_MISSION",
    (
        "Multi-drone mission. Drone 1 inspects the manor area from the west. "
        "Drone 2 surveys the windmill corridor to the south. Drone 3 patrols "
        "the northern tree line and acts as overwatch. Each drone should take "
        "off first, avoid duplicated coverage, and adapt based on onboard vision."
    ),
)
async def _prepare_plan(pool: SQLiteSwarmPool) -> list[dict]:
    logger.info("[%s] Waiting for cloud planner to publish initial tasks.", DRONE_ID)
    return await pool.wait_for_plan(DRONE_ID, timeout=PLAN_TIMEOUT)


async def _run() -> None:
    logger.info("Starting multi-drone Webots controller for %s", DRONE_ID)
    logger.info("Swarm IDs: %s", SWARM_IDS)
    logger.info("Mission: %s", MISSION)

    pool = SQLiteSwarmPool(DB_PATH, SWARM_IDS)
    initial_tasks = await _prepare_plan(pool)
    logger.info("[%s] Initial tasks: %s", DRONE_ID, " -> ".join(t["action"] for t in initial_tasks))

    vlm = VLMAgent()
    controller = WebotsController(DRONE_ID)
    await controller.connect()

    lifecycle = DroneLifecycle(
        drone_id=DRONE_ID,
        controller=controller,
        memory=pool,
        vlm=vlm,
        initial_tasks=initial_tasks,
        stop_when_empty=False,
    )

    try:
        await lifecycle.run()
    finally:
        await controller.disconnect()


def main() -> None:
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        logger.info("[%s] Interrupted by user.", DRONE_ID)


if __name__ == "__main__":
    main()
