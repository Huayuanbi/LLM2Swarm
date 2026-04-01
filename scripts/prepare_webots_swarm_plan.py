#!/usr/bin/env python3
"""
Prepare the initial multi-drone Webots mission plan before controllers start.

This script represents the cloud-side initialization step:
  1. reset the shared SQLite swarm state,
  2. generate a full per-drone task plan,
  3. write the plan so each drone controller can fetch only its own slice.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

import config
from memory.sqlite_pool import SQLiteSwarmPool
from operators.global_operator import GlobalOperator

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-7s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare-webots-swarm-plan")


def _parse_swarm_ids() -> list[str]:
    raw = os.getenv("WEBOTS_SWARM_DRONES", "drone_1,drone_2,drone_3")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _fallback_plan(drone_ids: list[str]) -> dict[str, list[dict]]:
    defaults = {
        "drone_1": [
            {"action": "takeoff", "params": {"altitude": 12.0}},
            {
                "action": "go_to_waypoint",
                "params": {"x": -35.0, "y": 12.0, "z": 12.0, "velocity": 3.0},
            },
            {
                "action": "search_pattern",
                "params": {
                    "center_x": -35.0,
                    "center_y": 12.0,
                    "radius": 18.0,
                    "altitude": 12.0,
                },
            },
        ],
        "drone_2": [
            {"action": "takeoff", "params": {"altitude": 10.0}},
            {
                "action": "go_to_waypoint",
                "params": {"x": -48.0, "y": -18.0, "z": 10.0, "velocity": 3.0},
            },
            {
                "action": "search_pattern",
                "params": {
                    "center_x": -48.0,
                    "center_y": -18.0,
                    "radius": 16.0,
                    "altitude": 10.0,
                },
            },
        ],
        "drone_3": [
            {"action": "takeoff", "params": {"altitude": 14.0}},
            {
                "action": "go_to_waypoint",
                "params": {"x": -12.0, "y": 28.0, "z": 14.0, "velocity": 3.0},
            },
            {
                "action": "search_pattern",
                "params": {
                    "center_x": -12.0,
                    "center_y": 28.0,
                    "radius": 14.0,
                    "altitude": 14.0,
                },
            },
        ],
    }
    return {did: defaults.get(did, defaults["drone_1"]) for did in drone_ids}


async def _prepare() -> None:
    swarm_ids = _parse_swarm_ids()
    mission = os.getenv(
        "WEBOTS_SWARM_MISSION",
        (
            "Multi-drone mission. Drone 1 inspects the manor area from the west. "
            "Drone 2 surveys the windmill corridor to the south. Drone 3 patrols "
            "the northern tree line and acts as overwatch. Each drone should take "
            "off first, avoid duplicated coverage, and adapt based on onboard vision."
        ),
    )
    db_path = os.getenv("WEBOTS_SWARM_DB", "/tmp/llm2swarm_webots_swarm.sqlite3")

    pool = SQLiteSwarmPool(db_path, swarm_ids)
    await pool.reset_run(mission)

    plan = _fallback_plan(swarm_ids)
    logger.info("Preparing cloud-side initial plan for drones: %s", swarm_ids)
    logger.info("Mission: %s", mission)

    if config.OPENAI_API_KEY or config.GLOBAL_LLM_BASE_URL:
        try:
            operator = GlobalOperator(drone_ids=swarm_ids)
            global_plan = await operator.plan_mission(mission)
            if global_plan.plan:
                plan = global_plan.plan
                logger.info("Cloud planner returned a global plan.")
        except Exception as exc:
            logger.warning("Cloud planner failed (%s); using fallback plan.", exc)
    else:
        logger.info("No planner configured; using fallback plan.")

    await pool.set_plan(plan, mission)
    logger.info("Initial plan written to %s", db_path)
    for did in swarm_ids:
        actions = " -> ".join(task["action"] for task in plan.get(did, []))
        logger.info("  %s: %s", did, actions or "(none)")


def main() -> None:
    asyncio.run(_prepare())


if __name__ == "__main__":
    main()
