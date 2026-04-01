#!/usr/bin/env python3
"""
Single-drone Webots demo controller for LLM2Swarm.

This script is meant to be launched directly by Webots as the robot controller.
It wires together:
  - WebotsController  → low-level flight loop + camera capture
  - GlobalOperator    → optional initial task planning
  - DroneLifecycle    → action execution + periodic VLM ticks
  - VLMAgent          → image-based local replanning

Environment variables:
  WEBOTS_DEMO_DRONE_ID   default: drone_1
  WEBOTS_DEMO_MISSION    natural-language mission string
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
from memory.pool import SharedMemoryPool
from operators.drone_lifecycle import DroneLifecycle
from operators.global_operator import GlobalOperator
from operators.vlm_agent import VLMAgent

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-7s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("webots-single")

DRONE_ID = os.getenv("WEBOTS_DEMO_DRONE_ID", "drone_1")
DEFAULT_MISSION = os.getenv(
    "WEBOTS_DEMO_MISSION",
    (
        "Single drone mission. Take off to 12 meters, inspect the rural area "
        "around the manor with a 20 meter radius search pattern, and adapt if "
        "the onboard vision model sees an obstacle or a better target."
    ),
)


def _fallback_tasks() -> list[dict]:
    return [
        {"action": "takeoff", "params": {"altitude": 12.0}},
        {
            "action": "go_to_waypoint",
            "params": {"x": -35.0, "y": 10.0, "z": 12.0, "velocity": 3.0},
        },
        {
            "action": "search_pattern",
            "params": {
                "center_x": -35.0,
                "center_y": 10.0,
                "radius": 20.0,
                "altitude": 12.0,
            },
        },
    ]


async def _get_initial_tasks(drone_id: str, mission: str) -> list[dict]:
    if not (config.OPENAI_API_KEY or config.GLOBAL_LLM_BASE_URL):
        logger.info("No global planner configured; using fallback task list.")
        return _fallback_tasks()

    try:
        operator = GlobalOperator(drone_ids=[drone_id])
        plan = await operator.plan_mission(mission)
        tasks = plan.get_tasks(drone_id)
        if tasks:
            logger.info("Initial task list received from GlobalOperator.")
            return tasks
        logger.warning("GlobalOperator returned no tasks for %s; using fallback plan.", drone_id)
    except Exception as exc:
        logger.warning("GlobalOperator failed (%s); using fallback plan.", exc)

    return _fallback_tasks()


async def _run_demo() -> None:
    logger.info("Starting single-drone Webots demo for %s", DRONE_ID)
    logger.info("Mission: %s", DEFAULT_MISSION)

    memory = SharedMemoryPool([DRONE_ID])
    vlm = VLMAgent()
    controller = WebotsController(DRONE_ID)
    initial_tasks = await _get_initial_tasks(DRONE_ID, DEFAULT_MISSION)

    logger.info("Initial tasks: %s", " -> ".join(t["action"] for t in initial_tasks))

    await controller.connect()
    lifecycle = DroneLifecycle(
        drone_id=DRONE_ID,
        controller=controller,
        memory=memory,
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
        asyncio.run(_run_demo())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")


if __name__ == "__main__":
    main()
