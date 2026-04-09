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
from models.schemas import DroneState, OnboardPlanningContext, RoleBrief
from operators.drone_lifecycle import DroneLifecycle
from operators.local_planner import build_task_graph_runtime
from operators.onboard_planner import OnboardPlannerAgent
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
    "search the area for fire",
)
async def _prepare_role(pool: SQLiteSwarmPool) -> RoleBrief:
    logger.info("[%s] Waiting for cloud planner to publish initial role brief.", DRONE_ID)
    raw_role = await pool.wait_for_role_brief(DRONE_ID, timeout=PLAN_TIMEOUT)
    return RoleBrief.model_validate(raw_role)


async def _run() -> None:
    logger.info("Starting multi-drone Webots controller for %s", DRONE_ID)
    logger.info("Swarm IDs: %s", SWARM_IDS)
    logger.info("Mission: %s", MISSION)

    pool = SQLiteSwarmPool(DB_PATH, SWARM_IDS)
    role = await _prepare_role(pool)
    logger.info("[%s] Role brief: %s", DRONE_ID, role.mission_role)

    vlm = VLMAgent()
    onboard_planner = OnboardPlannerAgent()
    controller = WebotsController(DRONE_ID)
    await controller.connect()
    pos = await controller.get_position()
    await pool.update_drone(DRONE_ID, position=list(pos), status="idle")
    peer_states = await pool.get_peer_states(DRONE_ID)
    planning_response = await onboard_planner.build_initial_task_graph(
        OnboardPlanningContext(
            drone_id=DRONE_ID,
            role_brief=role,
            self_state=DroneState(drone_id=DRONE_ID, position=list(pos), status="idle"),
            peer_states=peer_states,
            active_claims=await pool.get_claims(),
            active_events=await pool.get_events(),
            agent_profile=controller.get_agent_profile(),
            available_primitives=controller.get_available_primitive_specs(),
            available_capabilities=controller.get_capability_tags(),
        )
    )
    task_graph = build_task_graph_runtime(
        DRONE_ID,
        role,
        planning_response.task_graph,
        agent_profile=controller.get_agent_profile(),
    )
    logger.info("[%s] Initial task graph: %s", DRONE_ID, planning_response.task_graph.summary or planning_response.task_graph.graph_id)
    logger.info("[%s] Bootstrap actions: %s", DRONE_ID, " -> ".join(t["action"] for t in task_graph.preview_actions()) or "(none)")

    lifecycle = DroneLifecycle(
        drone_id=DRONE_ID,
        controller=controller,
        memory=pool,
        vlm=vlm,
        task_graph=task_graph,
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
