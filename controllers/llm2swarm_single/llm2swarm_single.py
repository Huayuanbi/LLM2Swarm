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
from models.schemas import DroneState, OnboardPlanningContext, RoleBrief
from operators.drone_lifecycle import DroneLifecycle
from operators.global_operator import GlobalOperator
from operators.local_planner import build_task_graph_runtime
from operators.onboard_planner import OnboardPlannerAgent
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


def _fallback_role() -> RoleBrief:
    return RoleBrief(
        mission_role="Single-drone perimeter inspection",
        mission_intent=(
            "Inspect the rural manor perimeter, adapt to local observations, "
            "and keep the shared memory updated with relevant findings."
        ),
        responsibilities=[
            "inspect the assigned perimeter zone",
            "report meaningful scene changes",
        ],
        coordination_contracts=[
            "publish notable detections immediately to the memory pool",
        ],
        event_watchlist=[
            "obstacle_detected",
            "target_detected",
            "region_complete",
        ],
        initial_hints=[
            "begin with a safe takeoff and move toward the manor perimeter context if available",
            "keep the initial graph small and rely on runtime replanning",
        ],
        shared_context={
            "area_label": "manor perimeter",
            "bootstrap_actions": [
                {"action": "takeoff", "params": {"altitude": 12.0}},
                {"action": "go_to_waypoint", "params": {"x": -35.0, "y": 10.0, "z": 12.0, "velocity": 3.0}},
                {"action": "search_pattern", "params": {"center_x": -35.0, "center_y": 10.0, "radius": 20.0, "altitude": 12.0}},
            ],
        },
    )


async def _get_initial_role(
    drone_id: str,
    mission: str,
    *,
    initial_state: dict,
    agent_profile: dict,
) -> RoleBrief:
    if not (config.OPENAI_API_KEY or config.GLOBAL_LLM_BASE_URL):
        logger.info("No global planner configured; using fallback role brief.")
        return _fallback_role()

    try:
        operator = GlobalOperator(drone_ids=[drone_id])
        role_plan = await operator.assign_roles(
            mission,
            initial_states={drone_id: initial_state},
            agent_profiles={drone_id: agent_profile},
        )
        role = role_plan.get_role(drone_id)
        if role is not None:
            logger.info("Initial role brief received from GlobalOperator.")
            return role
        logger.warning("GlobalOperator returned no role for %s; using fallback role.", drone_id)
    except Exception as exc:
        logger.warning("GlobalOperator failed (%s); using fallback role.", exc)

    return _fallback_role()


async def _run_demo() -> None:
    logger.info("Starting single-drone Webots demo for %s", DRONE_ID)
    logger.info("Mission: %s", DEFAULT_MISSION)

    memory = SharedMemoryPool([DRONE_ID])
    vlm = VLMAgent()
    onboard_planner = OnboardPlannerAgent()
    controller = WebotsController(DRONE_ID)
    await controller.connect()
    pos = await controller.get_position()
    await memory.update_drone(DRONE_ID, position=list(pos), status="idle")
    role = await _get_initial_role(
        DRONE_ID,
        DEFAULT_MISSION,
        initial_state={
            "position": list(pos),
            "status": "idle",
        },
        agent_profile=controller.get_agent_profile().model_dump(),
    )
    planning_response = await onboard_planner.build_initial_task_graph(
        OnboardPlanningContext(
            drone_id=DRONE_ID,
            role_brief=role,
            self_state=DroneState(drone_id=DRONE_ID, position=list(pos), status="idle"),
            peer_states={},
            active_claims=[],
            active_events=[],
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

    logger.info("Role brief: %s", role.mission_role)
    logger.info("Initial task graph: %s", planning_response.task_graph.summary or planning_response.task_graph.graph_id)
    logger.info("Bootstrap actions: %s", " -> ".join(t["action"] for t in task_graph.preview_actions()) or "(none)")

    lifecycle = DroneLifecycle(
        drone_id=DRONE_ID,
        controller=controller,
        memory=memory,
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
        asyncio.run(_run_demo())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")


if __name__ == "__main__":
    main()
