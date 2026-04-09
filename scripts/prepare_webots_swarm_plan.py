#!/usr/bin/env python3
"""
Prepare the initial multi-drone Webots role allocation before controllers start.

This script represents the cloud-side initialization step:
  1. reset the shared SQLite swarm state,
  2. generate a full per-drone role allocation,
  3. write the briefs so each local controller can synthesize its own task graph.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
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
from models.schemas import AgentProfile, RoleBrief
from operators.global_operator import GlobalOperator
from primitives import build_agent_profile

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-7s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare-webots-swarm-plan")


def _parse_swarm_ids() -> list[str]:
    raw = os.getenv("WEBOTS_SWARM_DRONES", "drone_1,drone_2,drone_3")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_initial_states_env(drone_ids: list[str]) -> dict[str, dict]:
    raw = os.getenv("WEBOTS_SWARM_INITIAL_STATES", "").strip()
    if not raw:
        return {}

    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("WEBOTS_SWARM_INITIAL_STATES must be a JSON object keyed by drone_id.")

    states: dict[str, dict] = {}
    for drone_id in drone_ids:
        entry = data.get(drone_id)
        if not isinstance(entry, dict):
            continue
        state = dict(entry)
        state.setdefault("status", "idle")
        states[drone_id] = state
    return states


def _parse_world_initial_states(world_file: Path, drone_ids: list[str]) -> dict[str, dict]:
    if not world_file.exists():
        return {}

    content = world_file.read_text(encoding="utf-8")
    blocks: list[str] = []
    lines = content.splitlines()
    index = 0

    while index < len(lines):
        line = lines[index]
        if "{" not in line:
            index += 1
            continue

        depth = 0
        block_lines: list[str] = []
        while index < len(lines):
            current = lines[index]
            block_lines.append(current)
            depth += current.count("{")
            depth -= current.count("}")
            index += 1
            if depth == 0:
                break
        blocks.append("\n".join(block_lines))

    states: dict[str, dict] = {}
    for block in blocks:
        name_match = re.search(r'^\s*name\s+"([^"]+)"\s*$', block, re.MULTILINE)
        if not name_match:
            continue
        drone_id = name_match.group(1)
        if drone_id not in drone_ids:
            continue

        translation_match = re.search(
            r'^\s*translation\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$',
            block,
            re.MULTILINE,
        )
        if not translation_match:
            continue

        states[drone_id] = {
            "position": [
                float(translation_match.group(1)),
                float(translation_match.group(2)),
                float(translation_match.group(3)),
            ],
            "status": "idle",
        }
    return states


def _collect_initial_states(drone_ids: list[str]) -> dict[str, dict]:
    try:
        env_states = _parse_initial_states_env(drone_ids)
    except Exception as exc:
        logger.warning("Failed to parse WEBOTS_SWARM_INITIAL_STATES (%s); ignoring it.", exc)
        env_states = {}

    if env_states:
        logger.info("Using externally supplied initial states for cloud role allocation.")
        return env_states

    world_file = Path(
        os.getenv(
            "WEBOTS_SWARM_WORLD_FILE",
            str(REPO_ROOT / "worlds" / "mavic_2_pro_swarm_demo.wbt"),
        )
    )
    world_states = _parse_world_initial_states(world_file, drone_ids)
    if world_states:
        logger.info("Using initial states parsed from %s for cloud role allocation.", world_file)
    else:
        logger.info("No initial states available for cloud role allocation; planner will see unknown states.")
    return world_states


def _parse_agent_profiles_env(drone_ids: list[str]) -> dict[str, AgentProfile]:
    raw = os.getenv("WEBOTS_SWARM_AGENT_PROFILES", "").strip()
    if not raw:
        return {}

    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("WEBOTS_SWARM_AGENT_PROFILES must be a JSON object keyed by drone_id.")

    profiles: dict[str, AgentProfile] = {}
    for drone_id in drone_ids:
        entry = data.get(drone_id)
        if not isinstance(entry, dict):
            continue
        profiles[drone_id] = AgentProfile.model_validate({"agent_id": drone_id, **entry})
    return profiles


def _collect_agent_profiles(drone_ids: list[str]) -> dict[str, AgentProfile]:
    try:
        env_profiles = _parse_agent_profiles_env(drone_ids)
    except Exception as exc:
        logger.warning("Failed to parse WEBOTS_SWARM_AGENT_PROFILES (%s); ignoring it.", exc)
        env_profiles = {}

    if env_profiles:
        logger.info("Using externally supplied agent profiles for cloud role allocation.")
        return env_profiles

    profiles = {
        drone_id: build_agent_profile(
            agent_id=drone_id,
            agent_kind="webots_mavic2pro",
        )
        for drone_id in drone_ids
    }
    logger.info("Using default Webots demo agent profiles for cloud role allocation.")
    return profiles


def _fallback_roles(drone_ids: list[str]) -> dict[str, dict]:
    defaults = {
        "drone_1": RoleBrief(
            mission_role="West sector lead",
            mission_intent="Own the western mission responsibility and maintain shared awareness there.",
            responsibilities=[
                "maintain coverage of the western sector",
                "publish meaningful detections and anomalies",
            ],
            coordination_contracts=[
                "avoid duplicating the southern sector workload owned by drone_2",
            ],
            event_watchlist=[
                "peer_lost",
                "battery_low",
                "target_detected",
            ],
            initial_hints=[
                "bootstrap from the west sector context and keep the first local graph small",
            ],
            shared_context={
                "sector_label": "west manor sector",
                "bootstrap_actions": [
                    {"action": "takeoff", "params": {"altitude": 30.0}},
                    {"action": "go_to_waypoint", "params": {"x": -50.0, "y": 0.0, "z": 30.0, "velocity": 5.0}},
                    {"action": "search_pattern", "params": {"center_x": -50.0, "center_y": 0.0, "radius": 40.0, "altitude": 30.0}},
                ],
            },
        ).model_dump(),
        "drone_2": RoleBrief(
            mission_role="South corridor owner",
            mission_intent="Own the southern corridor responsibility and keep overlap low.",
            responsibilities=[
                "maintain coverage of the southern corridor",
                "share high-value observations with the swarm",
            ],
            coordination_contracts=[
                "stay south of the west-sector lead unless reassigned",
            ],
            event_watchlist=[
                "peer_lost",
                "battery_low",
                "target_detected",
            ],
            initial_hints=[
                "bootstrap from the south corridor context and preserve spacing from peers",
            ],
            shared_context={
                "sector_label": "south corridor",
                "bootstrap_actions": [
                    {"action": "takeoff", "params": {"altitude": 30.0}},
                    {"action": "go_to_waypoint", "params": {"x": 0.0, "y": -80.0, "z": 30.0, "velocity": 5.0}},
                    {"action": "search_pattern", "params": {"center_x": 0.0, "center_y": -80.0, "radius": 50.0, "altitude": 30.0}},
                ],
            },
        ).model_dump(),
        "drone_3": RoleBrief(
            mission_role="North sector observer",
            mission_intent="Maintain broad situational awareness over the northern sector.",
            responsibilities=[
                "observe the northern sector",
                "publish cross-checkable observations for the swarm",
            ],
            coordination_contracts=[
                "maintain separation from the lower-altitude sectors when possible",
            ],
            event_watchlist=[
                "peer_lost",
                "battery_low",
                "target_detected",
            ],
            initial_hints=[
                "bootstrap from the northern context with higher altitude separation",
            ],
            shared_context={
                "sector_label": "north tree line",
                "bootstrap_actions": [
                    {"action": "takeoff", "params": {"altitude": 40.0}},
                    {"action": "go_to_waypoint", "params": {"x": 0.0, "y": 60.0, "z": 40.0, "velocity": 5.0}},
                    {"action": "search_pattern", "params": {"center_x": 0.0, "center_y": 60.0, "radius": 60.0, "altitude": 40.0}},
                ],
            },
        ).model_dump(),
    }
    return {did: defaults.get(did, defaults["drone_1"]) for did in drone_ids}


async def _prepare() -> None:
    swarm_ids = _parse_swarm_ids()
    mission = os.getenv(
        "WEBOTS_SWARM_MISSION",
        "search the area for fire",
    )
    db_path = os.getenv("WEBOTS_SWARM_DB", "/tmp/llm2swarm_webots_swarm.sqlite3")

    pool = SQLiteSwarmPool(db_path, swarm_ids)
    await pool.reset_run(mission)
    initial_states = _collect_initial_states(swarm_ids)
    agent_profiles = _collect_agent_profiles(swarm_ids)

    roles = _fallback_roles(swarm_ids)
    logger.info("Preparing cloud-side role allocation for drones: %s", swarm_ids)
    logger.info("Mission: %s", mission)

    if config.OPENAI_API_KEY or config.GLOBAL_LLM_BASE_URL:
        try:
            operator = GlobalOperator(drone_ids=swarm_ids)
            role_plan = await operator.assign_roles(
                mission,
                initial_states=initial_states or None,
                agent_profiles=agent_profiles,
            )
            if role_plan.roles:
                roles = {did: brief.model_dump() for did, brief in role_plan.roles.items()}
                logger.info("Cloud planner returned per-drone role briefs.")
        except Exception as exc:
            logger.warning("Cloud planner failed (%s); using fallback roles.", exc)
    else:
        logger.info("No planner configured; using fallback role briefs.")

    await pool.set_role_briefs(roles, mission)
    logger.info("Initial role briefs written to %s", db_path)
    for did in swarm_ids:
        role = RoleBrief.model_validate(roles[did])
        logger.info("  %s role: %s", did, role.mission_role)
        logger.info("  %s intent: %s", did, role.mission_intent)
        logger.info("  %s hints: %s", did, role.initial_hints or ["(none)"])


def main() -> None:
    asyncio.run(_prepare())


if __name__ == "__main__":
    main()
