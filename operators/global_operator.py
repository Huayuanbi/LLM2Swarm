"""
operators/global_operator.py — Cloud LLM Global Planner (OpenAI GPT-4o).

Responsibilities:
  - Accept a natural-language mission description.
  - Call GPT-4o with a strict system prompt that enforces JSON output.
  - Parse and validate the response into a GlobalPlan Pydantic model.
  - Return per-drone task lists ready for the lifecycle loops.

Retry logic:
  - Up to MAX_RETRIES attempts with exponential back-off.
  - On each retry, the previous malformed response is fed back to the model
    as a correction prompt so it can self-fix its output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Callable, Optional

import config
from models.schemas import AgentProfile, DroneState, GlobalPlan, GlobalRolePlan
from primitives.registry import format_primitives_for_prompt, list_registered_primitives
from utils.debug_gate import DEBUG_GATE
from utils.openai_client import build_async_openai_client, should_bypass_env_proxy

logger = logging.getLogger(__name__)

MAX_RETRIES      = config.GLOBAL_LLM_MAX_RETRIES
RETRY_BASE_DELAY = config.GLOBAL_LLM_RETRY_BASE_DELAY


# ─── System prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = f"""\
You are the Global Mission Planner for an autonomous multi-drone swarm system.

Your job is to translate a natural-language mission into a structured JSON task plan.

OUTPUT FORMAT — you must return ONLY valid JSON with no markdown, no code fences,
no commentary. The structure is:

{{
  "drone_1": [
    {{"action": "<skill_name>", "params": {{<param_key>: <value>, ...}}}},
    ...
  ],
  "drone_2": [...],
  ...
}}

AVAILABLE SKILLS and their required params:
{format_primitives_for_prompt(list_registered_primitives())}

RULES:
  1. Every drone must begin its task list with a takeoff action.
  2. Assign tasks only to drones listed in the mission context.
  3. Use realistic coordinates (metres, NED frame). Keep values under 200 m.
  4. Before every search_pattern action, always insert a go_to_waypoint to
     (center_x, center_y, altitude) so the drone explicitly flies to the
     search area first — do not assume it is already there.
  5. Return ONLY the JSON object — nothing else.
"""

_ROLE_SYSTEM_PROMPT = """\
You are the Cloud Mission Planner for an autonomous multi-drone swarm system.

Your job is to translate a coarse natural-language mission into per-drone role
briefs. You may use each drone's initial state (position, battery, status) to
decide how to split responsibility.

OUTPUT FORMAT — return ONLY valid JSON with no markdown, no code fences,
no commentary. The structure is:

{
  "drone_1": {
    "mission_role": "<short role name>",
    "mission_intent": "<why this agent exists in the mission>",
    "responsibilities": ["<responsibility>", "..."],
    "constraints": ["<hard rule or exclusion>", "..."],
    "coordination_contracts": ["<peer coordination rule>", "..."],
    "capability_requirements": ["<desired capability>", "..."],
    "capability_exclusions": ["<capability this role should avoid depending on>", "..."],
    "resource_requirements": ["<resource or payload that should exist>", "..."],
    "resource_permissions": ["<resource this role may use>", "..."],
    "success_criteria": ["<what counts as success>", "..."],
    "handoff_conditions": ["<when to escalate, reassign, or hand off>", "..."],
    "event_watchlist": ["<important event type>", "..."],
    "shared_context": {
      "<context_key>": "<context_value or nested object>"
    },
    "initial_hints": ["<non-binding local planning hint>", "..."],
    "metadata": {
      "<optional_key>": "<optional_value>"
    }
  },
  "drone_2": {...}
}

RULES:
  1. Assign a role brief only to drones listed in the mission context.
  2. Do NOT output a full primitive action list or detailed route plan.
  3. Keep the schema generic and reusable across different agent types and tasks.
  4. Put mission-specific spatial or semantic details inside shared_context, not
     in top-level task-specific fields.
  5. Use each agent's declared capabilities/resources to decide which roles are feasible.
  6. capability_requirements and capability_exclusions should reflect what the role
     needs or should avoid depending on.
  7. coordination_contracts should explicitly reduce duplicated work.
  8. event_watchlist should mention likely important events such as battery_low,
     peer_lost, target_detected, obstacle_detected, or region_complete when relevant.
  9. Return ONLY the JSON object — nothing else.
"""


class GlobalOperator:
    """
    Wraps the OpenAI async client and exposes a single high-level method:
        plan = await operator.plan_mission("Drone 1 patrol sector A …")
    """

    def __init__(self, drone_ids: Optional[list[str]] = None):
        self._drone_ids = drone_ids or config.DRONE_IDS
        self._client = build_async_openai_client(
            api_key=config.GLOBAL_LLM_API_KEY or "nokey",
            base_url=config.GLOBAL_LLM_BASE_URL or None,
            max_retries=0,   # disable SDK-level retries; our loop handles them
        )
        if should_bypass_env_proxy(config.GLOBAL_LLM_BASE_URL):
            logger.info(
                "GlobalOperator: bypassing environment HTTP proxy for local/private base URL %s",
                config.GLOBAL_LLM_BASE_URL,
            )

    async def plan_mission(self, mission_description: str) -> GlobalPlan:
        """
        Convert a free-text mission description into a validated GlobalPlan.

        Args:
            mission_description: Natural-language task for the swarm.

        Returns:
            GlobalPlan with task lists for each drone.

        Raises:
            RuntimeError: If the LLM fails to produce valid JSON after all retries.
        """
        user_message = (
            f"Active drones: {', '.join(self._drone_ids)}\n\n"
            f"Mission: {mission_description}"
        )

        messages = [
            {"role": "system",  "content": _SYSTEM_PROMPT},
            {"role": "user",    "content": user_message},
        ]

        while True:
            await DEBUG_GATE.checkpoint(
                "cloud_request",
                {
                    "mode": "plan_mission",
                    "model": config.GLOBAL_LLM_MODEL,
                    "mission": mission_description,
                    "messages": messages,
                },
                actor_id="cloud",
                summary="Inspect cloud action-plan request before sending.",
            )
            plan = await self._run_json_planner(
                messages=messages,
                parser=lambda raw: _parse_plan(raw, self._drone_ids),
                parse_target="plan",
            )
            command = await DEBUG_GATE.checkpoint(
                "cloud_response",
                {
                    "mode": "plan_mission",
                    "model": config.GLOBAL_LLM_MODEL,
                    "mission": mission_description,
                    "parsed_plan": plan.model_dump(),
                },
                actor_id="cloud",
                allow_regenerate=True,
                summary="Inspect parsed cloud action-plan output.",
            )
            if command == "regenerate":
                logger.info("GlobalOperator: regenerating action plan due to debug gate request.")
                continue
            logger.info(
                "GlobalOperator: plan created for drones %s",
                list(plan.plan.keys()),
            )
            return plan

    async def assign_roles(
        self,
        mission_description: str,
        initial_states: Optional[dict[str, DroneState | dict]] = None,
        agent_profiles: Optional[dict[str, AgentProfile | dict]] = None,
    ) -> GlobalRolePlan:
        """
        Convert a coarse mission into per-drone role briefs.

        Args:
            mission_description: Natural-language objective for the swarm.
            initial_states: Optional per-drone initial state snapshot. These do
                not need to be perfect; even approximate position / battery
                helps the cloud planner produce better role allocation.
        """
        state_context = _build_initial_state_context(self._drone_ids, initial_states)
        profile_context = _build_agent_profile_context(self._drone_ids, agent_profiles)
        user_message = (
            f"Active drones: {', '.join(self._drone_ids)}\n\n"
            f"Initial states:\n{state_context}\n\n"
            f"Agent profiles:\n{profile_context}\n\n"
            f"Mission: {mission_description}"
        )
        messages = [
            {"role": "system", "content": _ROLE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        while True:
            await DEBUG_GATE.checkpoint(
                "cloud_request",
                {
                    "mode": "assign_roles",
                    "model": config.GLOBAL_LLM_MODEL,
                    "mission": mission_description,
                    "initial_states": initial_states,
                    "agent_profiles": agent_profiles,
                    "messages": messages,
                },
                actor_id="cloud",
                summary="Inspect cloud role-allocation request before sending.",
            )
            roles = await self._run_json_planner(
                messages=messages,
                parser=lambda raw: _parse_role_plan(raw, self._drone_ids),
                validator=lambda plan: _validate_role_plan(plan, self._drone_ids, agent_profiles),
                parse_target="role plan",
            )
            command = await DEBUG_GATE.checkpoint(
                "cloud_response",
                {
                    "mode": "assign_roles",
                    "model": config.GLOBAL_LLM_MODEL,
                    "mission": mission_description,
                    "parsed_roles": roles.model_dump(),
                },
                actor_id="cloud",
                allow_regenerate=True,
                summary="Inspect parsed cloud role-allocation output.",
            )
            if command == "regenerate":
                logger.info("GlobalOperator: regenerating role briefs due to debug gate request.")
                continue
            logger.info(
                "GlobalOperator: role briefs created for drones %s",
                list(roles.roles.keys()),
            )
            return roles

    async def _run_json_planner(
        self,
        *,
        messages: list[dict[str, str]],
        parser: Callable[[str], object],
        validator: Optional[Callable[[object], object]] = None,
        parse_target: str,
    ):
        """
        Shared retry + parse loop used by both action-plan and role-brief modes.
        """
        working_messages = list(messages)
        last_error: Optional[str] = None
        raw_content = ""

        for attempt in range(1, MAX_RETRIES + 1):
            if last_error:
                working_messages.append({
                    "role": "user",
                    "content": (
                        f"Your previous response was invalid. Error: {last_error}\n"
                        "Please return ONLY a valid JSON object matching the required schema."
                    ),
                })

            try:
                logger.info(
                    "GlobalOperator: calling %s (attempt %d/%d)",
                    config.GLOBAL_LLM_MODEL, attempt, MAX_RETRIES,
                )
                extra = {}
                if not config.GLOBAL_LLM_BASE_URL:
                    extra["response_format"] = {"type": "json_object"}

                response = await self._client.chat.completions.create(
                    model=config.GLOBAL_LLM_MODEL,
                    messages=working_messages,
                    temperature=0.2,
                    timeout=config.GLOBAL_LLM_TIMEOUT,
                    **extra,
                )

                raw_content = response.choices[0].message.content or ""
                logger.debug("GlobalOperator raw %s response:\n%s", parse_target, raw_content)
                parsed = parser(raw_content)
                if validator is not None:
                    validator(parsed)
                return parsed

            except (json.JSONDecodeError, ValueError) as e:
                last_error = str(e)
                logger.warning(
                    "GlobalOperator: parse error on attempt %d: %s", attempt, last_error
                )
                working_messages.append({
                    "role": "assistant",
                    "content": raw_content,
                })

            except Exception as e:
                last_error = str(e)
                if last_error == "Connection error." and config.GLOBAL_LLM_BASE_URL:
                    logger.error(
                        "GlobalOperator: API error on attempt %d: %s "
                        "(base_url=%s; check SSH tunnel and local proxy settings)",
                        attempt,
                        e,
                        config.GLOBAL_LLM_BASE_URL,
                    )
                else:
                    logger.error("GlobalOperator: API error on attempt %d: %s", attempt, e)

            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.info("GlobalOperator: retrying in %.1f s …", delay)
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"GlobalOperator failed to produce a valid {parse_target} after {MAX_RETRIES} "
            f"attempts. Last error: {last_error}"
        )


# ─── Parsing helpers ──────────────────────────────────────────────────────────

def _parse_plan(raw: str, drone_ids: list[str]) -> GlobalPlan:
    """
    Parse raw LLM text into a validated GlobalPlan.

    Handles two common LLM quirks:
      1. JSON wrapped in ```json ... ``` markdown fences.
      2. Extra commentary before/after the JSON object.
    """
    # Strip thinking blocks (qwen3 thinking mode)
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip()

    # Extract the outermost {...} block in case there's surrounding text
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {cleaned!r}")

    data = json.loads(match.group())

    # Validate with Pydantic (accepts both flat dict and {"plan": {...}} shapes)
    plan = GlobalPlan.model_validate(data)

    # Warn if the LLM omitted some drones
    missing = [did for did in drone_ids if did not in plan.plan]
    if missing:
        logger.warning(
            "GlobalOperator: LLM did not assign tasks to %s. "
            "They will sit idle until next replanning.",
            missing,
        )

    return plan


def _parse_role_plan(raw: str, drone_ids: list[str]) -> GlobalRolePlan:
    """
    Parse raw LLM text into a validated GlobalRolePlan.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip()

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {cleaned!r}")

    data = json.loads(match.group())
    plan = GlobalRolePlan.model_validate(data)

    missing = [did for did in drone_ids if did not in plan.roles]
    if missing:
        logger.warning(
            "GlobalOperator: LLM did not assign role briefs to %s. "
            "They will use local fallbacks.",
            missing,
        )

    return plan


def _validate_role_plan(
    plan: GlobalRolePlan,
    drone_ids: list[str],
    agent_profiles: Optional[dict[str, AgentProfile | dict]],
) -> None:
    """
    Validate that each generated RoleBrief is actually feasible for the target
    agent profile before it is handed to onboard planners.

    This shifts the primary correction loop to the cloud stage so we can feed
    invalid role assignments back to the cloud planner for regeneration instead
    of letting them degrade into empty onboard graphs.
    """
    profiles = _normalise_agent_profiles(drone_ids, agent_profiles)
    if not profiles:
        return

    errors: list[str] = []
    for drone_id, role_brief in plan.roles.items():
        profile = profiles.get(drone_id)
        if profile is None:
            continue

        available_capabilities = set(profile.available_capabilities)
        available_resources = set(profile.available_resources)

        missing_capabilities = set(role_brief.capability_requirements) - available_capabilities
        if missing_capabilities:
            errors.append(
                f"{drone_id}: role '{role_brief.mission_role}' requires unavailable capabilities "
                f"{sorted(missing_capabilities)}"
            )

        contradictory_exclusions = set(role_brief.capability_requirements) & set(role_brief.capability_exclusions)
        if contradictory_exclusions:
            errors.append(
                f"{drone_id}: role '{role_brief.mission_role}' both requires and excludes capabilities "
                f"{sorted(contradictory_exclusions)}"
            )

        missing_resources = set(role_brief.resource_requirements) - available_resources
        if missing_resources:
            errors.append(
                f"{drone_id}: role '{role_brief.mission_role}' requires unavailable resources "
                f"{sorted(missing_resources)}"
            )

    if errors:
        raise ValueError(
            "Role plan violates agent-profile constraints:\n- " + "\n- ".join(errors)
        )


def _build_initial_state_context(
    drone_ids: list[str],
    initial_states: Optional[dict[str, DroneState | dict]],
) -> str:
    if not initial_states:
        return "\n".join(f"  {drone_id}: position=(unknown) battery=(unknown) status=idle" for drone_id in drone_ids)

    lines = []
    for drone_id in drone_ids:
        raw = initial_states.get(drone_id) if initial_states else None
        state = raw if isinstance(raw, DroneState) else None
        if state is None and isinstance(raw, dict):
            try:
                state = DroneState.model_validate({"drone_id": drone_id, **raw})
            except Exception:
                state = None

        if state is None:
            lines.append(f"  {drone_id}: position=(unknown) battery=(unknown) status=idle")
            continue

        px, py, pz = state.position
        battery = "unknown" if state.battery_level is None else f"{state.battery_level:.2f}"
        lines.append(
            f"  {drone_id}: position=({px:.1f},{py:.1f},{pz:.1f}) "
            f"battery={battery} status={state.status}"
        )
    return "\n".join(lines)


def _build_agent_profile_context(
    drone_ids: list[str],
    agent_profiles: Optional[dict[str, AgentProfile | dict]],
) -> str:
    profiles = _normalise_agent_profiles(drone_ids, agent_profiles)
    if not profiles:
        return "\n".join(
            f"  {drone_id}: kind=(unknown) capabilities=(unknown) primitives=(unknown) resources=(unknown)"
            for drone_id in drone_ids
        )

    lines = []
    for drone_id in drone_ids:
        profile = profiles.get(drone_id)
        if profile is None:
            lines.append(
                f"  {drone_id}: kind=(unknown) capabilities=(unknown) primitives=(unknown) resources=(unknown)"
            )
            continue

        primitive_names = [primitive.name for primitive in profile.available_primitives]
        lines.append(
            f"  {drone_id}: kind={profile.agent_kind} "
            f"capabilities={profile.available_capabilities or ['(none)']} "
            f"primitives={primitive_names or ['(none)']} "
            f"resources={profile.available_resources or ['(none)']}"
        )
    return "\n".join(lines)


def _normalise_agent_profiles(
    drone_ids: list[str],
    agent_profiles: Optional[dict[str, AgentProfile | dict]],
) -> dict[str, AgentProfile]:
    if not agent_profiles:
        return {}

    profiles: dict[str, AgentProfile] = {}
    for drone_id in drone_ids:
        raw = agent_profiles.get(drone_id)
        profile = raw if isinstance(raw, AgentProfile) else None
        if profile is None and isinstance(raw, dict):
            try:
                profile = AgentProfile.model_validate({"agent_id": drone_id, **raw})
            except Exception:
                profile = None
        if profile is not None:
            profiles[drone_id] = profile
    return profiles
