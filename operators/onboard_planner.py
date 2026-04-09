"""
operators/onboard_planner.py — Onboard task-graph synthesizer interface.

This module is the abstraction boundary between:
  - a generic RoleBrief from the cloud
  - an onboard model / agent that proposes an executable task graph
  - the local runtime that validates and executes primitive nodes

The planner can later be swapped from a plain model call to a richer agent
framework without changing the controller/lifecycle call sites.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Optional

import config
from models.schemas import (
    OnboardPlanningContext,
    OnboardPlanningResponse,
    TaskGraphSpec,
)
from operators.local_planner import build_task_graph_runtime, compatibility_task_graph_from_role_brief
from primitives.registry import format_primitives_for_prompt
from utils.debug_gate import DEBUG_GATE
from utils.openai_client import build_async_openai_client

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You are the onboard task-graph synthesizer for one autonomous agent in a multi-agent system.

Input:
  - a generic RoleBrief from the cloud
  - the agent's declared capability/resource profile
  - the agent's current state
  - peer states
  - active shared claims
  - active shared events

Your job is to propose an INITIAL executable task graph for this single agent.

OUTPUT FORMAT — return ONLY valid JSON, no markdown, no commentary.

{
  "task_graph": {
    "graph_id": "<string>",
    "summary": "<short description>",
    "entry_node_id": "<node id>",
    "nodes": [
      {
        "node_id": "<id>",
        "kind": "primitive",
        "label": "<short label>",
        "description": "<why this node exists>",
        "action": {"action": "<primitive>", "params": {...}},
        "metadata": {...}
      }
    ],
    "edges": [
      {
        "source_node_id": "<id>",
        "target_node_id": "<id>",
        "condition": "on_success",
        "metadata": {}
      }
    ],
    "metadata": {}
  },
  "planner_notes": ["<optional short note>", "..."],
  "memory_update": "<optional short observation>"
}

RULES:
  1. Keep the plan local and minimal. Do not solve the whole mission in detail.
  2. Use the RoleBrief's shared_context and initial_hints only as hints.
  3. Respect the RoleBrief's capability_requirements, capability_exclusions,
     resource_requirements, and resource_permissions.
  4. Respect the agent profile. Do not rely on capabilities, resources, or
     primitives that the agent does not have.
  5. If the context is insufficient, return a safe minimal graph rather than inventing specifics.
  6. Only use primitives listed in the prompt payload as available to this agent.
  7. Return ONLY the JSON object.
"""


class OnboardPlannerAgent:
    """
    Adapter-friendly onboard planner.

    Today it uses an OpenAI-compatible model endpoint. In the future the same
    interface can be backed by a multi-step agent framework.
    """

    def __init__(self):
        self._client = build_async_openai_client(
            api_key=config.ONBOARD_PLANNER_API_KEY,
            base_url=config.ONBOARD_PLANNER_BASE_URL,
        )

    async def build_initial_task_graph(
        self,
        context: OnboardPlanningContext,
    ) -> OnboardPlanningResponse:
        while True:
            await DEBUG_GATE.checkpoint(
                "onboard_request",
                {
                    "model": config.ONBOARD_PLANNER_MODEL,
                    "context": context,
                    "user_prompt": _build_user_prompt(context),
                },
                actor_id=context.drone_id,
                summary="Inspect onboard task-graph planning request before sending.",
            )
            try:
                response = await asyncio.wait_for(
                    self._call_planner(context),
                    timeout=config.ONBOARD_PLANNER_TIMEOUT,
                )
                build_task_graph_runtime(
                    context.drone_id,
                    context.role_brief,
                    response.task_graph,
                    agent_profile=context.agent_profile,
                )
                command = await DEBUG_GATE.checkpoint(
                    "onboard_response",
                    {
                        "model": config.ONBOARD_PLANNER_MODEL,
                        "drone_id": context.drone_id,
                        "fallback_used": False,
                        "response": response,
                    },
                    actor_id=context.drone_id,
                    allow_regenerate=True,
                    summary="Inspect validated onboard task-graph output.",
                )
                if command == "regenerate":
                    logger.info("[%s] Regenerating onboard task graph due to debug gate request.", context.drone_id)
                    continue
                logger.info("[%s] Onboard planner produced graph '%s'", context.drone_id, response.task_graph.graph_id)
                return response
            except Exception as exc:
                error_type = type(exc).__name__
                error_repr = repr(exc)
                logger.warning(
                    "[%s] Onboard planner failed (%s: %s); using compatibility bootstrap graph.",
                    context.drone_id,
                    error_type,
                    exc,
                )
                fallback = compatibility_task_graph_from_role_brief(context.drone_id, context.role_brief)
                fallback_notes = ["compatibility_fallback"]
                fallback_source = "compatibility_fallback"
                try:
                    build_task_graph_runtime(
                        context.drone_id,
                        context.role_brief,
                        fallback,
                        agent_profile=context.agent_profile,
                    )
                except Exception as fallback_exc:
                    logger.warning(
                        "[%s] Compatibility bootstrap graph also failed validation (%s); using empty graph.",
                        context.drone_id,
                        fallback_exc,
                    )
                    fallback = TaskGraphSpec(
                        graph_id=f"{context.drone_id}-empty-bootstrap",
                        summary=f"Empty bootstrap graph for {context.role_brief.mission_role}",
                        nodes=[],
                        edges=[],
                        metadata={"source": "validated_empty_fallback"},
                    )
                    fallback_notes.append(f"fallback_validation_error:{fallback_exc}")
                    fallback_source = "validated_empty_fallback"

                fallback_response = OnboardPlanningResponse(
                    task_graph=fallback,
                    planner_notes=fallback_notes,
                )
                command = await DEBUG_GATE.checkpoint(
                    "onboard_response",
                    {
                        "model": config.ONBOARD_PLANNER_MODEL,
                        "drone_id": context.drone_id,
                        "fallback_used": True,
                        "fallback_source": fallback_source,
                        "error": str(exc),
                        "error_type": error_type,
                        "error_repr": error_repr,
                        "response": fallback_response,
                    },
                    actor_id=context.drone_id,
                    allow_regenerate=True,
                    summary="Inspect onboard planner fallback output.",
                )
                if command == "regenerate":
                    logger.info("[%s] Regenerating onboard task graph after fallback due to debug gate request.", context.drone_id)
                    continue
                return fallback_response

    async def _call_planner(self, context: OnboardPlanningContext) -> OnboardPlanningResponse:
        response = await self._client.chat.completions.create(
            model=config.ONBOARD_PLANNER_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(context)},
            ],
            temperature=config.ONBOARD_PLANNER_TEMPERATURE,
        )
        raw = response.choices[0].message.content or ""
        logger.debug("[%s] Onboard planner raw response: %s", context.drone_id, raw)
        return _parse_planning_response(raw)


def _build_user_prompt(context: OnboardPlanningContext) -> str:
    self_state = context.self_state.model_dump() if context.self_state is not None else None
    peer_states = {drone_id: state.model_dump() for drone_id, state in context.peer_states.items()}
    claims = [claim.model_dump() for claim in context.active_claims]
    events = [event.model_dump() for event in context.active_events]
    primitive_names = [primitive.name for primitive in context.available_primitives]
    agent_profile_summary = None
    if context.agent_profile is not None:
        agent_profile_summary = {
            "agent_id": context.agent_profile.agent_id,
            "agent_kind": context.agent_profile.agent_kind,
            "available_resources": context.agent_profile.available_resources,
            "hard_constraints": context.agent_profile.hard_constraints,
            "metadata": context.agent_profile.metadata,
        }

    payload = {
        "drone_id": context.drone_id,
        "role_brief": context.role_brief.model_dump(),
        "agent_profile": agent_profile_summary,
        "self_state": self_state,
        "peer_states": peer_states,
        "active_claims": claims,
        "active_events": events,
        "available_capabilities": context.available_capabilities,
        "available_primitives": primitive_names,
        "has_image": bool(context.image_b64),
    }
    primitive_text = format_primitives_for_prompt(context.available_primitives)
    return (
        "=== AVAILABLE PRIMITIVES ===\n"
        f"{primitive_text}\n"
        "\n=== AVAILABLE CAPABILITIES ===\n"
        f"{context.available_capabilities or ['(unspecified)']}\n"
        "\n=== PLANNING CONTEXT JSON ===\n"
        f"{json.dumps(payload, ensure_ascii=True, indent=2)}"
    )


def _parse_planning_response(raw: str) -> OnboardPlanningResponse:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in onboard planner response: {cleaned!r}")

    data = json.loads(match.group())
    if "task_graph" not in data:
        data = {
            "task_graph": TaskGraphSpec.model_validate(data).model_dump(),
            "planner_notes": [],
            "memory_update": "",
        }
    return OnboardPlanningResponse.model_validate(data)
