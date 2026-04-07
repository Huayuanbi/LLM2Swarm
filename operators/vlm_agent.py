"""
operators/vlm_agent.py — Edge VLM Local Replanner (qwen3.5:4b via Ollama).

Called once per 10-second control loop tick for each drone.  Takes:
  - The drone's current telemetry (position, velocity, status)
  - A Base64 camera image
  - The active task description
  - Peer drone states from the Memory Pool

Returns a VLMDecision: either VLMContinue or VLMModify.

Timeout handling:
  - If the Ollama endpoint doesn't respond within VLM_TIMEOUT seconds,
    the fallback decision ("continue") is returned so the drone doesn't
    freeze mid-flight.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Optional

from openai import AsyncOpenAI

import config
from models.schemas import (
    DroneState,
    PrimitiveSpec,
    TaskClaim,
    TaskEvent,
    VLMDecision,
    parse_vlm_decision,
)
from primitives.registry import format_primitives_for_prompt
from utils.image_utils import build_image_message

logger = logging.getLogger(__name__)


# ─── System prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are the onboard AI co-pilot for a single autonomous drone in a multi-drone swarm.

Every 10 seconds you receive:
  - A front-facing camera image from the drone
  - The drone's current telemetry (position, velocity, status, active task)
  - The states of your teammate drones

Your job is to decide whether the current task should CONTINUE or be MODIFIED.

OUTPUT FORMAT — return ONLY valid JSON, no markdown, no commentary.

If the current task is appropriate:
{"decision": "continue"}

If the situation requires a change (obstacle, better strategy, teammate conflict):
{
  "decision":      "modify",
  "new_task":      "<human-readable description of new task>",
  "new_action":    {"action": "<skill_name>", "params": {<key>: <value>}},
  "task_graph_patch": {"reason": "<optional reason>", "prepend_nodes": [...], "prepend_edges": [...]},
  "event":         {"type": "<optional swarm event>", "source": "vlm", "priority": 1, "payload": {...}},
  "memory_update": "<brief observation to share with the swarm>"
}

RULES:
  1. Default to "continue" unless you observe a clear reason to change.
  2. Avoid sending two drones to the same waypoint simultaneously.
  3. If you observe a meaningful scene event, you MAY include a structured
     event in the "event" field. Prefer concise generic event types such as
     "target_detected", "obstacle_detected", "region_complete", or
     "task_blocked", with structured payload fields.
  4. You may use either new_action or task_graph_patch or both. Prefer the
     smallest valid change that helps this drone progress safely.
  5. memory_update should be a concise factual observation (max 20 words).
  6. Return ONLY the JSON object — nothing else.
  7. Only propose primitives that are explicitly listed as available to this agent.
  8. Respect active swarm claims shown in the prompt. If another drone already
     owns a target claim, avoid redirecting this drone to the same target.
  9. Use active swarm events shown in the prompt as shared context when making
     your decision, but only modify this drone's task when necessary.
"""


class VLMAgent:
    """
    Wraps the Ollama-hosted qwen3.5:4b endpoint with the same AsyncOpenAI
    client interface (Ollama's OpenAI-compatible /v1 API).
    """

    def __init__(self):
        self._client = AsyncOpenAI(
            api_key=config.EDGE_VLM_API_KEY,
            base_url=config.EDGE_VLM_BASE_URL,
        )

    async def decide(
        self,
        drone_id:     str,
        position:     tuple[float, float, float],
        velocity:     tuple[float, float, float],
        status:       str,
        current_task: Optional[str],
        image_b64:    str,
        peer_states:  dict[str, DroneState],
        claims:       Optional[list[TaskClaim]] = None,
        events:       Optional[list[TaskEvent]] = None,
        available_primitives: Optional[list[PrimitiveSpec]] = None,
        available_capabilities: Optional[list[str]] = None,
    ) -> VLMDecision:
        """
        Ask the edge VLM whether the drone should continue or modify its task.

        Falls back to VLMContinue on timeout or parse error.
        """
        user_content = _build_user_content(
            drone_id,
            position,
            velocity,
            status,
            current_task,
            image_b64,
            peer_states,
            claims or [],
            events or [],
            available_primitives or [],
            available_capabilities or [],
        )

        try:
            decision = await asyncio.wait_for(
                self._call_vlm(user_content),
                timeout=config.VLM_TIMEOUT,
            )
            logger.info("[%s] VLM decision: %s", drone_id, decision.decision)
            return decision

        except asyncio.TimeoutError:
            logger.warning(
                "[%s] VLM timed out after %.1f s — defaulting to '%s'",
                drone_id, config.VLM_TIMEOUT, config.VLM_FALLBACK_DECISION,
            )
            return parse_vlm_decision({"decision": config.VLM_FALLBACK_DECISION})

        except Exception as e:
            logger.error(
                "[%s] VLM error: %s — defaulting to '%s'",
                drone_id, e, config.VLM_FALLBACK_DECISION,
            )
            return parse_vlm_decision({"decision": config.VLM_FALLBACK_DECISION})

    async def _call_vlm(self, user_content: list[dict]) -> VLMDecision:
        """Raw API call + parse. Raises on any error."""
        response = await self._client.chat.completions.create(
            model=config.EDGE_VLM_MODEL,
            messages=[
                {"role": "system",  "content": _SYSTEM_PROMPT},
                {"role": "user",    "content": user_content},
            ],
            temperature=config.EDGE_VLM_TEMPERATURE,
        )
        raw = response.choices[0].message.content or ""
        logger.debug("VLM raw response: %s", raw)
        return _parse_vlm_response(raw)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_user_content(
    drone_id:     str,
    position:     tuple[float, float, float],
    velocity:     tuple[float, float, float],
    status:       str,
    current_task: Optional[str],
    image_b64:    str,
    peer_states:  dict[str, DroneState],
    claims:       list[TaskClaim],
    events:       list[TaskEvent],
    available_primitives: list[PrimitiveSpec],
    available_capabilities: list[str],
) -> list[dict]:
    """
    Build the multimodal message content list:
      [text block with telemetry + peer states, image block]
    """
    x, y, z    = position
    vx, vy, vz = velocity

    # Summarise peer states compactly
    peer_lines = []
    for pid, ps in peer_states.items():
        px, py, pz = ps.position
        obs_summary = (", ".join(ps.observations[-3:]) or "none")
        peer_lines.append(
            f"  {pid}: pos=({px:.1f},{py:.1f},{pz:.1f}) "
            f"status={ps.status} obs=[{obs_summary}]"
        )
    peer_summary = "\n".join(peer_lines) if peer_lines else "  (no peers)"

    claim_lines = []
    for claim in claims:
        remaining = max(0.0, claim.expires_at - time.time())
        claim_lines.append(
            f"  {claim.claim_type}:{claim.target_key} owned_by={claim.claimant_id} "
            f"payload={claim.payload} lease={remaining:.0f}s"
        )
    claim_summary = "\n".join(claim_lines) if claim_lines else "  (no active claims)"

    event_lines = []
    for event in events[-10:]:
        event_lines.append(
            f"  {event.type} from={event.source} priority={event.priority} payload={event.payload}"
        )
    event_summary = "\n".join(event_lines) if event_lines else "  (no active shared events)"
    primitive_summary = format_primitives_for_prompt(available_primitives)
    capability_summary = ", ".join(available_capabilities) if available_capabilities else "(unspecified)"

    text_prompt = (
        f"=== DRONE TELEMETRY ===\n"
        f"ID:           {drone_id}\n"
        f"Position:     x={x:.2f}  y={y:.2f}  z={z:.2f}  (metres)\n"
        f"Velocity:     vx={vx:.2f}  vy={vy:.2f}  vz={vz:.2f}  (m/s)\n"
        f"Status:       {status}\n"
        f"Active task:  {current_task or 'none'}\n"
        f"\n=== TEAMMATE STATES ===\n{peer_summary}\n"
        f"\n=== ACTIVE SWARM CLAIMS ===\n{claim_summary}\n"
        f"\n=== ACTIVE SWARM EVENTS ===\n{event_summary}\n"
        f"\n=== AVAILABLE PRIMITIVES ===\n{primitive_summary}\n"
        f"\n=== AVAILABLE CAPABILITIES ===\n  {capability_summary}\n"
        f"\n=== CAMERA IMAGE ===\n"
        f"The attached image is the current front-facing camera view.\n"
        f"Analyse it for obstacles, targets, points of interest, completion cues, or conflicts.\n"
        f"\nBased on this information, return your JSON decision."
    )

    return [
        {"type": "text",      "text": text_prompt},
        build_image_message(image_b64),
    ]


def _parse_vlm_response(raw: str) -> VLMDecision:
    """
    Extract and validate JSON from the VLM's text output.
    Strips <think>…</think> blocks (qwen3 thinking mode), markdown fences,
    and any leading/trailing prose.
    """
    # Remove thinking blocks — present when /no_think is ignored or unavailable
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip()
    match   = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in VLM response: {cleaned!r}")
    data = json.loads(match.group())
    return parse_vlm_decision(data)
