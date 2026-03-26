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
from typing import Optional

from openai import AsyncOpenAI

import config
from models.schemas import GlobalPlan

logger = logging.getLogger(__name__)

MAX_RETRIES      = 3
RETRY_BASE_DELAY = 2.0   # seconds; doubles each attempt


# ─── System prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are the Global Mission Planner for an autonomous multi-drone swarm system.

Your job is to translate a natural-language mission into a structured JSON task plan.

OUTPUT FORMAT — you must return ONLY valid JSON with no markdown, no code fences,
no commentary. The structure is:

{
  "drone_1": [
    {"action": "<skill_name>", "params": {<param_key>: <value>, ...}},
    ...
  ],
  "drone_2": [...],
  ...
}

AVAILABLE SKILLS and their required params:
  takeoff          → {"altitude": <float, metres>}
  go_to_waypoint   → {"x": <float>, "y": <float>, "z": <float>, "velocity": <float, m/s>}
  hover            → {"duration": <float, seconds>}
  search_pattern   → {"center_x": <float>, "center_y": <float>, "radius": <float>, "altitude": <float>}
  land             → {}

RULES:
  1. Every drone must begin its task list with a takeoff action.
  2. Assign tasks only to drones listed in the mission context.
  3. Use realistic coordinates (metres, NED frame). Keep values under 200 m.
  4. Before every search_pattern action, always insert a go_to_waypoint to
     (center_x, center_y, altitude) so the drone explicitly flies to the
     search area first — do not assume it is already there.
  5. Return ONLY the JSON object — nothing else.
"""


class GlobalOperator:
    """
    Wraps the OpenAI async client and exposes a single high-level method:
        plan = await operator.plan_mission("Drone 1 patrol sector A …")
    """

    def __init__(self, drone_ids: Optional[list[str]] = None):
        self._drone_ids = drone_ids or config.DRONE_IDS
        # Use Ollama-compatible base URL when configured, otherwise default to OpenAI
        client_kwargs: dict = {
            "api_key":     config.GLOBAL_LLM_API_KEY or "nokey",
            "max_retries": 0,   # disable SDK-level retries; our loop handles them
        }
        if config.GLOBAL_LLM_BASE_URL:
            client_kwargs["base_url"] = config.GLOBAL_LLM_BASE_URL
        self._client = AsyncOpenAI(**client_kwargs)

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

        last_error: Optional[str] = None

        for attempt in range(1, MAX_RETRIES + 1):
            # If a previous attempt failed, append a correction message
            if last_error:
                messages.append({
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
                # response_format=json_object is OpenAI-specific; omit for Ollama
                extra = {}
                if not config.GLOBAL_LLM_BASE_URL:
                    extra["response_format"] = {"type": "json_object"}

                response = await self._client.chat.completions.create(
                    model=config.GLOBAL_LLM_MODEL,
                    messages=messages,
                    temperature=0.2,
                    timeout=120.0,  # Ollama 9b may take ~90s for a multi-drone plan
                    **extra,
                )

                raw_content = response.choices[0].message.content or ""
                logger.debug("GlobalOperator raw response:\n%s", raw_content)

                plan = _parse_plan(raw_content, self._drone_ids)
                logger.info(
                    "GlobalOperator: plan created for drones %s",
                    list(plan.plan.keys()),
                )
                return plan

            except (json.JSONDecodeError, ValueError) as e:
                last_error = str(e)
                logger.warning(
                    "GlobalOperator: parse error on attempt %d: %s", attempt, last_error
                )
                # Append the bad response so GPT-4o can see what went wrong
                messages.append({
                    "role": "assistant",
                    "content": raw_content if "raw_content" in dir() else "",
                })

            except Exception as e:
                last_error = str(e)
                logger.error("GlobalOperator: API error on attempt %d: %s", attempt, e)

            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.info("GlobalOperator: retrying in %.1f s …", delay)
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"GlobalOperator failed to produce a valid plan after {MAX_RETRIES} "
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
