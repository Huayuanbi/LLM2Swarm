"""
models/schemas.py — Pydantic data models for all structured I/O in LLM2Swarm.

Every JSON payload that crosses a module boundary is validated here,
which catches hallucinated fields and missing keys before they crash the loop.
"""

from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, model_validator


# ─── Drone state (stored in the Memory Pool) ──────────────────────────────────

class DroneState(BaseModel):
    drone_id:     str
    position:     list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0],
                                      description="[x, y, z] in metres")
    velocity:     list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0],
                                      description="[vx, vy, vz] in m/s")
    status:       str         = "idle"   # idle | takeoff | moving | hovering | searching | error
    observations: list[str]   = Field(default_factory=list,
                                      description="Free-text observations logged this tick")
    current_task: Optional[str] = None   # human-readable label of active task


# ─── Skill / Action primitives ────────────────────────────────────────────────

class ActionTakeoff(BaseModel):
    action: Literal["takeoff"]
    params: dict = Field(default_factory=dict)
    # expected key: altitude (float, metres)

class ActionGoToWaypoint(BaseModel):
    action: Literal["go_to_waypoint"]
    params: dict = Field(default_factory=dict)
    # expected keys: x, y, z (float), velocity (float, m/s)

class ActionHover(BaseModel):
    action: Literal["hover"]
    params: dict = Field(default_factory=dict)
    # expected key: duration (float, seconds)

class ActionSearchPattern(BaseModel):
    action: Literal["search_pattern"]
    params: dict = Field(default_factory=dict)
    # expected keys: center_x, center_y, radius (float)

class ActionLand(BaseModel):
    action: Literal["land"]
    params: dict = Field(default_factory=dict)

# Union used for parsing an arbitrary task entry from the LLM
TaskAction = ActionTakeoff | ActionGoToWaypoint | ActionHover | ActionSearchPattern | ActionLand


# ─── Global Operator output (Cloud LLM → task lists) ─────────────────────────

class GlobalPlan(BaseModel):
    """
    Parsed output from the Cloud LLM global operator.
    Example:
        {
          "drone_1": [{"action": "takeoff", "params": {"altitude": 10}}, ...],
          "drone_2": [{"action": "go_to_waypoint", "params": {"x":10,"y":20,"z":5,"velocity":3}}]
        }
    """
    plan: dict[str, list[dict[str, Any]]] = Field(
        description="Maps each drone_id to its ordered task list"
    )

    @model_validator(mode="before")
    @classmethod
    def accept_flat_dict(cls, data: Any) -> Any:
        """Allow the LLM to return the plan either flat or wrapped in a 'plan' key."""
        if isinstance(data, dict) and "plan" not in data:
            return {"plan": data}
        return data

    def get_tasks(self, drone_id: str) -> list[dict[str, Any]]:
        return self.plan.get(drone_id, [])


# ─── Edge VLM output (per-drone replanning decision) ─────────────────────────

class VLMContinue(BaseModel):
    decision: Literal["continue"]

class VLMModify(BaseModel):
    decision:      Literal["modify"]
    new_task:      str           = Field(description="Human-readable description of new task")
    new_action:    dict[str, Any] = Field(
        description="Structured action to execute: {action, params}"
    )
    memory_update: str           = Field(
        default="",
        description="Observation/note to append to this drone's memory pool entry"
    )

# The VLM must return one of these two shapes
VLMDecision = VLMContinue | VLMModify


def parse_vlm_decision(raw: dict[str, Any]) -> VLMDecision:
    """
    Parse raw VLM JSON into a typed VLMDecision.
    Raises ValueError with a descriptive message on schema mismatch.
    """
    decision = raw.get("decision")
    if decision == "continue":
        return VLMContinue(**raw)
    elif decision == "modify":
        return VLMModify(**raw)
    else:
        raise ValueError(
            f"VLM returned unknown decision '{decision}'. "
            f"Expected 'continue' or 'modify'. Raw: {raw}"
        )
