"""
models/schemas.py — Pydantic data models for all structured I/O in LLM2Swarm.

Every JSON payload that crosses a module boundary is validated here,
which catches hallucinated fields and missing keys before they crash the loop.
"""

from __future__ import annotations
import time
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, model_validator


# ─── Drone state (stored in the Memory Pool) ──────────────────────────────────

class DroneState(BaseModel):
    drone_id:     str
    position:     list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0],
                                      description="[x, y, z] in metres")
    velocity:     list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0],
                                      description="[vx, vy, vz] in m/s")
    battery_level: Optional[float] = Field(
        default=None,
        description="Battery state of charge in [0, 1] if available",
    )
    status:       str         = "idle"   # idle | takeoff | moving | hovering | searching | error
    observations: list[str]   = Field(default_factory=list,
                                      description="Free-text observations logged this tick")
    current_task: Optional[str] = None   # human-readable label of active task
    updated_at:   float       = Field(
        default_factory=time.time,
        description="Unix timestamp of the last successful state update",
    )


# ─── Primitive / capability metadata ──────────────────────────────────────────

class PrimitiveParameterSpec(BaseModel):
    name: str = Field(description="Parameter name exposed in action JSON")
    type: str = Field(default="string", description="Logical type name for planners/prompts")
    description: str = Field(default="", description="Human-readable meaning of this parameter")
    required: bool = Field(default=False, description="Whether this parameter is required in action JSON")
    default: Optional[Any] = Field(default=None, description="Optional default value if omitted")
    handler_name: Optional[str] = Field(
        default=None,
        description="Optional controller handler kwarg name if different from the external action key",
    )


class PrimitiveSpec(BaseModel):
    name: str = Field(description="Stable primitive action id")
    handler_name: str = Field(description="Controller method used to execute this primitive")
    description: str = Field(default="", description="Human-readable primitive description")
    parameters: list[PrimitiveParameterSpec] = Field(default_factory=list)
    capability_tags: list[str] = Field(
        default_factory=list,
        description="Capability labels implied by supporting this primitive",
    )
    resource_tags: list[str] = Field(
        default_factory=list,
        description="Optional resources or payloads associated with this primitive",
    )
    continuous: bool = Field(
        default=False,
        description="Whether this primitive is expected to keep running until externally interrupted",
    )
    notes: list[str] = Field(default_factory=list, description="Optional planner-facing notes")


class AgentProfile(BaseModel):
    agent_id: str = Field(description="Stable agent identifier")
    agent_kind: str = Field(default="generic_agent", description="Embodiment/platform family label")
    available_primitives: list[PrimitiveSpec] = Field(
        default_factory=list,
        description="Primitive actions currently supported by this agent",
    )
    available_capabilities: list[str] = Field(
        default_factory=list,
        description="Capability tags exposed by this agent",
    )
    available_resources: list[str] = Field(
        default_factory=list,
        description="Resources / payloads / tools physically available to this agent",
    )
    hard_constraints: list[str] = Field(
        default_factory=list,
        description="Non-negotiable execution constraints for this agent",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Opaque platform metadata")

    @model_validator(mode="after")
    def fill_capability_defaults(self) -> "AgentProfile":
        if not self.available_capabilities:
            tags = {
                tag
                for primitive in self.available_primitives
                for tag in (primitive.capability_tags + primitive.resource_tags)
            }
            self.available_capabilities = sorted(tags)
        return self


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


# ─── Cloud Operator role briefs (Cloud LLM → per-drone responsibilities) ────

class RoleBrief(BaseModel):
    """
    A cloud-issued, per-drone mission responsibility.

    This schema is intentionally generic and capability-agnostic. It should be
    reusable across different embodied agents and mission types, not just
    aerial search tasks.

    The cloud planner should express:
      - what responsibility this agent owns
      - what constraints or contracts it must respect
      - what shared context or references are relevant
      - what conditions define success or handoff

    It should NOT encode a full concrete action sequence.
    """

    mission_role: str = Field(description="Short name for the drone's assigned role")
    mission_intent: str = Field(
        default="",
        description="Human-readable explanation of why this agent exists in the mission",
    )
    responsibilities: list[str] = Field(
        default_factory=list,
        description="Primary responsibilities or obligations owned by this agent",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Hard constraints, exclusions, or safety requirements",
    )
    coordination_contracts: list[str] = Field(
        default_factory=list,
        description="How this agent should coordinate with peers to reduce duplicate work",
    )
    capability_requirements: list[str] = Field(
        default_factory=list,
        description="Capabilities expected or preferred for this role",
    )
    capability_exclusions: list[str] = Field(
        default_factory=list,
        description="Capabilities or actuator classes this role must not depend on or invoke",
    )
    resource_requirements: list[str] = Field(
        default_factory=list,
        description="Resources or payloads that should be present for this role to be a good fit",
    )
    resource_permissions: list[str] = Field(
        default_factory=list,
        description="Resources or actuators this role may use when available",
    )
    success_criteria: list[str] = Field(
        default_factory=list,
        description="Conditions that indicate the role has succeeded",
    )
    handoff_conditions: list[str] = Field(
        default_factory=list,
        description="Conditions that should cause reassignment, escalation, or handoff",
    )
    event_watchlist: list[str] = Field(
        default_factory=list,
        description="Event types or situations that this agent should pay special attention to",
    )
    shared_context: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Mission-specific structured context. This may contain map anchors, "
            "region descriptors, sample bootstrap actions, semantic labels, or "
            "other scenario data, but the RoleBrief schema itself remains generic."
        ),
    )
    initial_hints: list[str] = Field(
        default_factory=list,
        description="Non-binding hints for the onboard planner when bootstrapping the local task graph",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional opaque metadata for future planners or agents",
    )

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_role_shape(cls, data: Any) -> Any:
        """
        Backward compatibility for the previous search-oriented RoleBrief shape.
        Legacy task-specific fields are folded into shared_context / hints so the
        outer schema remains generic.
        """
        if not isinstance(data, dict):
            return data

        raw = dict(data)
        raw.setdefault("mission_intent", raw.pop("objective", raw.get("mission_intent", "")))
        raw.setdefault("responsibilities", [])
        if not raw["responsibilities"] and raw.get("mission_intent"):
            raw["responsibilities"] = [raw["mission_intent"]]

        raw.setdefault("constraints", [])
        raw.setdefault("coordination_contracts", raw.pop("coordination_rules", []))
        raw.setdefault("capability_requirements", [])
        raw.setdefault("capability_exclusions", [])
        raw.setdefault("resource_requirements", [])
        raw.setdefault("resource_permissions", [])
        raw.setdefault("success_criteria", [])
        raw.setdefault("handoff_conditions", [])
        raw.setdefault("event_watchlist", raw.pop("contingencies", []))
        raw.setdefault("shared_context", {})
        raw.setdefault("initial_hints", [])
        raw.setdefault("metadata", {})

        legacy_context = {}
        for key in (
            "region_label",
            "search_strategy",
            "takeoff_altitude",
            "transit_waypoints",
            "search_region",
            "end_condition",
        ):
            if key in raw:
                legacy_context[key] = raw.pop(key)

        if legacy_context:
            raw["shared_context"] = {
                **legacy_context,
                **raw["shared_context"],
            }
            raw["initial_hints"] = [
                f"legacy_context_available:{key}" for key in legacy_context.keys()
            ] + list(raw["initial_hints"])

        return raw


# ─── Generic task-graph DSL (planner / runtime boundary) ────────────────────

class TaskGraphNode(BaseModel):
    node_id: str = Field(description="Stable identifier for this node within the graph")
    kind: Literal["primitive", "decision", "wait_event", "claim", "terminal", "note"] = Field(
        default="primitive",
        description="Node kind used by the task-graph runtime",
    )
    label: str = Field(description="Human-readable node label")
    description: str = Field(default="", description="Optional natural-language explanation")
    action: Optional[dict[str, Any]] = Field(
        default=None,
        description="Primitive action payload when kind='primitive'",
    )
    required_capabilities: list[str] = Field(
        default_factory=list,
        description="Capabilities that must be present to execute this node safely",
    )
    required_resources: list[str] = Field(
        default_factory=list,
        description="Resources / payloads that must be present to execute this node",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Opaque node-local metadata")


class TaskGraphEdge(BaseModel):
    source_node_id: str
    target_node_id: str
    condition: str = Field(
        default="on_success",
        description="Transition condition label such as on_success, on_event, on_failure",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskGraphSpec(BaseModel):
    graph_id: str = Field(default="", description="Stable graph identifier if available")
    summary: str = Field(default="", description="High-level summary of the task graph")
    nodes: list[TaskGraphNode] = Field(default_factory=list)
    edges: list[TaskGraphEdge] = Field(default_factory=list)
    entry_node_id: Optional[str] = Field(default=None, description="Entry node for traversal")
    required_capabilities: list[str] = Field(
        default_factory=list,
        description="Capabilities required somewhere in this graph or assumed by the planner",
    )
    required_resources: list[str] = Field(
        default_factory=list,
        description="Resources / payloads required somewhere in this graph",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def fill_defaults(self) -> "TaskGraphSpec":
        if not self.graph_id:
            self.graph_id = f"graph-{int(time.time() * 1000)}"
        if self.entry_node_id is None and self.nodes:
            self.entry_node_id = self.nodes[0].node_id
        return self


class TaskGraphPatch(BaseModel):
    reason: str = Field(description="Why this patch is being proposed")
    prepend_nodes: list[TaskGraphNode] = Field(
        default_factory=list,
        description="Nodes to prepend to the current runtime queue",
    )
    prepend_edges: list[TaskGraphEdge] = Field(
        default_factory=list,
        description="Edges associated with the prepended nodes if needed",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class OnboardPlanningContext(BaseModel):
    drone_id: str
    role_brief: RoleBrief
    self_state: Optional[DroneState] = None
    peer_states: dict[str, DroneState] = Field(default_factory=dict)
    active_claims: list["TaskClaim"] = Field(default_factory=list)
    active_events: list["TaskEvent"] = Field(default_factory=list)
    agent_profile: Optional[AgentProfile] = None
    available_primitives: list[PrimitiveSpec] = Field(default_factory=list)
    available_capabilities: list[str] = Field(default_factory=list)
    image_b64: str = Field(default="", description="Optional onboard image snapshot for multimodal planners")

    @model_validator(mode="after")
    def fill_profile_defaults(self) -> "OnboardPlanningContext":
        if self.agent_profile is not None:
            if not self.available_primitives:
                self.available_primitives = list(self.agent_profile.available_primitives)
            if not self.available_capabilities:
                self.available_capabilities = list(self.agent_profile.available_capabilities)
        return self


class OnboardPlanningResponse(BaseModel):
    task_graph: TaskGraphSpec
    planner_notes: list[str] = Field(default_factory=list)
    memory_update: str = Field(default="", description="Optional observation to publish after planning")


class GlobalRolePlan(BaseModel):
    """
    Parsed output from the cloud planner when it performs coarse role assignment.
    Example:
        {
          "drone_1": {
            "mission_role": "Northeast fire search",
            "objective": "...",
            ...
          },
          "drone_2": {...}
        }
    """

    roles: dict[str, RoleBrief] = Field(
        description="Maps each drone_id to its higher-level responsibility brief"
    )

    @model_validator(mode="before")
    @classmethod
    def accept_flat_dict(cls, data: Any) -> Any:
        if isinstance(data, dict) and "roles" not in data:
            return {"roles": data}
        return data

    def get_role(self, drone_id: str) -> Optional[RoleBrief]:
        return self.roles.get(drone_id)


# ─── Runtime event model (task graph patching / replanning) ──────────────────

class TaskEvent(BaseModel):
    type: str = Field(description="Event type, e.g. peer_lost, battery_low, target_detected")
    source: str = Field(default="system", description="Origin of the event")
    priority: int = Field(default=1, ge=0, le=3, description="Relative urgency; larger means more urgent")
    payload: dict[str, Any] = Field(default_factory=dict, description="Structured event payload")
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp when the event was raised")


class TaskClaim(BaseModel):
    claim_type: str = Field(description="Claim namespace, e.g. peer_lost_takeover")
    target_key: str = Field(description="Resource being claimed, e.g. drone_2")
    claimant_id: str = Field(description="Drone currently holding the claim")
    payload: dict[str, Any] = Field(default_factory=dict, description="Optional structured claim context")
    created_at: float = Field(default_factory=time.time, description="Unix timestamp when the claim was created")
    expires_at: float = Field(description="Unix timestamp when the claim lease expires")


# ─── Edge VLM output (per-drone replanning decision) ─────────────────────────

class VLMContinue(BaseModel):
    decision: Literal["continue"]

class VLMModify(BaseModel):
    decision:      Literal["modify"]
    new_task:      str           = Field(description="Human-readable description of new task")
    new_action:    Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional structured primitive action to execute immediately: {action, params}"
    )
    task_graph_patch: Optional[TaskGraphPatch] = Field(
        default=None,
        description="Optional generic task-graph patch proposed by the onboard planner/model",
    )
    event:         Optional[TaskEvent] = Field(
        default=None,
        description="Optional structured swarm event emitted alongside the modify decision",
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


OnboardPlanningContext.model_rebuild()
