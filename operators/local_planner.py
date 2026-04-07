"""
operators/local_planner.py — Generic task-graph runtime and compatibility adapters.

This module no longer treats RoleBrief as a task-specific search template.
Instead it provides:
  - a generic task-graph runtime that executes primitive nodes
  - a compatibility adapter that can turn sample/bootstrap actions into a graph
  - a small fallback bridge for legacy role briefs used by current demos/tests

Long term, the intended flow is:
  RoleBrief -> onboard planner model / agent -> TaskGraphSpec -> LocalTaskGraph runtime
"""

from __future__ import annotations

import copy
import logging
from collections import deque
from typing import Optional

from models.schemas import (
    AgentProfile,
    RoleBrief,
    TaskEvent,
    TaskGraphEdge,
    TaskGraphNode,
    TaskGraphPatch,
    TaskGraphSpec,
    VLMModify,
)
from primitives.registry import get_primitive_spec

logger = logging.getLogger(__name__)


class LocalTaskGraph:
    """
    Generic runtime wrapper around a TaskGraphSpec.

    The current executor still consumes a linear queue of primitive actions, so
    this runtime keeps a lightweight linearisation of the graph while preserving
    the richer graph structure for future planners and patchers.
    """

    def __init__(
        self,
        drone_id: str,
        role_brief: RoleBrief,
        graph_spec: TaskGraphSpec,
        agent_profile: Optional[AgentProfile] = None,
    ):
        self.drone_id = drone_id
        self.role_brief = role_brief
        self.agent_profile = copy.deepcopy(agent_profile)
        _validate_graph_spec(graph_spec, role_brief, self.agent_profile)
        self.graph_spec = copy.deepcopy(graph_spec)
        self._planned_nodes: deque[TaskGraphNode] = deque(_linearise_graph(self.graph_spec))
        self._interrupt_nodes: deque[TaskGraphNode] = deque()
        self._events: deque[TaskEvent] = deque(maxlen=64)
        self._current_source: Optional[str] = None
        self._current_label: str = role_brief.mission_role

    # ── Queue helpers ──────────────────────────────────────────────────────

    def has_pending_actions(self) -> bool:
        return bool(self._interrupt_nodes or self._planned_nodes)

    def pending_count(self) -> int:
        return len(self._interrupt_nodes) + len(self._planned_nodes)

    def pop_next_action(self) -> dict:
        node = self._pop_next_node()
        if node.action is None:
            raise ValueError(f"Node '{node.node_id}' has no primitive action payload")
        return copy.deepcopy(node.action)

    def preview_actions(self) -> list[dict]:
        items = [node.action for node in list(self._interrupt_nodes) + list(self._planned_nodes) if node.action]
        return copy.deepcopy(items)

    def current_task_label(self, current_action: Optional[dict] = None) -> str:
        if current_action is not None:
            return self._current_label
        next_node = self._peek_next_node()
        if next_node is not None:
            return next_node.label
        return self.role_brief.mission_role

    def on_action_completed(self, action: Optional[dict]) -> None:
        if action is None:
            return
        if self._current_source == "interrupt" and not self._interrupt_nodes:
            self._current_label = self.role_brief.mission_role
        self._current_source = None

    def graph_snapshot(self) -> TaskGraphSpec:
        return copy.deepcopy(self.graph_spec)

    # ── Runtime patching ───────────────────────────────────────────────────

    def apply_vlm_modify(
        self,
        decision: VLMModify,
        *,
        current_action: Optional[dict] = None,
    ) -> None:
        """
        Apply a model-proposed runtime change.

        Today this means:
          - prepend an explicit primitive action if provided
          - optionally apply a generic graph patch if the model proposes one
          - record the event/decision in runtime history
        """
        if decision.task_graph_patch is not None:
            self.apply_task_graph_patch(decision.task_graph_patch)

        if decision.new_action:
            node = TaskGraphNode(
                node_id=f"interrupt-{len(self._events)}",
                label=decision.new_task,
                description=decision.new_task,
                action=copy.deepcopy(decision.new_action),
                metadata={"source": "vlm_modify"},
            )
            self._validate_nodes([node])
            self._prepend_action_nodes(
                [node]
            )

        event = decision.event
        self._record_event(
            TaskEvent(
                type=event.type if event is not None else "vlm_modify",
                source="vlm",
                priority=event.priority if event is not None else 1,
                payload={
                    "new_task": decision.new_task,
                    "new_action": decision.new_action,
                    **(event.payload if event is not None else {}),
                },
            ),
        )

    def apply_task_graph_patch(self, patch: TaskGraphPatch) -> None:
        self._validate_nodes(patch.prepend_nodes)
        prependable = [node for node in patch.prepend_nodes if node.action is not None]
        if prependable:
            self._prepend_action_nodes(prependable)

        if patch.prepend_nodes:
            existing = {node.node_id for node in self.graph_spec.nodes}
            for node in patch.prepend_nodes:
                if node.node_id not in existing:
                    self.graph_spec.nodes.insert(0, copy.deepcopy(node))
            self.graph_spec.edges = list(patch.prepend_edges) + self.graph_spec.edges

        self._record_event(
            TaskEvent(
                type="task_graph_patch",
                source="runtime",
                priority=1,
                payload={"reason": patch.reason, "metadata": patch.metadata},
            )
        )

    def emit_event(self, event: TaskEvent) -> None:
        self._record_event(event)

    def event_history(self) -> list[TaskEvent]:
        return list(self._events)

    # ── Internals ──────────────────────────────────────────────────────────

    def _record_event(self, event: TaskEvent) -> None:
        self._events.append(event)
        logger.info("[%s] TaskGraph event: %s from %s", self.drone_id, event.type, event.source)

    def _pop_next_node(self) -> TaskGraphNode:
        if self._interrupt_nodes:
            node = self._interrupt_nodes.popleft()
            self._current_source = "interrupt"
            self._current_label = node.label
            return copy.deepcopy(node)

        node = self._planned_nodes.popleft()
        self._current_source = "planned"
        self._current_label = node.label
        return copy.deepcopy(node)

    def _peek_next_node(self) -> Optional[TaskGraphNode]:
        if self._interrupt_nodes:
            return self._interrupt_nodes[0]
        if self._planned_nodes:
            return self._planned_nodes[0]
        return None

    def _prepend_action_nodes(self, nodes: list[TaskGraphNode]) -> None:
        for node in reversed(nodes):
            self._interrupt_nodes.appendleft(copy.deepcopy(node))

    def _validate_nodes(self, nodes: list[TaskGraphNode]) -> None:
        if self.agent_profile is None:
            return
        scratch = TaskGraphSpec(
            graph_id="validation-only",
            summary="validation-only",
            nodes=nodes,
            edges=[],
        )
        _validate_graph_spec(scratch, self.role_brief, self.agent_profile)


def build_task_graph_runtime(
    drone_id: str,
    role_brief: RoleBrief,
    graph_spec: TaskGraphSpec,
    agent_profile: Optional[AgentProfile] = None,
) -> LocalTaskGraph:
    return LocalTaskGraph(
        drone_id=drone_id,
        role_brief=role_brief,
        graph_spec=graph_spec,
        agent_profile=agent_profile,
    )


def compile_role_brief(drone_id: str, role_brief: RoleBrief) -> LocalTaskGraph:
    """
    Compatibility helper for current demos/tests.

    This is NOT the desired long-term planning path. It only exists so that
    sample roles or legacy stored briefs can still be turned into a task graph
    until the onboard planner model/agent takes over completely.
    """
    spec = compatibility_task_graph_from_role_brief(drone_id, role_brief)
    return build_task_graph_runtime(drone_id, role_brief, spec)


def compatibility_task_graph_from_role_brief(
    drone_id: str,
    role_brief: RoleBrief,
) -> TaskGraphSpec:
    """
    Build a bootstrap graph from sample/legacy context embedded in RoleBrief.

    Priority:
      1. shared_context.bootstrap_actions
      2. legacy search-oriented context translated into primitive actions
      3. empty graph (the runtime will idle until a planner/model provides work)
    """
    ctx = role_brief.shared_context

    bootstrap_actions = ctx.get("bootstrap_actions")
    if isinstance(bootstrap_actions, list) and bootstrap_actions:
        return task_graph_from_actions(
            graph_id=f"{drone_id}-bootstrap",
            summary=f"Bootstrap graph for {role_brief.mission_role}",
            actions=bootstrap_actions,
            metadata={"source": "role_brief.shared_context.bootstrap_actions"},
        )

    legacy_actions = _legacy_actions_from_context(ctx)
    if legacy_actions:
        return task_graph_from_actions(
            graph_id=f"{drone_id}-legacy-bootstrap",
            summary=f"Legacy compatibility graph for {role_brief.mission_role}",
            actions=legacy_actions,
            metadata={"source": "role_brief.shared_context.legacy_context"},
        )

    return TaskGraphSpec(
        graph_id=f"{drone_id}-empty-bootstrap",
        summary=f"Empty bootstrap graph for {role_brief.mission_role}",
        nodes=[],
        edges=[],
        metadata={"source": "empty"},
    )


def task_graph_from_actions(
    *,
    graph_id: str,
    summary: str,
    actions: list[dict],
    metadata: Optional[dict] = None,
) -> TaskGraphSpec:
    nodes: list[TaskGraphNode] = []
    edges: list[TaskGraphEdge] = []

    for idx, action in enumerate(actions):
        node_id = f"n{idx}"
        label = action.get("action", f"action_{idx}")
        nodes.append(
            TaskGraphNode(
                node_id=node_id,
                kind="primitive",
                label=label,
                description=label,
                action=copy.deepcopy(action),
                metadata={"sequence_index": idx},
            )
        )
        if idx > 0:
            edges.append(
                TaskGraphEdge(
                    source_node_id=f"n{idx - 1}",
                    target_node_id=node_id,
                    condition="on_success",
                )
            )

    return TaskGraphSpec(
        graph_id=graph_id,
        summary=summary,
        nodes=nodes,
        edges=edges,
        entry_node_id=nodes[0].node_id if nodes else None,
        metadata=metadata or {},
    )


def _linearise_graph(graph_spec: TaskGraphSpec) -> list[TaskGraphNode]:
    """
    Convert a TaskGraphSpec into the linear primitive queue that today's runtime
    executor can handle.

    Strategy:
      - if there are no edges, preserve node order
      - otherwise, follow the entry node and on_success edges
      - ignore non-primitive nodes for execution, but keep them in the graph
    """
    if not graph_spec.nodes:
        return []

    node_map = {node.node_id: node for node in graph_spec.nodes}
    if not graph_spec.edges:
        return [copy.deepcopy(node) for node in graph_spec.nodes if node.action is not None]

    edge_map: dict[str, list[TaskGraphEdge]] = {}
    for edge in graph_spec.edges:
        edge_map.setdefault(edge.source_node_id, []).append(edge)

    ordered: list[TaskGraphNode] = []
    current = graph_spec.entry_node_id or graph_spec.nodes[0].node_id
    visited: set[str] = set()

    while current and current not in visited:
        visited.add(current)
        node = node_map.get(current)
        if node is None:
            break
        if node.action is not None:
            ordered.append(copy.deepcopy(node))

        next_edges = edge_map.get(current, [])
        next_edge = next((edge for edge in next_edges if edge.condition == "on_success"), None)
        current = next_edge.target_node_id if next_edge is not None else None

    if ordered:
        return ordered

    return [copy.deepcopy(node) for node in graph_spec.nodes if node.action is not None]


def _legacy_actions_from_context(ctx: dict) -> list[dict]:
    """
    Compatibility bridge for earlier sample roles that embedded search-specific
    fields directly. This is deliberately isolated so the core framework stays
    generic.
    """
    items: list[dict] = []

    takeoff_altitude = ctx.get("takeoff_altitude")
    if takeoff_altitude is not None:
        items.append({"action": "takeoff", "params": {"altitude": takeoff_altitude}})

    transit_waypoints = ctx.get("transit_waypoints")
    if isinstance(transit_waypoints, list):
        for waypoint in transit_waypoints:
            if not isinstance(waypoint, dict):
                continue
            items.append(
                {
                    "action": "go_to_waypoint",
                    "params": {
                        "x": waypoint.get("x", 0.0),
                        "y": waypoint.get("y", 0.0),
                        "z": waypoint.get("z", 0.0),
                        "velocity": waypoint.get("velocity", 5.0),
                    },
                }
            )

    search_region = ctx.get("search_region")
    if isinstance(search_region, dict):
        items.append(
            {
                "action": "go_to_waypoint",
                "params": {
                    "x": search_region.get("center_x", 0.0),
                    "y": search_region.get("center_y", 0.0),
                    "z": search_region.get("altitude", 0.0),
                    "velocity": 5.0,
                },
            }
        )
        items.append(
            {
                "action": "search_pattern",
                "params": {
                    "center_x": search_region.get("center_x", 0.0),
                    "center_y": search_region.get("center_y", 0.0),
                    "radius": search_region.get("radius", 0.0),
                    "altitude": search_region.get("altitude", 0.0),
                },
            }
        )

    return items


def _validate_graph_spec(
    graph_spec: TaskGraphSpec,
    role_brief: RoleBrief,
    agent_profile: Optional[AgentProfile],
) -> None:
    if agent_profile is None:
        return

    available_capabilities = set(agent_profile.available_capabilities)
    available_resources = set(agent_profile.available_resources)
    available_primitives = {primitive.name for primitive in agent_profile.available_primitives}

    role_required_capabilities = set(role_brief.capability_requirements)
    role_excluded_capabilities = set(role_brief.capability_exclusions)
    role_required_resources = set(role_brief.resource_requirements)
    role_permitted_resources = set(role_brief.resource_permissions)

    missing_role_capabilities = role_required_capabilities - available_capabilities
    if missing_role_capabilities:
        raise ValueError(
            f"Role '{role_brief.mission_role}' requires unavailable capabilities: {sorted(missing_role_capabilities)}"
        )

    missing_role_resources = role_required_resources - available_resources
    if missing_role_resources:
        raise ValueError(
            f"Role '{role_brief.mission_role}' requires unavailable resources: {sorted(missing_role_resources)}"
        )

    missing_graph_capabilities = set(graph_spec.required_capabilities) - available_capabilities
    if missing_graph_capabilities:
        raise ValueError(
            f"Task graph '{graph_spec.graph_id}' requires unavailable capabilities: {sorted(missing_graph_capabilities)}"
        )

    missing_graph_resources = set(graph_spec.required_resources) - available_resources
    if missing_graph_resources:
        raise ValueError(
            f"Task graph '{graph_spec.graph_id}' requires unavailable resources: {sorted(missing_graph_resources)}"
        )

    for node in graph_spec.nodes:
        node_required_capabilities = set(node.required_capabilities)
        node_required_resources = set(node.required_resources)

        missing_node_capabilities = node_required_capabilities - available_capabilities
        if missing_node_capabilities:
            raise ValueError(
                f"Node '{node.node_id}' requires unavailable capabilities: {sorted(missing_node_capabilities)}"
            )

        forbidden_node_capabilities = node_required_capabilities & role_excluded_capabilities
        if forbidden_node_capabilities:
            raise ValueError(
                f"Node '{node.node_id}' depends on role-excluded capabilities: {sorted(forbidden_node_capabilities)}"
            )

        missing_node_resources = node_required_resources - available_resources
        if missing_node_resources:
            raise ValueError(
                f"Node '{node.node_id}' requires unavailable resources: {sorted(missing_node_resources)}"
            )

        if role_permitted_resources and not node_required_resources.issubset(role_permitted_resources | role_required_resources):
            unexpected = node_required_resources - (role_permitted_resources | role_required_resources)
            if unexpected:
                raise ValueError(
                    f"Node '{node.node_id}' requires resources not permitted by the role: {sorted(unexpected)}"
                )

        action = node.action
        if action is None:
            continue

        action_name = action.get("action")
        if action_name not in available_primitives:
            raise ValueError(
                f"Node '{node.node_id}' uses unsupported primitive '{action_name}' for agent '{agent_profile.agent_id}'"
            )

        spec = get_primitive_spec(action_name)
        implied_capabilities = set(spec.capability_tags)
        implied_resources = set(spec.resource_tags)

        if not implied_capabilities.issubset(available_capabilities):
            missing = implied_capabilities - available_capabilities
            raise ValueError(
                f"Primitive '{action_name}' implies unavailable capabilities: {sorted(missing)}"
            )

        if implied_capabilities & role_excluded_capabilities:
            conflict = implied_capabilities & role_excluded_capabilities
            raise ValueError(
                f"Primitive '{action_name}' conflicts with role capability exclusions: {sorted(conflict)}"
            )

        if role_permitted_resources and implied_resources and not implied_resources.issubset(role_permitted_resources | role_required_resources):
            raise ValueError(
                f"Primitive '{action_name}' implies resources not permitted by the role: {sorted(implied_resources)}"
            )
