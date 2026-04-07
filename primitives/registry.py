"""
primitives/registry.py — Shared primitive/capability catalog for planners and controllers.

The goal is to keep action metadata in one place so that:
  - controller dispatch is data-driven
  - planner/VLM prompts can advertise the current primitive set dynamically
  - future agents or platforms can register new primitives without rewriting prompts
"""

from __future__ import annotations

import copy
from typing import Iterable, Optional

from models.schemas import AgentProfile, PrimitiveParameterSpec, PrimitiveSpec


_REGISTRY: dict[str, PrimitiveSpec] = {
    "takeoff": PrimitiveSpec(
        name="takeoff",
        handler_name="takeoff",
        description="Arm and climb to a target altitude.",
        parameters=[
            PrimitiveParameterSpec(
                name="altitude",
                type="float",
                description="Target altitude in metres",
                required=False,
                default=5.0,
            )
        ],
        capability_tags=["flight", "vertical_mobility", "takeoff_landing"],
    ),
    "land": PrimitiveSpec(
        name="land",
        handler_name="land",
        description="Descend and disarm / stop safely on the ground.",
        parameters=[],
        capability_tags=["flight", "takeoff_landing", "recovery"],
    ),
    "go_to_waypoint": PrimitiveSpec(
        name="go_to_waypoint",
        handler_name="go_to_waypoint",
        description="Move to an absolute waypoint in the environment.",
        parameters=[
            PrimitiveParameterSpec(name="x", type="float", description="Target x coordinate", required=True),
            PrimitiveParameterSpec(name="y", type="float", description="Target y coordinate", required=True),
            PrimitiveParameterSpec(name="z", type="float", description="Target z coordinate", required=True),
            PrimitiveParameterSpec(
                name="velocity",
                type="float",
                description="Desired travel speed in metres per second",
                required=False,
                default=3.0,
            ),
        ],
        capability_tags=["flight", "navigation", "point_to_point_motion"],
    ),
    "hover": PrimitiveSpec(
        name="hover",
        handler_name="hover",
        description="Hold position for a bounded duration.",
        parameters=[
            PrimitiveParameterSpec(
                name="duration",
                type="float",
                description="How long to hold position in seconds",
                required=False,
                default=5.0,
            )
        ],
        capability_tags=["flight", "station_keeping"],
    ),
    "search_pattern": PrimitiveSpec(
        name="search_pattern",
        handler_name="search_pattern",
        description="Execute a continuous area-coverage pattern around a centre point.",
        parameters=[
            PrimitiveParameterSpec(name="center_x", type="float", description="Pattern centre x", required=True),
            PrimitiveParameterSpec(name="center_y", type="float", description="Pattern centre y", required=True),
            PrimitiveParameterSpec(name="radius", type="float", description="Pattern radius in metres", required=True),
            PrimitiveParameterSpec(
                name="altitude",
                type="float",
                description="Operating altitude in metres",
                required=False,
                default=10.0,
            ),
        ],
        capability_tags=["flight", "area_coverage", "persistent_monitoring"],
        continuous=True,
        notes=["This primitive is continuous and may require an external modify/interrupt to stop."],
    ),
}


def register_primitive(spec: PrimitiveSpec) -> None:
    _REGISTRY[spec.name] = copy.deepcopy(spec)


def list_registered_primitives(names: Optional[Iterable[str]] = None) -> list[PrimitiveSpec]:
    if names is None:
        specs = _REGISTRY.values()
    else:
        specs = (_REGISTRY[name] for name in names if name in _REGISTRY)
    return [copy.deepcopy(spec) for spec in specs]


def get_primitive_spec(name: str) -> PrimitiveSpec:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown primitive '{name}'")
    return copy.deepcopy(_REGISTRY[name])


def get_supported_primitives_for_controller(controller) -> list[PrimitiveSpec]:
    supported: list[PrimitiveSpec] = []
    for spec in list_registered_primitives():
        handler = getattr(controller, spec.handler_name, None)
        if callable(handler):
            supported.append(spec)
    return supported


def derive_capability_tags(primitives: Iterable[PrimitiveSpec]) -> list[str]:
    tags = {
        tag
        for spec in primitives
        for tag in (spec.capability_tags + spec.resource_tags)
    }
    return sorted(tags)


def build_agent_profile(
    *,
    agent_id: str,
    agent_kind: str = "generic_agent",
    primitive_names: Optional[Iterable[str]] = None,
    available_resources: Optional[list[str]] = None,
    hard_constraints: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
) -> AgentProfile:
    primitives = list_registered_primitives(primitive_names)
    return AgentProfile(
        agent_id=agent_id,
        agent_kind=agent_kind,
        available_primitives=primitives,
        available_capabilities=derive_capability_tags(primitives),
        available_resources=list(available_resources or []),
        hard_constraints=list(hard_constraints or []),
        metadata=copy.deepcopy(metadata or {}),
    )


def normalize_primitive_handler_kwargs(spec: PrimitiveSpec, params: dict) -> dict:
    kwargs: dict = {}
    for param in spec.parameters:
        external_key = param.name
        handler_key = param.handler_name or param.name
        if external_key in params:
            kwargs[handler_key] = params[external_key]
        elif param.default is not None:
            kwargs[handler_key] = copy.deepcopy(param.default)
        elif param.required:
            raise ValueError(
                f"Primitive '{spec.name}' is missing required parameter '{external_key}'."
            )
    return kwargs


def format_primitives_for_prompt(primitives: Iterable[PrimitiveSpec]) -> str:
    lines: list[str] = []
    for spec in primitives:
        if spec.parameters:
            params = ", ".join(
                f'"{param.name}": <{param.type}>'
                + (f" // {param.description}" if param.description else "")
                for param in spec.parameters
            )
            param_block = "{" + params + "}"
        else:
            param_block = "{}"

        extras = []
        if spec.capability_tags:
            extras.append(f"capabilities={spec.capability_tags}")
        if spec.continuous:
            extras.append("continuous=true")
        if spec.notes:
            extras.append(f"notes={spec.notes}")
        suffix = f" [{' ; '.join(extras)}]" if extras else ""
        lines.append(f"  {spec.name:<16} → {param_block}{suffix}")
    return "\n".join(lines) if lines else "  (no primitives available)"
