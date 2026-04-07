"""
controllers/base_controller.py — Abstract interface for all simulator backends.

Any new simulator (Webots, AirSim, SITL …) just subclasses DroneController and
implements the abstract methods.  The rest of the system only ever calls methods
defined here, keeping the operator/VLM logic simulator-agnostic.
"""

from __future__ import annotations
import logging
from abc import ABC, abstractmethod

from models.schemas import AgentProfile, PrimitiveSpec
from primitives.registry import (
    build_agent_profile,
    derive_capability_tags,
    get_primitive_spec,
    get_supported_primitives_for_controller,
    normalize_primitive_handler_kwargs,
)

logger = logging.getLogger(__name__)


class DroneController(ABC):
    """
    High-level skill library for a single drone.

    All methods are async so the event loop never blocks waiting for a
    slow simulator command to complete.
    """

    def __init__(self, drone_id: str):
        self.drone_id = drone_id
        self._connected = False

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the simulator backend."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully release the simulator connection."""
        ...

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Telemetry ──────────────────────────────────────────────────────────────

    @abstractmethod
    async def get_position(self) -> tuple[float, float, float]:
        """Return current (x, y, z) in metres (NED or ENU — consistent per backend)."""
        ...

    @abstractmethod
    async def get_velocity(self) -> tuple[float, float, float]:
        """Return current (vx, vy, vz) in m/s."""
        ...

    @abstractmethod
    async def get_camera_image_base64(self) -> str:
        """
        Capture a front-facing camera frame and return it as a Base64-encoded
        JPEG string suitable for direct embedding in an OpenAI vision message.
        """
        ...

    # ── Skill Library ─────────────────────────────────────────────────────────

    @abstractmethod
    async def takeoff(self, altitude: float = 5.0) -> None:
        """
        Arm and take off to the specified altitude (metres).
        Returns once the drone has reached the target altitude.
        """
        ...

    @abstractmethod
    async def land(self) -> None:
        """Descend and disarm. Returns once on the ground."""
        ...

    @abstractmethod
    async def go_to_waypoint(
        self,
        x: float,
        y: float,
        z: float,
        velocity: float = 3.0,
    ) -> None:
        """
        Fly to absolute position (x, y, z) at the given velocity (m/s).
        Returns once the drone has reached within tolerance of the target.
        """
        ...

    @abstractmethod
    async def hover(self, duration: float = 5.0) -> None:
        """
        Hold current position for `duration` seconds.
        Used as the "pause" step at the start of each perception tick.
        """
        ...

    @abstractmethod
    async def search_pattern(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        altitude: float = 10.0,
    ) -> None:
        """
        Execute a continuous circular search orbit at the given centre and radius.
        Loops indefinitely — never returns on its own. The VLM must issue a
        "modify" decision to cancel it and move to the next task.
        """
        ...

    def get_available_primitive_specs(self) -> list[PrimitiveSpec]:
        return get_supported_primitives_for_controller(self)

    def get_available_primitive_names(self) -> list[str]:
        return [spec.name for spec in self.get_available_primitive_specs()]

    def get_capability_tags(self) -> list[str]:
        return derive_capability_tags(self.get_available_primitive_specs())

    def get_agent_profile(self) -> AgentProfile:
        return build_agent_profile(
            agent_id=self.drone_id,
            agent_kind=self.__class__.__name__,
            primitive_names=self.get_available_primitive_names(),
        )

    # ── Convenience helpers (shared across all backends) ──────────────────────

    async def execute_action(self, action: dict) -> None:
        """
        Dispatch a raw action dict (as produced by the LLM) to the matching skill.
        Raises ValueError for unknown action names.
        """
        name   = action.get("action", "")
        params = action.get("params", {})

        try:
            spec = get_primitive_spec(name)
        except KeyError:
            raise ValueError(
                f"[{self.drone_id}] Unknown action '{name}'. "
                f"Valid actions: {self.get_available_primitive_names()}"
            )

        handler = getattr(self, spec.handler_name, None)
        if not callable(handler):
            raise ValueError(
                f"[{self.drone_id}] Unsupported action '{name}' for this controller. "
                f"Available actions: {self.get_available_primitive_names()}"
            )

        kwargs = normalize_primitive_handler_kwargs(spec, params)
        logger.info("[%s] Executing skill: %s  params=%s", self.drone_id, name, params)
        await handler(**kwargs)
