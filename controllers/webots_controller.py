"""
controllers/webots_controller.py — Webots extern-controller backend.

Architecture note
─────────────────
Webots supports "extern controllers": the robot controller runs as an external
process and communicates with Webots via a TCP socket.  Each drone in the
Webots world file must be configured with:

    supervisor { TRUE }
    controller  "<extern>"
    controllerArgs ["--port=10021"]   # 10020 + drone index

Then run the Webots simulation; this class connects to each drone's port.

The Webots Python library (`controller` package) is bundled with Webots and
must be on PYTHONPATH.  Typical path on macOS:

    export PYTHONPATH="$WEBOTS_HOME/lib/controller/python:$PYTHONPATH"

where WEBOTS_HOME is usually /Applications/Webots.app/Contents.

This file is a *stub with full interface*: all methods raise NotImplementedError
until wired up to the Webots Robot / Supervisor objects.  The goal is to show
exactly where simulator-specific code belongs so integration is straightforward.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import Optional

import config
from controllers.base_controller import DroneController

logger = logging.getLogger(__name__)

try:
    # Only importable when PYTHONPATH includes Webots controller lib
    from controller import Robot, Camera, Motor, GPS, InertialUnit  # type: ignore
    _WEBOTS_AVAILABLE = True
except ImportError:
    _WEBOTS_AVAILABLE = False
    logger.warning(
        "Webots controller library not found on PYTHONPATH. "
        "WebotsController will raise NotImplementedError on all calls. "
        "Set SIMULATOR_BACKEND=mock to use the pure-Python mock instead."
    )


class WebotsController(DroneController):
    """
    Thin async wrapper around the synchronous Webots Python controller API.

    All blocking Webots calls are dispatched via asyncio.get_event_loop()
    .run_in_executor(None, …) so they don't block the async event loop.
    """

    def __init__(self, drone_id: str, webots_port: Optional[int] = None):
        super().__init__(drone_id)
        # Port = base port + drone index (e.g. drone_1 → 10021)
        idx = int(drone_id.split("_")[-1]) if "_" in drone_id else 0
        self._port = webots_port or (config.WEBOTS_PORT + idx)
        self._robot: Optional[object]   = None
        self._camera: Optional[object]  = None
        self._gps: Optional[object]     = None
        self._timestep: int             = 32   # ms, must match Webots world setting

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        if not _WEBOTS_AVAILABLE:
            raise RuntimeError(
                "Webots controller library is not available. "
                "Install Webots and add its controller/python directory to PYTHONPATH."
            )
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_connect)
        self._connected = True
        logger.info("[%s] WebotsController connected on port %d", self.drone_id, self._port)

    def _sync_connect(self) -> None:
        """Synchronous Webots initialisation (runs in thread pool)."""
        self._robot  = Robot()
        # Device names must match what is in the Webots .wbt world file
        self._camera = self._robot.getDevice("camera")       # type: ignore[union-attr]
        self._gps    = self._robot.getDevice("gps")          # type: ignore[union-attr]
        self._camera.enable(self._timestep)                  # type: ignore[union-attr]
        self._gps.enable(self._timestep)                     # type: ignore[union-attr]

    async def disconnect(self) -> None:
        # Webots Robot object is cleaned up when process exits; nothing explicit needed
        self._connected = False
        logger.info("[%s] WebotsController disconnected.", self.drone_id)

    # ── Telemetry ──────────────────────────────────────────────────────────────

    async def get_position(self) -> tuple[float, float, float]:
        loop = asyncio.get_event_loop()
        values = await loop.run_in_executor(None, self._gps.getValues)  # type: ignore[union-attr]
        return float(values[0]), float(values[1]), float(values[2])

    async def get_velocity(self) -> tuple[float, float, float]:
        # TODO: wire up to Webots Accelerometer or InertialUnit as needed
        raise NotImplementedError("Velocity telemetry not yet implemented for Webots backend.")

    async def get_camera_image_base64(self) -> str:
        loop = asyncio.get_event_loop()
        raw_bytes = await loop.run_in_executor(None, self._camera.getImage)  # type: ignore[union-attr]
        width  = self._camera.getWidth()   # type: ignore[union-attr]
        height = self._camera.getHeight()  # type: ignore[union-attr]

        # Webots returns BGRA bytes; convert to RGB PIL image
        from PIL import Image as PILImage
        img = PILImage.frombytes("RGBA", (width, height), raw_bytes)
        img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ── Skill Library ─────────────────────────────────────────────────────────
    # These are stubs.  Real Webots drone control uses Motor velocity commands
    # on the four rotors (or the DJI Mavic2Pro model's built-in motion manager).
    # See: https://www.cyberbotics.com/doc/guide/mavic-2-pro

    async def takeoff(self, altitude: float = 5.0) -> None:
        # TODO: ramp up motor thrust until GPS z-component reaches altitude
        raise NotImplementedError("takeoff not yet implemented for Webots backend.")

    async def land(self) -> None:
        raise NotImplementedError("land not yet implemented for Webots backend.")

    async def go_to_waypoint(
        self, x: float, y: float, z: float, velocity: float = 3.0
    ) -> None:
        # TODO: PID controller driving motor velocities toward (x, y, z)
        raise NotImplementedError("go_to_waypoint not yet implemented for Webots backend.")

    async def hover(self, duration: float = 5.0) -> None:
        # TODO: hold current GPS position via PID for `duration` seconds
        raise NotImplementedError("hover not yet implemented for Webots backend.")

    async def search_pattern(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        altitude: float = 10.0,
    ) -> None:
        raise NotImplementedError("search_pattern not yet implemented for Webots backend.")
