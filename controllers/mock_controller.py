"""
controllers/mock_controller.py — Pure-Python mock simulator backend.

Simulates drone physics good enough to exercise all control logic, the memory
pool, the VLM loop, and the LLM planner — without any external simulator.

Physics model:
  - Position advances toward the target waypoint at MOCK_MOVE_SPEED each tick.
  - Camera images are procedurally generated coloured frames (Pillow) that
    include the drone ID and current position as overlay text, making them
    visually distinct and parseable by a real VLM.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import math
import time
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

import config
from controllers.base_controller import DroneController

logger = logging.getLogger(__name__)

# Colour palette so different drones produce visually distinct images
_DRONE_COLOURS = {
    "drone_1": (30, 120, 200),   # blue-ish sky
    "drone_2": (200,  80,  30),  # amber/fire
    "drone_3": ( 40, 160,  80),  # green/forest
}
_DEFAULT_COLOUR = (100, 100, 100)


class MockDroneController(DroneController):
    """
    Simulated drone: no external dependencies, runs fully inside asyncio.
    Suitable for unit tests and end-to-end logic validation.
    """

    def __init__(
        self,
        drone_id: str,
        sim_positions: Optional[dict] = None,
    ):
        super().__init__(drone_id)
        # --- State ---
        self._pos: list[float] = [0.0, 0.0, 0.0]    # [x, y, z]
        self._vel: list[float] = [0.0, 0.0, 0.0]    # [vx, vy, vz]
        self._armed: bool = False
        self._target: Optional[list[float]] = None
        self._move_speed: float = config.MOCK_MOVE_SPEED
        self._tick_task: Optional[asyncio.Task] = None
        self._frame_counter: int = 0                 # increments each image capture
        # Shared simulator state dict written by physics loop, read by visualizer
        self._sim_positions: Optional[dict] = sim_positions

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        self._connected = True
        # Start the background physics tick loop
        self._tick_task = asyncio.create_task(
            self._physics_loop(), name=f"{self.drone_id}_physics"
        )
        logger.info("[%s] MockController connected.", self.drone_id)

    async def disconnect(self) -> None:
        if self._tick_task and not self._tick_task.done():
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass
        self._connected = False
        logger.info("[%s] MockController disconnected.", self.drone_id)

    # ── Physics loop (runs in background) ─────────────────────────────────────

    async def _physics_loop(self) -> None:
        """
        Advances position toward _target at _move_speed.
        Runs at ~20 Hz (MOCK_TICK_RATE seconds per tick).
        """
        while True:
            await asyncio.sleep(config.MOCK_TICK_RATE)
            if self._target is None or not self._armed:
                self._vel = [0.0, 0.0, 0.0]
                continue

            dx = self._target[0] - self._pos[0]
            dy = self._target[1] - self._pos[1]
            dz = self._target[2] - self._pos[2]
            dist = math.sqrt(dx**2 + dy**2 + dz**2)

            if dist < 0.1:   # close enough — snap and stop
                self._pos = list(self._target)
                self._vel = [0.0, 0.0, 0.0]
                self._target = None
                continue

            # Move one tick's worth toward target
            step = min(self._move_speed * config.MOCK_TICK_RATE, dist)
            scale = step / dist
            self._vel  = [dx * scale / config.MOCK_TICK_RATE,
                           dy * scale / config.MOCK_TICK_RATE,
                           dz * scale / config.MOCK_TICK_RATE]
            self._pos[0] += dx * scale
            self._pos[1] += dy * scale
            self._pos[2] += dz * scale

            # Publish to simulator bridge so visualizer can read without going
            # through the memory pool (which is only updated every 10 s)
            if self._sim_positions is not None:
                self._sim_positions[self.drone_id] = list(self._pos)

    async def _fly_to(self, x: float, y: float, z: float, velocity: float) -> None:
        """
        Set target and block until the physics loop reaches it (or times out).
        """
        self._move_speed = velocity
        self._target = [x, y, z]

        deadline = time.monotonic() + 120.0   # 2-minute hard timeout
        while self._target is not None:
            if time.monotonic() > deadline:
                logger.warning("[%s] fly_to(%s,%s,%s) timed out.", self.drone_id, x, y, z)
                self._target = None
                break
            await asyncio.sleep(config.MOCK_TICK_RATE * 5)  # poll at 4 Hz

    # ── Telemetry ──────────────────────────────────────────────────────────────

    async def get_position(self) -> tuple[float, float, float]:
        return tuple(self._pos)                          # type: ignore[return-value]

    async def get_velocity(self) -> tuple[float, float, float]:
        return tuple(self._vel)                          # type: ignore[return-value]

    async def get_camera_image_base64(self) -> str:
        """
        Generate a synthetic 320×240 JPEG with drone ID, position, and a
        frame counter overlay.  The background hue is unique per drone so the
        VLM can visually differentiate agents in multi-drone prompts.
        """
        self._frame_counter += 1
        colour = _DRONE_COLOURS.get(self.drone_id, _DEFAULT_COLOUR)
        x, y, z = self._pos

        # Build a simple gradient background in the drone's colour
        img = Image.new("RGB", (config.MOCK_IMAGE_WIDTH, config.MOCK_IMAGE_HEIGHT), color=colour)
        draw = ImageDraw.Draw(img)

        # Lighter rectangle to simulate horizon/ground split
        horizon_y = int(config.MOCK_IMAGE_HEIGHT * 0.55)
        ground_col = tuple(max(0, c - 60) for c in colour)
        draw.rectangle(
            [(0, horizon_y), (config.MOCK_IMAGE_WIDTH, config.MOCK_IMAGE_HEIGHT)],
            fill=ground_col,
        )

        # Overlay text: drone ID, position, frame number
        overlay_lines = [
            f"ID: {self.drone_id}",
            f"Pos: ({x:.1f}, {y:.1f}, {z:.1f}) m",
            f"Frame: {self._frame_counter}",
            f"Status: {'armed' if self._armed else 'disarmed'}",
        ]
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except (IOError, OSError):
            font = ImageFont.load_default()

        margin = 8
        for i, line in enumerate(overlay_lines):
            draw.text((margin, margin + i * 24), line, fill=(255, 255, 255), font=font)

        # Encode to Base64 JPEG
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ── Skill Library ─────────────────────────────────────────────────────────

    async def takeoff(self, altitude: float = 5.0) -> None:
        logger.info("[%s] Takeoff → %.1f m", self.drone_id, altitude)
        self._armed = True
        await self._fly_to(self._pos[0], self._pos[1], altitude, velocity=2.0)
        logger.info("[%s] Reached altitude %.1f m", self.drone_id, altitude)

    async def land(self) -> None:
        logger.info("[%s] Landing …", self.drone_id)
        await self._fly_to(self._pos[0], self._pos[1], 0.0, velocity=1.5)
        self._armed = False
        logger.info("[%s] Landed.", self.drone_id)

    async def go_to_waypoint(
        self,
        x: float,
        y: float,
        z: float,
        velocity: float = 3.0,
    ) -> None:
        logger.info("[%s] go_to_waypoint → (%.1f, %.1f, %.1f) @ %.1f m/s",
                    self.drone_id, x, y, z, velocity)
        await self._fly_to(x, y, z, velocity)
        logger.info("[%s] Reached waypoint (%.1f, %.1f, %.1f)", self.drone_id, x, y, z)

    async def hover(self, duration: float = 5.0) -> None:
        logger.info("[%s] Hovering for %.1f s …", self.drone_id, duration)
        self._target = None   # stop any active movement
        self._vel    = [0.0, 0.0, 0.0]
        await asyncio.sleep(duration)
        logger.info("[%s] Hover complete.", self.drone_id)

    async def search_pattern(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        altitude: float = 10.0,
    ) -> None:
        """
        Continuous circular search orbit: loops forever until cancelled.
        The VLM is expected to issue a "modify" decision to stop the search.
        """
        logger.info(
            "[%s] search_pattern: centre=(%.1f,%.1f) r=%.1f alt=%.1f — looping until VLM stops",
            self.drone_id, center_x, center_y, radius, altitude,
        )
        n_points = 8
        lap = 0
        while True:
            lap += 1
            logger.info("[%s] search_pattern: starting lap %d", self.drone_id, lap)
            for i in range(n_points):
                angle = 2 * math.pi * i / n_points
                wp_x  = center_x + radius * math.cos(angle)
                wp_y  = center_y + radius * math.sin(angle)
                await self.go_to_waypoint(wp_x, wp_y, altitude, velocity=4.0)
            logger.info("[%s] search_pattern: lap %d complete.", self.drone_id, lap)
