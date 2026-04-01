"""
controllers/webots_controller.py — Webots backend with a persistent flight loop.

Design
------
Webots devices are tied to the simulator's synchronous ``robot.step()`` loop,
so all low-level flight control lives inside one dedicated background thread
owned by this controller. High-level async actions such as ``takeoff`` and
``go_to_waypoint`` only publish targets / modes and then wait for completion.

This mirrors the project architecture:
  - high-frequency flight loop     → stays inside WebotsController
  - low-frequency agent / VLM loop → stays in DroneLifecycle

The controller continuously:
  1. steps the simulator,
  2. updates telemetry and a cached camera frame,
  3. computes motor commands from the current control mode,
  4. marks finite actions complete once their target is reached.

Important caveat
----------------
Webots' controller API allows only one ``Robot`` instance per process. That
means this class is conceptually "per drone", but in a real multi-drone Webots
setup each drone should typically be driven by its own extern-controller
process. The higher-level swarm architecture still applies one controller /
agent loop per drone.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import math
import threading
import time
from typing import Optional

import config
from controllers.base_controller import DroneController

logger = logging.getLogger(__name__)

try:
    from controller import Robot  # type: ignore

    _WEBOTS_AVAILABLE = True
except ImportError:
    _WEBOTS_AVAILABLE = False
    logger.warning(
        "Webots controller library not found on PYTHONPATH. "
        "WebotsController will raise RuntimeError when used. "
        "Set SIMULATOR_BACKEND=mock to use the pure-Python mock instead."
    )


def _clamp(value: float, value_min: float, value_max: float) -> float:
    return min(max(value, value_min), value_max)


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return angle


class WebotsController(DroneController):
    """
    Async facade around a synchronous Webots DJI Mavic 2 Pro control loop.

    Finite actions set a new mode/target and await completion.
    Long-running actions such as ``search_pattern`` keep running until the
    caller cancels them, at which point the controller snaps back to hover.
    """

    # Constants adapted from the official DJI Mavic 2 Pro sample controller.
    K_VERTICAL_THRUST = 68.5
    K_VERTICAL_OFFSET = 0.6
    K_VERTICAL_P = 3.0
    K_ROLL_P = 50.0
    K_PITCH_P = 30.0
    MAX_YAW_DISTURBANCE = 0.8
    MAX_PITCH_DISTURBANCE = -1.0

    TARGET_XY_TOL = 0.6
    TARGET_Z_TOL = 0.35
    LAND_Z_TOL = 0.15
    STABLE_VEL_TOL = 0.25
    ACTION_POLL_INTERVAL = 0.05
    SEARCH_WAYPOINTS = 8
    IMAGE_UPDATE_INTERVAL_STEPS = 4
    _process_robot_lock = threading.Lock()
    _process_robot_owner: Optional[str] = None

    def __init__(self, drone_id: str, webots_port: Optional[int] = None):
        super().__init__(drone_id)
        idx = int(drone_id.split("_")[-1]) if "_" in drone_id else 0
        self._port = webots_port or (config.WEBOTS_PORT + idx)

        self._robot: Optional[object] = None
        self._camera: Optional[object] = None
        self._gps: Optional[object] = None
        self._imu: Optional[object] = None
        self._gyro: Optional[object] = None
        self._camera_pitch_motor: Optional[object] = None
        self._motors: list[object] = []
        self._timestep: int = 8

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._image_ready_event = threading.Event()
        self._state_lock = threading.Lock()
        self._thread_error: Optional[BaseException] = None

        self._position = [0.0, 0.0, 0.0]
        self._velocity = [0.0, 0.0, 0.0]
        self._rpy = [0.0, 0.0, 0.0]  # roll, pitch, yaw
        self._latest_image_b64 = ""
        self._last_sensor_time: Optional[float] = None
        self._last_sensor_position: Optional[list[float]] = None

        self._mode = "idle"
        self._target_position = [0.0, 0.0, 0.0]
        self._hold_position = [0.0, 0.0, 0.0]
        self._hover_deadline: Optional[float] = None
        self._search_waypoints: list[tuple[float, float, float]] = []
        self._search_index = 0

        self._action_id = 0
        self._completed_action_id = 0
        self._action_done = threading.Event()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def connect(self) -> None:
        if not _WEBOTS_AVAILABLE:
            raise RuntimeError(
                "Webots controller library is not available. "
                "Install Webots and add its controller/python directory to PYTHONPATH."
            )

        with self._process_robot_lock:
            owner = self._process_robot_owner
            if owner is not None and owner != self.drone_id:
                raise RuntimeError(
                    "Webots allows only one Robot() instance per process. "
                    "This controller is per-drone, but multi-drone Webots runs "
                    "must launch one extern-controller process per drone."
                )
            self.__class__._process_robot_owner = self.drone_id

        self._stop_event.clear()
        self._ready_event.clear()
        self._image_ready_event.clear()
        self._thread_error = None

        self._thread = threading.Thread(
            target=self._controller_loop,
            name=f"{self.drone_id}-webots",
            daemon=True,
        )
        self._thread.start()

        try:
            await self._wait_for_ready(timeout=10.0)
        except Exception:
            with self._process_robot_lock:
                if self._process_robot_owner == self.drone_id:
                    self.__class__._process_robot_owner = None
            raise
        self._connected = True
        logger.info("[%s] WebotsController connected.", self.drone_id)

    async def disconnect(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            await asyncio.to_thread(self._thread.join, 5.0)
        with self._process_robot_lock:
            if self._process_robot_owner == self.drone_id:
                self.__class__._process_robot_owner = None
        self._connected = False
        logger.info("[%s] WebotsController disconnected.", self.drone_id)

    # ── Telemetry ──────────────────────────────────────────────────────────

    async def get_position(self) -> tuple[float, float, float]:
        with self._state_lock:
            return tuple(self._position)  # type: ignore[return-value]

    async def get_velocity(self) -> tuple[float, float, float]:
        with self._state_lock:
            return tuple(self._velocity)  # type: ignore[return-value]

    async def get_camera_image_base64(self) -> str:
        deadline = time.monotonic() + 2.0
        while not self._image_ready_event.is_set():
            await self._ensure_no_thread_error()
            if time.monotonic() > deadline:
                break
            await asyncio.sleep(self.ACTION_POLL_INTERVAL)
        await self._ensure_no_thread_error()
        with self._state_lock:
            return self._latest_image_b64

    # ── Skill Library ──────────────────────────────────────────────────────

    async def takeoff(self, altitude: float = 5.0) -> None:
        action_id = self._set_mode_takeoff(altitude)
        await self._wait_for_action(action_id)

    async def land(self) -> None:
        action_id = self._set_mode_land()
        await self._wait_for_action(action_id)

    async def go_to_waypoint(
        self,
        x: float,
        y: float,
        z: float,
        velocity: float = 3.0,
    ) -> None:
        del velocity  # The sample controller uses fixed tuning rather than explicit cruise speed.
        action_id = self._set_mode_goto(x, y, z)
        await self._wait_for_action(action_id)

    async def hover(self, duration: float = 5.0) -> None:
        action_id = self._set_mode_hover(duration)
        await self._wait_for_action(action_id)

    async def search_pattern(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        altitude: float = 10.0,
    ) -> None:
        action_id = self._set_mode_search(center_x, center_y, radius, altitude)
        try:
            while True:
                await self._ensure_no_thread_error()
                if not self.is_connected:
                    raise RuntimeError("Webots controller disconnected during search_pattern.")
                with self._state_lock:
                    if self._action_id != action_id or self._mode != "search":
                        return
                await asyncio.sleep(self.ACTION_POLL_INTERVAL)
        except asyncio.CancelledError:
            self._hold_current_position()
            raise

    # ── Mode setters ───────────────────────────────────────────────────────

    def _set_mode_takeoff(self, altitude: float) -> int:
        with self._state_lock:
            action_id = self._begin_action("takeoff")
            self._hold_position[0] = self._position[0]
            self._hold_position[1] = self._position[1]
            self._target_position = [self._position[0], self._position[1], max(0.5, altitude)]
            logger.info("[%s] Webots mode → takeoff(%.2f)", self.drone_id, altitude)
            return action_id

    def _set_mode_land(self) -> int:
        with self._state_lock:
            action_id = self._begin_action("land")
            self._hold_position[0] = self._position[0]
            self._hold_position[1] = self._position[1]
            self._target_position = [self._position[0], self._position[1], 0.0]
            logger.info("[%s] Webots mode → land", self.drone_id)
            return action_id

    def _set_mode_goto(self, x: float, y: float, z: float) -> int:
        with self._state_lock:
            action_id = self._begin_action("goto")
            self._target_position = [x, y, max(0.5, z)]
            logger.info("[%s] Webots mode → goto(%.2f, %.2f, %.2f)", self.drone_id, x, y, z)
            return action_id

    def _set_mode_hover(self, duration: float) -> int:
        with self._state_lock:
            action_id = self._begin_action("hover")
            self._hold_position = list(self._position)
            self._target_position = list(self._hold_position)
            self._hover_deadline = time.monotonic() + max(0.0, duration)
            logger.info("[%s] Webots mode → hover(%.2f)", self.drone_id, duration)
            return action_id

    def _set_mode_search(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        altitude: float,
    ) -> int:
        with self._state_lock:
            action_id = self._begin_action("search")
            self._search_waypoints = self._build_search_waypoints(
                center_x=center_x,
                center_y=center_y,
                radius=radius,
                altitude=max(0.5, altitude),
            )
            self._search_index = 0
            self._target_position = list(self._search_waypoints[0])
            logger.info(
                "[%s] Webots mode → search(center=(%.2f, %.2f), radius=%.2f, altitude=%.2f)",
                self.drone_id,
                center_x,
                center_y,
                radius,
                altitude,
            )
            return action_id

    def _hold_current_position(self) -> None:
        with self._state_lock:
            self._mode = "hover"
            self._hold_position = list(self._position)
            self._target_position = list(self._hold_position)
            self._hover_deadline = None
            self._action_done.clear()

    def _begin_action(self, mode: str) -> int:
        self._action_id += 1
        self._mode = mode
        self._hover_deadline = None
        self._action_done.clear()
        return self._action_id

    # ── Thread loop ────────────────────────────────────────────────────────

    def _controller_loop(self) -> None:
        try:
            self._sync_connect()
            self._ready_event.set()

            image_step_counter = 0
            while not self._stop_event.is_set():
                assert self._robot is not None
                if self._robot.step(self._timestep) == -1:
                    break

                now = time.monotonic()
                self._update_sensor_cache(now)

                image_step_counter += 1
                if image_step_counter >= self.IMAGE_UPDATE_INTERVAL_STEPS:
                    self._refresh_camera_cache()
                    image_step_counter = 0

                self._update_search_progress()
                motor_inputs = self._compute_motor_inputs(now)
                self._apply_motor_inputs(*motor_inputs)
        except BaseException as exc:
            self._thread_error = exc
            logger.error("[%s] Webots control loop crashed: %s", self.drone_id, exc, exc_info=True)
        finally:
            self._ready_event.set()
            self._stop_motors()

    def _sync_connect(self) -> None:
        self._robot = Robot()
        self._timestep = int(self._robot.getBasicTimeStep())

        self._camera = self._robot.getDevice("camera")
        self._imu = self._robot.getDevice("inertial unit")
        self._gps = self._robot.getDevice("gps")
        self._gyro = self._robot.getDevice("gyro")
        self._camera_pitch_motor = self._robot.getDevice("camera pitch")

        self._camera.enable(self._timestep)
        self._imu.enable(self._timestep)
        self._gps.enable(self._timestep)
        self._gyro.enable(self._timestep)
        self._camera_pitch_motor.setPosition(0.7)

        motor_names = [
            "front left propeller",
            "front right propeller",
            "rear left propeller",
            "rear right propeller",
        ]
        self._motors = [self._robot.getDevice(name) for name in motor_names]
        for motor in self._motors:
            motor.setPosition(float("inf"))
            motor.setVelocity(1.0)

    def _update_sensor_cache(self, now: float) -> None:
        assert self._imu is not None
        assert self._gps is not None

        roll, pitch, yaw = self._imu.getRollPitchYaw()
        x_pos, y_pos, altitude = self._gps.getValues()
        current_position = [float(x_pos), float(y_pos), float(altitude)]

        if self._last_sensor_time is None or self._last_sensor_position is None:
            velocity = [0.0, 0.0, 0.0]
        else:
            dt = max(now - self._last_sensor_time, 1e-6)
            velocity = [
                (current_position[i] - self._last_sensor_position[i]) / dt
                for i in range(3)
            ]

        with self._state_lock:
            self._position = current_position
            self._velocity = velocity
            self._rpy = [float(roll), float(pitch), float(yaw)]
            if self._mode == "idle" and self._target_position == [0.0, 0.0, 0.0]:
                self._hold_position = list(current_position)
                self._target_position = list(current_position)

        self._last_sensor_time = now
        self._last_sensor_position = current_position

    def _refresh_camera_cache(self) -> None:
        assert self._camera is not None
        raw_bytes = self._camera.getImage()
        if not raw_bytes:
            return

        width = self._camera.getWidth()
        height = self._camera.getHeight()

        from PIL import Image as PILImage

        img = PILImage.frombytes("RGBA", (width, height), raw_bytes, "raw", "BGRA")
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

        with self._state_lock:
            self._latest_image_b64 = encoded
        self._image_ready_event.set()

    def _update_search_progress(self) -> None:
        with self._state_lock:
            if self._mode != "search" or not self._search_waypoints:
                return
            target = self._search_waypoints[self._search_index]
            if self._target_reached(target):
                self._search_index = (self._search_index + 1) % len(self._search_waypoints)
                self._target_position = list(self._search_waypoints[self._search_index])

    def _compute_motor_inputs(self, now: float) -> tuple[float, float, float, float]:
        assert self._gyro is not None

        with self._state_lock:
            x_pos, y_pos, altitude = self._position
            vx, vy, vz = self._velocity
            roll, pitch, yaw = self._rpy
            mode = self._mode
            target_x, target_y, target_z = self._target_position
            hover_deadline = self._hover_deadline
            action_id = self._action_id

        roll_acceleration, pitch_acceleration, _ = self._gyro.getValues()
        roll_disturbance = 0.0
        yaw_disturbance = 0.0
        pitch_disturbance = 0.0

        dx = target_x - x_pos
        dy = target_y - y_pos
        dz = target_z - altitude
        planar_distance = math.hypot(dx, dy)

        if mode in {"takeoff", "goto", "search"}:
            desired_yaw = yaw if planar_distance < 1e-3 else math.atan2(dy, dx)
            angle_left = _normalize_angle(desired_yaw - yaw)
            yaw_disturbance = _clamp(
                self.MAX_YAW_DISTURBANCE * angle_left / math.pi,
                -self.MAX_YAW_DISTURBANCE,
                self.MAX_YAW_DISTURBANCE,
            )

            if abs(angle_left) < 1.0:
                pitch_disturbance = _clamp(
                    -0.08 * planar_distance,
                    self.MAX_PITCH_DISTURBANCE,
                    0.2,
                )
            else:
                pitch_disturbance = 0.0

        elif mode in {"hover", "land", "idle"}:
            yaw_disturbance = 0.0
            pitch_disturbance = 0.0

        clamped_difference_altitude = _clamp(
            target_z - altitude + self.K_VERTICAL_OFFSET,
            -1.0,
            1.0,
        )
        vertical_input = self.K_VERTICAL_P * pow(clamped_difference_altitude, 3.0)
        roll_input = self.K_ROLL_P * _clamp(roll, -1.0, 1.0) + roll_acceleration + roll_disturbance
        pitch_input = self.K_PITCH_P * _clamp(pitch, -1.0, 1.0) + pitch_acceleration + pitch_disturbance
        yaw_input = yaw_disturbance

        self._maybe_mark_action_complete(
            mode=mode,
            now=now,
            target=(target_x, target_y, target_z),
            velocity=(vx, vy, vz),
            action_id=action_id,
            hover_deadline=hover_deadline,
        )

        front_left = self.K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input
        front_right = self.K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input
        rear_left = self.K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input
        rear_right = self.K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input
        return front_left, front_right, rear_left, rear_right

    def _apply_motor_inputs(
        self,
        front_left: float,
        front_right: float,
        rear_left: float,
        rear_right: float,
    ) -> None:
        if len(self._motors) != 4:
            return

        self._motors[0].setVelocity(front_left)
        self._motors[1].setVelocity(-front_right)
        self._motors[2].setVelocity(-rear_left)
        self._motors[3].setVelocity(rear_right)

    def _stop_motors(self) -> None:
        if len(self._motors) != 4:
            return
        try:
            for motor in self._motors:
                motor.setVelocity(0.0)
        except Exception:
            pass

    # ── Action completion ──────────────────────────────────────────────────

    def _maybe_mark_action_complete(
        self,
        *,
        mode: str,
        now: float,
        target: tuple[float, float, float],
        velocity: tuple[float, float, float],
        action_id: int,
        hover_deadline: Optional[float],
    ) -> None:
        if mode == "takeoff":
            if self._target_reached(target) and abs(velocity[2]) < self.STABLE_VEL_TOL:
                self._complete_action(action_id, hold_at_target=True)
        elif mode == "goto":
            if self._target_reached(target):
                self._complete_action(action_id, hold_at_target=True)
        elif mode == "hover":
            if hover_deadline is not None and now >= hover_deadline:
                self._complete_action(action_id, hold_at_target=True)
        elif mode == "land":
            with self._state_lock:
                altitude = self._position[2]
            if altitude <= self.LAND_Z_TOL and abs(velocity[2]) < self.STABLE_VEL_TOL:
                self._complete_action(action_id, hold_at_target=False)

    def _complete_action(self, action_id: int, *, hold_at_target: bool) -> None:
        with self._state_lock:
            if action_id != self._action_id:
                return
            self._completed_action_id = action_id
            self._action_done.set()

            if hold_at_target:
                self._mode = "hover"
                self._hold_position = list(self._target_position)
                self._target_position = list(self._hold_position)
                self._hover_deadline = time.monotonic()
            else:
                self._mode = "idle"
                self._hold_position = list(self._position)
                self._target_position = list(self._position)
                self._hover_deadline = None

    def _target_reached(self, target: tuple[float, float, float] | list[float]) -> bool:
        x, y, z = target
        dx = x - self._position[0]
        dy = y - self._position[1]
        dz = z - self._position[2]
        return math.hypot(dx, dy) <= self.TARGET_XY_TOL and abs(dz) <= self.TARGET_Z_TOL

    # ── Async wait helpers ─────────────────────────────────────────────────

    async def _wait_for_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while not self._ready_event.is_set():
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out while waiting for Webots controller thread to start.")
            await asyncio.sleep(self.ACTION_POLL_INTERVAL)
        await self._ensure_no_thread_error()

    async def _wait_for_action(self, action_id: int) -> None:
        try:
            while True:
                await self._ensure_no_thread_error()
                with self._state_lock:
                    if self._completed_action_id >= action_id and self._action_done.is_set():
                        return
                await asyncio.sleep(self.ACTION_POLL_INTERVAL)
        except asyncio.CancelledError:
            self._hold_current_position()
            raise

    async def _ensure_no_thread_error(self) -> None:
        if self._thread_error is not None:
            raise RuntimeError(f"Webots control loop failed: {self._thread_error}") from self._thread_error

    # ── Search helpers ─────────────────────────────────────────────────────

    def _build_search_waypoints(
        self,
        *,
        center_x: float,
        center_y: float,
        radius: float,
        altitude: float,
    ) -> list[tuple[float, float, float]]:
        points = []
        n_points = max(4, self.SEARCH_WAYPOINTS)
        for i in range(n_points):
            angle = 2.0 * math.pi * i / n_points
            points.append(
                (
                    center_x + radius * math.cos(angle),
                    center_y + radius * math.sin(angle),
                    altitude,
                )
            )
        return points
