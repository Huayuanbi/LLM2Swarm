"""
operators/drone_lifecycle.py — The 10-second VLM control loop for a single drone.

Loop structure (repeats indefinitely until task queue is empty or shutdown):

  ┌─────────────────────────────────────────────────┐
  │  EXECUTE phase (up to CONTROL_LOOP_INTERVAL s)  │
  │    Run current action from task queue.           │
  │    If action finishes early → advance queue,     │
  │    start next action, keep running until 10 s.  │
  │    If 10 s elapses mid-action → pause.          │
  └───────────────────┬─────────────────────────────┘
                      │ 10 s tick
  ┌───────────────────▼─────────────────────────────┐
  │  PERCEIVE  — hover briefly, capture telemetry   │
  │             + front camera image (Base64)        │
  └───────────────────┬─────────────────────────────┘
                      │
  ┌───────────────────▼─────────────────────────────┐
  │  SYNCHRONISE — read all peer states from pool   │
  │                update own state in pool          │
  └───────────────────┬─────────────────────────────┘
                      │
  ┌───────────────────▼─────────────────────────────┐
  │  THINK — call edge VLM with multimodal prompt   │
  │          (image + telemetry + peers + task)      │
  └───────────────────┬─────────────────────────────┘
                      │
  ┌───────────────────▼─────────────────────────────┐
  │  ACT  — parse VLM response                      │
  │    "continue" → resume current action            │
  │    "modify"   → replace head of task queue,     │
  │                 write observation to pool        │
  └─────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Optional

import config
from controllers.base_controller import DroneController
from memory.pool import SharedMemoryPool
from models.schemas import VLMContinue, VLMModify
from operators.vlm_agent import VLMAgent

logger = logging.getLogger(__name__)

# How long to hover during the perception pause (included in the 10 s budget)
PERCEPTION_HOVER_S = 2.0


class DroneLifecycle:
    """
    Manages the full autonomous lifecycle of a single drone.

    Args:
        drone_id:   Unique identifier, e.g. "drone_1".
        controller: Connected DroneController instance.
        memory:     Shared swarm memory pool.
        vlm:        VLMAgent pointing at the Ollama endpoint.
        initial_tasks: Ordered list of action dicts from the GlobalPlan.
    """

    def __init__(
        self,
        drone_id:        str,
        controller:      DroneController,
        memory:          SharedMemoryPool,
        vlm:             VLMAgent,
        initial_tasks:   list[dict],
        stop_when_empty: bool = False,
    ):
        self.drone_id        = drone_id
        self._ctrl           = controller
        self._memory         = memory
        self._vlm            = vlm
        self._queue:         deque[dict] = deque(initial_tasks)
        self._running        = False
        self._tick_count     = 0
        self._stop_when_empty = stop_when_empty
        # Track which action is currently mid-execution (may survive across ticks)
        self._current_action: Optional[dict] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Spin the control loop until the task queue is exhausted or stop() is called.
        Call this as an asyncio.Task so multiple drones run concurrently.
        """
        self._running = True
        logger.info("[%s] Lifecycle starting. Task queue depth: %d",
                    self.drone_id, len(self._queue))

        await self._memory.update_drone(self.drone_id, status="starting")

        try:
            while self._running:
                # If nothing left in queue:
                if not self._queue and self._current_action is None:
                    if self._stop_when_empty:
                        # Clean exit — used in tests and finite missions
                        logger.info("[%s] Task queue empty — exiting cleanly.", self.drone_id)
                        await self._memory.update_drone(self.drone_id, status="idle")
                        break
                    # Production: hover and keep polling the VLM for reassignment
                    logger.info("[%s] Task queue empty — hovering idle.", self.drone_id)
                    await self._memory.update_drone(self.drone_id, status="idle")
                    await self._ctrl.hover(duration=config.CONTROL_LOOP_INTERVAL)
                    await self._perception_and_replan()
                    continue

                await self._execute_phase()
                await self._perception_and_replan()
                self._tick_count += 1

        except asyncio.CancelledError:
            logger.info("[%s] Lifecycle cancelled — landing.", self.drone_id)
            await self._safe_land()
        except Exception as e:
            logger.error("[%s] Lifecycle crashed: %s", self.drone_id, e, exc_info=True)
            await self._memory.update_drone(self.drone_id, status="error")
            raise
        finally:
            self._running = False
            logger.info("[%s] Lifecycle stopped after %d ticks.",
                        self.drone_id, self._tick_count)

    def stop(self) -> None:
        """Signal the loop to exit cleanly after the current tick."""
        self._running = False

    # ── Execute phase ──────────────────────────────────────────────────────────

    async def _execute_phase(self) -> None:
        """
        Run actions from the queue for up to CONTROL_LOOP_INTERVAL seconds.

        - If the current action completes, pop the next one and keep going.
        - If the interval expires mid-action, let the VLM decide whether to
          continue or replace it on the next tick.
        """
        deadline = time.monotonic() + config.CONTROL_LOOP_INTERVAL - PERCEPTION_HOVER_S

        while time.monotonic() < deadline and self._running:
            # Grab next action if we don't have one in flight
            if self._current_action is None:
                if not self._queue:
                    break
                self._current_action = self._queue.popleft()
                logger.info("[%s] Starting action: %s", self.drone_id, self._current_action)
                await self._memory.update_drone(
                    self.drone_id,
                    status="executing",
                    current_task=self._current_action.get("action", "unknown"),
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            try:
                await asyncio.wait_for(
                    self._ctrl.execute_action(self._current_action),
                    timeout=remaining,
                )
                # Action completed within the remaining window
                logger.info("[%s] Action completed: %s",
                            self.drone_id, self._current_action.get("action"))
                self._current_action = None   # ready for next

            except asyncio.TimeoutError:
                # 10 s tick fired while action was still running — that's fine,
                # the VLM will decide whether to continue or change course
                logger.debug(
                    "[%s] Tick elapsed mid-action '%s' — pausing for perception.",
                    self.drone_id, self._current_action.get("action"),
                )
                break

            except Exception as e:
                logger.error(
                    "[%s] Action '%s' raised an error: %s — skipping.",
                    self.drone_id, self._current_action.get("action"), e,
                )
                self._current_action = None   # drop the failed action

    # ── Perception + VLM phase ────────────────────────────────────────────────

    async def _perception_and_replan(self) -> None:
        """
        1. Hover briefly (stabilise).
        2. Capture telemetry + camera.
        3. Sync memory pool.
        4. Call VLM.
        5. Act on decision.
        """
        # ── 1. Stabilise ──
        await self._ctrl.hover(duration=PERCEPTION_HOVER_S)

        # ── 2. Capture ──
        pos      = await self._ctrl.get_position()
        vel      = await self._ctrl.get_velocity()
        image_b64 = await self._ctrl.get_camera_image_base64()

        # ── 3. Sync pool ──
        await self._memory.update_drone(
            self.drone_id,
            position=list(pos),
            velocity=list(vel),
            status="perceiving",
        )
        peer_states = await self._memory.get_peer_states(self.drone_id)

        logger.info(
            "[%s] Tick %d | pos=(%.1f,%.1f,%.1f) | queue depth=%d | peers=%s",
            self.drone_id, self._tick_count,
            pos[0], pos[1], pos[2],
            len(self._queue) + (1 if self._current_action else 0),
            {pid: s.status for pid, s in peer_states.items()},
        )

        # ── 4. Think ──
        current_label = (
            self._current_action.get("action") if self._current_action
            else (self._queue[0].get("action") if self._queue else "idle")
        )

        decision = await self._vlm.decide(
            drone_id     = self.drone_id,
            position     = pos,
            velocity     = vel,
            status       = "executing" if self._current_action else "idle",
            current_task = current_label,
            image_b64    = image_b64,
            peer_states  = peer_states,
        )

        # ── 5. Act ──
        if isinstance(decision, VLMContinue):
            logger.info("[%s] VLM → continue current task.", self.drone_id)
            await self._memory.update_drone(self.drone_id, status="executing")

        elif isinstance(decision, VLMModify):
            logger.info(
                "[%s] VLM → modify: '%s'  new_action=%s",
                self.drone_id, decision.new_task, decision.new_action,
            )
            # Prepend the new action (overrides whatever was in flight)
            self._current_action = None
            self._queue.appendleft(decision.new_action)

            if decision.memory_update:
                await self._memory.update_drone(
                    self.drone_id,
                    add_observation=decision.memory_update,
                )

            await self._memory.update_drone(
                self.drone_id,
                status="replanning",
                current_task=decision.new_task,
            )

    # ── Helpers ────────────────────────────────────────────────────────────────

    async def _safe_land(self) -> None:
        """Best-effort landing — does not raise."""
        try:
            await self._ctrl.land()
            await self._memory.update_drone(self.drone_id, status="landed")
        except Exception as e:
            logger.warning("[%s] Safe-land failed: %s", self.drone_id, e)


# ── Convenience factory ────────────────────────────────────────────────────────

async def launch_drone(
    drone_id:        str,
    memory:          SharedMemoryPool,
    vlm:             VLMAgent,
    initial_tasks:   list[dict],
    stop_when_empty: bool = False,
) -> None:
    """
    Create + connect a controller, then run the lifecycle to completion.
    Designed to be used directly with asyncio.gather():

        await asyncio.gather(
            launch_drone("drone_1", pool, vlm, tasks_1),
            launch_drone("drone_2", pool, vlm, tasks_2),
        )

    Args:
        stop_when_empty: If True, the lifecycle exits when the task queue is
                         exhausted (useful for tests and finite missions).
                         If False (default), the drone hovers and polls the
                         VLM indefinitely, waiting for new assignments.
    """
    from controllers import make_controller   # lazy import avoids circular deps

    ctrl = make_controller(drone_id)
    await ctrl.connect()
    try:
        lifecycle = DroneLifecycle(
            drone_id        = drone_id,
            controller      = ctrl,
            memory          = memory,
            vlm             = vlm,
            initial_tasks   = initial_tasks,
            stop_when_empty = stop_when_empty,
        )
        await lifecycle.run()
    finally:
        await ctrl.disconnect()
