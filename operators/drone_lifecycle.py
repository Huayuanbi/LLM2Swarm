"""
operators/drone_lifecycle.py — The 10-second VLM control loop for a single drone.

Loop structure (repeats indefinitely until task queue is exhausted or shutdown):

  ┌─────────────────────────────────────────────────────────────┐
  │  Snapshot + fire VLM query (async task, runs in background) │
  │    Capture pos / vel / camera image at tick start.          │
  │    VLM call happens concurrently — drone never stops.       │
  └──────────────┬──────────────────────────────────────────────┘
                 │  (VLM thinking in background)
  ┌──────────────▼──────────────────────────────────────────────┐
  │  EXECUTE — run task queue for CONTROL_LOOP_INTERVAL seconds │
  │    Actions continue uninterrupted while VLM thinks.         │
  │    If all tasks finish early → keep running until VLM done. │
  └──────────────┬──────────────────────────────────────────────┘
                 │  (await VLM result — usually already ready)
  ┌──────────────▼──────────────────────────────────────────────┐
  │  ACT — apply VLM decision                                   │
  │    "continue" → no change, next tick begins immediately     │
  │    "modify"   → prepend new action, write observation       │
  └─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Optional, Union

import config
from controllers.base_controller import DroneController
from memory.pool import SharedMemoryPool
from models.schemas import VLMContinue, VLMModify
from operators.vlm_agent import VLMAgent

logger = logging.getLogger(__name__)



def _fmt_action(task: dict) -> str:
    """Compact one-line summary of an action dict for display."""
    action = task.get("action", "?")
    p = task.get("params", {})
    if action == "takeoff":
        return f"takeoff({p.get('altitude', 0):.0f}m)"
    if action == "go_to_waypoint":
        return f"goto({p.get('x', 0):.0f},{p.get('y', 0):.0f},{p.get('z', 0):.0f})"
    if action == "search_pattern":
        return f"search(r={p.get('radius', 0):.0f}m)"
    if action == "hover":
        return f"hover({p.get('duration', 0):.0f}s)"
    if action == "land":
        return "land"
    return action


class DroneLifecycle:
    """
    Manages the full autonomous lifecycle of a single drone.

    The VLM query fires concurrently at the start of each tick so the drone
    never pauses to wait for the model — it keeps executing its task queue
    throughout the inference window and only applies the decision afterwards.

    Args:
        drone_id:     Unique identifier, e.g. "drone_1".
        controller:   Connected DroneController instance.
        memory:       Shared swarm memory pool.
        vlm:          VLMAgent pointing at the Ollama endpoint.
        initial_tasks: Ordered list of action dicts from the GlobalPlan.
        task_queues:  Optional shared dict ``{drone_id: [label, ...]}`` kept
                      current so the visualizer can display remaining tasks.
    """

    def __init__(
        self,
        drone_id:        str,
        controller:      DroneController,
        memory:          SharedMemoryPool,
        vlm:             VLMAgent,
        initial_tasks:   list[dict],
        stop_when_empty: bool = False,
        vlm_log:         Optional[deque] = None,
        task_queues:     Optional[dict]  = None,
    ):
        self.drone_id         = drone_id
        self._ctrl            = controller
        self._memory          = memory
        self._vlm             = vlm
        self._queue:          deque[dict] = deque(initial_tasks)
        self._running         = False
        self._tick_count      = 0
        self._stop_when_empty = stop_when_empty
        self._vlm_log         = vlm_log
        self._task_queues     = task_queues
        # Track which action is currently mid-execution (survives across ticks)
        self._current_action: Optional[dict] = None

        # Publish initial queue to the display dict
        self._sync_task_queues()

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Spin the control loop until the task queue is exhausted or stop() is called.

        Each action runs to natural completion.  Every CONTROL_LOOP_INTERVAL seconds
        a VLM tick fires concurrently via asyncio.shield — the action is never
        cancelled by the timer, only by an explicit VLM "modify" decision.
        """
        self._running = True
        logger.info("[%s] Lifecycle starting. Task queue depth: %d",
                    self.drone_id, len(self._queue))

        await self._memory.update_drone(self.drone_id, status="starting")

        try:
            while self._running:
                # ── Empty queue ──────────────────────────────────────────────
                if not self._queue and self._current_action is None:
                    if self._stop_when_empty:
                        logger.info("[%s] Task queue empty — exiting cleanly.", self.drone_id)
                        await self._memory.update_drone(self.drone_id, status="idle")
                        break
                    logger.info("[%s] Task queue empty — idling.", self.drone_id)
                    await self._memory.update_drone(self.drone_id, status="idle")
                    await asyncio.sleep(config.CONTROL_LOOP_INTERVAL)
                    decision = await self._query_vlm()
                    await self._apply_decision(decision)
                    self._tick_count += 1
                    continue

                # ── Pop next action ───────────────────────────────────────────
                if self._current_action is None:
                    self._current_action = self._pop_next()
                    logger.info("[%s] Starting action: %s", self.drone_id, self._current_action)
                    await self._memory.update_drone(
                        self.drone_id,
                        status="executing",
                        current_task=self._current_action.get("action", "unknown"),
                    )
                    self._sync_task_queues()

                # ── Run action; VLM fires every CONTROL_LOOP_INTERVAL ─────────
                # asyncio.shield keeps the action running when wait_for times out.
                # Only a "modify" decision cancels it explicitly.
                action_task = asyncio.create_task(
                    self._ctrl.execute_action(self._current_action),
                    name=f"{self.drone_id}-action",
                )
                try:
                    await self._run_action_with_vlm_ticks(action_task)
                except Exception:
                    if not action_task.done():
                        action_task.cancel()
                        try:
                            await action_task
                        except (asyncio.CancelledError, Exception):
                            pass
                    raise

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

    async def _run_action_with_vlm_ticks(self, action_task: asyncio.Task) -> None:
        """
        Wait for ``action_task`` to finish, firing a VLM tick every
        CONTROL_LOOP_INTERVAL seconds while it runs.

        The VLM query is launched as a background task at the START of each
        tick window so it runs concurrently with the action — the drone never
        pauses to wait for the model.

        - ``continue`` → let the action keep running.
        - ``modify``   → cancel the action, apply new task.
        """
        while self._running:
            # Fire VLM at tick start — runs concurrently with action execution
            vlm_task = asyncio.create_task(
                self._query_vlm(),
                name=f"{self.drone_id}-vlm",
            )
            try:
                # Wait up to one tick interval; shield keeps action alive on timeout
                await asyncio.wait_for(
                    asyncio.shield(action_task),
                    timeout=config.CONTROL_LOOP_INTERVAL,
                )
                # Action completed within this tick window
                logger.info("[%s] Action completed: %s",
                            self.drone_id,
                            self._current_action.get("action") if self._current_action else "?")
                self._current_action = None
                self._sync_task_queues()

                if vlm_task.done():
                    # VLM already finished — apply decision now
                    decision = await vlm_task
                    self._tick_count += 1
                    await self._apply_decision(decision)
                elif self._queue:
                    # VLM still running but more actions are queued —
                    # cancel this VLM (a fresh one fires at the next tick start)
                    # so the drone doesn't sit idle waiting for inference.
                    vlm_task.cancel()
                    try:
                        await vlm_task
                    except (asyncio.CancelledError, Exception):
                        pass
                else:
                    # Queue empty — wait for VLM before deciding what to do next
                    decision = await vlm_task
                    self._tick_count += 1
                    await self._apply_decision(decision)
                return

            except asyncio.TimeoutError:
                # Tick elapsed — action still running; collect VLM result
                # (action keeps running via shield while we await vlm_task)
                decision = await vlm_task
                self._tick_count += 1

                if isinstance(decision, VLMModify):
                    # Explicitly cancel the in-flight action
                    action_task.cancel()
                    try:
                        await action_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    self._current_action = None
                    self._sync_task_queues()
                    await self._apply_decision(decision)
                    return
                else:
                    await self._apply_decision(decision)
                    # Loop — action_task is still running via shield

            except Exception as e:
                if not vlm_task.done():
                    vlm_task.cancel()
                logger.error("[%s] Action '%s' raised: %s — skipping.",
                             self.drone_id,
                             self._current_action.get("action") if self._current_action else "?",
                             e)
                self._current_action = None
                self._sync_task_queues()
                return

    def stop(self) -> None:
        """Signal the loop to exit cleanly after the current tick."""
        self._running = False

    # ── Action helpers ─────────────────────────────────────────────────────────

    def _pop_next(self) -> dict:
        """Pop the next action from the queue."""
        return self._queue.popleft()

    # ── VLM query (runs concurrently with execute) ────────────────────────────

    async def _query_vlm(self) -> Union[VLMContinue, VLMModify]:
        """
        Snapshot telemetry, sync the memory pool, call the VLM, and return
        the decision.  Runs as a background task while the drone keeps flying.
        """
        # Capture snapshot — no hover, drone keeps moving
        pos       = await self._ctrl.get_position()
        vel       = await self._ctrl.get_velocity()
        image_b64 = await self._ctrl.get_camera_image_base64()

        # Sync pool (position update at 10 s cadence, as intended)
        # Status is NOT changed here — drone stays "executing" while VLM thinks.
        await self._memory.update_drone(
            self.drone_id,
            position=list(pos),
            velocity=list(vel),
        )
        peer_states = await self._memory.get_peer_states(self.drone_id)

        logger.info(
            "[%s] Tick %d | pos=(%.1f,%.1f,%.1f) | queue=%d | peers=%s",
            self.drone_id, self._tick_count,
            pos[0], pos[1], pos[2],
            len(self._queue) + (1 if self._current_action else 0),
            {pid: s.status for pid, s in peer_states.items()},
        )

        current_label = (
            self._current_action.get("action") if self._current_action
            else (self._queue[0].get("action") if self._queue else "idle")
        )

        return await self._vlm.decide(
            drone_id     = self.drone_id,
            position     = pos,
            velocity     = vel,
            status       = "executing" if self._current_action else "idle",
            current_task = current_label,
            image_b64    = image_b64,
            peer_states  = peer_states,
        )

    # ── Apply decision ────────────────────────────────────────────────────────

    async def _apply_decision(
        self, decision: Union[VLMContinue, VLMModify]
    ) -> None:
        ts = time.strftime("%H:%M:%S")

        if isinstance(decision, VLMContinue):
            logger.info("[%s] VLM → continue.", self.drone_id)
            await self._memory.update_drone(self.drone_id, status="executing")
            if self._vlm_log is not None:
                self._vlm_log.append(f"  {ts}  {self.drone_id:<10}  continue")

        elif isinstance(decision, VLMModify):
            logger.info("[%s] VLM → MODIFY: '%s'  action=%s",
                        self.drone_id, decision.new_task, decision.new_action)
            self._current_action = None
            self._queue.appendleft(decision.new_action)
            self._sync_task_queues()

            if decision.memory_update:
                await self._memory.update_drone(
                    self.drone_id, add_observation=decision.memory_update
                )
            await self._memory.update_drone(
                self.drone_id,
                status="replanning",
                current_task=decision.new_task,
            )
            if self._vlm_log is not None:
                self._vlm_log.append(
                    f"  {ts}  {self.drone_id:<10}  MODIFY → {decision.new_task}"
                )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _sync_task_queues(self) -> None:
        """Push current queue snapshot to the shared display dict."""
        if self._task_queues is None:
            return
        items = []
        if self._current_action:
            items.append(f"▶ {_fmt_action(self._current_action)}")
        items.extend(_fmt_action(t) for t in self._queue)
        self._task_queues[self.drone_id] = items

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
    sim_positions:   dict | None = None,
    vlm_log:         Optional[deque] = None,
    task_queues:     Optional[dict]  = None,
) -> None:
    """
    Create + connect a controller, then run the lifecycle to completion.

    Args:
        stop_when_empty: Exit when queue empties (tests / finite missions).
        sim_positions:   Simulator bridge dict for smooth visualisation.
        vlm_log:         Rolling deque of VLM decision strings.
        task_queues:     Shared ``{drone_id: [label, ...]}`` for the dashboard.
    """
    from controllers import make_controller   # lazy import avoids circular deps

    ctrl = make_controller(drone_id, sim_positions=sim_positions)
    await ctrl.connect()
    try:
        lifecycle = DroneLifecycle(
            drone_id        = drone_id,
            controller      = ctrl,
            memory          = memory,
            vlm             = vlm,
            initial_tasks   = initial_tasks,
            stop_when_empty = stop_when_empty,
            vlm_log         = vlm_log,
            task_queues     = task_queues,
        )
        await lifecycle.run()
    finally:
        await ctrl.disconnect()
