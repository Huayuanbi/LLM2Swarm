"""
memory/pool.py — Async-safe Shared Memory Pool.

All drone agents read and write to a single SharedMemoryPool instance.
asyncio.Lock ensures no two coroutines corrupt state simultaneously.

Design:
  - One DroneState entry per drone_id, pre-seeded on initialisation.
  - All mutations go through update_drone() — never direct dict access.
  - get_all_states() returns a snapshot (deep copy) so callers can't
    accidentally mutate shared state.
  - Observation history is capped at MAX_OBSERVATIONS to prevent unbounded growth.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from typing import Optional

from models.schemas import DroneState

logger = logging.getLogger(__name__)

MAX_OBSERVATIONS = 20   # keep only the N most recent observations per drone


class SharedMemoryPool:
    """
    Central in-process state store for the drone swarm.

    Usage:
        pool = SharedMemoryPool(["drone_1", "drone_2"])
        await pool.update_drone("drone_1", position=[10, 5, 8], status="moving")
        state = await pool.get_drone("drone_1")
        all_states = await pool.get_all_states()
    """

    def __init__(self, drone_ids: list[str]):
        self._lock: asyncio.Lock = asyncio.Lock()
        self._store: dict[str, DroneState] = {
            did: DroneState(drone_id=did) for did in drone_ids
        }
        logger.info("SharedMemoryPool initialised for drones: %s", drone_ids)

    # ── Reads ──────────────────────────────────────────────────────────────────

    async def get_drone(self, drone_id: str) -> DroneState:
        """Return a snapshot of a single drone's state."""
        async with self._lock:
            if drone_id not in self._store:
                raise KeyError(f"Unknown drone_id '{drone_id}'. "
                               f"Registered: {list(self._store.keys())}")
            return copy.deepcopy(self._store[drone_id])

    async def get_all_states(self) -> dict[str, DroneState]:
        """Return a snapshot of every drone's state."""
        async with self._lock:
            return copy.deepcopy(self._store)

    async def get_peer_states(self, exclude_drone_id: str) -> dict[str, DroneState]:
        """Return snapshots for all drones *except* the caller."""
        all_states = await self.get_all_states()
        return {did: s for did, s in all_states.items() if did != exclude_drone_id}

    # ── Writes ─────────────────────────────────────────────────────────────────

    async def update_drone(
        self,
        drone_id: str,
        *,
        position: Optional[list[float]]  = None,
        velocity: Optional[list[float]]  = None,
        status: Optional[str]            = None,
        current_task: Optional[str]      = None,
        add_observation: Optional[str]   = None,
    ) -> None:
        """
        Partially update a drone's state.  Only non-None kwargs are applied.

        Args:
            drone_id:        Target drone.
            position:        New [x, y, z] in metres.
            velocity:        New [vx, vy, vz] in m/s.
            status:          New status string.
            current_task:    Human-readable label of the active task.
            add_observation: Append a new observation string to the history
                             (automatically capped at MAX_OBSERVATIONS).
        """
        async with self._lock:
            if drone_id not in self._store:
                raise KeyError(f"Unknown drone_id '{drone_id}'.")

            state = self._store[drone_id]

            if position is not None:
                state.position = position
            if velocity is not None:
                state.velocity = velocity
            if status is not None:
                state.status = status
            if current_task is not None:
                state.current_task = current_task
            if add_observation is not None:
                state.observations.append(add_observation)
                # Trim to cap
                if len(state.observations) > MAX_OBSERVATIONS:
                    state.observations = state.observations[-MAX_OBSERVATIONS:]

        logger.debug(
            "[%s] Memory updated — status=%s pos=%s obs=%s",
            drone_id, status, position, add_observation,
        )

    async def clear_observations(self, drone_id: str) -> None:
        """Wipe the observation log for a drone (e.g. after a VLM replanning tick)."""
        async with self._lock:
            self._store[drone_id].observations = []
