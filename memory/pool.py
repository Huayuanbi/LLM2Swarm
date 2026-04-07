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
import time
from collections import deque
from typing import Optional

from models.schemas import DroneState, TaskClaim, TaskEvent

logger = logging.getLogger(__name__)

MAX_OBSERVATIONS = 20   # keep only the N most recent observations per drone
MAX_EVENTS       = 50   # keep only the N most recent shared swarm events


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
        self._claims: dict[tuple[str, str], TaskClaim] = {}
        self._events: deque[TaskEvent] = deque(maxlen=MAX_EVENTS)
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

    async def get_claims(self) -> list[TaskClaim]:
        """Return the active, non-expired claim leases."""
        async with self._lock:
            self._prune_expired_claims_locked()
            return copy.deepcopy(list(self._claims.values()))

    async def get_events(self) -> list[TaskEvent]:
        """Return the recent shared swarm events."""
        async with self._lock:
            return copy.deepcopy(list(self._events))

    # ── Writes ─────────────────────────────────────────────────────────────────

    async def update_drone(
        self,
        drone_id: str,
        *,
        position: Optional[list[float]]  = None,
        velocity: Optional[list[float]]  = None,
        battery_level: Optional[float]   = None,
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
            if battery_level is not None:
                state.battery_level = battery_level
            if status is not None:
                state.status = status
            if current_task is not None:
                state.current_task = current_task
            if add_observation is not None:
                state.observations.append(add_observation)
                # Trim to cap
                if len(state.observations) > MAX_OBSERVATIONS:
                    state.observations = state.observations[-MAX_OBSERVATIONS:]
            state.updated_at = time.time()

        logger.debug(
            "[%s] Memory updated — status=%s pos=%s battery=%s obs=%s",
            drone_id, status, position, battery_level, add_observation,
        )

    async def clear_observations(self, drone_id: str) -> None:
        """Wipe the observation log for a drone (e.g. after a VLM replanning tick)."""
        async with self._lock:
            self._store[drone_id].observations = []

    async def emit_event(self, event: TaskEvent) -> None:
        """Append a structured swarm event to the shared event stream."""
        async with self._lock:
            self._events.append(event)

    async def acquire_claim(
        self,
        *,
        claim_type: str,
        target_key: str,
        claimant_id: str,
        ttl: float,
        payload: Optional[dict] = None,
    ) -> bool:
        """
        Try to acquire a time-limited claim lease for a shared swarm resource.

        Returns True only if the claim was free/expired or already owned by the
        same claimant. This is the primitive used for takeover coordination.
        """
        expires_at = time.time() + ttl
        key = (claim_type, target_key)
        async with self._lock:
            self._prune_expired_claims_locked()
            existing = self._claims.get(key)
            if existing and existing.claimant_id != claimant_id:
                return False

            self._claims[key] = TaskClaim(
                claim_type=claim_type,
                target_key=target_key,
                claimant_id=claimant_id,
                payload=payload or {},
                expires_at=expires_at,
            )
            return True

    async def release_claim(
        self,
        *,
        claim_type: str,
        target_key: str,
        claimant_id: Optional[str] = None,
    ) -> bool:
        """
        Release a claim if it exists and is either unowned-filtered or owned by
        the provided claimant_id.
        """
        key = (claim_type, target_key)
        async with self._lock:
            self._prune_expired_claims_locked()
            existing = self._claims.get(key)
            if existing is None:
                return False
            if claimant_id is not None and existing.claimant_id != claimant_id:
                return False
            self._claims.pop(key, None)
            return True

    def _prune_expired_claims_locked(self) -> None:
        now = time.time()
        expired = [key for key, claim in self._claims.items() if claim.expires_at <= now]
        for key in expired:
            self._claims.pop(key, None)
