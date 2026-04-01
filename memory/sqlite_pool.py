"""
memory/sqlite_pool.py — SQLite-backed swarm state for multi-process Webots runs.

Webots launches one controller process per robot when using direct controllers,
so the in-process SharedMemoryPool is not enough for multi-drone simulation.
This module provides a small async-compatible, file-backed coordination layer
that mirrors the SharedMemoryPool API and also stores the global mission plan.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from models.schemas import DroneState

logger = logging.getLogger(__name__)

MAX_OBSERVATIONS = 20


class SQLiteSwarmPool:
    """
    SQLite-backed shared state store for multi-process drone controllers.

    API mirrors SharedMemoryPool closely enough for DroneLifecycle:
      - get_drone
      - get_all_states
      - get_peer_states
      - update_drone
      - clear_observations

    Additional helpers support global plan coordination:
      - reset_run
      - set_plan
      - get_plan
      - wait_for_plan
    """

    def __init__(self, db_path: str | Path, drone_ids: list[str]):
        self._db_path = str(db_path)
        self._drone_ids = list(drone_ids)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_db()
        self._seed_drones()
        logger.info("SQLiteSwarmPool ready at %s for drones: %s", self._db_path, self._drone_ids)

    # ── Public reads ───────────────────────────────────────────────────────

    async def get_drone(self, drone_id: str) -> DroneState:
        return await asyncio.to_thread(self._get_drone_sync, drone_id)

    async def get_all_states(self) -> dict[str, DroneState]:
        return await asyncio.to_thread(self._get_all_states_sync)

    async def get_peer_states(self, exclude_drone_id: str) -> dict[str, DroneState]:
        all_states = await self.get_all_states()
        return {did: state for did, state in all_states.items() if did != exclude_drone_id}

    # ── Public writes ──────────────────────────────────────────────────────

    async def update_drone(
        self,
        drone_id: str,
        *,
        position: Optional[list[float]] = None,
        velocity: Optional[list[float]] = None,
        status: Optional[str] = None,
        current_task: Optional[str] = None,
        add_observation: Optional[str] = None,
    ) -> None:
        await asyncio.to_thread(
            self._update_drone_sync,
            drone_id,
            position,
            velocity,
            status,
            current_task,
            add_observation,
        )

    async def clear_observations(self, drone_id: str) -> None:
        await asyncio.to_thread(self._clear_observations_sync, drone_id)

    # ── Plan helpers ───────────────────────────────────────────────────────

    async def reset_run(self, mission: str) -> None:
        await asyncio.to_thread(self._reset_run_sync, mission)

    async def set_plan(self, plan: dict[str, list[dict]], mission: str) -> None:
        await asyncio.to_thread(self._set_plan_sync, plan, mission)

    async def get_plan(self, drone_id: str) -> list[dict]:
        return await asyncio.to_thread(self._get_plan_sync, drone_id)

    async def wait_for_plan(self, drone_id: str, timeout: float = 180.0) -> list[dict]:
        deadline = time.monotonic() + timeout
        while True:
            tasks = await self.get_plan(drone_id)
            if tasks:
                return tasks
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for plan for {drone_id}.")
            await asyncio.sleep(0.5)

    # ── Sync implementation ────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS drone_states (
                    drone_id TEXT PRIMARY KEY,
                    position TEXT NOT NULL,
                    velocity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_task TEXT,
                    observations TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mission_plan (
                    drone_id TEXT PRIMARY KEY,
                    tasks TEXT NOT NULL,
                    mission TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

    def _seed_drones(self) -> None:
        now = time.time()
        with self._lock, self._conn:
            for did in self._drone_ids:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO drone_states
                    (drone_id, position, velocity, status, current_task, observations, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        did,
                        json.dumps([0.0, 0.0, 0.0]),
                        json.dumps([0.0, 0.0, 0.0]),
                        "idle",
                        None,
                        json.dumps([]),
                        now,
                    ),
                )

    def _get_drone_sync(self, drone_id: str) -> DroneState:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM drone_states WHERE drone_id = ?",
                (drone_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown drone_id '{drone_id}'. Registered: {self._drone_ids}")
        return self._row_to_state(row)

    def _get_all_states_sync(self) -> dict[str, DroneState]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM drone_states").fetchall()
        return {row["drone_id"]: self._row_to_state(row) for row in rows}

    def _update_drone_sync(
        self,
        drone_id: str,
        position: Optional[list[float]],
        velocity: Optional[list[float]],
        status: Optional[str],
        current_task: Optional[str],
        add_observation: Optional[str],
    ) -> None:
        current = self._get_drone_sync(drone_id)

        if position is not None:
            current.position = position
        if velocity is not None:
            current.velocity = velocity
        if status is not None:
            current.status = status
        if current_task is not None:
            current.current_task = current_task
        if add_observation is not None:
            current.observations.append(add_observation)
            current.observations = current.observations[-MAX_OBSERVATIONS:]

        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE drone_states
                SET position = ?, velocity = ?, status = ?, current_task = ?,
                    observations = ?, updated_at = ?
                WHERE drone_id = ?
                """,
                (
                    json.dumps(current.position),
                    json.dumps(current.velocity),
                    current.status,
                    current.current_task,
                    json.dumps(current.observations),
                    time.time(),
                    drone_id,
                ),
            )

    def _clear_observations_sync(self, drone_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE drone_states SET observations = ?, updated_at = ? WHERE drone_id = ?",
                (json.dumps([]), time.time(), drone_id),
            )

    def _reset_run_sync(self, mission: str) -> None:
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM mission_plan")
            self._conn.execute("DELETE FROM run_meta")
            self._conn.execute(
                "INSERT INTO run_meta(key, value) VALUES (?, ?)",
                ("mission", mission),
            )
            for did in self._drone_ids:
                self._conn.execute(
                    """
                    UPDATE drone_states
                    SET position = ?, velocity = ?, status = ?, current_task = ?,
                        observations = ?, updated_at = ?
                    WHERE drone_id = ?
                    """,
                    (
                        json.dumps([0.0, 0.0, 0.0]),
                        json.dumps([0.0, 0.0, 0.0]),
                        "idle",
                        None,
                        json.dumps([]),
                        now,
                        did,
                    ),
                )

    def _set_plan_sync(self, plan: dict[str, list[dict]], mission: str) -> None:
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM mission_plan")
            for did, tasks in plan.items():
                self._conn.execute(
                    """
                    INSERT INTO mission_plan(drone_id, tasks, mission, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (did, json.dumps(tasks), mission, now),
                )
            self._conn.execute(
                """
                INSERT INTO run_meta(key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                ("plan_ready", str(now)),
            )

    def _get_plan_sync(self, drone_id: str) -> list[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT tasks FROM mission_plan WHERE drone_id = ?",
                (drone_id,),
            ).fetchone()
        if row is None:
            return []
        return json.loads(row["tasks"])

    @staticmethod
    def _row_to_state(row: sqlite3.Row) -> DroneState:
        return DroneState(
            drone_id=row["drone_id"],
            position=json.loads(row["position"]),
            velocity=json.loads(row["velocity"]),
            status=row["status"],
            observations=json.loads(row["observations"]),
            current_task=row["current_task"],
        )
