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

from models.schemas import DroneState, TaskClaim, TaskEvent

logger = logging.getLogger(__name__)

MAX_OBSERVATIONS = 20
MAX_EVENTS = 50


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
      - set_role_briefs
      - get_role_brief
      - wait_for_role_brief
      - acquire_claim
      - release_claim
      - get_claims
      - emit_event
      - get_events
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
        battery_level: Optional[float] = None,
        status: Optional[str] = None,
        current_task: Optional[str] = None,
        add_observation: Optional[str] = None,
    ) -> None:
        await asyncio.to_thread(
            self._update_drone_sync,
            drone_id,
            position,
            velocity,
            battery_level,
            status,
            current_task,
            add_observation,
        )

    async def clear_observations(self, drone_id: str) -> None:
        await asyncio.to_thread(self._clear_observations_sync, drone_id)

    async def get_claims(self) -> list[TaskClaim]:
        return await asyncio.to_thread(self._get_claims_sync)

    async def get_events(self) -> list[TaskEvent]:
        return await asyncio.to_thread(self._get_events_sync)

    async def acquire_claim(
        self,
        *,
        claim_type: str,
        target_key: str,
        claimant_id: str,
        ttl: float,
        payload: Optional[dict] = None,
    ) -> bool:
        return await asyncio.to_thread(
            self._acquire_claim_sync,
            claim_type,
            target_key,
            claimant_id,
            ttl,
            payload or {},
        )

    async def release_claim(
        self,
        *,
        claim_type: str,
        target_key: str,
        claimant_id: Optional[str] = None,
    ) -> bool:
        return await asyncio.to_thread(
            self._release_claim_sync,
            claim_type,
            target_key,
            claimant_id,
        )

    async def emit_event(self, event: TaskEvent) -> None:
        await asyncio.to_thread(self._emit_event_sync, event)

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

    async def set_role_briefs(self, roles: dict[str, dict], mission: str) -> None:
        await asyncio.to_thread(self._set_role_briefs_sync, roles, mission)

    async def get_role_brief(self, drone_id: str) -> dict:
        return await asyncio.to_thread(self._get_role_brief_sync, drone_id)

    async def wait_for_role_brief(self, drone_id: str, timeout: float = 180.0) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            role = await self.get_role_brief(drone_id)
            if role:
                return role
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for role brief for {drone_id}.")
            await asyncio.sleep(0.5)

    # ── Sync implementation ────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            columns = {
                row["name"]
                for row in self._conn.execute("PRAGMA table_info(drone_states)").fetchall()
            }
            if columns and "battery_level" not in columns:
                self._conn.execute("ALTER TABLE drone_states ADD COLUMN battery_level REAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS drone_states (
                    drone_id TEXT PRIMARY KEY,
                    position TEXT NOT NULL,
                    velocity TEXT NOT NULL,
                    battery_level REAL,
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
                CREATE TABLE IF NOT EXISTS role_briefs (
                    drone_id TEXT PRIMARY KEY,
                    brief TEXT NOT NULL,
                    mission TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS swarm_claims (
                    claim_type TEXT NOT NULL,
                    target_key TEXT NOT NULL,
                    claimant_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    PRIMARY KEY (claim_type, target_key)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS swarm_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    timestamp REAL NOT NULL
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
                    (drone_id, position, velocity, battery_level, status, current_task, observations, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        did,
                        json.dumps([0.0, 0.0, 0.0]),
                        json.dumps([0.0, 0.0, 0.0]),
                        None,
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
        battery_level: Optional[float],
        status: Optional[str],
        current_task: Optional[str],
        add_observation: Optional[str],
    ) -> None:
        current = self._get_drone_sync(drone_id)

        if position is not None:
            current.position = position
        if velocity is not None:
            current.velocity = velocity
        if battery_level is not None:
            current.battery_level = battery_level
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
                SET position = ?, velocity = ?, battery_level = ?, status = ?, current_task = ?,
                    observations = ?, updated_at = ?
                WHERE drone_id = ?
                """,
                (
                    json.dumps(current.position),
                    json.dumps(current.velocity),
                    current.battery_level,
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

    def _get_claims_sync(self) -> list[TaskClaim]:
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM swarm_claims WHERE expires_at <= ?", (now,))
            rows = self._conn.execute("SELECT * FROM swarm_claims").fetchall()
        return [
            TaskClaim(
                claim_type=row["claim_type"],
                target_key=row["target_key"],
                claimant_id=row["claimant_id"],
                payload=json.loads(row["payload"]),
                created_at=row["created_at"],
                expires_at=row["expires_at"],
            )
            for row in rows
        ]

    def _get_events_sync(self) -> list[TaskEvent]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT event_type, source, priority, payload, timestamp
                FROM (
                    SELECT event_type, source, priority, payload, timestamp
                    FROM swarm_events
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY timestamp ASC
                """,
                (MAX_EVENTS,),
            ).fetchall()
        return [
            TaskEvent(
                type=row["event_type"],
                source=row["source"],
                priority=row["priority"],
                payload=json.loads(row["payload"]),
                timestamp=row["timestamp"],
            )
            for row in rows
        ]

    def _emit_event_sync(self, event: TaskEvent) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO swarm_events(event_type, source, priority, payload, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event.type,
                    event.source,
                    event.priority,
                    json.dumps(event.payload),
                    event.timestamp,
                ),
            )
            self._conn.execute(
                """
                DELETE FROM swarm_events
                WHERE id NOT IN (
                    SELECT id FROM swarm_events ORDER BY id DESC LIMIT ?
                )
                """,
                (MAX_EVENTS,),
            )

    def _acquire_claim_sync(
        self,
        claim_type: str,
        target_key: str,
        claimant_id: str,
        ttl: float,
        payload: dict,
    ) -> bool:
        now = time.time()
        expires_at = now + ttl
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM swarm_claims WHERE expires_at <= ?", (now,))
            row = self._conn.execute(
                """
                SELECT claimant_id FROM swarm_claims
                WHERE claim_type = ? AND target_key = ?
                """,
                (claim_type, target_key),
            ).fetchone()
            if row is not None and row["claimant_id"] != claimant_id:
                return False

            self._conn.execute(
                """
                INSERT INTO swarm_claims(
                    claim_type, target_key, claimant_id, payload, created_at, expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(claim_type, target_key) DO UPDATE SET
                    claimant_id = excluded.claimant_id,
                    payload = excluded.payload,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
                """,
                (
                    claim_type,
                    target_key,
                    claimant_id,
                    json.dumps(payload),
                    now,
                    expires_at,
                ),
            )
            return True

    def _release_claim_sync(
        self,
        claim_type: str,
        target_key: str,
        claimant_id: Optional[str],
    ) -> bool:
        with self._lock, self._conn:
            row = self._conn.execute(
                """
                SELECT claimant_id FROM swarm_claims
                WHERE claim_type = ? AND target_key = ?
                """,
                (claim_type, target_key),
            ).fetchone()
            if row is None:
                return False
            if claimant_id is not None and row["claimant_id"] != claimant_id:
                return False
            self._conn.execute(
                "DELETE FROM swarm_claims WHERE claim_type = ? AND target_key = ?",
                (claim_type, target_key),
            )
            return True

    def _reset_run_sync(self, mission: str) -> None:
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM mission_plan")
            self._conn.execute("DELETE FROM role_briefs")
            self._conn.execute("DELETE FROM swarm_claims")
            self._conn.execute("DELETE FROM swarm_events")
            self._conn.execute("DELETE FROM run_meta")
            self._conn.execute(
                "INSERT INTO run_meta(key, value) VALUES (?, ?)",
                ("mission", mission),
            )
            for did in self._drone_ids:
                self._conn.execute(
                    """
                    UPDATE drone_states
                    SET position = ?, velocity = ?, battery_level = ?, status = ?, current_task = ?,
                        observations = ?, updated_at = ?
                    WHERE drone_id = ?
                    """,
                    (
                        json.dumps([0.0, 0.0, 0.0]),
                        json.dumps([0.0, 0.0, 0.0]),
                        None,
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

    def _set_role_briefs_sync(self, roles: dict[str, dict], mission: str) -> None:
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM role_briefs")
            for did, brief in roles.items():
                self._conn.execute(
                    """
                    INSERT INTO role_briefs(drone_id, brief, mission, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (did, json.dumps(brief), mission, now),
                )
            self._conn.execute(
                """
                INSERT INTO run_meta(key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                ("role_briefs_ready", str(now)),
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

    def _get_role_brief_sync(self, drone_id: str) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT brief FROM role_briefs WHERE drone_id = ?",
                (drone_id,),
            ).fetchone()
        if row is None:
            return {}
        return json.loads(row["brief"])

    @staticmethod
    def _row_to_state(row: sqlite3.Row) -> DroneState:
        return DroneState(
            drone_id=row["drone_id"],
            position=json.loads(row["position"]),
            velocity=json.loads(row["velocity"]),
            battery_level=row["battery_level"],
            status=row["status"],
            observations=json.loads(row["observations"]),
            current_task=row["current_task"],
            updated_at=row["updated_at"],
        )
