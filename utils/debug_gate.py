"""
utils/debug_gate.py — File-backed human-in-the-loop debug pauses.

This module lets long-running planners/controllers pause at key checkpoints,
dump request/response payloads to disk, and wait for a human decision:

  - continue
  - regenerate
  - abort

The design is intentionally file-based so it still works when different agents
run in separate processes (for example, Webots multi-controller mode).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import config

logger = logging.getLogger(__name__)

CONTINUE_COMMAND = "continue"
REGENERATE_COMMAND = "regenerate"
ABORT_COMMAND = "abort"


class DebugAbort(RuntimeError):
    """Raised when a human operator aborts execution via the debug gate."""


def _parse_filter(raw: str) -> set[str] | None:
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@dataclass(slots=True)
class DebugGate:
    enabled: bool
    root: Path
    session_id: str
    stages: set[str] | None = None
    targets: set[str] | None = None
    poll_interval: float = 0.5
    vlm_pause_once_per_actor: bool = True

    @classmethod
    def from_config(cls) -> "DebugGate":
        return cls(
            enabled=config.DEBUG_MODE,
            root=Path(config.DEBUG_DIR),
            session_id=config.DEBUG_SESSION_ID,
            stages=_parse_filter(config.DEBUG_STAGES),
            targets=_parse_filter(config.DEBUG_TARGETS),
            poll_interval=config.DEBUG_POLL_INTERVAL,
            vlm_pause_once_per_actor=config.DEBUG_VLM_PAUSE_ONCE_PER_ACTOR,
        )

    def is_enabled_for(self, stage: str, actor_id: Optional[str] = None) -> bool:
        if not self.enabled:
            return False
        if self.stages is not None and stage not in self.stages:
            return False
        if self.targets is not None and actor_id is not None and actor_id not in self.targets:
            return False
        if self.targets is not None and actor_id is None:
            return False
        if self._is_vlm_stage(stage) and self.vlm_pause_once_per_actor:
            if actor_id is None:
                return False
            if self._vlm_seen_marker(stage, actor_id).exists():
                return False
        return True

    async def checkpoint(
        self,
        stage: str,
        payload: Any,
        *,
        actor_id: Optional[str] = None,
        allow_regenerate: bool = False,
        summary: str = "",
    ) -> str:
        if not self.is_enabled_for(stage, actor_id):
            return CONTINUE_COMMAND

        allowed_commands = [CONTINUE_COMMAND, ABORT_COMMAND]
        if allow_regenerate:
            allowed_commands.insert(1, REGENERATE_COMMAND)

        step_dir = self._create_step_dir(stage, actor_id)
        payload_path = step_dir / "payload.json"
        meta_path = step_dir / "meta.json"
        command_path = step_dir / "command.txt"
        instructions_path = step_dir / "instructions.txt"
        resolved_path = step_dir / "resolved.json"

        payload_path.write_text(
            json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        meta = {
            "stage": stage,
            "actor_id": actor_id,
            "session_id": self.session_id,
            "summary": summary,
            "created_at": time.time(),
            "allowed_commands": allowed_commands,
            "payload_file": str(payload_path),
            "command_file": str(command_path),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        instructions_path.write_text(
            "\n".join(
                [
                    f"Stage: {stage}",
                    f"Actor: {actor_id or 'global'}",
                    "",
                    f"Inspect: {payload_path}",
                    f"Then write one of {allowed_commands} to:",
                    f"  {command_path}",
                    "",
                    "Or use:",
                    "  conda run -n llm2swarm python scripts/debug_gate.py list",
                    "  conda run -n llm2swarm python scripts/debug_gate.py show latest",
                    f"  conda run -n llm2swarm python scripts/debug_gate.py reply {step_dir} {CONTINUE_COMMAND}",
                    "  conda run -n llm2swarm python scripts/debug_gate.py serve",
                ]
            ),
            encoding="utf-8",
        )

        logger.warning(
            "Debug pause at stage '%s' for %s. Inspect %s and reply via %s "
            "with one of %s.",
            stage,
            actor_id or "global",
            payload_path,
            command_path,
            allowed_commands,
        )

        while True:
            if command_path.exists():
                command = command_path.read_text(encoding="utf-8").strip().lower()
                if command not in allowed_commands:
                    logger.warning(
                        "Ignoring invalid debug command '%s' for %s. Allowed: %s",
                        command,
                        step_dir,
                        allowed_commands,
                    )
                else:
                    resolved_path.write_text(
                        json.dumps(
                            {
                                "command": command,
                                "resolved_at": time.time(),
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    if command == ABORT_COMMAND:
                        raise DebugAbort(f"Aborted by debug gate at stage '{stage}'")
                    if command == CONTINUE_COMMAND and self._is_vlm_stage(stage) and actor_id is not None:
                        marker = self._vlm_seen_marker(stage, actor_id)
                        marker.parent.mkdir(parents=True, exist_ok=True)
                        marker.write_text(
                            json.dumps(
                                {
                                    "stage": stage,
                                    "actor_id": actor_id,
                                    "resolved_at": time.time(),
                                    "step_dir": str(step_dir),
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                    return command
            await asyncio.sleep(self.poll_interval)

    def _create_step_dir(self, stage: str, actor_id: Optional[str]) -> Path:
        actor_slug = (actor_id or "global").replace(os.sep, "_")
        step_id = (
            f"{time.strftime('%Y%m%d_%H%M%S')}"
            f"_{stage}_{actor_slug}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        )
        step_dir = self.root / self.session_id / step_id
        step_dir.mkdir(parents=True, exist_ok=False)
        return step_dir

    def _vlm_seen_marker(self, stage: str, actor_id: str) -> Path:
        actor_slug = actor_id.replace(os.sep, "_")
        return self.root / self.session_id / ".vlm_once" / f"{stage}_{actor_slug}.json"

    @staticmethod
    def _is_vlm_stage(stage: str) -> bool:
        return stage in {"vlm_request", "vlm_response"}


def list_pending_steps(root: str | Path, session_id: Optional[str] = None) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []

    session_paths: list[Path]
    if session_id:
        session_path = root_path / session_id
        session_paths = [session_path] if session_path.exists() else []
    else:
        session_paths = [path for path in root_path.iterdir() if path.is_dir()]

    pending: list[Path] = []
    for session_path in sorted(session_paths):
        for step_dir in sorted(path for path in session_path.iterdir() if path.is_dir()):
            if not (step_dir / "resolved.json").exists():
                pending.append(step_dir)
    return pending


def latest_pending_step(root: str | Path, session_id: Optional[str] = None) -> Optional[Path]:
    pending = list_pending_steps(root, session_id=session_id)
    return pending[-1] if pending else None


def read_step_metadata(step_dir: str | Path) -> dict[str, Any]:
    step_path = Path(step_dir)
    meta_path = step_path / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def read_step_payload(step_dir: str | Path) -> Any:
    step_path = Path(step_dir)
    payload_path = step_path / "payload.json"
    if not payload_path.exists():
        return None
    raw = payload_path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except Exception:
        return raw


DEBUG_GATE = DebugGate.from_config()
