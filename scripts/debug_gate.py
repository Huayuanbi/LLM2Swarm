#!/usr/bin/env python3
"""
scripts/debug_gate.py — Inspect and reply to file-backed debug pauses.

Examples:
    conda run -n llm2swarm python scripts/debug_gate.py list
    conda run -n llm2swarm python scripts/debug_gate.py show latest
    conda run -n llm2swarm python scripts/debug_gate.py reply latest continue
    conda run -n llm2swarm python scripts/debug_gate.py serve
"""

from __future__ import annotations

import argparse
import html
import json
import os
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import config
from utils.debug_gate import (
    latest_pending_step,
    list_pending_steps,
    read_step_metadata,
    read_step_payload,
)


def _serialize_step(step: Path) -> dict[str, Any]:
    meta = read_step_metadata(step)
    return {
        "path": str(step),
        "stage": meta.get("stage", "?"),
        "actor_id": meta.get("actor_id", "global"),
        "summary": meta.get("summary", ""),
        "allowed_commands": meta.get("allowed_commands", []),
        "payload_path": str(step / "payload.json"),
        "created_at": meta.get("created_at"),
    }


def _resolve_root(root: str | Path) -> Path:
    return Path(root).expanduser().resolve()


def _resolve_step(root: Path, step_arg: str, session: Optional[str] = None) -> Optional[Path]:
    if step_arg == "latest":
        return latest_pending_step(root, session_id=session)

    step = Path(step_arg).expanduser().resolve()
    try:
        step.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Step path must stay under debug root {root}") from exc
    return step


def _print_step(step: Path) -> None:
    record = _serialize_step(step)
    print(f"step:    {record['path']}")
    print(f"stage:   {record['stage']}")
    print(f"actor:   {record['actor_id']}")
    if record["summary"]:
        print(f"summary: {record['summary']}")
    print(f"allowed: {record['allowed_commands']}")
    print(f"inspect: {record['payload_path']}")


def _cmd_list(args) -> int:
    root = _resolve_root(args.root)
    steps = list_pending_steps(root, session_id=args.session)
    if not steps:
        print("No pending debug steps.")
        return 0

    for index, step in enumerate(steps, start=1):
        record = _serialize_step(step)
        print(f"[{index}] {record['path']}")
        print(f"  stage:   {record['stage']}")
        print(f"  actor:   {record['actor_id']}")
        if record["summary"]:
            print(f"  summary: {record['summary']}")
        print(f"  allowed: {record['allowed_commands']}")
        print(f"  inspect first: {record['payload_path']}")
        print()
    return 0


def _cmd_show(args) -> int:
    root = _resolve_root(args.root)
    step = _resolve_step(root, args.step, session=args.session)
    if step is None or not step.exists():
        print("No matching pending debug step.")
        return 1

    _print_step(step)
    payload = read_step_payload(step)
    print()
    print("payload:")
    if isinstance(payload, (dict, list)):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload)
    return 0


def _cmd_reply(args) -> int:
    root = _resolve_root(args.root)
    step = _resolve_step(root, args.step, session=args.session)
    if step is None or not step.exists():
        print("No matching pending debug step.")
        return 1

    command = args.command.strip().lower()
    command_path = step / "command.txt"
    command_path.write_text(command, encoding="utf-8")
    print(f"Wrote '{command}' to {command_path}")
    return 0


def _find_available_port(host: str, preferred_port: int, max_tries: int = 100) -> int:
    for port in range(preferred_port, preferred_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise OSError(f"No available port found in range {preferred_port}-{preferred_port + max_tries - 1}")


def _build_html() -> str:
    return """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>LLM2Swarm Debug Gate</title>
    <style>
      body { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin: 0; background: #f4f1ea; color: #1f1f1f; }
      .layout { display: grid; grid-template-columns: 360px 1fr; min-height: 100vh; }
      .sidebar { border-right: 1px solid #d4cfc4; background: #fbf8f3; padding: 16px; overflow: auto; }
      .main { padding: 16px; overflow: auto; }
      h1, h2 { margin: 0 0 12px 0; font-size: 18px; }
      .step { border: 1px solid #d4cfc4; border-radius: 10px; padding: 12px; margin-bottom: 10px; background: white; cursor: pointer; }
      .step.active { border-color: #7a4f28; box-shadow: 0 0 0 2px rgba(122,79,40,0.12); }
      .label { font-size: 12px; color: #666; margin-bottom: 6px; }
      .title { font-weight: 600; margin-bottom: 6px; }
      .summary { color: #444; font-size: 13px; }
      .toolbar { display: flex; gap: 8px; margin-bottom: 14px; }
      button { padding: 8px 12px; border-radius: 8px; border: 1px solid #7a4f28; background: white; cursor: pointer; }
      button.primary { background: #7a4f28; color: white; }
      pre { white-space: pre-wrap; word-break: break-word; background: white; border: 1px solid #d4cfc4; border-radius: 10px; padding: 14px; }
      .muted { color: #666; font-size: 13px; }
    </style>
  </head>
  <body>
    <div class="layout">
      <div class="sidebar">
        <h1>LLM2Swarm Debug Gate</h1>
        <p class="muted">Click a pending step, inspect its payload, then choose continue / regenerate / abort.</p>
        <div class="toolbar">
          <button onclick="refreshSteps()">Refresh</button>
        </div>
        <div id="steps"></div>
      </div>
      <div class="main">
        <h2 id="detail-title">No step selected</h2>
        <p class="muted" id="detail-meta">Select a pending step from the left.</p>
        <div class="toolbar">
          <button class="primary" onclick="replyCurrent('continue')">Continue</button>
          <button onclick="replyCurrent('regenerate')">Regenerate</button>
          <button onclick="replyCurrent('abort')">Abort</button>
        </div>
        <pre id="payload">{}</pre>
      </div>
    </div>
    <script>
      let currentPath = null;
      let currentSteps = [];

      async function fetchJson(url, options) {
        const res = await fetch(url, options);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
        return data;
      }

      async function refreshSteps() {
        const data = await fetchJson('/api/steps');
        currentSteps = data.steps;
        const container = document.getElementById('steps');
        container.innerHTML = '';
        if (!currentSteps.length) {
          container.innerHTML = '<p class="muted">No pending debug steps.</p>';
          currentPath = null;
          document.getElementById('detail-title').textContent = 'No step selected';
          document.getElementById('detail-meta').textContent = 'No pending debug steps.';
          document.getElementById('payload').textContent = '{}';
          return;
        }
        if (!currentPath || !currentSteps.some(step => step.path === currentPath)) {
          currentPath = currentSteps[currentSteps.length - 1].path;
        }
        for (const step of currentSteps) {
          const el = document.createElement('div');
          el.className = 'step' + (step.path === currentPath ? ' active' : '');
          el.onclick = () => selectStep(step.path);
          el.innerHTML = `
            <div class="label">${step.stage} · ${step.actor_id}</div>
            <div class="title">${step.path.split('/').slice(-1)[0]}</div>
            <div class="summary">${step.summary || '(no summary)'}</div>
          `;
          container.appendChild(el);
        }
        await selectStep(currentPath, false);
      }

      async function selectStep(path, rerender = true) {
        currentPath = path;
        const data = await fetchJson('/api/step?path=' + encodeURIComponent(path));
        document.getElementById('detail-title').textContent = `${data.meta.stage} · ${data.meta.actor_id || 'global'}`;
        document.getElementById('detail-meta').textContent = data.meta.summary || data.path;
        document.getElementById('payload').textContent = JSON.stringify(data.payload, null, 2);
        if (rerender) {
          await refreshSteps();
        }
      }

      async function replyCurrent(command) {
        if (!currentPath) return;
        await fetchJson('/api/reply', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({path: currentPath, command}),
        });
        await refreshSteps();
      }

      refreshSteps();
      setInterval(refreshSteps, 2000);
    </script>
  </body>
</html>"""


def _make_handler(root: Path, session: Optional[str]):
    html_page = _build_html().encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: Any, *, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _resolve_step_request(self, step_arg: str) -> Path:
            step = _resolve_step(root, step_arg, session=session)
            if step is None or not step.exists():
                raise FileNotFoundError("No matching pending debug step.")
            return step

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(html_page)
                return
            if parsed.path == "/api/steps":
                steps = [_serialize_step(step) for step in list_pending_steps(root, session_id=session)]
                self._send_json({"steps": steps})
                return
            if parsed.path == "/api/step":
                qs = parse_qs(parsed.query)
                step_arg = qs.get("path", ["latest"])[0]
                try:
                    step = self._resolve_step_request(step_arg)
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=404)
                    return
                self._send_json(
                    {
                        "path": str(step),
                        "meta": read_step_metadata(step),
                        "payload": read_step_payload(step),
                    }
                )
                return

            self._send_json({"error": "Not found"}, status=404)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/reply":
                self._send_json({"error": "Not found"}, status=404)
                return

            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                self._send_json({"error": "Invalid JSON body"}, status=400)
                return

            command = str(body.get("command", "")).strip().lower()
            if command not in {"continue", "regenerate", "abort"}:
                self._send_json({"error": "Invalid command"}, status=400)
                return

            step_arg = str(body.get("path", "latest"))
            try:
                step = self._resolve_step_request(step_arg)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=404)
                return

            (step / "command.txt").write_text(command, encoding="utf-8")
            self._send_json({"ok": True, "path": str(step), "command": command})

        def log_message(self, fmt: str, *args) -> None:  # noqa: A003
            return

    return Handler


def _cmd_serve(args) -> int:
    root = _resolve_root(args.root)
    handler = _make_handler(root, args.session)
    port = _find_available_port(args.host, args.port) if args.auto_port else args.port
    if port != args.port:
        print(f"Port {args.port} is busy; using {port} instead.")

    server = ThreadingHTTPServer((args.host, port), handler)
    url = f"http://{args.host}:{port}"
    print(f"Debug gate UI: {url}")
    print(f"Debug root:    {root}")
    if args.session:
        print(f"Session:       {args.session}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and reply to LLM2Swarm debug pauses.")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="List pending debug pauses.")
    list_parser.add_argument("--root", default=config.DEBUG_DIR)
    list_parser.add_argument("--session", default=None)
    list_parser.set_defaults(func=_cmd_list)

    show_parser = sub.add_parser("show", help="Print one pending step and its payload.")
    show_parser.add_argument("step", help="Absolute step path or 'latest'")
    show_parser.add_argument("--root", default=config.DEBUG_DIR)
    show_parser.add_argument("--session", default=None)
    show_parser.set_defaults(func=_cmd_show)

    reply_parser = sub.add_parser("reply", help="Reply to a pending debug pause.")
    reply_parser.add_argument("step", help="Absolute step path or 'latest'")
    reply_parser.add_argument("command", choices=["continue", "regenerate", "abort"])
    reply_parser.add_argument("--root", default=config.DEBUG_DIR)
    reply_parser.add_argument("--session", default=None)
    reply_parser.set_defaults(func=_cmd_reply)

    serve_parser = sub.add_parser("serve", help="Run a small local web UI for pending debug pauses.")
    serve_parser.add_argument("--root", default=config.DEBUG_DIR)
    serve_parser.add_argument("--session", default=None)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--no-auto-port", dest="auto_port", action="store_false")
    serve_parser.set_defaults(auto_port=True)
    serve_parser.set_defaults(func=_cmd_serve)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
