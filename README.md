# LLM2Swarm

An edge-cloud collaborative multi-drone swarm system driven by large language models.

Each drone runs an independent 10-second control loop: a **Cloud LLM** (GPT-4o or Ollama) generates the initial mission plan, and an **Edge VLM** (qwen3.5:9b via Ollama) replans in real time based on camera perception and swarm-wide state.

```
┌─────────────────────────────────────────────────────────────┐
│                      Cloud LLM (GPT-4o)                     │
│              Natural language → per-drone task plan         │
└───────────────────────────┬─────────────────────────────────┘
                            │ initial plan
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                  ▼
      [drone_1]         [drone_2]          [drone_3]
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │  10s loop    │  │  10s loop    │  │  10s loop    │
   │  Execute     │  │  Execute     │  │  Execute     │
   │  Perceive ───┼──┼── Memory ───┼──┼─── Perceive  │
   │  VLM decide  │  │  Pool       │  │  VLM decide  │
   └──────────────┘  └──────────────┘  └──────────────┘
          │                 │                  │
          └─────────────────▼──────────────────┘
                    Edge VLM (qwen3.5:9b)
                  Ollama on remote Linux server
```

## Architecture

| Module | Purpose |
|---|---|
| `main.py` | Entry point — wires all components and launches drones via `asyncio.gather` |
| `config.py` | All constants and env-var overrides |
| `controllers/` | Simulator backends — `MockDroneController` (built-in) and `WebotsController` (stub) |
| `memory/pool.py` | Async-safe shared state store for the entire swarm |
| `operators/global_operator.py` | Cloud LLM planner — converts mission text to per-drone task lists |
| `operators/vlm_agent.py` | Edge VLM replanner — sends camera image + telemetry, returns `continue`/`modify` |
| `operators/drone_lifecycle.py` | Per-drone 10-second control loop |
| `models/schemas.py` | Pydantic models for all structured I/O |
| `utils/image_utils.py` | Base64 image helpers for the VLM vision messages |
| `scripts/start_tunnel.sh` | SSH tunnel manager for the remote Ollama server |

## Requirements

- **macOS** (primary dev platform) or Linux
- **Python 3.11+** via conda
- **Ollama** running on a remote Linux server with `qwen3.5:9b` pulled
- **OpenAI API key** with credits (optional — falls back to Ollama if unavailable)
- SSH access to the Ollama server

## Quick Start

### 1. Clone and create the environment

```bash
git clone https://github.com/<your-username>/LLM2Swarm.git
cd LLM2Swarm

conda create -n llm2swarm python=3.11 -y
conda activate llm2swarm
pip install -r requirements.txt
```

### 2. Configure

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required for cloud planning (optional if using Ollama substitute)
OPENAI_API_KEY=sk-...

# SSH tunnel target — your remote Ollama server
# Run: ssh -N -f -L 11435:localhost:11434 <user>@<server-ip>
EDGE_VLM_BASE_URL=http://localhost:11435/v1

# To use qwen3.5:9b as GPT-4o substitute (no OpenAI credits needed):
GLOBAL_LLM_BASE_URL=http://localhost:11435/v1
GLOBAL_LLM_API_KEY=ollama
GLOBAL_LLM_MODEL=qwen3.5:9b

# Simulator: "mock" (no simulator needed) or "webots"
SIMULATOR_BACKEND=mock
```

### 3. Start the SSH tunnel to the Ollama server

```bash
./scripts/start_tunnel.sh          # start (or confirm already running)
./scripts/start_tunnel.sh --status # check tunnel + list available models
./scripts/start_tunnel.sh --kill   # stop the tunnel
```

> **Proxy note:** If your machine routes traffic through a local proxy (e.g. Clash/V2Ray), the script and `.env` already handle bypassing it for `localhost` via `NO_PROXY=localhost,127.0.0.1`.

### 4. Run

```bash
conda activate llm2swarm

# Default 3-drone mission
python main.py

# Custom mission description
python main.py --mission "Drone 1 patrol sector A at altitude 15m, Drone 2 standby at origin"

# Override drone count
python main.py --drones 2

# Stop at any time with Ctrl-C — all drones land gracefully
```

## Testing

All tests run fully offline (mock controller, stub VLM — no simulator or network required).

```bash
conda activate llm2swarm

# Phase 1 — Skill library & controllers
python tests/test_phase1.py

# Phase 2 — Memory pool (offline) + GlobalOperator (live, skipped if no API key)
python tests/test_phase2.py

# Phase 3 — Drone lifecycle loop (offline) + live VLM (skipped if tunnel is down)
python tests/test_phase3.py

# Run all phases in sequence
python tests/test_phase1.py && python tests/test_phase2.py && python tests/test_phase3.py
```

Expected output (no network):
```
Phase 1 results: 8 passed, 0 failed
Phase 2 results: 6 passed, 0 failed   # GlobalOperator skipped (no key)
Phase 3 results: 4 passed, 0 failed   # live VLM skipped (tunnel down)
```

Expected output (with SSH tunnel active):
```
Phase 1 results: 8 passed, 0 failed
Phase 2 results: 6 passed, 0 failed   # GlobalOperator skipped (no key / quota)
Phase 3 results: 5 passed, 0 failed   # live VLM test passes
```

## Switching Between Planners

| Scenario | `.env` settings |
|---|---|
| GPT-4o (production) | `OPENAI_API_KEY=sk-...` and no `GLOBAL_LLM_*` vars |
| qwen3.5:9b via Ollama (no OpenAI credits) | `GLOBAL_LLM_BASE_URL=http://localhost:11435/v1`, `GLOBAL_LLM_API_KEY=ollama`, `GLOBAL_LLM_MODEL=qwen3.5:9b` |

No code changes needed — just update `.env`.

## Simulator Backends

| Backend | How to use |
|---|---|
| `mock` (default) | Set `SIMULATOR_BACKEND=mock`. Pure-Python physics, generates synthetic camera images. No install needed. |
| `webots` | Install [Webots](https://cyberbotics.com), add its controller library to `PYTHONPATH`, set `SIMULATOR_BACKEND=webots`. See `controllers/webots_controller.py` for wiring details. |

## Project Structure

```
LLM2Swarm/
├── main.py                      # Entry point
├── config.py                    # All configuration
├── requirements.txt
├── .env.example                 # Template — copy to .env
├── .gitignore
├── scripts/
│   └── start_tunnel.sh          # SSH tunnel manager
├── controllers/
│   ├── base_controller.py       # Abstract DroneController
│   ├── mock_controller.py       # Pure-Python mock (default)
│   └── webots_controller.py     # Webots extern-controller stub
├── memory/
│   └── pool.py                  # Async-safe SharedMemoryPool
├── operators/
│   ├── global_operator.py       # Cloud LLM mission planner
│   ├── vlm_agent.py             # Edge VLM per-tick replanner
│   └── drone_lifecycle.py       # 10-second control loop
├── models/
│   └── schemas.py               # Pydantic I/O models
├── utils/
│   └── image_utils.py           # Image encoding helpers
└── tests/
    ├── test_phase1.py            # Skill library tests
    ├── test_phase2.py            # Memory pool + planner tests
    └── test_phase3.py            # Lifecycle + VLM integration tests
```

## VLM Decision Protocol

Every 10 seconds each drone sends its camera image, telemetry, and swarm state to the edge VLM. The VLM returns one of two JSON shapes:

```json
{ "decision": "continue" }
```
```json
{
  "decision": "modify",
  "new_task": "avoid obstacle ahead",
  "new_action": { "action": "go_to_waypoint", "params": { "x": 5, "y": 5, "z": 10, "velocity": 3 } },
  "memory_update": "obstacle detected at 20,30"
}
```

If the VLM times out or returns malformed JSON, the drone automatically defaults to `continue`.
