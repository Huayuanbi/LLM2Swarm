"""
tests/test_phase3.py — Phase 3 integration tests.

Tests 1-3: Lifecycle logic with a StubVLM (no network).
Test  4:   3-drone concurrent run with StubVLM (no network).
Test  5:   Single-drone run calling the real Ollama VLM endpoint (skipped if unreachable).

Run:
    conda run -n llm2swarm python tests/test_phase3.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

import config
from controllers import make_controller
from memory.pool import SharedMemoryPool
from models.schemas import VLMContinue, VLMModify, DroneState
from operators.drone_lifecycle import DroneLifecycle, launch_drone
from operators.vlm_agent import VLMAgent


# ─── Stub VLM — returns scripted decisions, no network ───────────────────────

class StubVLM:
    """
    Replaces VLMAgent in tests.  Cycles through a predefined decision list so
    we can assert exact behaviour without hitting the remote server.
    """
    def __init__(self, decisions: list):
        self._decisions = list(decisions)
        self._idx = 0

    async def decide(self, **kwargs) -> VLMContinue | VLMModify:
        d = self._decisions[self._idx % len(self._decisions)]
        self._idx += 1
        return d


def run(coro):
    return asyncio.run(coro)


# ─── Minimal task lists for testing ──────────────────────────────────────────

TASKS_SIMPLE = [
    {"action": "takeoff",        "params": {"altitude": 5.0}},
    {"action": "go_to_waypoint", "params": {"x": 10.0, "y": 10.0, "z": 5.0, "velocity": 5.0}},
    {"action": "hover",          "params": {"duration": 0.2}},
]

TASKS_SEARCH = [
    {"action": "takeoff",        "params": {"altitude": 8.0}},
    {"action": "search_pattern", "params": {"center_x": 0.0, "center_y": 0.0,
                                             "radius": 5.0, "altitude": 8.0}},
]


# ─── Test 1: Lifecycle runs and exhausts a simple task queue ─────────────────

async def _test_lifecycle_completes():
    pool = SharedMemoryPool(["drone_1"])
    vlm  = StubVLM([VLMContinue(decision="continue")])

    ctrl = make_controller("drone_1")
    await ctrl.connect()

    lifecycle = DroneLifecycle(
        drone_id        = "drone_1",
        controller      = ctrl,
        memory          = pool,
        vlm             = vlm,
        initial_tasks   = list(TASKS_SIMPLE),
        stop_when_empty = True,
    )

    await asyncio.wait_for(lifecycle.run(), timeout=60.0)

    state = await pool.get_drone("drone_1")
    assert state.status == "idle", f"Expected 'idle' after queue empty, got '{state.status}'"

    await ctrl.disconnect()

def test_lifecycle_completes():
    run(_test_lifecycle_completes())
    print("✓ test_lifecycle_completes")


# ─── Test 2: VLM "modify" replaces the active task ───────────────────────────

async def _test_vlm_modify():
    pool = SharedMemoryPool(["drone_1"])

    modify_decision = VLMModify(
        decision      = "modify",
        new_task      = "avoid obstacle",
        new_action    = {"action": "hover", "params": {"duration": 0.1}},
        memory_update = "obstacle_spotted_at_5_5",
    )
    # First tick: modify. Subsequent: continue (so loop can finish).
    vlm = StubVLM([modify_decision, VLMContinue(decision="continue")])

    ctrl = make_controller("drone_1")
    await ctrl.connect()

    lifecycle = DroneLifecycle(
        drone_id        = "drone_1",
        controller      = ctrl,
        memory          = pool,
        vlm             = vlm,
        initial_tasks   = [
            {"action": "takeoff", "params": {"altitude": 5.0}},
            {"action": "hover",   "params": {"duration": 0.1}},
        ],
        stop_when_empty = True,
    )

    await asyncio.wait_for(lifecycle.run(), timeout=60.0)

    state = await pool.get_drone("drone_1")
    assert "obstacle_spotted_at_5_5" in state.observations, \
        f"Observation not written to pool. Got: {state.observations}"

    await ctrl.disconnect()

def test_vlm_modify():
    run(_test_vlm_modify())
    print("✓ test_vlm_modify")


# ─── Test 3: Memory pool reflects live telemetry during execution ─────────────

async def _test_memory_sync():
    pool = SharedMemoryPool(["drone_1"])
    vlm  = StubVLM([VLMContinue(decision="continue")])

    ctrl = make_controller("drone_1")
    await ctrl.connect()

    lifecycle = DroneLifecycle(
        drone_id        = "drone_1",
        controller      = ctrl,
        memory          = pool,
        vlm             = vlm,
        initial_tasks   = [
            {"action": "takeoff",        "params": {"altitude": 10.0}},
            {"action": "go_to_waypoint", "params": {"x": 20.0, "y": 0.0, "z": 10.0, "velocity": 5.0}},
            {"action": "hover",          "params": {"duration": 0.1}},
        ],
        stop_when_empty = True,
    )

    await asyncio.wait_for(lifecycle.run(), timeout=60.0)

    state = await pool.get_drone("drone_1")
    # After completing the waypoint mission the drone should be near (20, 0, 10)
    x, y, z = state.position
    assert abs(x - 20.0) < 2.0, f"Expected x≈20, got {x}"
    assert abs(z - 10.0) < 2.0, f"Expected z≈10, got {z}"

    await ctrl.disconnect()

def test_memory_sync():
    run(_test_memory_sync())
    print("✓ test_memory_sync")


# ─── Test 4: 3 drones run concurrently without deadlock ──────────────────────

async def _test_concurrent_drones():
    drone_ids = ["drone_1", "drone_2", "drone_3"]
    pool = SharedMemoryPool(drone_ids)
    vlm  = StubVLM([VLMContinue(decision="continue")])

    task_plans = {
        "drone_1": [
            {"action": "takeoff",        "params": {"altitude": 5.0}},
            {"action": "go_to_waypoint", "params": {"x": 10.0, "y": 0.0, "z": 5.0, "velocity": 5.0}},
            {"action": "hover",          "params": {"duration": 0.1}},
        ],
        "drone_2": [
            {"action": "takeoff",        "params": {"altitude": 8.0}},
            {"action": "go_to_waypoint", "params": {"x": -10.0, "y": 10.0, "z": 8.0, "velocity": 5.0}},
            {"action": "hover",          "params": {"duration": 0.1}},
        ],
        "drone_3": [
            {"action": "takeoff",        "params": {"altitude": 12.0}},
            {"action": "search_pattern", "params": {"center_x": 0.0, "center_y": 0.0,
                                                     "radius": 3.0, "altitude": 12.0}},
        ],
    }

    async def run_one(drone_id):
        await launch_drone(
            drone_id        = drone_id,
            memory          = pool,
            vlm             = vlm,
            initial_tasks   = task_plans[drone_id],
            stop_when_empty = True,
        )

    # All three drones in parallel — must complete within 120 s
    await asyncio.wait_for(
        asyncio.gather(*[run_one(did) for did in drone_ids]),
        timeout=120.0,
    )

    # Every drone should have reported its final position to the pool
    for did in drone_ids:
        state = await pool.get_drone(did)
        assert state.position != [0.0, 0.0, 0.0], \
            f"{did} never updated its position in the pool"
        print(f"  {did}: final pos={[f'{v:.1f}' for v in state.position]}  status={state.status}")

def test_concurrent_drones():
    run(_test_concurrent_drones())
    print("✓ test_concurrent_drones (3-drone asyncio.gather)")


# ─── Test 5: Live Ollama VLM endpoint (skipped if unreachable) ────────────────

async def _check_ollama_reachable() -> bool:
    """Return True if the Ollama server responds within 3 seconds."""
    import aiohttp
    url = config.EDGE_VLM_BASE_URL.replace("/v1", "") + "/api/tags"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                return resp.status == 200
    except Exception:
        return False

async def _test_live_vlm():
    reachable = await _check_ollama_reachable()
    if not reachable:
        print(f"  ⚠  Ollama not reachable at {config.EDGE_VLM_BASE_URL} — skipping live VLM test")
        return

    pool = SharedMemoryPool(["drone_1"])
    vlm  = VLMAgent()  # real endpoint

    ctrl = make_controller("drone_1")
    await ctrl.connect()
    await ctrl.takeoff(altitude=5.0)

    pos       = await ctrl.get_position()
    vel       = await ctrl.get_velocity()
    image_b64 = await ctrl.get_camera_image_base64()

    await pool.update_drone("drone_1", position=list(pos), velocity=list(vel))
    peers = await pool.get_peer_states("drone_1")

    decision = await vlm.decide(
        drone_id     = "drone_1",
        position     = pos,
        velocity     = vel,
        status       = "executing",
        current_task = "patrol sector A",
        image_b64    = image_b64,
        peer_states  = peers,
    )

    assert decision.decision in ("continue", "modify"), \
        f"Unexpected VLM decision: {decision.decision}"
    print(f"  Live VLM decision: {decision}")

    await ctrl.disconnect()

def test_live_vlm():
    run(_test_live_vlm())
    print("✓ test_live_vlm (Ollama qwen3.5:9b)")


# ─── Runner ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_lifecycle_completes,
        test_vlm_modify,
        test_memory_sync,
        test_concurrent_drones,
        test_live_vlm,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"✗ {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"Phase 3 results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
