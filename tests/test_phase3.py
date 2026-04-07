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
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

import config
from controllers import make_controller
from memory.pool import SharedMemoryPool
from models.schemas import AgentProfile, RoleBrief, TaskEvent, TaskGraphNode, TaskGraphSpec, VLMContinue, VLMModify, DroneState
from operators.drone_lifecycle import DroneLifecycle, launch_drone
from operators.local_planner import build_task_graph_runtime, compile_role_brief
from operators.vlm_agent import VLMAgent
from primitives import build_agent_profile


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


def _generic_role_with_bootstrap() -> RoleBrief:
    return RoleBrief(
        mission_role="Generic test role",
        mission_intent="Exercise the generic role/task-graph interfaces.",
        responsibilities=["maintain the assigned test context"],
        coordination_contracts=["avoid duplicate work in the shared test context"],
        event_watchlist=["peer_lost", "target_detected"],
        shared_context={
            "bootstrap_actions": [
                {"action": "takeoff", "params": {"altitude": 8.0}},
                {"action": "go_to_waypoint", "params": {"x": 0.0, "y": 0.0, "z": 8.0, "velocity": 5.0}},
                {"action": "search_pattern", "params": {"center_x": 0.0, "center_y": 0.0, "radius": 5.0, "altitude": 8.0}},
            ]
        },
    )


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


def test_task_graph_records_vlm_event_generically():
    role = _generic_role_with_bootstrap()
    task_graph = compile_role_brief("drone_1", role)

    decision = VLMModify(
        decision="modify",
        new_task="inspect potential target",
        new_action={
            "action": "go_to_waypoint",
            "params": {"x": 20.0, "y": 20.0, "z": 10.0, "velocity": 4.0},
        },
        event=TaskEvent(
            type="target_detected",
            source="vlm",
            priority=2,
            payload={"target_position": [20.0, 20.0, 10.0]},
        ),
    )

    task_graph.apply_vlm_modify(
        decision,
        current_action={
            "action": "search_pattern",
            "params": {"center_x": 0.0, "center_y": 0.0, "radius": 5.0, "altitude": 8.0},
        },
    )
    actions = [item["action"] for item in task_graph.preview_actions()[:2]]
    assert actions[0] == "go_to_waypoint", actions
    assert task_graph.event_history()[-1].type == "target_detected"

def test_task_graph_records_vlm_event_generically_wrapper():
    test_task_graph_records_vlm_event_generically()
    print("✓ test_task_graph_records_vlm_event_generically")


async def _test_target_claim_blocks_duplicate_modify():
    pool = SharedMemoryPool(["drone_1", "drone_2"])
    await pool.acquire_claim(
        claim_type="target_claim",
        target_key="grid:4:4:2",
        claimant_id="drone_2",
        ttl=30.0,
        payload={"target_position": [20.0, 20.0, 10.0]},
    )

    ctrl = make_controller("drone_1")
    await ctrl.connect()

    role = _generic_role_with_bootstrap()
    task_graph = compile_role_brief("drone_1", role)
    before = task_graph.preview_actions()

    lifecycle = DroneLifecycle(
        drone_id="drone_1",
        controller=ctrl,
        memory=pool,
        vlm=StubVLM([VLMContinue(decision="continue")]),
        task_graph=task_graph,
        stop_when_empty=True,
    )

    decision = VLMModify(
        decision="modify",
        new_task="inspect potential target",
        new_action={"action": "hover", "params": {"duration": 2.0}},
        event=TaskEvent(
            type="target_detected",
            source="vlm",
            priority=2,
            payload={"target_position": [20.0, 20.0, 10.0]},
        ),
        memory_update="potential fire at 20,20",
    )

    await lifecycle._apply_decision(decision)

    after = task_graph.preview_actions()
    assert after == before, "Task graph should stay unchanged when target claim is already owned"

    claims = await pool.get_claims()
    owners = [claim.claimant_id for claim in claims if claim.claim_type == "target_claim"]
    assert owners == ["drone_2"], f"Expected drone_2 to keep the claim, got {owners}"

    await ctrl.disconnect()

def test_target_claim_blocks_duplicate_modify():
    run(_test_target_claim_blocks_duplicate_modify())
    print("✓ test_target_claim_blocks_duplicate_modify")


async def _test_peer_loss_emits_shared_event_without_reallocation():
    pool = SharedMemoryPool(["drone_1", "drone_2"])
    ctrl = make_controller("drone_1")
    await ctrl.connect()

    role = _generic_role_with_bootstrap()
    task_graph = compile_role_brief("drone_1", role)
    before = task_graph.preview_actions()

    lifecycle = DroneLifecycle(
        drone_id="drone_1",
        controller=ctrl,
        memory=pool,
        vlm=StubVLM([VLMContinue(decision="continue")]),
        task_graph=task_graph,
        stop_when_empty=True,
    )

    await pool.update_drone("drone_2", status="executing")
    pool._store["drone_2"].updated_at = time.time() - config.PEER_LOST_TIMEOUT - 1.0

    peer_states = await pool.get_peer_states("drone_1")
    await lifecycle._detect_runtime_events(peer_states)

    events = await pool.get_events()
    assert events, "Expected a shared event to be emitted"
    assert events[-1].type == "peer_lost"
    assert events[-1].payload["peer_id"] == "drone_2"
    assert task_graph.preview_actions() == before, "Framework should not auto-reallocate actions"

    await ctrl.disconnect()

def test_peer_loss_emits_shared_event_without_reallocation():
    run(_test_peer_loss_emits_shared_event_without_reallocation())
    print("✓ test_peer_loss_emits_shared_event_without_reallocation")


async def _test_vlm_event_is_published_to_shared_stream():
    pool = SharedMemoryPool(["drone_1"])
    ctrl = make_controller("drone_1")
    await ctrl.connect()

    role = _generic_role_with_bootstrap()
    task_graph = compile_role_brief("drone_1", role)

    lifecycle = DroneLifecycle(
        drone_id="drone_1",
        controller=ctrl,
        memory=pool,
        vlm=StubVLM([VLMContinue(decision="continue")]),
        task_graph=task_graph,
        stop_when_empty=True,
    )

    decision = VLMModify(
        decision="modify",
        new_task="inspect scene change",
        new_action={"action": "hover", "params": {"duration": 2.0}},
        event=TaskEvent(
            type="obstacle_detected",
            source="vlm",
            priority=1,
            payload={"bearing_deg": 20.0},
        ),
    )

    await lifecycle._apply_decision(decision)

    events = await pool.get_events()
    assert events[-1].type == "obstacle_detected"
    assert task_graph.preview_actions()[0]["action"] == "hover"

    await ctrl.disconnect()

def test_vlm_event_is_published_to_shared_stream():
    run(_test_vlm_event_is_published_to_shared_stream())
    print("✓ test_vlm_event_is_published_to_shared_stream")


def test_task_graph_runtime_accepts_generic_graph_spec():
    role = RoleBrief(
        mission_role="Generic runtime role",
        mission_intent="Validate the task-graph runtime against a model-produced graph.",
    )
    spec = TaskGraphSpec(
        graph_id="runtime-test",
        summary="Two-step generic runtime graph",
        nodes=[
            TaskGraphNode(
                node_id="n0",
                label="takeoff",
                action={"action": "takeoff", "params": {"altitude": 5.0}},
            ),
            TaskGraphNode(
                node_id="n1",
                label="hover",
                action={"action": "hover", "params": {"duration": 0.1}},
            ),
        ],
    )
    runtime = build_task_graph_runtime("drone_1", role, spec)
    preview = runtime.preview_actions()
    assert [item["action"] for item in preview] == ["takeoff", "hover"]

def test_task_graph_runtime_accepts_generic_graph_spec_wrapper():
    test_task_graph_runtime_accepts_generic_graph_spec()
    print("✓ test_task_graph_runtime_accepts_generic_graph_spec")


def test_task_graph_runtime_rejects_capability_mismatch():
    role = RoleBrief(
        mission_role="Ground-only role",
        mission_intent="Validate capability-aware runtime checks.",
        capability_exclusions=["flight"],
    )
    profile = build_agent_profile(agent_id="drone_1", agent_kind="test_drone")
    spec = TaskGraphSpec(
        graph_id="runtime-mismatch",
        summary="A graph that should be rejected",
        nodes=[
            TaskGraphNode(
                node_id="n0",
                label="takeoff",
                action={"action": "takeoff", "params": {"altitude": 5.0}},
            )
        ],
    )

    try:
        build_task_graph_runtime("drone_1", role, spec, agent_profile=profile)
        assert False, "Expected capability-aware validation to reject the graph"
    except ValueError as exc:
        assert "role capability exclusions" in str(exc)

def test_task_graph_runtime_rejects_capability_mismatch_wrapper():
    test_task_graph_runtime_rejects_capability_mismatch()
    print("✓ test_task_graph_runtime_rejects_capability_mismatch")


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
            {"action": "go_to_waypoint", "params": {"x": 0.0, "y": -10.0, "z": 12.0, "velocity": 5.0}},
            {"action": "hover",          "params": {"duration": 0.1}},
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
        test_task_graph_records_vlm_event_generically_wrapper,
        test_target_claim_blocks_duplicate_modify,
        test_peer_loss_emits_shared_event_without_reallocation,
        test_vlm_event_is_published_to_shared_stream,
        test_task_graph_runtime_accepts_generic_graph_spec_wrapper,
        test_task_graph_runtime_rejects_capability_mismatch_wrapper,
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
