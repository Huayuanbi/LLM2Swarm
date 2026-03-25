"""
tests/test_phase2.py — Phase 2 sanity checks.

Tests 1-5: Memory pool (no network).
Test  6:   GlobalOperator live call to GPT-4o (requires OPENAI_API_KEY in .env).

Run:
    conda run -n llm2swarm python tests/test_phase2.py
"""

from __future__ import annotations

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.pool import SharedMemoryPool
from models.schemas import DroneState, GlobalPlan
from operators.global_operator import GlobalOperator
import config


def run(coro):
    return asyncio.run(coro)


# ─── Memory Pool tests ────────────────────────────────────────────────────────

async def _test_pool_init():
    pool = SharedMemoryPool(["drone_1", "drone_2"])
    s1 = await pool.get_drone("drone_1")
    assert s1.drone_id == "drone_1"
    assert s1.position == [0.0, 0.0, 0.0]
    assert s1.status == "idle"

def test_pool_init():
    run(_test_pool_init())
    print("✓ test_pool_init")


async def _test_pool_update():
    pool = SharedMemoryPool(["drone_1"])
    await pool.update_drone("drone_1", position=[10.0, 20.0, 5.0], status="moving")
    s = await pool.get_drone("drone_1")
    assert s.position == [10.0, 20.0, 5.0]
    assert s.status == "moving"

def test_pool_update():
    run(_test_pool_update())
    print("✓ test_pool_update")


async def _test_pool_observation_cap():
    pool = SharedMemoryPool(["drone_1"])
    for i in range(25):
        await pool.update_drone("drone_1", add_observation=f"obs_{i}")
    s = await pool.get_drone("drone_1")
    assert len(s.observations) == 20, f"Expected 20, got {len(s.observations)}"
    assert s.observations[-1] == "obs_24"

def test_pool_observation_cap():
    run(_test_pool_observation_cap())
    print("✓ test_pool_observation_cap")


async def _test_pool_peer_states():
    pool = SharedMemoryPool(["drone_1", "drone_2", "drone_3"])
    peers = await pool.get_peer_states("drone_2")
    assert "drone_2" not in peers
    assert "drone_1" in peers and "drone_3" in peers

def test_pool_peer_states():
    run(_test_pool_peer_states())
    print("✓ test_pool_peer_states")


async def _test_pool_snapshot_isolation():
    """Mutating a returned snapshot must not change the pool's internal state."""
    pool = SharedMemoryPool(["drone_1"])
    snap = await pool.get_drone("drone_1")
    snap.position = [999.0, 999.0, 999.0]
    fresh = await pool.get_drone("drone_1")
    assert fresh.position == [0.0, 0.0, 0.0], \
        "Snapshot mutation leaked into pool!"

def test_pool_snapshot_isolation():
    run(_test_pool_snapshot_isolation())
    print("✓ test_pool_snapshot_isolation")


async def _test_pool_unknown_drone():
    pool = SharedMemoryPool(["drone_1"])
    try:
        await pool.get_drone("drone_99")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

def test_pool_unknown_drone():
    run(_test_pool_unknown_drone())
    print("✓ test_pool_unknown_drone")


# ─── GlobalOperator live test ─────────────────────────────────────────────────

async def _test_global_operator():
    if not config.OPENAI_API_KEY:
        print("⚠  OPENAI_API_KEY not set — skipping live GlobalOperator test")
        return

    op = GlobalOperator(drone_ids=["drone_1", "drone_2"])
    try:
        plan = await op.plan_mission(
            "Drone 1 takes off and patrols a circular area centred at (50, 50) "
            "with radius 20 m at altitude 15 m. "
            "Drone 2 takes off, flies to waypoint (80, 80, 10) and hovers for 10 seconds."
        )
    except RuntimeError as e:
        if "insufficient_quota" in str(e) or "429" in str(e):
            print("⚠  OpenAI quota exceeded — skipping live test. "
                  "Add billing at platform.openai.com/billing to enable.")
            return
        raise

    assert isinstance(plan, GlobalPlan)
    assert "drone_1" in plan.plan, f"drone_1 missing from plan: {plan.plan.keys()}"
    assert "drone_2" in plan.plan, f"drone_2 missing from plan: {plan.plan.keys()}"

    # Every drone list must start with takeoff
    for did, tasks in plan.plan.items():
        assert len(tasks) > 0, f"{did} has empty task list"
        assert tasks[0]["action"] == "takeoff", \
            f"{did}'s first action is '{tasks[0]['action']}', expected 'takeoff'"

    print(f"  Plan received: {plan.plan}")

def test_global_operator():
    run(_test_global_operator())
    print("✓ test_global_operator (live GPT-4o)")


# ─── Runner ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_pool_init,
        test_pool_update,
        test_pool_observation_cap,
        test_pool_peer_states,
        test_pool_snapshot_isolation,
        test_pool_unknown_drone,
        test_global_operator,
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
    print(f"Phase 2 results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
