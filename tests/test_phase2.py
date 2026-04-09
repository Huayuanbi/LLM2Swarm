"""
tests/test_phase2.py — Phase 2 sanity checks.

Tests 1-5: Memory pool (no network).
Test  6:   GlobalOperator live call to GPT-4o (requires OPENAI_API_KEY in .env).

Run:
    conda run -n llm2swarm python tests/test_phase2.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.pool import SharedMemoryPool
from models.schemas import (
    AgentProfile,
    DroneState,
    GlobalPlan,
    GlobalRolePlan,
    OnboardPlanningContext,
    RoleBrief,
    TaskEvent,
)
from primitives import build_agent_profile
from operators.global_operator import GlobalOperator
from operators.onboard_planner import _SYSTEM_PROMPT as ONBOARD_SYSTEM_PROMPT, _build_user_prompt
from utils.debug_gate import DebugGate, list_pending_steps, read_step_metadata
from utils.openai_client import should_bypass_env_proxy
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


async def _test_pool_claim_lease():
    pool = SharedMemoryPool(["drone_1", "drone_2"])

    ok = await pool.acquire_claim(
        claim_type="peer_lost_takeover",
        target_key="drone_3",
        claimant_id="drone_1",
        ttl=5.0,
        payload={"reason": "peer lost"},
    )
    assert ok, "First claimant should acquire the lease"

    denied = await pool.acquire_claim(
        claim_type="peer_lost_takeover",
        target_key="drone_3",
        claimant_id="drone_2",
        ttl=5.0,
    )
    assert not denied, "Second claimant should be rejected while lease is active"

    claims = await pool.get_claims()
    assert len(claims) == 1
    assert claims[0].claimant_id == "drone_1"

    released = await pool.release_claim(
        claim_type="peer_lost_takeover",
        target_key="drone_3",
        claimant_id="drone_1",
    )
    assert released, "Owning claimant should be able to release its lease"

    re_acquired = await pool.acquire_claim(
        claim_type="peer_lost_takeover",
        target_key="drone_3",
        claimant_id="drone_2",
        ttl=5.0,
    )
    assert re_acquired, "Resource should be claimable after release"

def test_pool_claim_lease():
    run(_test_pool_claim_lease())
    print("✓ test_pool_claim_lease")


async def _test_pool_claim_expiry():
    pool = SharedMemoryPool(["drone_1", "drone_2"])

    ok = await pool.acquire_claim(
        claim_type="peer_lost_takeover",
        target_key="drone_3",
        claimant_id="drone_1",
        ttl=0.01,
    )
    assert ok
    await asyncio.sleep(0.05)

    re_acquired = await pool.acquire_claim(
        claim_type="peer_lost_takeover",
        target_key="drone_3",
        claimant_id="drone_2",
        ttl=1.0,
    )
    assert re_acquired, "Expired lease should not block a new claimant"

def test_pool_claim_expiry():
    run(_test_pool_claim_expiry())
    print("✓ test_pool_claim_expiry")


async def _test_pool_event_stream():
    pool = SharedMemoryPool(["drone_1", "drone_2"])
    await pool.emit_event(
        TaskEvent(
            type="peer_lost",
            source="memory_pool",
            priority=2,
            payload={"peer_id": "drone_2"},
        )
    )
    events = await pool.get_events()
    assert len(events) == 1
    assert events[0].type == "peer_lost"
    assert events[0].payload["peer_id"] == "drone_2"

def test_pool_event_stream():
    run(_test_pool_event_stream())
    print("✓ test_pool_event_stream")


def test_role_brief_legacy_mapping():
    role = RoleBrief.model_validate(
        {
            "mission_role": "Legacy search role",
            "objective": "Inspect the assigned sector.",
            "search_strategy": "orbit_search",
            "takeoff_altitude": 12.0,
            "search_region": {"center_x": 1.0, "center_y": 2.0, "radius": 3.0, "altitude": 4.0},
            "coordination_rules": ["avoid overlap"],
            "contingencies": ["peer_lost"],
        }
    )
    assert role.mission_intent == "Inspect the assigned sector."
    assert role.coordination_contracts == ["avoid overlap"]
    assert role.event_watchlist == ["peer_lost"]
    assert role.shared_context["search_region"]["center_x"] == 1.0
    assert role.capability_exclusions == []
    assert role.resource_requirements == []
    print("✓ test_role_brief_legacy_mapping")


def test_agent_profile_derives_capabilities_from_primitives():
    profile = build_agent_profile(agent_id="drone_1", agent_kind="test_drone")
    assert isinstance(profile, AgentProfile)
    primitive_names = [primitive.name for primitive in profile.available_primitives]
    assert "takeoff" in primitive_names
    assert "search_pattern" in primitive_names
    assert "flight" in profile.available_capabilities
    assert "navigation" in profile.available_capabilities
    print("✓ test_agent_profile_derives_capabilities_from_primitives")


def test_openai_client_proxy_bypass_detection():
    assert should_bypass_env_proxy("http://localhost:11435/v1")
    assert should_bypass_env_proxy("http://127.0.0.1:11435/v1")
    assert should_bypass_env_proxy("http://10.130.138.37:11434/v1")
    assert not should_bypass_env_proxy("https://api.openai.com/v1")
    assert not should_bypass_env_proxy(None)
    print("✓ test_openai_client_proxy_bypass_detection")


async def _test_assign_roles_prompt_contract():
    mission = "search the area for fire"
    initial_states = {
        "drone_1": {
            "position": [1.0, 2.0, 3.0],
            "battery_level": 0.75,
            "status": "idle",
        },
        "drone_2": {
            "position": [-4.0, 5.0, 1.0],
            "battery_level": 0.50,
            "status": "idle",
        },
    }
    agent_profiles = {
        "drone_1": build_agent_profile(agent_id="drone_1", agent_kind="test_uav"),
        "drone_2": build_agent_profile(agent_id="drone_2", agent_kind="test_uav"),
    }
    op = GlobalOperator(drone_ids=["drone_1", "drone_2"])

    captured: dict = {}

    async def fake_run_json_planner(*, messages, parser, parse_target, validator=None):
        captured["messages"] = messages
        captured["parse_target"] = parse_target
        return GlobalRolePlan.model_validate(
            {
                "drone_1": {
                    "mission_role": "sector scout",
                    "mission_intent": "Search the assigned area for fire indicators.",
                },
                "drone_2": {
                    "mission_role": "support scout",
                    "mission_intent": "Maintain adjacent area coverage.",
                },
            }
        )

    op._run_json_planner = fake_run_json_planner  # type: ignore[method-assign]
    plan = await op.assign_roles(
        mission,
        initial_states=initial_states,
        agent_profiles=agent_profiles,
    )

    assert captured["parse_target"] == "role plan"
    assert isinstance(plan, GlobalRolePlan)

    system_prompt = captured["messages"][0]["content"]
    user_prompt = captured["messages"][1]["content"]

    assert "Do NOT output a full primitive action list or detailed route plan" in system_prompt
    assert "Keep the schema generic and reusable across different agent types and tasks" in system_prompt
    assert "capability_requirements" in system_prompt
    assert "capability_exclusions" in system_prompt

    assert f"Mission: {mission}" in user_prompt
    assert "Initial states:" in user_prompt
    assert "drone_1: position=(1.0,2.0,3.0) battery=0.75 status=idle" in user_prompt
    assert "drone_2: position=(-4.0,5.0,1.0) battery=0.50 status=idle" in user_prompt
    assert "Agent profiles:" in user_prompt
    assert "kind=test_uav" in user_prompt
    assert "primitives=['takeoff', 'land', 'go_to_waypoint', 'hover', 'search_pattern']" in user_prompt
    assert "capabilities=['area_coverage', 'flight', 'navigation', 'persistent_monitoring', 'point_to_point_motion', 'recovery', 'station_keeping', 'takeoff_landing', 'vertical_mobility']" in user_prompt

def test_assign_roles_prompt_contract():
    run(_test_assign_roles_prompt_contract())
    print("✓ test_assign_roles_prompt_contract")


async def _test_assign_roles_profile_validation_regenerates():
    mission = "search the area for fire"
    agent_profiles = {
        "drone_1": build_agent_profile(agent_id="drone_1", agent_kind="test_uav"),
        "drone_2": build_agent_profile(agent_id="drone_2", agent_kind="test_uav"),
        "drone_3": build_agent_profile(agent_id="drone_3", agent_kind="test_uav"),
    }
    op = GlobalOperator(drone_ids=["drone_1", "drone_2", "drone_3"])

    invalid_role_json = json.dumps(
        {
            "drone_1": {
                "mission_role": "searcher",
                "mission_intent": "search the area for fire",
            },
            "drone_2": {
                "mission_role": "searcher",
                "mission_intent": "search the area for fire",
            },
            "drone_3": {
                "mission_role": "swarm_coordinator",
                "mission_intent": "coordinate the swarm",
                "capability_requirements": ["communication"],
            },
        }
    )
    valid_role_json = json.dumps(
        {
            "drone_1": {
                "mission_role": "searcher",
                "mission_intent": "search the area for fire",
            },
            "drone_2": {
                "mission_role": "searcher",
                "mission_intent": "search the area for fire",
            },
            "drone_3": {
                "mission_role": "observer",
                "mission_intent": "maintain local observation and publish events",
                "capability_requirements": ["persistent_monitoring"],
            },
        }
    )

    call_messages: list[list[dict[str, str]]] = []
    responses = [invalid_role_json, valid_role_json]

    async def fake_create(*, model, messages, temperature, timeout, **extra):
        call_messages.append(messages)
        content = responses[len(call_messages) - 1]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    op._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=fake_create)
        )
    )

    plan = await op.assign_roles(mission, agent_profiles=agent_profiles)

    assert isinstance(plan, GlobalRolePlan)
    assert plan.roles["drone_3"].mission_role == "observer"
    assert len(call_messages) == 2
    second_attempt_messages = call_messages[1]
    assert any(
        "requires unavailable capabilities ['communication']" in message["content"]
        for message in second_attempt_messages
        if message["role"] == "user"
    )


def test_assign_roles_profile_validation_regenerates():
    run(_test_assign_roles_profile_validation_regenerates())
    print("✓ test_assign_roles_profile_validation_regenerates")


def test_onboard_planner_prompt_contract():
    role = RoleBrief(
        mission_role="local search role",
        mission_intent="search the area for fire",
        responsibilities=["inspect local surroundings for fire indicators"],
        capability_requirements=["flight", "area_coverage"],
        resource_permissions=["camera"],
        event_watchlist=["target_detected", "peer_lost"],
        initial_hints=["start small and safe"],
        shared_context={"region_hint": "unknown surrounding area"},
    )
    profile = build_agent_profile(agent_id="drone_1", agent_kind="test_uav")
    context = OnboardPlanningContext(
        drone_id="drone_1",
        role_brief=role,
        self_state=DroneState(
            drone_id="drone_1",
            position=[10.0, 20.0, 5.0],
            battery_level=0.8,
            status="idle",
        ),
        peer_states={
            "drone_2": DroneState(
                drone_id="drone_2",
                position=[-5.0, 4.0, 6.0],
                battery_level=0.6,
                status="executing",
            )
        },
        active_claims=[],
        active_events=[
            TaskEvent(
                type="peer_lost",
                source="memory_pool",
                priority=2,
                payload={"peer_id": "drone_3"},
            )
        ],
        agent_profile=profile,
    )

    user_prompt = _build_user_prompt(context)

    assert "RoleBrief's capability_requirements, capability_exclusions" in ONBOARD_SYSTEM_PROMPT
    assert "Only use primitives listed in the prompt payload as available to this agent" in ONBOARD_SYSTEM_PROMPT
    assert "search the area for fire" in user_prompt
    assert '"agent_profile"' in user_prompt
    assert '"agent_kind": "test_uav"' in user_prompt
    assert '"available_capabilities"' in user_prompt
    assert '"available_primitives"' in user_prompt
    assert '"drone_id": "drone_1"' in user_prompt
    assert '"peer_states"' in user_prompt
    assert '"active_events"' in user_prompt
    assert "takeoff" in user_prompt
    assert "search_pattern" in user_prompt
    assert "flight" in user_prompt
    assert "area_coverage" in user_prompt
    print("✓ test_onboard_planner_prompt_contract")


async def _test_debug_gate_stage_filter():
    with TemporaryDirectory() as tmp:
        gate = DebugGate(
            enabled=True,
            root=Path(tmp),
            session_id="test_session",
            stages={"cloud_response"},
            targets=None,
            poll_interval=0.01,
        )
        command = await gate.checkpoint(
            "vlm_response",
            {"decision": "continue"},
            actor_id="drone_1",
            allow_regenerate=True,
        )
        assert command == "continue"
        assert list(Path(tmp).rglob("*")) == []


def test_debug_gate_stage_filter():
    run(_test_debug_gate_stage_filter())
    print("✓ test_debug_gate_stage_filter")


async def _test_debug_gate_roundtrip():
    with TemporaryDirectory() as tmp:
        gate = DebugGate(
            enabled=True,
            root=Path(tmp),
            session_id="test_session",
            stages={"cloud_response"},
            targets=None,
            poll_interval=0.01,
        )
        wait_task = asyncio.create_task(
            gate.checkpoint(
                "cloud_response",
                {"parsed_roles": {"drone_1": {"mission_role": "observer"}}},
                actor_id="cloud",
                allow_regenerate=True,
                summary="Inspect response",
            )
        )
        await asyncio.sleep(0.05)
        pending = list_pending_steps(Path(tmp), session_id="test_session")
        assert len(pending) == 1
        meta = read_step_metadata(pending[0])
        assert meta["stage"] == "cloud_response"
        assert meta["actor_id"] == "cloud"
        assert "regenerate" in meta["allowed_commands"]
        (pending[0] / "command.txt").write_text("regenerate", encoding="utf-8")
        command = await asyncio.wait_for(wait_task, timeout=1.0)
        assert command == "regenerate"
        assert (pending[0] / "resolved.json").exists()


def test_debug_gate_roundtrip():
    run(_test_debug_gate_roundtrip())
    print("✓ test_debug_gate_roundtrip")


async def _test_debug_gate_vlm_only_pauses_once_per_actor():
    with TemporaryDirectory() as tmp:
        gate = DebugGate(
            enabled=True,
            root=Path(tmp),
            session_id="test_session",
            stages={"vlm_request", "vlm_response"},
            targets=None,
            poll_interval=0.01,
            vlm_pause_once_per_actor=True,
        )
        first_wait = asyncio.create_task(
            gate.checkpoint(
                "vlm_request",
                {"tick": 1},
                actor_id="drone_1",
                allow_regenerate=True,
            )
        )
        await asyncio.sleep(0.05)
        pending = list_pending_steps(Path(tmp), session_id="test_session")
        assert len(pending) == 1
        (pending[0] / "command.txt").write_text("continue", encoding="utf-8")
        first_command = await asyncio.wait_for(first_wait, timeout=1.0)
        assert first_command == "continue"

        second_command = await gate.checkpoint(
            "vlm_request",
            {"tick": 2},
            actor_id="drone_1",
            allow_regenerate=True,
        )
        assert second_command == "continue"

        all_step_dirs = [path for path in (Path(tmp) / "test_session").iterdir() if path.is_dir() and not path.name.startswith(".")]
        assert len(all_step_dirs) == 1, all_step_dirs


def test_debug_gate_vlm_only_pauses_once_per_actor():
    run(_test_debug_gate_vlm_only_pauses_once_per_actor())
    print("✓ test_debug_gate_vlm_only_pauses_once_per_actor")


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
        test_pool_claim_lease,
        test_pool_claim_expiry,
        test_pool_event_stream,
        test_role_brief_legacy_mapping,
        test_agent_profile_derives_capabilities_from_primitives,
        test_openai_client_proxy_bypass_detection,
        test_assign_roles_prompt_contract,
        test_assign_roles_profile_validation_regenerates,
        test_onboard_planner_prompt_contract,
        test_debug_gate_stage_filter,
        test_debug_gate_roundtrip,
        test_debug_gate_vlm_only_pauses_once_per_actor,
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
