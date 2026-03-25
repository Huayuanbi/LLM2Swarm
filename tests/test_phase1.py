"""
tests/test_phase1.py — Phase 1 sanity checks.

Runs without any external simulator, LLM, or network access.
Execute with:
    conda run -n llm2swarm python -m pytest tests/test_phase1.py -v
or simply:
    conda run -n llm2swarm python tests/test_phase1.py
"""

from __future__ import annotations

import asyncio
import sys
import os

# Allow running from the repo root without installing as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from controllers import make_controller
from controllers.mock_controller import MockDroneController
from models.schemas import (
    DroneState, GlobalPlan, VLMContinue, VLMModify, parse_vlm_decision
)
from utils.image_utils import build_image_message, describe_image_mock


# ─── helpers ──────────────────────────────────────────────────────────────────

def run(coro):
    return asyncio.run(coro)


# ─── Test 1: factory returns the right type ───────────────────────────────────

def test_factory_returns_mock():
    assert config.SIMULATOR_BACKEND == "mock", \
        f"Expected SIMULATOR_BACKEND='mock', got '{config.SIMULATOR_BACKEND}'"
    ctrl = make_controller("drone_1")
    assert isinstance(ctrl, MockDroneController), \
        f"Expected MockDroneController, got {type(ctrl)}"
    print("✓ test_factory_returns_mock")


# ─── Test 2: connect / disconnect lifecycle ───────────────────────────────────

async def _test_connect():
    ctrl = make_controller("drone_1")
    assert not ctrl.is_connected
    await ctrl.connect()
    assert ctrl.is_connected
    await ctrl.disconnect()
    assert not ctrl.is_connected

def test_connect_lifecycle():
    run(_test_connect())
    print("✓ test_connect_lifecycle")


# ─── Test 3: takeoff changes altitude ─────────────────────────────────────────

async def _test_takeoff():
    ctrl = make_controller("drone_2")
    await ctrl.connect()
    _, _, z_before = await ctrl.get_position()
    assert z_before == 0.0, f"Expected z=0 before takeoff, got {z_before}"
    await ctrl.takeoff(altitude=5.0)
    _, _, z_after = await ctrl.get_position()
    assert abs(z_after - 5.0) < 0.5, f"Expected z≈5 after takeoff, got {z_after}"
    await ctrl.disconnect()

def test_takeoff():
    run(_test_takeoff())
    print("✓ test_takeoff")


# ─── Test 4: go_to_waypoint moves drone ───────────────────────────────────────

async def _test_go_to_waypoint():
    ctrl = make_controller("drone_3")
    await ctrl.connect()
    await ctrl.takeoff(altitude=10.0)
    await ctrl.go_to_waypoint(20.0, 30.0, 10.0, velocity=5.0)
    x, y, z = await ctrl.get_position()
    assert abs(x - 20.0) < 0.5, f"x mismatch: {x}"
    assert abs(y - 30.0) < 0.5, f"y mismatch: {y}"
    assert abs(z - 10.0) < 0.5, f"z mismatch: {z}"
    await ctrl.disconnect()

def test_go_to_waypoint():
    run(_test_go_to_waypoint())
    print("✓ test_go_to_waypoint")


# ─── Test 5: camera image is a non-empty Base64 string ────────────────────────

async def _test_camera():
    ctrl = make_controller("drone_1")
    await ctrl.connect()
    b64 = await ctrl.get_camera_image_base64()
    assert isinstance(b64, str) and len(b64) > 100, \
        "Camera image should be a non-trivial Base64 string"
    # Verify it wraps correctly into the OpenAI content format
    msg = build_image_message(b64)
    assert msg["type"] == "image_url"
    assert msg["image_url"]["url"].startswith("data:image/jpeg;base64,")
    await ctrl.disconnect()

def test_camera_image():
    run(_test_camera())
    print("✓ test_camera_image")


# ─── Test 6: hover does not move the drone ────────────────────────────────────

async def _test_hover():
    ctrl = make_controller("drone_1")
    await ctrl.connect()
    await ctrl.takeoff(altitude=8.0)
    pos_before = await ctrl.get_position()
    await ctrl.hover(duration=0.3)   # short duration for fast tests
    pos_after = await ctrl.get_position()
    for a, b in zip(pos_before, pos_after):
        assert abs(a - b) < 0.1, f"Position changed during hover: {pos_before} → {pos_after}"
    await ctrl.disconnect()

def test_hover():
    run(_test_hover())
    print("✓ test_hover")


# ─── Test 7: execute_action dispatch ─────────────────────────────────────────

async def _test_execute_action():
    ctrl = make_controller("drone_1")
    await ctrl.connect()
    await ctrl.execute_action({"action": "takeoff", "params": {"altitude": 6.0}})
    _, _, z = await ctrl.get_position()
    assert abs(z - 6.0) < 0.5, f"z after execute_action takeoff: {z}"

    # Unknown action should raise
    try:
        await ctrl.execute_action({"action": "fly_backwards", "params": {}})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    await ctrl.disconnect()

def test_execute_action():
    run(_test_execute_action())
    print("✓ test_execute_action")


# ─── Test 8: Pydantic schema validation ──────────────────────────────────────

def test_schemas():
    # DroneState defaults
    s = DroneState(drone_id="drone_1")
    assert s.position == [0.0, 0.0, 0.0]
    assert s.status == "idle"

    # GlobalPlan — flat dict (LLM often returns this)
    gp = GlobalPlan.model_validate({
        "drone_1": [{"action": "takeoff", "params": {"altitude": 10}}],
        "drone_2": [{"action": "hover",   "params": {"duration": 5}}],
    })
    assert len(gp.get_tasks("drone_1")) == 1
    assert gp.get_tasks("drone_99") == []

    # VLMDecision — continue
    d = parse_vlm_decision({"decision": "continue"})
    assert isinstance(d, VLMContinue)

    # VLMDecision — modify
    d2 = parse_vlm_decision({
        "decision":      "modify",
        "new_task":      "avoid obstacle",
        "new_action":    {"action": "go_to_waypoint", "params": {"x": 5, "y": 5, "z": 10, "velocity": 3}},
        "memory_update": "obstacle_at_10_10",
    })
    assert isinstance(d2, VLMModify)
    assert d2.new_task == "avoid obstacle"

    # Bad decision key
    try:
        parse_vlm_decision({"decision": "explode"})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("✓ test_schemas")


# ─── Runner ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_factory_returns_mock,
        test_connect_lifecycle,
        test_takeoff,
        test_go_to_waypoint,
        test_camera_image,
        test_hover,
        test_execute_action,
        test_schemas,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"✗ {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Phase 1 results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
