"""
controllers/__init__.py — Factory for creating the right DroneController
based on the SIMULATOR_BACKEND config value.
"""

import config
from controllers.base_controller import DroneController


def make_controller(
    drone_id: str,
    sim_positions: dict | None = None,
) -> DroneController:
    """
    Return the appropriate DroneController subclass for the configured backend.

    Args:
        sim_positions: Optional shared dict ``{drone_id: [x, y, z]}`` written
                       by the simulator's physics loop so the visualizer can
                       read smooth positions without touching the memory pool.

    Usage:
        ctrl = make_controller("drone_1", sim_positions=sim_positions)
        await ctrl.connect()
    """
    backend = config.SIMULATOR_BACKEND.lower()

    if backend == "mock":
        from controllers.mock_controller import MockDroneController
        return MockDroneController(drone_id, sim_positions=sim_positions)

    elif backend == "webots":
        from controllers.webots_controller import WebotsController
        return WebotsController(drone_id)

    else:
        raise ValueError(
            f"Unknown SIMULATOR_BACKEND '{backend}'. "
            "Valid options: 'mock', 'webots'"
        )
