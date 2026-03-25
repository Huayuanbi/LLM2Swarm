"""
controllers/__init__.py — Factory for creating the right DroneController
based on the SIMULATOR_BACKEND config value.
"""

import config
from controllers.base_controller import DroneController


def make_controller(drone_id: str) -> DroneController:
    """
    Return the appropriate DroneController subclass for the configured backend.

    Usage:
        ctrl = make_controller("drone_1")
        await ctrl.connect()
    """
    backend = config.SIMULATOR_BACKEND.lower()

    if backend == "mock":
        from controllers.mock_controller import MockDroneController
        return MockDroneController(drone_id)

    elif backend == "webots":
        from controllers.webots_controller import WebotsController
        return WebotsController(drone_id)

    else:
        raise ValueError(
            f"Unknown SIMULATOR_BACKEND '{backend}'. "
            "Valid options: 'mock', 'webots'"
        )
