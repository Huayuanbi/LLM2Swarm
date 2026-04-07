"""
Primitive registry exports.
"""

from primitives.registry import (
    build_agent_profile,
    derive_capability_tags,
    format_primitives_for_prompt,
    get_primitive_spec,
    get_supported_primitives_for_controller,
    list_registered_primitives,
    normalize_primitive_handler_kwargs,
    register_primitive,
)

__all__ = [
    "build_agent_profile",
    "derive_capability_tags",
    "format_primitives_for_prompt",
    "get_primitive_spec",
    "get_supported_primitives_for_controller",
    "list_registered_primitives",
    "normalize_primitive_handler_kwargs",
    "register_primitive",
]
