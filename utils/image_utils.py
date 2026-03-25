"""
utils/image_utils.py — Helpers for image handling between the simulator and VLM.

The VLM (qwen3.5:9b via Ollama) expects images embedded in the OpenAI message
format as:
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<B64>"}}

This module provides:
  - build_image_message()   — wrap a raw Base64 string into the correct dict
  - resize_and_encode()     — resize a PIL Image and produce a Base64 JPEG string
  - describe_image_mock()   — deterministic textual description for tests
"""

from __future__ import annotations

import base64
import io
from typing import Optional

from PIL import Image


def build_image_message(base64_jpeg: str) -> dict:
    """
    Wrap a Base64-encoded JPEG string in the OpenAI vision content format.

    Args:
        base64_jpeg: Raw Base64 string (no 'data:' prefix).

    Returns:
        A content block dict ready to be inserted into a messages list.
    """
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_jpeg}"},
    }


def resize_and_encode(
    image: Image.Image,
    max_width: int = 320,
    max_height: int = 240,
    quality: int = 85,
) -> str:
    """
    Resize a PIL Image to fit within max_width×max_height (preserving aspect
    ratio), then encode as Base64 JPEG.

    Keeping images small reduces VLM inference latency significantly.
    """
    image.thumbnail((max_width, max_height), Image.LANCZOS)
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def decode_base64_to_image(base64_jpeg: str) -> Image.Image:
    """Decode a Base64 JPEG string back to a PIL Image (useful for debugging)."""
    raw = base64.b64decode(base64_jpeg)
    return Image.open(io.BytesIO(raw))


def describe_image_mock(drone_id: str, position: tuple[float, float, float]) -> str:
    """
    Return a deterministic textual scene description for unit tests that don't
    want to call a real VLM but still exercise the prompt-building logic.
    """
    x, y, z = position
    return (
        f"Synthetic camera view from {drone_id} at position "
        f"({x:.1f}, {y:.1f}, {z:.1f}) m. "
        "Scene: clear sky, flat terrain, no obstacles detected."
    )
