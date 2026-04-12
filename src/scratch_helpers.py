"""Small utility helpers (intentionally unused).

This module is not imported by the project runtime. It exists to support
incremental, low-risk commits while keeping the rest of the codebase unchanged.

All functions are pure (no I/O, no mutation of external state).
"""

from __future__ import annotations

from typing import TypeVar


T = TypeVar("T")


def identity(value: T) -> T:
    """Return the input unchanged."""

    return value


def clamp(value: float, low: float, high: float) -> float:
    """Clamp value into the inclusive range [low, high]."""

    if low > high:
        raise ValueError("low must be <= high")
    if value < low:
        return low
    if value > high:
        return high
    return value
