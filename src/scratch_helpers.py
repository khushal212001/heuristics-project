"""Small utility helpers (intentionally unused).

This module is not imported by the project runtime. It exists to support
incremental, low-risk commits while keeping the rest of the codebase unchanged.

All functions are pure (no I/O, no mutation of external state).
"""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence, Tuple, TypeVar


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


def safe_div(numerator: float, denominator: float, *, default: float = 0.0) -> float:
    """Return numerator/denominator, or default if denominator is 0."""

    if denominator == 0:
        return default
    return numerator / denominator


def is_sorted(values: Iterable[float]) -> bool:
    """Return True if the iterable is non-decreasing."""

    it = iter(values)
    try:
        prev = next(it)
    except StopIteration:
        return True
    for x in it:
        if x < prev:
            return False
        prev = x
    return True


def pairwise(values: Iterable[T]) -> Iterator[Tuple[T, T]]:
    """Yield consecutive pairs: (v0,v1), (v1,v2), ..."""

    it = iter(values)
    try:
        prev = next(it)
    except StopIteration:
        return
    for x in it:
        yield prev, x
        prev = x


def chunked(values: Iterable[T], size: int) -> Iterator[Tuple[T, ...]]:
    """Yield tuples of length size (last chunk may be shorter)."""

    if size <= 0:
        raise ValueError("size must be positive")

    buf = []
    for x in values:
        buf.append(x)
        if len(buf) == size:
            yield tuple(buf)
            buf.clear()
    if buf:
        yield tuple(buf)


def flatten(chunks: Iterable[Sequence[T]]) -> list[T]:
    """Flatten an iterable of sequences into a single list."""

    out: list[T] = []
    for seq in chunks:
        out.extend(seq)
    return out


def rotate_left(values: Sequence[T], k: int) -> list[T]:
    """Return a new list rotated left by k steps."""

    if not values:
        return []
    n = len(values)
    kk = k % n
    return list(values[kk:]) + list(values[:kk])


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single spaces and strip ends."""

    return " ".join(text.split())


def parse_int(text: str) -> int:
    """Parse an integer from text, rejecting surrounding whitespace."""

    if text != text.strip():
        raise ValueError("unexpected surrounding whitespace")
    return int(text)
