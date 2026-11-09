"""Time the Python bindings on the scenes the AoE2 repo's bench_temporal.py measures.

Conversion overhead counts, so this runs the full round trip through NumPy. Only release
builds give meaningful numbers: run `just py` first.
"""

from __future__ import annotations

import math
import statistics
import time

import numpy as np
import screentrack

WIDTH = 1920
HEIGHT = 1080
BOX = 64.0
COLUMNS = 10
ORIGIN = 200.0
JITTER = 3
FRAMES = 30
HISTORY_SECONDS = 1.0
SETTLING_FRAMES = 3


def _scene(rng: np.random.Generator, objects: int, spacing: int) -> np.ndarray:
    indices = np.arange(objects)
    x = ORIGIN + indices % COLUMNS * spacing + rng.integers(-JITTER, JITTER + 1, objects)
    y = ORIGIN + indices // COLUMNS * spacing + rng.integers(-JITTER, JITTER + 1, objects)
    return np.stack((x, y, x + BOX, y + BOX), axis=1, dtype=np.float64)


def _percentiles(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    return statistics.median(ordered), ordered[math.ceil(len(ordered) * 0.95) - 1]


def run_case(objects: int, spacing: int, rate_hz: int) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(3)
    pixels = rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    settled_after = math.ceil(HISTORY_SECONDS * rate_hz) + SETTLING_FRAMES
    tracker = screentrack.Tracker(screentrack.Config.aoe2(), {"villager"})
    labels = ["villager"] * objects
    confidences = np.full(objects, 0.8)
    prepare: list[float] = []
    associate: list[float] = []
    for frame in range(1, FRAMES + 1):
        boxes = _scene(rng, objects, spacing)
        started = time.perf_counter()
        signatures = screentrack.rgb_mean_std(pixels, boxes)
        prepared_at = time.perf_counter()
        tracker.observe(
            capture_id=frame,
            captured_at=1.0 + frame / rate_hz,
            motion="stable",
            boxes=boxes,
            labels=labels,
            confidences=confidences,
            signatures=signatures,
        )
        finished = time.perf_counter()
        prepare.append((prepared_at - started) * 1000)
        associate.append((finished - prepared_at) * 1000)
    return prepare[settled_after:], associate[settled_after:]


def main() -> int:
    print("Steady-state milliseconds per frame, through the Python bindings")
    print(
        f"{'layout':<10}{'n':>4}{'Hz':>6}{'prep p50':>10}{'prep p95':>10}"
        f"{'asso p50':>10}{'asso p95':>10}{'total p50':>11}{'total p95':>11}"
    )
    for layout, spacing in (("separated", 100), ("crowded", 6)):
        for objects in (20, 40, 60):
            for rate_hz in (5, 10):
                prepare, associate = run_case(objects, spacing, rate_hz)
                total = [a + b for a, b in zip(prepare, associate, strict=True)]
                print(
                    f"{layout:<10}{objects:>4}{rate_hz:>6}"
                    f"{_percentiles(prepare)[0]:>10.3f}{_percentiles(prepare)[1]:>10.3f}"
                    f"{_percentiles(associate)[0]:>10.3f}{_percentiles(associate)[1]:>10.3f}"
                    f"{_percentiles(total)[0]:>11.3f}{_percentiles(total)[1]:>11.3f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
