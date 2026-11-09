"""Signatures through the bindings, including the buffers the agent really passes."""

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import screentrack

FIXTURES = Path(__file__).parent.parent / "fixtures" / "signatures"
ABSOLUTE_TOLERANCE = 1e-9


def _boxes(count: int) -> np.ndarray:
    return np.tile(np.array([0.0, 0.0, 8.0, 8.0]), (count, 1))


def test_signatures_match_numpy_fixtures() -> None:
    cases = json.loads((FIXTURES / "cases.json").read_text())
    for case in cases:
        data = (FIXTURES / case["pixels"]).read_bytes()
        pixels = np.frombuffer(data, dtype=np.uint8).reshape(case["height"], case["width"], 3)
        boxes = np.asarray(case["boxes"], dtype=np.float64)
        expected = np.asarray(case["expected"], dtype=np.float64)
        actual = screentrack.rgb_mean_std(pixels, boxes)
        np.testing.assert_array_equal(actual[:, :3], expected[:, :3])
        np.testing.assert_allclose(actual[:, 3:], expected[:, 3:], rtol=0, atol=ABSOLUTE_TOLERANCE)


def test_read_only_capture_buffers_are_accepted() -> None:
    pixels = np.frombuffer(bytes(3 * 16 * 16), dtype=np.uint8).reshape(16, 16, 3)
    assert not pixels.flags.writeable
    assert screentrack.rgb_mean_std(pixels, _boxes(2)).shape == (2, 6)


def test_no_boxes_return_an_empty_array() -> None:
    pixels = np.zeros((16, 16, 3), dtype=np.uint8)
    assert screentrack.rgb_mean_std(pixels, np.empty((0, 4))).shape == (0, 6)


@pytest.mark.parametrize(
    "pixels",
    [
        np.zeros((16, 16, 4), dtype=np.uint8),
        np.zeros((32, 32, 3), dtype=np.uint8)[::2],
        np.zeros((16, 16, 3), dtype=np.uint8, order="F"),
    ],
    ids=["four_channels", "strided_view", "fortran_order"],
)
def test_unsupported_pixel_layouts_are_rejected(pixels: np.ndarray) -> None:
    with pytest.raises(ValueError, match="pixels must"):
        screentrack.rgb_mean_std(pixels, _boxes(1))


def test_boxes_must_have_four_columns() -> None:
    pixels = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="boxes must"):
        screentrack.rgb_mean_std(pixels, np.zeros((1, 3)))


def test_the_gil_is_released_during_extraction() -> None:
    """Another Python thread keeps running while signatures are computed.

    The other thread spins rather than sleeping. A 1 ms sleep can overshoot by several
    milliseconds on a busy CI runner and miss the whole call, which made an earlier
    version of this test fail at random. A spinning thread can't run at all while the
    GIL is held, so a held GIL still fails every time.
    """
    pixels = np.zeros((1080, 1920, 3), dtype=np.uint8)
    boxes = np.tile(np.array([0.0, 0.0, 1920.0, 1080.0]), (24, 1))
    beats: list[float] = []
    stop = threading.Event()

    def spin() -> None:
        while not stop.is_set():
            beats.append(time.perf_counter())

    thread = threading.Thread(target=spin)
    thread.start()
    try:
        started = time.perf_counter()
        screentrack.rgb_mean_std(pixels, boxes)
        finished = time.perf_counter()
    finally:
        stop.set()
        thread.join()
    # A thread waiting at either end of the call would run around it anyway, so only the
    # middle of the call proves anything.
    margin = (finished - started) / 4
    assert any(started + margin < beat_at < finished - margin for beat_at in beats)
