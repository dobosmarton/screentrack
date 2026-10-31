"""The binding's own behaviour: argument handling and the Python-facing API."""

import math
from typing import Any

import numpy as np
import pytest
import screentrack

BOX = np.array([[100.0, 100.0, 120.0, 140.0]])
SIGNATURE = np.zeros((1, 6))
CONFIDENCE = np.array([0.8])


def _tracker() -> screentrack.Tracker:
    return screentrack.Tracker(screentrack.Config.aoe2())


def _observe(
    tracker: screentrack.Tracker,
    capture_id: int,
    captured_at: float,
    motion: Any = "stable",
    label: str = "villager",
) -> tuple[np.ndarray, list[str], np.ndarray] | None:
    return tracker.observe(
        capture_id=capture_id,
        captured_at=captured_at,
        motion=motion,
        boxes=BOX,
        labels=[label],
        confidences=CONFIDENCE,
        signatures=SIGNATURE,
    )


def test_identity_persists_across_frames() -> None:
    tracker = _tracker()
    first = _observe(tracker, 1, 1.0)
    second = _observe(tracker, 2, 1.2)
    assert first is not None
    assert second is not None
    assert first[0].tolist() == second[0].tolist()
    assert first[0].dtype == np.uint64
    assert second[1] == ["villager"]
    assert math.isnan(first[2][0])
    assert second[2][0] == pytest.approx(1.0)


def test_output_arrays_are_writable_by_the_caller() -> None:
    result = _observe(_tracker(), 1, 1.0)
    assert result is not None
    assert result[0].flags.writeable
    assert result[2].flags.writeable


def test_stale_frames_return_none_and_change_nothing() -> None:
    tracker = _tracker()
    _observe(tracker, 5, 5.0)
    assert not tracker.accepts(5, 6.0)
    assert _observe(tracker, 5, 6.0) is None
    assert tracker.accepts(6, 6.0)


def test_motion_forms() -> None:
    tracker = _tracker()
    first = _observe(tracker, 1, 1.0)
    panned = tracker.observe(
        capture_id=2,
        captured_at=1.2,
        motion=("pan", 300.0, 0.0),
        boxes=BOX + np.array([300.0, 0.0, 300.0, 0.0]),
        labels=["villager"],
        confidences=CONFIDENCE,
        signatures=SIGNATURE,
    )
    reset = _observe(tracker, 3, 1.4, motion="reset")
    assert first is not None
    assert panned is not None
    assert reset is not None
    assert panned[0][0] == first[0][0]
    assert reset[0][0] != first[0][0]


@pytest.mark.parametrize(
    "motion",
    ["sideways", ("zoom", 1.0, 2.0)],
    ids=["unknown_string", "unknown_tuple"],
)
def test_unknown_motion_is_rejected(motion: Any) -> None:
    with pytest.raises(ValueError, match="unknown motion"):
        _observe(_tracker(), 1, 1.0, motion=motion)


def test_a_short_motion_tuple_is_rejected() -> None:
    with pytest.raises(ValueError, match="length 3"):
        _observe(_tracker(), 1, 1.0, motion=("pan", 1.0))


def test_a_motion_of_the_wrong_type_is_rejected() -> None:
    with pytest.raises(TypeError):
        _observe(_tracker(), 1, 1.0, motion=7)


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        _tracker().observe(
            capture_id=1,
            captured_at=1.0,
            motion="stable",
            boxes=BOX,
            labels=["villager", "sheep"],
            confidences=CONFIDENCE,
            signatures=SIGNATURE,
        )


def test_wrong_column_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="signatures must"):
        _tracker().observe(
            capture_id=1,
            captured_at=1.0,
            motion="stable",
            boxes=BOX,
            labels=["villager"],
            confidences=CONFIDENCE,
            signatures=np.zeros((1, 5)),
        )


def test_empty_frames_are_valid() -> None:
    result = _tracker().observe(
        capture_id=1,
        captured_at=1.0,
        motion="stable",
        boxes=np.empty((0, 4)),
        labels=[],
        confidences=np.empty(0),
        signatures=np.empty((0, 6)),
    )
    assert result is not None
    assert result[0].shape == (0,)
    assert result[1] == []


def test_labels_are_interned_per_tracker() -> None:
    tracker = _tracker()
    for capture_id, label in enumerate(("sheep", "villager", "sheep"), start=1):
        result = _observe(tracker, capture_id, float(capture_id) / 10 + 1, label=label)
        assert result is not None
        assert result[1] == [label]


def test_reset_keeps_the_id_counter_and_the_guard() -> None:
    tracker = _tracker()
    first = _observe(tracker, 1, 1.0)
    tracker.reset()
    assert not tracker.accepts(1, 1.0)
    second = _observe(tracker, 2, 1.2)
    assert first is not None
    assert second is not None
    assert second[0][0] == first[0][0] + 1


def test_stats_describe_the_last_frame() -> None:
    tracker = _tracker()
    _observe(tracker, 1, 1.0)
    _observe(tracker, 2, 1.2)
    assert tracker.candidate_count == 1
    assert tracker.ambiguous_count == 0
    assert tracker.candidate_ms >= 0.0
    assert tracker.matching_ms >= 0.0
