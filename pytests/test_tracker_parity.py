"""Every golden fixture, replayed through the Python bindings."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import screentrack

FIXTURES = Path(__file__).parent.parent / "fixtures"
ABSOLUTE_TOLERANCE = 1e-9
SEQUENCES = json.loads((FIXTURES / "manifest.json").read_text())["tracker_sequences"]


def _motion(motion: dict) -> Any:
    if motion["kind"] == "pan":
        return ("pan", motion["dx"], motion["dy"])
    return motion["kind"]


def _replay_frame(tracker: screentrack.Tracker, frame: dict) -> None:
    detections = frame["detections"]
    expected = frame["expected"]
    result = tracker.observe(
        capture_id=frame["capture_id"],
        captured_at=frame["captured_at"],
        motion=_motion(frame["motion"]),
        boxes=np.array([item["bbox"] for item in detections], dtype=np.float64).reshape(-1, 4),
        labels=[item["label"] for item in detections],
        confidences=np.array([item["confidence"] for item in detections], dtype=np.float64),
        signatures=np.array([item["signature"] for item in detections], dtype=np.float64).reshape(
            -1, 6
        ),
    )
    if not expected["accepted"]:
        assert result is None
        return
    assert result is not None
    track_ids, labels, agreements = result
    wanted = expected["tracked"]
    assert track_ids.tolist() == [item["track_id"] for item in wanted]
    assert labels == [item["label"] for item in wanted]
    assert tracker.ambiguous_count == expected["ambiguous_count"]
    assert tracker.candidate_count == expected["candidate_count"]
    for agreement, item in zip(agreements.tolist(), wanted, strict=True):
        if item["agreement"] is None:
            assert math.isnan(agreement)
        else:
            assert agreement == pytest.approx(item["agreement"], rel=0, abs=ABSOLUTE_TOLERANCE)


@pytest.mark.parametrize("name", SEQUENCES)
def test_fixture_replays_identically(name: str) -> None:
    sequence = json.loads((FIXTURES / "tracker" / f"{name}.json").read_text())
    tracker = screentrack.Tracker(screentrack.Config.aoe2())
    for frame in sequence["frames"]:
        _replay_frame(tracker, frame)
