from typing import Literal, TypeAlias, final

import numpy as np
from numpy.typing import NDArray

__all__ = ["Config", "Tracker", "__version__", "rgb_mean_std"]

__version__: str

_Motion: TypeAlias = Literal["stable", "reset"] | tuple[Literal["pan"], float, float]
_Observation: TypeAlias = tuple[NDArray[np.uint64], list[str], NDArray[np.float64]]

def rgb_mean_std(pixels: NDArray[np.uint8], boxes: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute `(N, 6)` signatures for `(N, 4)` boxes, without holding the GIL."""

@final
class Config:
    """Tuned tracker values."""

    @staticmethod
    def aoe2() -> Config:
        """Return the values tuned for Age of Empires II."""

@final
class Tracker:
    """Links detections across frames. Keep one per video stream."""

    @property
    def ambiguous_count(self) -> int:
        """Detections with two near-equal candidates in the last accepted frame."""

    @property
    def candidate_count(self) -> int:
        """Feasible track-to-detection pairs in the last accepted frame."""

    @property
    def candidate_ms(self) -> float:
        """Time spent generating candidates in the last accepted frame."""

    @property
    def matching_ms(self) -> float:
        """Time spent on ambiguity, matching and output in the last accepted frame."""

    def __new__(cls, config: Config) -> Tracker: ...
    def accepts(self, capture_id: int, captured_at: float) -> bool:
        """Return whether a frame is newer than the last by both ID and time."""

    def reset(self) -> None:
        """Drop every track, keeping the ID counter and the ordering guard."""

    def observe(
        self,
        *,
        capture_id: int,
        captured_at: float,
        motion: _Motion,
        boxes: NDArray[np.float64],
        labels: list[str],
        confidences: NDArray[np.float64],
        signatures: NDArray[np.float64],
    ) -> _Observation | None:
        """Process one frame, returning `None` if it is not newer than the last.

        The result is `(track_ids, labels, agreements)`, in detection order. An agreement
        is `NaN` until a track has two evidence entries.
        """
