"""The clip V-JEPA 2 sees: ``frames`` images ``period`` apart, by their stamps, the newest last.

A camera's frames come at its own rate (10, 15, 30 frames/s, or fewer when the network drops some); the model wants a
fixed time between its frames. ``Clip`` keeps the frames of the last clip length (each already made the model's input,
a few hundred kB on the GPU) and ``take`` picks, for the times ``newest - k * period``, the frame nearest to each. A
frame further than one period from its time means the stream had a gap (or has not been running for a clip yet): no
clip. A stamp older than the newest by more than a clip (a replay starting over, a camera's clock reset) starts the
clip anew.
"""

from __future__ import annotations

import bisect


def pick(stamps: list[int], frames: int, period_ns: int) -> list[int] | None:
    """The stamps (sorted ``stamps``, oldest first) of a clip of ``frames`` ending at the newest one, ``period_ns``
    apart: each the stamp nearest its time (a frame may be taken twice when the camera is slower than the clip); None
    if a time has no frame within one period."""
    if not stamps:
        return None
    newest, chosen = stamps[-1], []
    for k in range(frames - 1, -1, -1):
        target = newest - k * period_ns
        i = bisect.bisect_left(stamps, target)
        near = min((s for s in stamps[max(i - 1, 0):i + 1]), key=lambda s: abs(s - target))
        if abs(near - target) > period_ns:
            return None
        chosen.append(near)
    return chosen


class Clip:
    """The frames of the last clip (stamp -> the model's input of that frame)."""

    def __init__(self, frames: int, period_ns: int):
        if frames < 1 or period_ns <= 0:
            raise ValueError("a clip of at least one frame, a positive period")
        self.frames, self.period_ns = frames, period_ns
        self.span_ns = (frames - 1) * period_ns
        self.inputs: dict[int, object] = {}
        self.restarts = 0

    @property
    def newest(self) -> int | None:
        return max(self.inputs) if self.inputs else None

    def put(self, stamp_ns: int, item):
        newest = self.newest
        if newest is not None and stamp_ns < newest - self.span_ns - self.period_ns:
            self.inputs.clear()  # time went back by more than a clip: a new stream
            self.restarts += 1
        self.inputs[stamp_ns] = item
        newest = max(newest or stamp_ns, stamp_ns)
        for old in [s for s in self.inputs if s < newest - self.span_ns - self.period_ns]:
            del self.inputs[old]

    def take(self) -> tuple[list[int], list] | None:
        """(stamps, inputs) of the clip ending at the newest frame, oldest first; None while there is none."""
        stamps = pick(sorted(self.inputs), self.frames, self.period_ns)
        return None if stamps is None else (stamps, [self.inputs[s] for s in stamps])
