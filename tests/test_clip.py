"""clip.py: which frames make a clip, whatever the camera's rate; gaps, the start and a stream starting over."""

from xblick_vjepa2.clip import Clip, pick

MS = 1_000_000
PERIOD = 125 * MS  # 8 frames/s


def stamps(rate_hz: float, seconds: float, start_ns: int = 10**12) -> list[int]:
    return [start_ns + round(i * 1e9 / rate_hz) for i in range(int(seconds * rate_hz) + 1)]


def test_a_camera_at_the_clips_rate_gives_its_frames():
    s = stamps(8, 3)
    assert pick(s, 16, PERIOD) == s[-16:]


def test_a_faster_camera_gives_the_nearest_frames():
    s = stamps(30, 3)
    chosen = pick(s, 16, PERIOD)
    assert len(chosen) == 16 and chosen[-1] == s[-1]
    for k, c in enumerate(reversed(chosen)):
        assert abs(c - (s[-1] - k * PERIOD)) <= 1e9 / 30 / 2 + 1  # within half a camera period of its time


def test_a_slower_camera_gives_frames_twice():
    chosen = pick(stamps(5, 4), 16, PERIOD)
    assert len(chosen) == 16 and len(set(chosen)) < 16


def test_no_clip_before_its_time_or_across_a_gap():
    assert pick(stamps(30, 1.5), 16, PERIOD) is None  # 1.5 s of a 1.875 s clip
    assert pick([], 16, PERIOD) is None
    s = stamps(30, 3)
    gap = [t for t in s if not s[-1] - 1.2e9 < t < s[-1] - 0.7e9]  # 500 ms missing: a time 250 ms from any frame
    assert pick(gap, 16, PERIOD) is None
    short = [t for t in s if not s[-1] - 0.9e9 < t < s[-1] - 0.7e9]  # 200 ms: every time within a period of a frame
    assert pick(short, 16, PERIOD) is not None


def test_the_clip_keeps_a_clip_and_starts_over_when_time_goes_back():
    clip = Clip(16, PERIOD)
    s = stamps(30, 5)
    for t in s:
        clip.put(t, t)
    assert min(clip.inputs) >= s[-1] - 16 * PERIOD  # older frames dropped
    got_stamps, inputs = clip.take()
    assert got_stamps == inputs and got_stamps[-1] == s[-1]
    for t in stamps(30, 0.5, start_ns=s[0]):  # a replay from its start
        clip.put(t, t)
    assert clip.restarts == 1 and clip.newest == stamps(30, 0.5, start_ns=s[0])[-1] and clip.take() is None


def test_max_hz_limits_the_clips_not_their_frames():
    """--max-hz: of a 30 frames/s stream's clips, the first of each 1/max-hz period is encoded; without a rate all."""
    from xblick_vjepa2.node import clip_due

    for max_hz, expected in ((None, 91), (4.0, 13), (1.0, 4)):
        period = None if max_hz is None else round(1e9 / max_hz)
        clipped, encoded = -1, 0
        for t in stamps(30, 3.0):
            if clip_due(t, clipped, period):
                clipped, encoded = t, encoded + 1
        assert encoded == expected, (max_hz, encoded)
