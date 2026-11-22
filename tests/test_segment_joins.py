"""Holds joined from restarted segments: frame spacing and lag pairs."""

import numpy as np
import pytest

from opcfilm.io.trajectory import (
    JOIN_LAG_ATOL_PS,
    continuation_frame_spacing_ps,
    frame_spacing_ps,
    lag_pair_ok,
)


def joined_times(gaps, frames=50, spacing=10.0):
    """Segments on a 10 ps grid, each starting ``gap`` after the last frame."""
    out, t = [], 0.0
    for gap in [None] + list(gaps):
        if gap is not None:
            t = out[-1] + gap
        out.extend(t + spacing * np.arange(frames))
    return np.asarray(out)


def test_uniform_check_still_rejects_joins():
    with pytest.raises(ValueError):
        frame_spacing_ps(joined_times([16.797, 12.223]))


def test_joins_are_accepted_and_counted():
    spacing, joins = continuation_frame_spacing_ps(joined_times([16.797, 12.223, 19.609]))
    assert spacing == pytest.approx(10.0)
    assert joins == 3


def test_float32_rounding_is_regular():
    times = np.float32(34943.781) + np.float32(10.0) * np.arange(400, dtype=np.float32)
    spacing, joins = continuation_frame_spacing_ps(times.astype(float))
    assert spacing == pytest.approx(10.0, abs=0.02)
    assert joins == 0


@pytest.mark.parametrize("gap", [5.0, 25.0])
def test_short_or_missing_frame_gaps_are_errors(gap):
    with pytest.raises(ValueError, match="not a segment join"):
        continuation_frame_spacing_ps(joined_times([gap]))


def test_overlap_is_an_error():
    times = joined_times([10.0])
    times[50:] -= 30.0
    with pytest.raises(ValueError, match="strictly increasing"):
        continuation_frame_spacing_ps(times)


def test_lag_pairs_across_a_join():
    times = joined_times([16.0], frames=200)
    start = np.arange(0, 300)
    short = lag_pair_ok(times, start, 1, 10.0)
    # Only the pair straddling the join is off by 6 ps at a 10 ps lag.
    assert np.flatnonzero(~short).tolist() == [199]
    # At 100 frames (1 ns) a 6 ps error is under 1 percent: every pair kept.
    assert lag_pair_ok(times, start[:300], 100, 10.0).all()
    assert JOIN_LAG_ATOL_PS < 1.0


def test_float32_rounding_late_in_a_long_hold():
    # hold-r1-4.44-150 runs from 35 ns to 343 ns. Above 262144 ps a float32
    # time step is 0.03125 ps, three times the old 0.01 ps tolerance.
    segments, t = [], 34943.781
    for gap in (None, 16.797, 12.223, 19.609, 10.0):
        if gap is not None:
            t = segments[-1][-1] + gap
        segments.append(t + 10.0 * np.arange(8000))
    times = np.concatenate(segments).astype(np.float32).astype(float)
    assert times[-1] > 262144.0
    assert np.abs(np.diff(times) - 10.0).max() > 0.01
    spacing, joins = continuation_frame_spacing_ps(times)
    assert spacing == pytest.approx(10.0, abs=0.04)
    assert joins == 3
    tail = times[-8000:]
    assert tail[0] > 262144.0
    assert frame_spacing_ps(tail) == pytest.approx(10.0, abs=0.04)
