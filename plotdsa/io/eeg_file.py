"""
Raw EEG CSV file import.

Expected format (one sample per line, no header required):

    20:12:06.2613,-82
    20:12:06.2638,-82

The time column may also carry a date ("2024-05-01 20:12:06.2613").
Files without a date are placed on ``base_date`` (default: today).
"""
import datetime
import math
import re

import numpy as np

from .. import config
from ..core.buffers import EEGBuffer

_SPLIT_RE = re.compile(r"[,;\t]")
_SECONDS_PER_DAY = 24 * 3600
# Gaps longer than this many sample periods are kept as gaps (NaN) when resampling.
_GAP_FACTOR = 3.0


def _parse_time_of_day(text):
    hours, minutes, seconds = text.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _parse_absolute(text):
    return datetime.datetime.fromisoformat(text.replace("T", " ")).timestamp()


def _parse_lines(lines):
    """Return (time_values, eeg_values, has_date) for all parseable lines."""
    times, values = [], []
    has_date = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = _SPLIT_RE.split(line)
        if len(parts) < 2:
            continue
        time_text = parts[0].strip()
        try:
            if has_date is None:
                has_date = " " in time_text or "T" in time_text
            t = _parse_absolute(time_text) if has_date else _parse_time_of_day(time_text)
            v = float(parts[1])
        except ValueError:
            if not times:
                has_date = None  # header line, decide again on the next line
            continue
        times.append(t)
        values.append(v if math.isfinite(v) else np.nan)
    return times, values, bool(has_date)


def _unwrap_midnight(seconds):
    """Add a day whenever time-of-day jumps back by more than 12 h."""
    jumps = np.diff(seconds) < -_SECONDS_PER_DAY / 2
    if not jumps.any():
        return seconds
    offsets = np.concatenate(([0.0], np.cumsum(jumps) * _SECONDS_PER_DAY))
    return seconds + offsets


def _resample(timestamps, values, rate_hz):
    """Linearly resample to a uniform grid at rate_hz; gaps stay NaN."""
    period = 1.0 / rate_hz
    source_period = float(np.median(np.diff(timestamps)))
    n = int(math.floor((timestamps[-1] - timestamps[0]) / period)) + 1
    new_ts = timestamps[0] + np.arange(n) * period

    finite = np.isfinite(values)
    new_values = np.interp(new_ts, timestamps[finite], values[finite]) if finite.any() \
        else np.full(n, np.nan)

    idx = np.clip(np.searchsorted(timestamps, new_ts, side="right"), 1, len(timestamps) - 1)
    interval = timestamps[idx] - timestamps[idx - 1]
    gap = interval > _GAP_FACTOR * max(source_period, period)
    gap |= ~finite[idx] | ~finite[idx - 1]
    new_values[gap] = np.nan
    return new_ts, new_values


def read_eeg_csv(path, base_date=None):
    """
    Read a raw EEG CSV file.

    Returns (timestamps, values): epoch seconds (float64) and EEG values (float64),
    resampled to config.SAMPLE_RATE_HZ if the file uses a different rate.
    Raises ValueError if the file contains no usable samples.
    """
    with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
        times, values, has_date = _parse_lines(handle)

    if len(times) < 2:
        raise ValueError("No valid EEG samples found in file.")

    timestamps = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    if not has_date:
        day = base_date or datetime.date.today()
        midnight = datetime.datetime.combine(day, datetime.time()).timestamp()
        timestamps = midnight + _unwrap_midnight(timestamps)

    order = np.argsort(timestamps, kind="stable")
    timestamps, values = timestamps[order], values[order]
    keep = np.concatenate(([True], np.diff(timestamps) > 0))
    timestamps, values = timestamps[keep], values[keep]
    if len(timestamps) < 2:
        raise ValueError("No valid EEG samples found in file.")

    file_rate = 1.0 / float(np.median(np.diff(timestamps)))
    if abs(file_rate - config.SAMPLE_RATE_HZ) > 0.01 * config.SAMPLE_RATE_HZ:
        timestamps, values = _resample(timestamps, values, config.SAMPLE_RATE_HZ)

    return timestamps, values


def snap_dsa_column(ts, hop_sec):
    """Snap a DSA column start to the display grid and return (snapped_ts, steps)."""
    slot = int(math.floor(ts / config.TIME_RESOLUTION + 0.5))
    next_slot = int(math.floor((ts + hop_sec) / config.TIME_RESOLUTION + 0.5))
    next_slot = max(slot + 1, next_slot)
    return slot * config.TIME_RESOLUTION, next_slot - slot


def dsa_columns_from_samples(eeg_buffer, samples, method):
    """Feed (timestamp, value) samples into eeg_buffer and return [(ts, psd, steps)]."""
    hop_sec = eeg_buffer.hop_len / config.SAMPLE_RATE_HZ
    columns = []
    for ts, psd in eeg_buffer.get_dsa_columns(samples, method=method):
        if psd is None:
            continue
        snapped_ts, steps = snap_dsa_column(ts, hop_sec)
        columns.append((snapped_ts, psd, steps))
    return columns


def compute_dsa_columns(timestamps, values, user_config):
    """Compute all DSA columns for a whole recording."""
    eeg_buffer = EEGBuffer(user_config.window_sec, user_config.window_overlap)
    method = "multitaper" if user_config.use_multitaper else "welch"
    samples = list(zip(np.asarray(timestamps).tolist(), np.asarray(values).tolist()))
    return dsa_columns_from_samples(eeg_buffer, samples, method)
