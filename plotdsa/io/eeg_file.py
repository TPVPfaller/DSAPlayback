"""
Raw EEG CSV file import.

Expected format (one sample per line, no header required):

    20:12:06.2613,-82
    20:12:06.2638,-82

The time column may also carry a date ("2024-05-01 20:12:06.2613").
Files without a date are placed on ``base_date`` (default: today).

More columns are more EEG channels. If the first column does not look like
timestamps (time strings or regularly increasing numbers, see ``_looks_regular``),
every column is a channel: the sampling rate must then be given and the recording
starts at midnight of ``base_date``. The same applies to single-column files.
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
# Timestamp column detection: most steps must be whole multiples of the typical step
# (missing samples), give or take this fraction of a step (rounding jitter).
_TIMESTAMP_JITTER = 0.35
_TIMESTAMP_MIN_REGULAR = 0.9
_TIMESTAMP_MIN_INCREASING = 0.95
_DETECTION_ROWS = 5000
_RESAMPLE_JITTER = 0.1  # steps further than this fraction from the sample period are resampled
_EPOCH_MIN = 1e8  # numeric timestamps above this are epoch seconds (after 1973)


def _parse_time_of_day(text):
    hours, minutes, seconds = text.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _parse_absolute(text):
    return datetime.datetime.fromisoformat(text.replace("T", " ")).timestamp()


def _time_kind(text):
    """'date', 'time' (time of day) or 'number' for a first-column field."""
    if ":" in text:
        return "date" if (" " in text or "T" in text) else "time"
    return "number"


def _parse_time(text, kind):
    if kind == "date":
        return _parse_absolute(text)
    if kind == "time":
        return _parse_time_of_day(text)
    return float(text)


def _parse_value(text):
    try:
        v = float(text)
    except ValueError:
        return np.nan
    return v if math.isfinite(v) else np.nan


def _is_data_row(parts):
    """True if every field parses as a number or a time."""
    for text in parts:
        try:
            _parse_time(text, _time_kind(text))
        except ValueError:
            return False
    return True


def _read_rows(lines):
    """Return (header, rows): the split fields of the header line (or None) and of all data lines."""
    header, rows = None, []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in _SPLIT_RE.split(line)]
        if not rows and not _is_data_row(parts):
            header = parts  # last non-data line before the data
            continue
        rows.append(parts)
    return header, rows


def _first_column_times(rows, kind):
    """First-column times of rows (NaN where unparseable)."""
    times = np.empty(len(rows), dtype=np.float64)
    for i, parts in enumerate(rows):
        try:
            times[i] = _parse_time(parts[0], kind)
        except (ValueError, IndexError):
            times[i] = np.nan
    return times


def _typical_step(steps):
    """Sample period from time steps, robust to rounding jitter and missing samples."""
    positive = steps[steps > 0]
    rough = float(np.median(positive))
    return float(np.mean(positive[positive <= 1.5 * rough]))


def _looks_regular(times):
    """
    True if times increase in roughly equal steps. Gaps may be whole multiples of the
    typical step (missing samples) and may jitter by rounding.
    """
    times = times[np.isfinite(times)]
    if len(times) < 3:
        return False
    steps = np.diff(times)
    positive = steps[steps > 0]
    if len(positive) < _TIMESTAMP_MIN_INCREASING * len(steps):
        return False
    ratio = steps / _typical_step(steps)
    multiple = np.round(ratio)
    regular = (multiple >= 1) & (np.abs(ratio - multiple) <= _TIMESTAMP_JITTER)
    return bool(regular.mean() >= _TIMESTAMP_MIN_REGULAR)


def _timestamp_kind(rows):
    """
    How the first column holds timestamps ('date', 'time', 'seconds' or 'ms'),
    or None if it is an EEG channel (or the file has a single column).
    """
    if not rows or len(rows[0]) < 2:
        return None
    kind = _time_kind(rows[0][0])
    if kind != "number":
        return kind  # time strings are always timestamps
    times = _first_column_times(rows[:_DETECTION_ROWS], kind)
    if not _looks_regular(times):
        return None
    step = _typical_step(np.diff(times[np.isfinite(times)]))
    return "ms" if step >= 1.0 else "seconds"


def needs_sample_rate(path):
    """True if the file has data but no timestamp column, so its sampling rate must be given."""
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
            _, rows = _read_rows(handle)
    except OSError:
        return False
    return bool(rows) and _timestamp_kind(rows) is None


def is_single_column(path):
    """True if the first data line of the file holds a single value (no timestamp)."""
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
            for line in handle:
                parts = [p for p in _SPLIT_RE.split(line.strip()) if p.strip()]
                if not parts:
                    continue
                if len(parts) >= 2:
                    return False
                try:
                    float(parts[0])
                except ValueError:
                    continue  # header line
                return True
    except OSError:
        return False
    return False


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


def read_eeg_channels(path, base_date=None, sample_rate_hz=None):
    """
    Read a raw EEG CSV file with one or more channels.

    If the first of several columns looks like timestamps (time strings, or numbers
    increasing in roughly regular steps), it is the time axis and every other column is
    one channel. Otherwise every column is a channel sampled at sample_rate_hz, which is
    then required.

    Returns (timestamps, values, names): epoch seconds (float64), EEG values with shape
    (n_channels, n_samples) resampled to config.SAMPLE_RATE_HZ if needed, and channel names.
    Raises ValueError if the file contains no usable samples.
    """
    with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
        header, rows = _read_rows(handle)
    if len(rows) < 2:
        raise ValueError("No valid EEG samples found in file.")

    kind = _timestamp_kind(rows)
    n_columns = len(rows[0])
    first_value = 1 if kind else 0
    n_channels = n_columns - first_value

    if kind:
        times = _first_column_times(rows, "number" if kind in ("seconds", "ms") else kind)
        valid = np.isfinite(times)
        rows = [r for r, ok in zip(rows, valid) if ok]
        times = times[valid]
        if kind == "ms":
            times = times / 1000.0
        # Epoch timestamps carry their own date; small numbers are relative seconds.
        has_date = kind == "date" or (kind in ("seconds", "ms") and len(times) and times[0] > _EPOCH_MIN)
    else:
        if sample_rate_hz is None:
            raise ValueError("The file has no timestamp column; its sampling rate is required.")
        if sample_rate_hz <= 0:
            raise ValueError("Sampling rate must be positive.")
        times = np.arange(len(rows)) / float(sample_rate_hz)
        has_date = False

    if len(times) < 2:
        raise ValueError("No valid EEG samples found in file.")

    values = np.full((n_channels, len(rows)), np.nan, dtype=np.float64)
    for i, parts in enumerate(rows):
        for ch, text in enumerate(parts[first_value:first_value + n_channels]):
            values[ch, i] = _parse_value(text)

    if header is not None and len(header) == n_columns:
        names = [name or f"Ch {i + 1}" for i, name in enumerate(header[first_value:])]
    else:
        names = [f"Ch {i + 1}" for i in range(n_channels)]

    timestamps = np.asarray(times, dtype=np.float64)
    if not has_date:
        day = base_date or datetime.date.today()
        midnight = datetime.datetime.combine(day, datetime.time()).timestamp()
        if kind == "time":
            timestamps = _unwrap_midnight(timestamps)
        timestamps = midnight + timestamps

    order = np.argsort(timestamps, kind="stable")
    timestamps, values = timestamps[order], values[:, order]
    keep = np.concatenate(([True], np.diff(timestamps) > 0))
    timestamps, values = timestamps[keep], values[:, keep]
    if len(timestamps) < 2:
        raise ValueError("No valid EEG samples found in file.")

    steps = np.diff(timestamps)
    period = _typical_step(steps)
    other_rate = abs(1.0 / period - config.SAMPLE_RATE_HZ) > 0.01 * config.SAMPLE_RATE_HZ
    # Rounded timestamps or missing samples: put the samples on a uniform grid.
    irregular = np.any(np.abs(steps - period) > _RESAMPLE_JITTER * period)
    if other_rate or irregular:
        resampled = [_resample(timestamps, channel, config.SAMPLE_RATE_HZ) for channel in values]
        timestamps = resampled[0][0]
        values = np.vstack([channel for _, channel in resampled])

    return timestamps, values, names


def read_eeg_csv(path, base_date=None, sample_rate_hz=None):
    """
    Read the first EEG channel of a raw EEG CSV file (see read_eeg_channels).

    sample_rate_hz is required for files without a timestamp column and ignored otherwise.
    Returns (timestamps, values): epoch seconds (float64) and EEG values (float64).
    """
    timestamps, values, _ = read_eeg_channels(path, base_date=base_date, sample_rate_hz=sample_rate_hz)
    return timestamps, values[0]


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
