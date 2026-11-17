import time

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from ..core.buffers import EEGBuffer
from ..io.eeg_file import snap_dsa_column
from .. import config


def _method(user_config):
    return "multitaper" if user_config.use_multitaper else "welch"


class EEGFilePlayer(QObject):
    """
    Replays a loaded EEG recording in real time on the GUI thread.

    All DSA columns are computed up front; during playback each column is
    emitted when the sample completing its window is played, exactly as if
    it had been computed while streaming.
    """

    new_dsa_column = Signal(float, object, int)  # timestamp, psd, steps
    new_samples = Signal(object)  # list of eeg values
    position_changed = Signal(float)  # seconds since recording start
    finished = Signal()
    # Channels after the first of a multi-channel recording: (channel index, ...)
    new_extra_dsa_column = Signal(int, float, object, int)  # channel, timestamp, psd, steps
    new_extra_samples = Signal(int, object)  # channel, list of eeg values

    TICK_MS = 50

    def __init__(self, user_config, parent=None):
        super().__init__(parent)
        self.user_config = user_config
        self.is_playing = False
        self.speed = 1.0
        self._timestamps = np.empty(0)
        self._values = np.empty(0)
        self._pos = 0
        self._anchor_pos = 0
        self._anchor_wall = 0.0
        self.dsa_columns = []  # [(ts, psd, steps)] of the whole recording
        self.column_starts = np.empty(0, dtype=np.int64)  # first sample index of each column
        self._column_ready = np.empty(0, dtype=np.int64)  # samples played before a column is shown
        self._next_column = 0
        # Per extra channel (index 1, 2, ...): values, columns, ready sample counts, next column.
        self._extra = {}

        self._timer = QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)

    def start(self, timestamps, values):
        """values: one channel (n_samples,) or several channels (n_channels, n_samples)."""
        self.stop()
        self._timestamps = np.asarray(timestamps, dtype=np.float64)
        channels = np.atleast_2d(np.asarray(values, dtype=np.float64))
        self._values = channels[0]
        self._extra = {ch: {"values": channels[ch]} for ch in range(1, len(channels))}
        self._pos = 0
        self._compute_columns()
        self._next_column = 0
        self._reset_anchor()
        self.is_playing = True
        self._timer.start()

    def stop(self):
        self._timer.stop()
        self.is_playing = False

    pause = stop

    def resume(self):
        """Continue playback from the current position (restarts if at the end)."""
        if len(self._timestamps) == 0 or self.is_playing:
            return
        if self._pos >= len(self._timestamps):
            self.seek(0.0)
        self._reset_anchor()
        self.is_playing = True
        self._timer.start()

    def set_speed(self, speed):
        """Play `speed` recording seconds per wall-clock second from the current position."""
        self.speed = float(speed)
        self._reset_anchor()

    @property
    def duration(self):
        return len(self._timestamps) / config.SAMPLE_RATE_HZ

    @property
    def position(self):
        return self._pos / config.SAMPLE_RATE_HZ

    def seek(self, seconds):
        """
        Jump to `seconds` and return the DSA columns [(ts, psd, steps)] of the
        recording up to there, so the caller can rebuild the DSA view.
        """
        self._pos = int(min(max(seconds, 0.0), self.duration) * config.SAMPLE_RATE_HZ)
        self._next_column = int(np.searchsorted(self._column_ready, self._pos, side="right"))
        self._sync_extra_next_columns()
        self._reset_anchor()
        self.position_changed.emit(self.position)
        return self.dsa_columns[:self._next_column]

    @property
    def extra_channels(self):
        return sorted(self._extra)

    def extra_columns_played(self, channel):
        """DSA columns [(ts, psd, steps)] of an extra channel up to the current position."""
        extra = self._extra[channel]
        return extra["columns"][:extra["next"]]

    def _sync_extra_next_columns(self):
        for extra in self._extra.values():
            extra["next"] = int(np.searchsorted(extra["ready"], self._pos, side="right"))

    def apply_config(self, user_config):
        """Returns True if the DSA columns were recomputed (DSA settings changed)."""
        old_config = self.user_config
        self.user_config = user_config
        if len(self._timestamps) and (
            old_config.window_sec != user_config.window_sec
            or old_config.window_overlap != user_config.window_overlap
            or old_config.use_multitaper != user_config.use_multitaper
        ):
            self._compute_columns()
            self._next_column = int(np.searchsorted(self._column_ready, self._pos, side="right"))
            self._sync_extra_next_columns()
            return True
        return False

    def _compute_columns(self):
        self.dsa_columns, self.column_starts, self._column_ready = self._channel_columns(self._values)
        for extra in self._extra.values():
            extra["columns"], _, extra["ready"] = self._channel_columns(extra["values"])
            extra["next"] = 0

    def _channel_columns(self, values):
        """Return (columns, first sample index of each column, samples played before each is shown)."""
        eeg_buffer = EEGBuffer(self.user_config.window_sec, self.user_config.window_overlap)
        hop_sec = eeg_buffer.hop_len / config.SAMPLE_RATE_HZ
        samples = list(zip(self._timestamps.tolist(), values.tolist()))
        columns, starts = [], []
        for ts, psd in eeg_buffer.get_dsa_columns(samples, method=_method(self.user_config)):
            if psd is None:
                continue
            snapped_ts, steps = snap_dsa_column(ts, hop_sec)
            columns.append((snapped_ts, psd, steps))
            starts.append(int(np.searchsorted(self._timestamps, ts)))
        starts = np.asarray(starts, dtype=np.int64)
        # A window is complete once its last sample has been played.
        return columns, starts, starts + eeg_buffer.window_len

    def _reset_anchor(self):
        self._anchor_pos = self._pos
        self._anchor_wall = time.perf_counter()

    def _tick(self):
        total = len(self._timestamps)
        elapsed = time.perf_counter() - self._anchor_wall
        target = min(total, self._anchor_pos + int(elapsed * self.speed * config.SAMPLE_RATE_HZ))
        if target > self._pos:
            start, self._pos = self._pos, target
            self._emit_chunk(start, target)
            self.position_changed.emit(self.position)

        if self._pos >= total:
            self.stop()
            self.finished.emit()

    def _emit_chunk(self, start, end):
        # Extra channels first, so their data is in place when channel 1 redraws the views.
        for channel, extra in sorted(self._extra.items()):
            self.new_extra_samples.emit(channel, extra["values"][start:end].tolist())
            while extra["next"] < len(extra["columns"]) and extra["ready"][extra["next"]] <= end:
                self.new_extra_dsa_column.emit(channel, *extra["columns"][extra["next"]])
                extra["next"] += 1

        self.new_samples.emit(self._values[start:end].tolist())

        while self._next_column < len(self.dsa_columns) and self._column_ready[self._next_column] <= end:
            self.new_dsa_column.emit(*self.dsa_columns[self._next_column])
            self._next_column += 1
