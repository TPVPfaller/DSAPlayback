import time

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from ..core.buffers import EEGBuffer
from ..io.eeg_file import dsa_columns_from_samples
from .. import config


class EEGFilePlayer(QObject):
    """Replays a loaded EEG recording in real time on the GUI thread."""

    new_dsa_column = Signal(float, object, int)  # timestamp, psd, steps
    new_samples = Signal(object)  # list of eeg values
    finished = Signal()

    TICK_MS = 50

    def __init__(self, user_config, parent=None):
        super().__init__(parent)
        self.user_config = user_config
        self.is_playing = False
        self._timestamps = np.empty(0)
        self._values = np.empty(0)
        self._pos = 0
        self._anchor_pos = 0
        self._anchor_wall = 0.0
        self._eeg_buffer = None

        self._timer = QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)

    def start(self, timestamps, values):
        self.stop()
        self._timestamps = np.asarray(timestamps, dtype=np.float64)
        self._values = np.asarray(values, dtype=np.float64)
        self._pos = 0
        self._eeg_buffer = EEGBuffer(self.user_config.window_sec, self.user_config.window_overlap)
        self._reset_anchor()
        self.is_playing = True
        self._timer.start()

    def stop(self):
        self._timer.stop()
        self.is_playing = False

    def apply_config(self, user_config):
        old_config = self.user_config
        self.user_config = user_config
        if self._eeg_buffer is not None and (
            old_config.window_sec != user_config.window_sec
            or old_config.window_overlap != user_config.window_overlap
        ):
            self._eeg_buffer.apply_config(user_config.window_sec, user_config.window_overlap)

    def _reset_anchor(self):
        self._anchor_pos = self._pos
        self._anchor_wall = time.perf_counter()

    def _tick(self):
        total = len(self._timestamps)
        elapsed = time.perf_counter() - self._anchor_wall
        target = min(total, self._anchor_pos + int(elapsed * config.SAMPLE_RATE_HZ))
        if target > self._pos:
            start, self._pos = self._pos, target
            self._emit_chunk(start, target)

        if self._pos >= total:
            self.stop()
            self.finished.emit()

    def _emit_chunk(self, start, end):
        ts = self._timestamps[start:end]
        vals = self._values[start:end]

        self.new_samples.emit(vals.tolist())

        method = "multitaper" if self.user_config.use_multitaper else "welch"
        samples = list(zip(ts.tolist(), vals.tolist()))
        for col_ts, psd, steps in dsa_columns_from_samples(self._eeg_buffer, samples, method):
            self.new_dsa_column.emit(col_ts, psd, steps)
