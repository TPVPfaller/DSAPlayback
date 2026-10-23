import time
import math
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import QObject, Signal, Slot
from ..core.buffers import EEGBuffer
from ..io.output import Output
from ..io.input import EEGStream
from .. import config


class ProcessingWorker(QObject):
    new_dsa_column = Signal(float, object, int)  # timestamp, power spectral density array (psd), steps
    new_samples = Signal(object)  # list of eeg values

    def __init__(self, user_config):
        super().__init__()
        self.running = True
        self.user_config = user_config
        self._new_config = None

        self.stream = EEGStream()
        self._last_connection_state = self.stream.receiving
        self._next_dsa_slot = None
        self._expected_dsa_ts = None
        self.eeg_buffer = EEGBuffer(
            self.user_config.window_sec,
            self.user_config.window_overlap
        )
        self._io_executor = ThreadPoolExecutor(max_workers=1)

    @Slot(object)
    def apply_config(self, user_config):
        self._new_config = user_config

    def _apply_pending_config(self):
        if self._new_config is None:
            return False

        processing_config_changed = self._processing_config_changed(self.user_config, self._new_config)
        self.user_config = self._new_config
        self._new_config = None

        if processing_config_changed:
            self.eeg_buffer.apply_config(self.user_config.window_sec, self.user_config.window_overlap)
            self._next_dsa_slot = None
            self._expected_dsa_ts = None
        return True

    @Slot()
    def run(self):
        next_time = time.time()
        try:
            while self.running:

                self._apply_pending_config()
                if not self.stream.receiving:
                    self.stream.connect()
                    # Emit on change
                    if self.stream.receiving != self._last_connection_state:
                        self._last_connection_state = self.stream.receiving
                    time.sleep(0.5)
                    continue

                samples = self.stream.read_lsl_samples()

                method = 'multitaper' if self.user_config.use_multitaper else 'welch'
                dsa_columns = self.eeg_buffer.get_dsa_columns(samples, method=method)

                raw_samples = [sample[1] for sample in samples]
                self.new_samples.emit(raw_samples)

                for ts, psd in dsa_columns:
                    if psd is None:
                        print(f"DSA gap cause: worker skipped None PSD column at {ts}")
                        continue

                    ts, steps = self._discretize_dsa_column(ts)

                    duration = steps * config.TIME_RESOLUTION

                    self.new_dsa_column.emit(ts, psd, steps)

                    self._io_executor.submit(Output.save_psd_to_csv, ts, duration, psd)

                next_time += config.TIME_RESOLUTION
                sleep = max(0.0, next_time - time.time())
                time.sleep(sleep)
        finally:
            self._io_executor.shutdown(wait=True)


    def stop(self):
        self.running = False

    def _discretize_dsa_column(self, ts):
        hop_sec = self.eeg_buffer.hop_len / config.SAMPLE_RATE_HZ
        if self._next_dsa_slot is None:
            current_slot = self._get_dsa_slot(ts)
        elif self._is_continuous_dsa_ts(ts):
            current_slot = self._next_dsa_slot
        else:
            expected_ts = self._expected_dsa_ts
            expected_slot = self._next_dsa_slot
            current_slot = max(self._next_dsa_slot, self._get_dsa_slot(ts))
            skipped_slots = current_slot - expected_slot
            print(
                "DSA gap cause: DSA column timestamp discontinuity "
                f"(timestamp={ts:.6f}, expected={expected_ts:.6f}, "
                f"current_slot={current_slot}, expected_slot={expected_slot}, "
                f"skipped_slots={skipped_slots})"
            )

        next_slot = self._get_dsa_slot(ts + hop_sec)
        next_slot = max(current_slot + 1, next_slot)

        self._next_dsa_slot = next_slot
        self._expected_dsa_ts = ts + hop_sec
        snapped_ts = current_slot * config.TIME_RESOLUTION
        return snapped_ts, next_slot - current_slot

    def _is_continuous_dsa_ts(self, ts):
        if self._expected_dsa_ts is None:
            return False
        return abs(ts - self._expected_dsa_ts) <= config.DSA_TIME_DIFF_TOLERANCE

    def _get_dsa_slot(self, ts):
        offset = ts / config.TIME_RESOLUTION
        return int(math.floor(offset + 0.5))

    @staticmethod
    def _processing_config_changed(old_config, new_config):
        return (
            old_config.window_sec != new_config.window_sec
            or old_config.window_overlap != new_config.window_overlap
        )
