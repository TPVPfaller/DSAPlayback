import pytest
import numpy as np
from worker import ProcessingWorker
from config import UserConfig
import config


class DummyStream:
    def __init__(self):
        self.receiving = False

    def connect(self):
        self.receiving = False


def test_apply_config_stashes_new_config(monkeypatch):
    monkeypatch.setattr("worker.EEGStream", DummyStream)

    worker = ProcessingWorker(UserConfig())
    new_config = worker.user_config.update(window_sec=4)

    worker.apply_config(new_config)
    worker.stop()

    assert worker._new_config == new_config


def test_run_emits_samples_and_saves_generated_columns(monkeypatch):
    saved_calls = []
    emitted_samples = []
    emitted_columns = []

    class ConnectedStream:
        def __init__(self):
            self.receiving = True

        def connect(self):
            self.receiving = True

        def read_lsl_samples(self):
            return [(0, 1.0)]

    class FakeExecutor:
        def submit(self, fn, *args):
            saved_calls.append((fn, args))

        def shutdown(self, wait=True):
            pass

    monkeypatch.setattr("worker.EEGStream", ConnectedStream)
    monkeypatch.setattr("worker.Output.save_psd_to_csv", lambda *args: None)

    worker = ProcessingWorker(UserConfig(window_sec=2, window_overlap=0.5))
    worker._io_executor = FakeExecutor()

    def fake_get_dsa_columns(samples, method="multitaper"):
        worker.running = False
        return [(123.0, np.array([1.0, 2.0], dtype=np.float32))]

    worker.eeg_buffer.get_dsa_columns = fake_get_dsa_columns
    worker.new_samples.connect(emitted_samples.append)
    worker.new_dsa_column.connect(lambda ts, psd, steps: emitted_columns.append((ts, psd, steps)))

    worker.run()

    assert emitted_samples == [[1.0]]
    assert len(emitted_columns) == 1
    assert emitted_columns[0][0] == 123.0
    expected_steps = int(round(1.0 / config.TIME_RESOLUTION))
    assert emitted_columns[0][2] == expected_steps
    assert saved_calls[0][1][0] == 123.0
    assert saved_calls[0][1][1] == expected_steps * config.TIME_RESOLUTION


def test_discretize_dsa_column_keeps_continuous_one_second_hops_contiguous(monkeypatch):
    monkeypatch.setattr("worker.EEGStream", DummyStream)

    worker = ProcessingWorker(UserConfig(window_sec=2, window_overlap=0.5))

    emitted = [
        worker._discretize_dsa_column(1000.26 + i * 1.0)
        for i in range(4)
    ]

    # snaps to time grid
    expected = [
            (1000.25, int(round(1.0 / config.TIME_RESOLUTION))),
            (1001.25, int(round(1.0 / config.TIME_RESOLUTION))),
            (1002.25, int(round(1.0 / config.TIME_RESOLUTION))),
            (1003.25, int(round(1.0 / config.TIME_RESOLUTION))),
    ]
    for (actual_ts, actual_steps), (expected_ts, expected_steps) in zip(emitted, expected):
        assert actual_ts == pytest.approx(expected_ts)
        assert actual_steps == expected_steps


def test_apply_pending_display_only_config_keeps_worker_and_eeg_buffer_state(monkeypatch):
    monkeypatch.setattr("worker.EEGStream", DummyStream)

    worker = ProcessingWorker(UserConfig(window_sec=2, window_overlap=0.5))
    worker._next_dsa_slot = 123
    worker._expected_dsa_ts = 456.0
    worker.eeg_buffer.eeg_values = [1.0, 2.0]
    worker.eeg_buffer.timestamps = [1.0, 2.0]
    worker.eeg_buffer.last_ts = 2.0

    worker.apply_config(worker.user_config.update(psd_db_min=-10, psd_db_max=15))

    assert worker._apply_pending_config() is True
    assert worker._next_dsa_slot == 123
    assert worker._expected_dsa_ts == 456.0
    assert worker.eeg_buffer.window_len == 2 * config.SAMPLE_RATE_HZ
    assert worker.eeg_buffer.hop_len == int(worker.eeg_buffer.window_len * 0.5)
    assert worker.eeg_buffer.eeg_values == [1.0, 2.0]
    assert worker.eeg_buffer.timestamps == [1.0, 2.0]
    assert worker.eeg_buffer.last_ts == 2.0
