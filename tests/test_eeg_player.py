import types

import numpy as np
import pytest

from plotdsa import config
from plotdsa.app import player as player_module
from plotdsa.app.player import EEGFilePlayer
from plotdsa.config import UserConfig
from plotdsa.io.eeg_file import compute_dsa_columns


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(player_module, "time", types.SimpleNamespace(perf_counter=fake.perf_counter))
    return fake


@pytest.fixture
def recording():
    rate = config.SAMPLE_RATE_HZ
    n = 6 * rate
    timestamps = 1_000_000.0 + np.arange(n) / rate
    values = 50 * np.sin(2 * np.pi * 10 * np.arange(n) / rate)
    return timestamps, values


@pytest.fixture
def player(qapp, clock):
    p = EEGFilePlayer(UserConfig())
    p.samples, p.columns, p.finish_count = [], [], [0]
    p.new_samples.connect(p.samples.extend)
    p.new_dsa_column.connect(lambda ts, psd, steps: p.columns.append((ts, psd, steps)))
    p.finished.connect(lambda: p.finish_count.__setitem__(0, p.finish_count[0] + 1))
    yield p
    p.stop()


def _start(player, recording):
    player.start(*recording)
    player._timer.stop()  # ticks are driven manually by the tests


def test_player_emits_samples_paced_by_elapsed_time(player, clock, recording):
    _start(player, recording)

    clock.now = 0.5
    player._tick()
    assert len(player.samples) == int(0.5 * config.SAMPLE_RATE_HZ)

    clock.now = 1.0
    player._tick()
    assert len(player.samples) == config.SAMPLE_RATE_HZ
    assert player.samples == pytest.approx(recording[1][:config.SAMPLE_RATE_HZ].tolist())
    assert player.is_playing is True
    assert player.finish_count[0] == 0


def test_player_produces_same_dsa_columns_as_loading_the_file(player, clock, recording):
    _start(player, recording)

    while player.is_playing:
        clock.now += 0.037  # chunk boundaries that do not line up with the DSA hop
        player._tick()

    expected = compute_dsa_columns(*recording, UserConfig())
    assert len(player.columns) == len(expected) > 0
    for (ts, psd, steps), (exp_ts, exp_psd, exp_steps) in zip(player.columns, expected):
        assert ts == pytest.approx(exp_ts)
        assert steps == exp_steps
        np.testing.assert_allclose(psd, exp_psd)


def test_player_finishes_once_after_last_sample(player, clock, recording):
    _start(player, recording)

    clock.now = 100.0
    player._tick()

    assert len(player.samples) == len(recording[0])
    assert player.is_playing is False
    assert player.finish_count[0] == 1


def test_player_stop_halts_playback(player, recording):
    player.start(*recording)
    assert player._timer.isActive() is True

    player.stop()

    assert player.is_playing is False
    assert player._timer.isActive() is False


def test_player_restart_begins_new_recording_from_first_sample(player, clock, recording):
    _start(player, recording)
    clock.now = 2.0
    player._tick()

    played_before_restart = len(player.samples)
    clock.now = 10.0
    _start(player, (recording[0][:200], recording[1][:200] + 1.0))
    clock.now = 10.25
    player._tick()

    restarted = player.samples[played_before_restart:]
    assert restarted == pytest.approx((recording[1][:100] + 1.0).tolist())


def test_player_apply_config_reconfigures_buffer_only_when_window_changes(player, recording, monkeypatch):
    _start(player, recording)
    calls = []
    monkeypatch.setattr(player._eeg_buffer, "apply_config", lambda sec, overlap: calls.append((sec, overlap)))

    player.apply_config(player.user_config.update(psd_db_min=-30))
    assert calls == []

    player.apply_config(player.user_config.update(window_sec=4))
    assert calls == [(4, player.user_config.window_overlap)]
    assert player.user_config.window_sec == 4
