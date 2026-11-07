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


def test_player_apply_config_recomputes_columns_only_when_dsa_settings_change(player, recording):
    _start(player, recording)
    columns_before = player.dsa_columns

    assert player.apply_config(player.user_config.update(psd_db_min=-30)) is False
    assert player.dsa_columns is columns_before

    assert player.apply_config(player.user_config.update(window_sec=4)) is True
    assert player.user_config.window_sec == 4
    expected = compute_dsa_columns(*recording, player.user_config)
    assert [c[0] for c in player.dsa_columns] == [e[0] for e in expected]


def test_player_precomputes_all_columns_on_start(player, recording):
    _start(player, recording)

    expected = compute_dsa_columns(*recording, UserConfig())
    assert len(player.dsa_columns) == len(expected) > 0
    assert player.column_starts[0] == 0
    assert player.samples == []  # nothing is emitted before playback advances


def test_player_pause_holds_position_and_resume_continues(player, clock, recording):
    _start(player, recording)
    clock.now += 1.0
    player._tick()
    player.pause()
    assert player.is_playing is False

    clock.now += 2.0  # time passing while paused must not be played back
    player.resume()
    player._timer.stop()
    clock.now += 0.5
    player._tick()

    assert len(player.samples) == int(1.5 * config.SAMPLE_RATE_HZ)
    assert player.samples == pytest.approx(recording[1][:len(player.samples)].tolist())


def test_player_seek_returns_dsa_up_to_position_and_continues_from_there(player, clock, recording):
    _start(player, recording)
    positions = []
    player.position_changed.connect(positions.append)

    columns = player.seek(4.0)

    expected = compute_dsa_columns(recording[0][:4 * config.SAMPLE_RATE_HZ],
                                   recording[1][:4 * config.SAMPLE_RATE_HZ], UserConfig())
    assert len(columns) == len(expected) > 0
    assert [c[0] for c in columns] == [e[0] for e in expected]
    assert player.position == pytest.approx(4.0)
    assert positions == [pytest.approx(4.0)]

    clock.now += 0.5
    player._tick()
    assert player.samples == pytest.approx(recording[1][4 * config.SAMPLE_RATE_HZ:4 * config.SAMPLE_RATE_HZ + len(player.samples)].tolist())


def test_player_seek_backward_and_out_of_range_is_clamped(player, clock, recording):
    _start(player, recording)
    clock.now += 3.0
    player._tick()

    assert player.seek(-5.0) == []
    assert player.position == 0.0
    player.seek(100.0)
    assert player.position == pytest.approx(player.duration)


def test_player_resume_at_end_restarts_from_beginning(player, clock, recording):
    _start(player, recording)
    clock.now += 10.0
    player._tick()
    assert player.finish_count[0] == 1

    player.resume()
    player._timer.stop()

    assert player.is_playing is True
    assert player.position == 0.0


def test_player_speed_scales_samples_per_wall_second(player, clock, recording):
    _start(player, recording)
    clock.now += 0.5
    player._tick()
    player.set_speed(4.0)

    clock.now += 0.5
    player._tick()

    assert len(player.samples) == int(0.5 * config.SAMPLE_RATE_HZ) + int(2.0 * config.SAMPLE_RATE_HZ)
    assert player.position == pytest.approx(2.5)


def test_overview_image_uses_latest_column_without_averaging():
    from plotdsa.ui.dsa_overview import overview_image

    psds = [np.full(3, 10.0), np.full(3, 100.0)]  # 10 dB, 20 dB
    image = overview_image([1.0, 2.0], psds, duration=4.0, bins=4)

    assert np.isnan(image[0]).all()  # before the first column
    assert image[1:, 0].tolist() == pytest.approx([10.0, 20.0, 20.0])
