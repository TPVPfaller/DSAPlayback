import datetime

import numpy as np
import pytest

from plotdsa import config
from plotdsa.config import UserConfig
from plotdsa.io.eeg_file import (
    compute_dsa_columns,
    is_single_column,
    needs_sample_rate,
    read_eeg_channels,
    read_eeg_csv,
    snap_dsa_column,
)

DAY = datetime.date(2026, 10, 19)


def _write_csv(path, start_sec, rate, n, header=None):
    lines = [header] if header else []
    for i in range(n):
        sec = (start_sec + i / rate) % 86400
        value = 50 * np.sin(2 * np.pi * 10 * i / rate)
        lines.append(f"{int(sec // 3600):02d}:{int(sec % 3600 // 60):02d}:{sec % 60:07.4f},{value:.0f}")
    path.write_text("\n".join(lines) + "\n")


def test_read_eeg_csv_parses_time_of_day_format(tmp_path):
    path = tmp_path / "eeg.csv"
    path.write_text("20:12:06.2613,-82\n20:12:06.2638,-82\n20:12:06.2663,-80\n")

    timestamps, values = read_eeg_csv(path, base_date=DAY)

    start = datetime.datetime(2026, 10, 19, 20, 12, 6, 261300).timestamp()
    assert timestamps[0] == pytest.approx(start)
    assert np.diff(timestamps).tolist() == pytest.approx([0.0025, 0.0025], abs=1e-6)
    assert values.tolist() == [-82, -82, -80]


def test_read_eeg_csv_skips_header_and_handles_midnight(tmp_path):
    path = tmp_path / "eeg.csv"
    _write_csv(path, 23 * 3600 + 59 * 60 + 59, config.SAMPLE_RATE_HZ, 800, header="time,value")

    timestamps, _ = read_eeg_csv(path, base_date=DAY)

    assert len(timestamps) == 800
    assert np.all(np.diff(timestamps) > 0)
    assert datetime.datetime.fromtimestamp(timestamps[-1]).date() == DAY + datetime.timedelta(days=1)


def test_read_eeg_csv_resamples_other_rates(tmp_path):
    path = tmp_path / "eeg.csv"
    _write_csv(path, 3600, 250, 2500)  # 10 s at 250 Hz

    timestamps, values = read_eeg_csv(path, base_date=DAY)

    assert np.diff(timestamps).tolist() == pytest.approx([1 / config.SAMPLE_RATE_HZ] * (len(timestamps) - 1), abs=1e-6)
    assert len(values) == pytest.approx(10 * config.SAMPLE_RATE_HZ, abs=2)


def test_read_eeg_csv_rejects_files_without_samples(tmp_path):
    path = tmp_path / "eeg.csv"
    path.write_text("not,an\neeg,file\n")

    with pytest.raises(ValueError):
        read_eeg_csv(path)


def test_compute_dsa_columns_are_continuous(tmp_path):
    path = tmp_path / "eeg.csv"
    _write_csv(path, 3600, config.SAMPLE_RATE_HZ, 10 * config.SAMPLE_RATE_HZ)
    timestamps, values = read_eeg_csv(path, base_date=DAY)

    columns = compute_dsa_columns(timestamps, values, UserConfig())

    assert len(columns) > 1
    for (ts, _, steps), (next_ts, _, _) in zip(columns, columns[1:]):
        assert ts + steps * config.TIME_RESOLUTION == pytest.approx(next_ts)


def test_snap_dsa_column_returns_at_least_one_step():
    ts, steps = snap_dsa_column(100.01, 0.0)
    assert ts == pytest.approx(100.0)
    assert steps == 1


def test_read_eeg_csv_uses_dates_from_file_instead_of_base_date(tmp_path):
    path = tmp_path / "eeg.csv"
    path.write_text(
        "2024-05-01 20:12:06.2600,-82\n"
        "2024-05-01 20:12:06.2625,-81\n"
        "2024-05-01T20:12:06.2650,-80\n"
    )

    timestamps, values = read_eeg_csv(path, base_date=DAY)

    start = datetime.datetime(2024, 5, 1, 20, 12, 6, 260000).timestamp()
    assert timestamps[0] == pytest.approx(start)
    assert np.diff(timestamps).tolist() == pytest.approx([0.0025, 0.0025], abs=1e-6)
    assert values.tolist() == [-82, -81, -80]


@pytest.mark.parametrize("separator", [";", "\t"])
def test_read_eeg_csv_accepts_semicolon_and_tab_separators(tmp_path, separator):
    path = tmp_path / "eeg.csv"
    path.write_text(f"20:12:06.2600{separator}1\n20:12:06.2625{separator}2\n20:12:06.2650{separator}3\n")

    _, values = read_eeg_csv(path, base_date=DAY)

    assert values.tolist() == [1, 2, 3]


def test_read_eeg_csv_turns_non_finite_values_into_nan(tmp_path):
    path = tmp_path / "eeg.csv"
    path.write_text("20:12:06.2600,1\n20:12:06.2625,inf\n20:12:06.2650,nan\n20:12:06.2675,4\n")

    _, values = read_eeg_csv(path, base_date=DAY)

    assert values[0] == 1
    assert np.isnan(values[1]) and np.isnan(values[2])
    assert values[3] == 4


def test_read_eeg_csv_sorts_samples_and_drops_duplicate_timestamps(tmp_path):
    path = tmp_path / "eeg.csv"
    path.write_text(
        "20:12:06.2650,3\n"
        "20:12:06.2600,1\n"
        "20:12:06.2625,2\n"
        "20:12:06.2625,99\n"
    )

    timestamps, values = read_eeg_csv(path, base_date=DAY)

    assert np.all(np.diff(timestamps) > 0)
    assert values.tolist() == [1, 2, 3]


def test_read_eeg_csv_keeps_recording_gaps_as_nan_when_resampling(tmp_path):
    rate = 250
    first = [3600 + i / rate for i in range(2 * rate)]
    second = [first[-1] + 1.0 + i / rate for i in range(2 * rate)]  # 1 s gap
    lines = [f"01:00:{sec - 3600:07.4f},10" for sec in first + second]
    path = tmp_path / "eeg.csv"
    path.write_text("\n".join(lines) + "\n")

    timestamps, values = read_eeg_csv(path, base_date=DAY)

    gap_start = timestamps[0] + (first[-1] - first[0])
    gap_end = gap_start + 1.0
    inside_gap = (timestamps > gap_start + 0.01) & (timestamps < gap_end - 0.01)
    outside_gap = (timestamps < gap_start - 0.01) | (timestamps > gap_end + 0.01)
    assert inside_gap.any()
    assert np.isnan(values[inside_gap]).all()
    assert np.isfinite(values[outside_gap]).all()


def test_compute_dsa_columns_returns_nothing_for_recording_shorter_than_window(tmp_path):
    path = tmp_path / "eeg.csv"
    user_config = UserConfig()
    _write_csv(path, 3600, config.SAMPLE_RATE_HZ, user_config.window_sec * config.SAMPLE_RATE_HZ - 1)
    timestamps, values = read_eeg_csv(path, base_date=DAY)

    assert compute_dsa_columns(timestamps, values, user_config) == []


def test_is_single_column_detects_value_only_files(tmp_path):
    single = tmp_path / "single.csv"
    single.write_text("value\n-82\n-80\n")
    double = tmp_path / "double.csv"
    double.write_text("20:12:06.2613,-82\n")

    assert is_single_column(single) is True
    assert is_single_column(double) is False
    assert is_single_column(tmp_path / "missing.csv") is False


def test_read_eeg_csv_single_column_uses_given_sample_rate(tmp_path):
    path = tmp_path / "eeg.csv"
    path.write_text("value\n" + "\n".join(str(i) for i in range(500)) + "\n")  # 2 s at 250 Hz

    timestamps, values = read_eeg_csv(path, base_date=DAY, sample_rate_hz=250)

    midnight = datetime.datetime.combine(DAY, datetime.time()).timestamp()
    assert timestamps[0] == pytest.approx(midnight)
    assert np.diff(timestamps).tolist() == pytest.approx([1 / config.SAMPLE_RATE_HZ] * (len(timestamps) - 1), abs=1e-6)
    assert timestamps[-1] - timestamps[0] == pytest.approx(499 / 250, abs=1 / config.SAMPLE_RATE_HZ)


def test_read_eeg_csv_single_column_rejects_non_positive_rate(tmp_path):
    path = tmp_path / "eeg.csv"
    path.write_text("1\n2\n3\n")

    with pytest.raises(ValueError):
        read_eeg_csv(path, sample_rate_hz=0)


def _channel_values(n, rate, n_channels):
    t = np.arange(n) / rate
    return [np.round(20 * (ch + 1) * np.sin(2 * np.pi * (5 + ch) * t)) for ch in range(n_channels)]


def test_read_eeg_channels_uses_jittery_numeric_first_column_with_missing_samples_as_time(tmp_path):
    rate = config.SAMPLE_RATE_HZ
    n = 2 * rate
    channels = _channel_values(n, rate, 3)
    keep = np.ones(n, dtype=bool)
    keep[[50, 51, 52, 300]] = False  # missing samples
    lines = ["time,Fp1,Fp2,Cz"]
    for i in np.flatnonzero(keep):
        # Rounded to milliseconds: steps of 2 or 3 ms instead of exactly 2.5 ms.
        lines.append(f"{i / rate:.3f}," + ",".join(f"{ch[i]:.0f}" for ch in channels))
    path = tmp_path / "eeg.csv"
    path.write_text("\n".join(lines) + "\n")

    assert needs_sample_rate(path) is False
    timestamps, values, names = read_eeg_channels(path, base_date=DAY)

    midnight = datetime.datetime.combine(DAY, datetime.time()).timestamp()
    assert names == ["Fp1", "Fp2", "Cz"]
    assert values.shape == (3, len(timestamps))
    assert timestamps[0] == pytest.approx(midnight)
    assert timestamps[-1] - timestamps[0] == pytest.approx((n - 1) / rate, abs=2 / rate)
    # Rounded timestamps and the missing samples are put back on a uniform grid.
    assert np.diff(timestamps) == pytest.approx(1 / rate, abs=1e-6)
    assert len(timestamps) == n
    for ch in range(3):
        assert values[ch, :40] == pytest.approx(channels[ch][:40], abs=3)
        assert np.isfinite(values[ch, 300])  # one missing sample is interpolated


def test_read_eeg_channels_reads_millisecond_epoch_timestamps(tmp_path):
    start_ms = 1_700_000_000_000
    lines = [f"{start_ms + i * 4},{i},{-i}" for i in range(500)]  # 250 Hz
    path = tmp_path / "eeg.csv"
    path.write_text("\n".join(lines) + "\n")

    timestamps, values, names = read_eeg_channels(path, base_date=DAY)

    assert names == ["Ch 1", "Ch 2"]
    assert timestamps[0] == pytest.approx(start_ms / 1000.0)
    assert timestamps[-1] - timestamps[0] == pytest.approx(499 * 0.004, abs=1 / config.SAMPLE_RATE_HZ)
    assert values.shape[0] == 2


def test_read_eeg_channels_treats_irregular_first_column_as_a_channel(tmp_path):
    rate = config.SAMPLE_RATE_HZ
    channels = _channel_values(rate, rate, 2)
    path = tmp_path / "eeg.csv"
    path.write_text("".join(f"{a:.0f},{b:.0f}\n" for a, b in zip(*channels)))

    assert needs_sample_rate(path) is True
    with pytest.raises(ValueError):
        read_eeg_channels(path, base_date=DAY)

    timestamps, values, names = read_eeg_channels(path, base_date=DAY, sample_rate_hz=rate)

    assert names == ["Ch 1", "Ch 2"]
    assert values[0].tolist() == channels[0].tolist()
    assert values[1].tolist() == channels[1].tolist()
    assert np.diff(timestamps) == pytest.approx(1 / rate, abs=1e-6)


def test_read_eeg_channels_reads_time_of_day_files_with_several_channels(tmp_path):
    path = tmp_path / "eeg.csv"
    path.write_text("20:12:06.2600,1,10\n20:12:06.2625,2,20\n20:12:06.2650,3,30\n")

    assert needs_sample_rate(path) is False
    _, values, _ = read_eeg_channels(path, base_date=DAY)

    assert values.tolist() == [[1, 2, 3], [10, 20, 30]]
