import numpy as np
from datetime import datetime, timedelta
from input_output import Output
import config


def test_save_psd_to_csv_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(Output, "_file_states", {})

    psd = np.arange(len(config.FREQ_BINS), dtype=np.float32)

    Output.save_psd_to_csv(1_700_000_000.0, 1.0, psd)

    files = list(tmp_path.glob("*.csv"))
    assert len(files) == 1

    content = files[0].read_text()
    assert "timestamp" in content
    assert f"f_{config.FREQ_BINS[0]:.1f}_Hz" in content


def test_save_psd_to_csv_preserves_fractional_duration(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(Output, "_file_states", {})

    psd = np.arange(len(config.FREQ_BINS), dtype=np.float32)

    Output.save_psd_to_csv(1_700_000_000.0, config.TIME_RESOLUTION, psd)

    [csv_file] = list(tmp_path.glob("*.csv"))
    duration = csv_file.read_text().splitlines()[1].split(",")[1]

    assert float(duration) == config.TIME_RESOLUTION


def test_save_psd_to_csv_skips_duplicate_timestamp(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(Output, "_file_states", {})

    first_psd = np.arange(len(config.FREQ_BINS), dtype=np.float32)
    second_psd = first_psd + 10.0

    Output.save_psd_to_csv(1_700_000_000.0, 1.0, first_psd)
    Output.save_psd_to_csv(1_700_000_000.0, 1.0, second_psd)

    [csv_file] = list(tmp_path.glob("*.csv"))
    lines = csv_file.read_text().splitlines()

    assert len(lines) == 2


def test_save_psd_to_csv_truncates_rows_after_time_rewind(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(Output, "_file_states", {})
    psd = np.arange(len(config.FREQ_BINS), dtype=np.float32)

    earlier = 1_700_000_000.0
    later = earlier + 60.0
    rewind = earlier + 30.0

    Output.save_psd_to_csv(earlier, 1.0, psd)
    Output.save_psd_to_csv(later, 1.0, psd + 10)
    Output.save_psd_to_csv(rewind, 1.0, psd + 20)

    [csv_file] = list(tmp_path.glob("*.csv"))
    lines = csv_file.read_text().splitlines()
    timestamps = [line.split(",")[0] for line in lines[1:]]

    assert len(lines) == 3
    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) == 2


def test_save_psd_to_csv_reuses_cached_state_for_monotonic_appends(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(Output, "_file_states", {})

    psd = np.arange(len(config.FREQ_BINS), dtype=np.float32)

    Output.save_psd_to_csv(1_700_000_000.0, 1.0, psd)
    monkeypatch.setattr(
        Output,
        "_scan_file_state",
        classmethod(lambda cls, filepath: (_ for _ in ()).throw(AssertionError("unexpected rescan"))),
    )

    Output.save_psd_to_csv(1_700_000_060.0, 1.0, psd + 1)

    [csv_file] = list(tmp_path.glob("*.csv"))
    lines = csv_file.read_text().splitlines()

    assert len(lines) == 3


def test_load_psd_from_time_ignores_malformed_rows_and_sorts_results(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BASE_DIR", str(tmp_path))
    start_dt = datetime(2023, 11, 14, 22, 13, 20)
    next_dt = start_dt + timedelta(seconds=10)
    header = ["timestamp", "duration"] + [f"f_{freq:.1f}_Hz" for freq in config.FREQ_BINS]
    valid_psd = ["1.0"] * len(config.FREQ_BINS)
    csv_path = tmp_path / "dsa_2023-11-14.csv"
    rows = [
        ["22:13:30.000", "1"] + valid_psd,
        ["bad-row"],
        ["2023-11-14T22:13:20", "1"] + valid_psd,
    ]

    with csv_path.open("w", newline="") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(row) + "\n")

    loaded = Output.load_psd_from_time(start_dt)

    assert [item[0] for item in loaded] == [start_dt.timestamp(), next_dt.timestamp()]
    assert loaded[0][1] == 1.0
    assert loaded[0][2].shape == config.FREQ_BINS.shape


def test_load_psd_from_time_skips_rows_with_truncated_psd_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BASE_DIR", str(tmp_path))
    start_dt = datetime(2023, 11, 14, 22, 13, 20)
    header = ["timestamp", "duration"] + [f"f_{freq:.1f}_Hz" for freq in config.FREQ_BINS]
    good_row = ["22:13:20.000", "1"] + ["1.0"] * len(config.FREQ_BINS)
    bad_row = ["22:13:21.000", "1"] + ["1.0"] * (len(config.FREQ_BINS) - 1)
    csv_path = tmp_path / "dsa_2023-11-14.csv"

    with csv_path.open("w", newline="") as handle:
        handle.write(",".join(header) + "\n")
        handle.write(",".join(good_row) + "\n")
        handle.write(",".join(bad_row) + "\n")

    loaded = Output.load_psd_from_time(start_dt)

    assert len(loaded) == 1
    assert loaded[0][0] == start_dt.timestamp()


def test_load_psd_from_time_reads_fractional_duration(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BASE_DIR", str(tmp_path))
    file_date = datetime(2023, 11, 14)
    header = ["timestamp", "duration"] + [f"f_{freq:.1f}_Hz" for freq in config.FREQ_BINS]
    row = ["22:13:20.000", f"{config.TIME_RESOLUTION:g}"] + ["2.0"] * len(config.FREQ_BINS)
    csv_path = tmp_path / "dsa_2023-11-14.csv"

    with csv_path.open("w", newline="") as handle:
        handle.write(",".join(header) + "\n")
        handle.write(",".join(row) + "\n")

    loaded = Output.load_psd_from_time(file_date)

    assert len(loaded) == 1
    assert loaded[0][1] == config.TIME_RESOLUTION


def test_load_psd_from_time_supports_legacy_header_without_duration(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BASE_DIR", str(tmp_path))
    file_date = datetime(2023, 11, 14)
    header = ["timestamp"] + [f"f_{freq:.1f}_Hz" for freq in config.FREQ_BINS]
    row = ["22:13:20.000"] + ["2.0"] * len(config.FREQ_BINS)
    csv_path = tmp_path / "dsa_2023-11-14.csv"

    with csv_path.open("w", newline="") as handle:
        handle.write(",".join(header) + "\n")
        handle.write(",".join(row) + "\n")

    loaded = Output.load_psd_from_time(file_date)

    assert len(loaded) == 1
    assert loaded[0][1] == config.TIME_RESOLUTION
    assert loaded[0][0] == datetime(2023, 11, 14, 22, 13, 20).timestamp()
