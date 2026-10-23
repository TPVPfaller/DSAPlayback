import numpy as np
import datetime
from buffers import EEGBuffer
import config


def make_timestamp_series(start, n, sr):
    dt = datetime.timedelta(seconds=1 / sr)
    return [start + i * dt for i in range(n)]


def generate_samples(n, start_time):
    samples = []
    delta = datetime.timedelta(milliseconds=1000 / config.SAMPLE_RATE_HZ)

    ts = start_time
    for i in range(n):
        samples.append((ts, 10.0))
        ts += delta

    return samples


def test_timestamp_fault_resets_buffer():
    buffer = EEGBuffer(4, 0.0)

    start = datetime.datetime.now()
    samples = generate_samples(100, start)

    # inject fault
    samples.append((start + datetime.timedelta(seconds=10), 10.0))

    dsa_cols = buffer.get_dsa_columns(samples)

    assert len(buffer.eeg_values) == 0


def test_eegbuffer_sliding_window_count():
    sr = config.SAMPLE_RATE_HZ
    window = 2.0
    overlap = 0.5

    buf = EEGBuffer(window, overlap)

    n = int(5 * sr)
    ts = make_timestamp_series(datetime.datetime.now(), n, sr)
    sig = np.random.randn(n)

    samples = list(zip(ts, sig))

    out = buf.get_dsa_columns(samples)

    expected = int((5 - window) / (window * (1 - overlap))) + 1

    assert abs(len(out) - expected) <= 1


def test_eegbuffer_resets_on_gap():
    sr = config.SAMPLE_RATE_HZ
    buf = EEGBuffer(2.0, 0.0)

    t0 = datetime.datetime.now()

    good = [(t0 + datetime.timedelta(seconds=i / sr), 1.0) for i in range(200)]

    gap = [(t0 + datetime.timedelta(seconds=10), 1.0)]

    out1 = buf.get_dsa_columns(good)
    out2 = buf.get_dsa_columns(gap)

    assert len(out2) == 0, "Gap must reset EEGBuffer"


def test_eegbuffer_resets_on_timestamp_gap():
    window_sec = 1.0
    buf = EEGBuffer(window_sec, 0.0)

    base = datetime.datetime.now()

    good_sample = (base, 10.0)
    bad_sample = (base + datetime.timedelta(seconds=1), 10.0)  # big gap

    out = buf.get_dsa_columns([good_sample, bad_sample])

    # Gap should cause reset → no DSA output
    assert len(out) == 0


def test_eegbuffer_produces_dsa_column():
    window_sec = 2.0
    sample_rate = config.SAMPLE_RATE_HZ
    buf = EEGBuffer(window_sec, 0.0)

    base = datetime.datetime.now()

    data = []
    for i in range(int(window_sec * sample_rate)):
        ts = base + datetime.timedelta(milliseconds=i * (1000 / sample_rate))
        data.append((ts, 1.0))

    out = buf.get_dsa_columns(data)

    assert len(out) >= 1


def test_eegbuffer_uses_exact_window_start_timestamp():
    window_sec = 2.0
    sample_rate = config.SAMPLE_RATE_HZ
    buf = EEGBuffer(window_sec, 0.0)
    base = datetime.datetime.now()

    data = []
    for i in range(int(window_sec * sample_rate)):
        ts = base + datetime.timedelta(milliseconds=i * (1000 / sample_rate))
        data.append((ts, 1.0))

    out = buf.get_dsa_columns(data)

    assert len(out) == 1
    assert out[0][0] == base.timestamp()


def test_apply_config_recomputes_window_and_hop_lengths():
    buf = EEGBuffer(window_sec=4.0, overlap=0.0)

    buf.apply_config(window_sec=2.0, overlap=0.5)

    assert buf.window_sec == 2.0
    assert buf.window_len == int(2.0 * config.SAMPLE_RATE_HZ)
    assert buf.hop_len == int(buf.window_len * 0.5)


def test_eegbuffer_jitter_within_dsa_tolerance_does_not_reset():
    buf = EEGBuffer(2.0, 0.0)
    base = datetime.datetime.now()
    nominal_step = datetime.timedelta(milliseconds=1000 / config.SAMPLE_RATE_HZ)
    jitter = datetime.timedelta(seconds=config.EEG_TIME_DIFF_TOLERANCE * 1.5)

    samples = []
    ts = base
    for idx in range(int(2.0 * config.SAMPLE_RATE_HZ)):
        if idx == 50:
            ts += jitter
        samples.append((ts, 1.0))
        ts += nominal_step

    out = buf.get_dsa_columns(samples)

    assert len(out) == 1
    assert not any(np.isnan(value) for value in buf.eeg_values)


def test_eegbuffer_high_overlap_uses_minimum_hop_of_one_sample():
    buf = EEGBuffer(2.0, 0.99)
    base = datetime.datetime.now()
    samples = generate_samples(int(2.0 * config.SAMPLE_RATE_HZ) + 3, base)

    out = buf.get_dsa_columns(samples)

    assert buf.hop_len == 8
    assert len(out) >= 1


def test_eegbuffer_drops_initial_nan_sample_without_crashing():
    buf = EEGBuffer(2.0, 0.0)
    base = datetime.datetime.now()

    out = buf.get_dsa_columns([(base, np.nan)])

    assert out == []
    assert buf.last_ts is None
