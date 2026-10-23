import numpy as np
from calculations import DSACalculator
import config

def generate_sine(freq, seconds, sr, noise=0.0):
    t = np.arange(0, seconds, 1 / sr)
    sig = np.sin(2 * np.pi * freq * t)
    if noise > 0:
        sig += np.random.normal(0, noise, size=len(sig))
    return sig

def test_psd_returns_correct_frequency_range():
    calc = DSACalculator(window_sec=4)

    # Generate synthetic sine at 10 Hz
    fs = config.SAMPLE_RATE_HZ
    t = np.arange(0, 4, 1/fs)
    signal = np.sin(2 * np.pi * 10 * t)

    psd = calc.compute_psd_column(signal)
    f = config.FREQ_BINS

    assert f is not None
    assert psd is not None
    assert len(f) == len(psd)
    assert np.all(f >= config.LOWEST_FREQ_HZ)
    assert np.all(f <= config.MAX_FREQ_HZ_BOUNDS[1])


def test_psd_detects_10hz_peak():
    calc = DSACalculator(4)

    fs = config.SAMPLE_RATE_HZ
    t = np.arange(0, 4, 1/fs)
    signal = np.sin(2 * np.pi * 10 * t)

    psd = calc.compute_psd_column(signal)
    f = config.FREQ_BINS

    peak_freq = f[np.argmax(psd)]
    assert abs(peak_freq - 10) < 0.5


def test_psd_returns_nan_on_invalid_signal():
    calc = DSACalculator(4)
    signal = np.zeros(int(4 * config.SAMPLE_RATE_HZ))

    psd = calc.compute_psd_column(signal)

    assert np.all(np.isnan(psd))

def test_psd_detects_peak_in_noisy_signal():
    rng = np.random.default_rng(0)
    sr = config.SAMPLE_RATE_HZ
    t = np.arange(0, 4, 1 / sr)
    signal = np.sin(2 * np.pi * 12 * t) + rng.normal(0, 0.25, size=len(t))

    calc = DSACalculator(4.0)
    psd = calc.compute_psd_column(signal)
    peak_freq = config.FREQ_BINS[np.argmax(psd)]

    assert abs(peak_freq - 12.0) < 1.0


def test_psd_returns_none_for_short_signal():
    calc = DSACalculator(window_sec=4)

    psd = calc.compute_psd_column(np.ones(int(config.SAMPLE_RATE_HZ)))

    assert psd == (None, None)


def test_psd_welch_mode_uses_same_frequency_bins():
    calc = DSACalculator(window_sec=4)
    sr = config.SAMPLE_RATE_HZ
    t = np.arange(0, 4, 1 / sr)
    signal = np.sin(2 * np.pi * 8 * t)

    psd = calc.compute_psd_column(signal, method="welch")

    assert psd.shape == config.FREQ_BINS.shape
    assert np.isfinite(psd).all()


def test_minimum_supported_two_second_window_keeps_dsa_frequency_shape_for_both_psd_methods():
    calc = DSACalculator(window_sec=2)
    sr = config.SAMPLE_RATE_HZ
    t = np.arange(0, 2, 1 / sr)
    signal = np.sin(2 * np.pi * 10 * t)

    for method in ("multitaper", "welch"):
        psd = calc.compute_psd_column(signal, method=method)

        assert psd.shape == config.FREQ_BINS.shape
        assert np.isfinite(psd).all()
        peak_freq = config.FREQ_BINS[np.argmax(psd)]
        assert abs(peak_freq - 10.0) <= 1.0
