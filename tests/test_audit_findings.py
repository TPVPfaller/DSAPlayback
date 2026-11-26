"""
Regression tests for audit findings: a DC offset or slow drift must not change the DSA,
and a loaded file's DSA must follow DSA settings changes.
"""
import sys

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from plotdsa import config
from plotdsa.core.calculations import DSACalculator

FS = config.SAMPLE_RATE_HZ


def _db(psd):
    return 10 * np.log10(psd)


# --------------------------------------------------------------------------------------
# Finding 1: the short FIR "0.1-47 Hz band-pass" passes DC, so without detrending an
# electrode offset showed up as large spurious low-frequency power in the DSA.
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["multitaper", "welch"])
def test_slow_drift_does_not_change_dsa(method):
    rng = np.random.default_rng(3)
    noise = rng.normal(0.0, 10.0, 2 * FS)
    drift = np.linspace(-150.0, -50.0, len(noise))  # offset drifting by 100 uV over the window
    calc = DSACalculator(window_sec=2)

    clean = _db(calc.compute_psd_column(noise, method=method))
    drifting = _db(calc.compute_psd_column(noise + drift, method=method))

    assert np.max(np.abs(drifting - clean)) < 3.0


@pytest.mark.parametrize("method", ["multitaper", "welch"])
def test_dc_offset_does_not_change_dsa(method):
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 10.0, 2 * FS)  # 2 s of 10 uV white noise
    calc = DSACalculator(window_sec=2)

    clean = _db(calc.compute_psd_column(noise, method=method))
    offset = _db(calc.compute_psd_column(noise - 82.0, method=method))  # -82 uV offset

    delta = offset - clean
    # Same EEG, only a constant offset added: the DSA should not change by more than 3 dB.
    assert np.max(np.abs(delta)) < 3.0, (
        f"{method}: offset raised power by up to {delta.max():.1f} dB "
        f"(0.5 Hz: {delta[0]:+.1f} dB, 2 Hz: {delta[3]:+.1f} dB)"
    )


# --------------------------------------------------------------------------------------
# Finding 3: for a file opened with "Load" (not "Play"), changing Window / Overlap /
# Multitaper did not recompute the DSA: the view kept showing the old analysis.
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


def _write_csv(path, seconds=60):
    t = np.arange(int(seconds * FS)) / FS
    values = 20 * np.sin(2 * np.pi * 10 * t) + np.random.default_rng(2).normal(0, 5, len(t))
    lines = []
    for ti, vi in zip(t, values):
        sec = 3600.0 + ti
        lines.append(f"{int(sec // 3600):02d}:{int(sec % 3600 // 60):02d}:{sec % 60:07.4f},{vi:.3f}")
    path.write_text("\n".join(lines) + "\n")


def _base_res_snapshot(view):
    buf = view.dsa_buffer.buffers[view.dsa_buffer.RESOLUTIONS[0]]
    return {slot: col.copy() for slot, col in buf["data"].items()}


def test_loaded_file_dsa_follows_settings_change(qapp, qtbot, tmp_path):
    from plotdsa.app.main import DSAApplication

    window = DSAApplication()
    qtbot.addWidget(window)
    window.show()
    path = tmp_path / "eeg.csv"
    _write_csv(path)
    assert window._load_eeg_file(str(path)) is True
    before = _base_res_snapshot(window.dsa_view)

    # Switch Multitaper -> Welch and 2 s -> 10 s window, as from the menu / settings dialog.
    window._on_config_change(window.user_config.update(use_multitaper=False, window_sec=10))

    after = _base_res_snapshot(window.dsa_view)
    changed = before.keys() != after.keys() or any(
        not np.array_equal(before[s], after[s], equal_nan=True) for s in before
    )
    assert changed, "DSA of the loaded file still shows the 2 s multitaper analysis"
