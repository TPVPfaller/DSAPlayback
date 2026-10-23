import numpy as np
from buffers import DSABuffer
import config


def get_freq_bins():
    return config.FREQ_BINS


# ------------------------------------------------------------
# DSA time alignment
# ------------------------------------------------------------

def test_get_view_at_basic_pan():
    buf = DSABuffer()
    f = get_freq_bins()
    psd = np.ones(len(f))

    t0 = 1000.0
    for i in range(50):
        buf.append(t0 + i * config.TIME_RESOLUTION, psd * i)

    # Pan 2 time slots forward
    pan_sec = t0 + 2 * config.TIME_RESOLUTION
    t_start, view, res = buf.get_view_at(
        width=3 * config.TIME_RESOLUTION,
        height=10,
        pan_sec=pan_sec,
        target_resolution=config.TIME_RESOLUTION,
    )

    assert abs(t_start - pan_sec) < 1e-6
    assert not np.isnan(view[0]).all()


def test_get_view_at_clamps_to_last_slot():
    buf = DSABuffer()
    f = get_freq_bins()
    psd = np.ones(len(f))

    t0 = 1000.0
    buf.append(t0, psd)

    # Request far beyond available data
    t_start, view, res = buf.get_view_at(width=5, height=10, pan_sec=t0 + 1000, target_resolution=1)

    # Should not crash and should contain valid first frame
    assert not np.isnan(view[0]).all()


def test_get_view_at_empty_buffer_returns_nan_frame():
    buf = DSABuffer()

    t_start, view, res = buf.get_view_at(width=4, height=6, pan_sec=1000.0, target_resolution=1)

    assert isinstance(t_start, float)
    assert view.shape == (4, 6)
    assert np.isnan(view).all()
    assert res == 1


def test_get_view_at_clamps_pan_before_buffer_start():
    buf = DSABuffer()
    psd = np.arange(len(get_freq_bins()), dtype=np.float32)

    t0 = 1000.0
    buf.append(t0, psd)
    buf.append(t0 + 1.0, psd + 1)

    t_start, view, _ = buf.get_view_at(width=2, height=3, pan_sec=t0 - 50.0, target_resolution=1)

    assert t_start == t0
    np.testing.assert_array_equal(view[0], psd[:3])


def test_append_nan_psd_does_not_populate_frame():
    buf = DSABuffer()
    nan_psd = np.full(len(get_freq_bins()), np.nan, dtype=np.float32)

    buf.append(1000.0, nan_psd)
    t_start, view, _ = buf.get_view_at(width=1, height=4, pan_sec=1000.0, target_resolution=1)

    assert t_start == 1000.0
    assert np.isnan(view).all()


def test_buffer_t0_is_snapped_to_time_resolution():
    buf = DSABuffer()
    psd = np.ones(len(get_freq_bins()), dtype=np.float32)

    buf.append(1000.17, psd)

    assert buf.t0 == 1000.0


def test_finest_resolution_overwrites_within_same_time_slot():
    buf = DSABuffer()
    psd_a = np.ones(len(get_freq_bins()), dtype=np.float32)
    psd_b = np.full(len(get_freq_bins()), 3.0, dtype=np.float32)

    buf.append(1000.01, psd_a)
    buf.append(1000.04, psd_b)

    np.testing.assert_array_equal(buf.buffers[config.TIME_RESOLUTION]["data"][0], psd_b)


def test_coarse_resolution_averages_columns_within_same_slot():
    buf = DSABuffer()
    psd_a = np.ones(len(get_freq_bins()), dtype=np.float32)
    psd_b = np.full(len(get_freq_bins()), 3.0, dtype=np.float32)

    buf.append(1000.0, psd_a)
    buf.append(1004.0, psd_b)

    slot_zero = buf.buffers[10]["data"][0]
    np.testing.assert_allclose(slot_zero, 2.0)


def test_old_frames_are_trimmed_when_capacity_is_exceeded():
    buf = DSABuffer()
    buf.buffers[1]["max_frames"] = 2
    psd = np.ones(len(get_freq_bins()), dtype=np.float32)

    for i in range(4):
        buf.append(1000.0 + i, psd * i)

    t_start, view, _ = buf.get_view_at(width=4, height=1, pan_sec=1000.0, target_resolution=1)

    assert t_start == 1002.0
    np.testing.assert_array_equal(view[:, 0], np.array([2.0, 3.0], dtype=np.float32))

