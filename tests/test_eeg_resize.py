import pytest
import numpy as np
import time
from plotdsa.ui.views import DSAView, EEGView, PSDView
from plotdsa.config import TEXT_COLOR_STR, UserConfig
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QRectF
import sys

@pytest.fixture(scope="session")
def app():
    test_app = QApplication.instance()
    if test_app is None:
        test_app = QApplication(sys.argv)
    return test_app

def test_update_time_scale_reuses_latest_history_samples(app, monkeypatch):
    conf = UserConfig()
    view = EEGView(conf)

    class FakeScreen:
        def logicalDotsPerInch(self):
            return 96

    class FakeHandle:
        def screen(self):
            return FakeScreen()

    class FakeWindow:
        def windowHandle(self):
            return FakeHandle()

    class FakeViewBox:
        def sceneBoundingRect(self):
            return QRectF(0, 0, 40, 100)

    monkeypatch.setattr(view, "window", lambda: FakeWindow())
    monkeypatch.setattr(view, "getViewBox", lambda: FakeViewBox())

    view.history.extend(float(i) for i in range(200))
    view.display_head = 168

    view._update_time_scale()

    assert view.N == 169
    assert view.display[view.display_head] == 199.0
    assert view.display[view.display_head - 1] == 198.0


def test_update_time_scale_ignores_tiny_viewports(app, monkeypatch):
    conf = UserConfig()
    view = EEGView(conf)
    original_n = view.N

    class FakeScreen:
        def logicalDotsPerInch(self):
            return 96

    class FakeHandle:
        def screen(self):
            return FakeScreen()

    class FakeWindow:
        def windowHandle(self):
            return FakeHandle()

    class TinyViewBox:
        def sceneBoundingRect(self):
            return QRectF(0, 0, 2, 100)

    monkeypatch.setattr(view, "window", lambda: FakeWindow())
    monkeypatch.setattr(view, "getViewBox", lambda: TinyViewBox())

    view._update_time_scale()

    assert view.N == original_n


def test_plot_axes_use_white_larger_text(app):
    conf = UserConfig()
    dsa = DSAView(conf, lambda _: None, lambda _: None)
    psd = PSDView(conf)
    eeg = EEGView(conf)

    def assert_axis_style(axis, has_label):
        expected_color = TEXT_COLOR_STR.lower()
        assert axis.textPen().color().name() == expected_color
        assert axis.style["tickFont"].pointSize() == 11
        if has_label:
            label_html = axis.label.toHtml().lower()
            assert f"color:{expected_color}" in label_html
            assert "font-size:12pt" in label_html

    assert_axis_style(dsa.freq_axis, has_label=True)
    assert_axis_style(dsa.time_axis, has_label=False)
    assert_axis_style(dsa.colorbar.getAxis("left"), has_label=True)
    assert dsa.colorbar.colorMapMenu is False
    assert_axis_style(psd.getAxis("left"), has_label=True)
    assert_axis_style(psd.getAxis("bottom"), has_label=True)
    assert_axis_style(eeg.getAxis("left"), has_label=True)
    assert_axis_style(eeg.getAxis("bottom"), has_label=True)


def test_pause_button_sits_under_settings_and_blocks_samples(app, qtbot):
    view = EEGView(UserConfig())
    qtbot.addWidget(view)
    view.resize(500, 220)
    view.show()

    view._update_settings_button_pos()

    assert view.pause_button.x() == view.settings_button.x()
    assert view.pause_button.y() > view.settings_button.y()
    assert view.is_paused is False

    view.pause_button.click()

    assert view.is_paused is True
    assert len(view._pending) == 0

    view.append_sample(1.0)
    assert len(view._pending) == 0
    assert len(view.history) == 0

    view.pause_button.click()

    assert view.is_paused is False
    view.append_sample(2.0)
    assert len(view._pending) == 1


def test_resuming_eeg_keeps_sweep_position_while_refreshing_latest_samples(app, qtbot):
    view = EEGView(UserConfig())
    qtbot.addWidget(view)

    view.append_sample(10.0)
    view._pending[0] = (time.perf_counter() - 1.0, 10.0)
    view._render_frame_stable()
    initial_head = view.display_head

    view.pause_button.click()
    for value in (20.0, 30.0, 40.0):
        view.append_sample(value)

    assert view.display_head == initial_head
    assert len(view._pending) == 0

    view.pause_button.click()

    assert view.is_paused is False
    assert view.display_head == initial_head
    assert view.display[view.display_head] == 10.0
    assert np.count_nonzero(~np.isnan(view.display)) == 1

    view.append_sample(50.0)
    assert len(view._pending) == 1


def test_dsa_calibrate_emits_python_int_config_values(app, qtbot, monkeypatch):
    emitted = []
    view = DSAView(UserConfig(), emitted.append, lambda _: None)
    qtbot.addWidget(view)

    monkeypatch.setattr(
        view,
        "_get_visible_dsa_data",
        lambda _: (1000.0, np.array([[-120.0, 25.0, 80.0]], dtype=np.float32), 1.0),
    )

    view.calibrate()

    assert len(emitted) == 1
    new_config = emitted[0]
    assert type(new_config.psd_db_min) is int
    assert type(new_config.psd_db_max) is int

    view.apply_config(new_config)
