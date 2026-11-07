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
