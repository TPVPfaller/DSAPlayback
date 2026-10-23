import sys

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

import config
from config import UserConfig
from views import DSAView


@pytest.fixture(scope="session")
def app():
    test_app = QApplication.instance()
    if test_app is None:
        test_app = QApplication(sys.argv)
    return test_app


def test_zoomed_historical_view_keeps_right_edge_and_can_pan_left(qtbot, app):
    view = DSAView(UserConfig(), lambda _: None, lambda _: None)
    qtbot.addWidget(view)

    start_ts = 1_000.0
    total_steps = int(120 / config.TIME_RESOLUTION)
    for offset in range(total_steps):
        psd = np.full(len(config.FREQ_BINS), float(offset), dtype=np.float32)
        view.append(start_ts + offset * config.TIME_RESOLUTION, psd)

    view.jump_to_live()
    view.apply_zoom(1.0)

    expected_live_start = start_ts + 120 - 60
    assert view._pan_sec == pytest.approx(expected_live_start)
    assert view.t0 == pytest.approx(expected_live_start)
    visible_steps = int(60.0 / config.TIME_RESOLUTION)
    expected_db = 10.0 * np.log10(np.arange(total_steps - visible_steps, total_steps, dtype=np.float32))
    np.testing.assert_allclose(view.dsa_rect[:, 0], expected_db)

    view.pan(-0.5)

    assert view.live_mode is False
    assert view._pan_sec == pytest.approx(expected_live_start - 30)


def test_zoom_out_keeps_same_slice_when_rendered_column_count_is_unchanged(qtbot, app):
    view = DSAView(UserConfig(display_minutes=32.0 / 60.0), lambda _: None, lambda _: None)
    qtbot.addWidget(view)

    start_ts = 1_000.0
    for offset in range(int(120 / config.TIME_RESOLUTION)):
        psd = np.full(len(config.FREQ_BINS), float(offset + 1), dtype=np.float32)
        view.append(start_ts + offset * config.TIME_RESOLUTION, psd)

    view.live_mode = False
    view._pan_sec = start_ts + 40.2
    view.update()

    original_start = view.t0
    original_rect = view.dsa_rect.copy()

    view.apply_zoom(32.05 / 60.0)

    assert view.t0 == pytest.approx(original_start)
    np.testing.assert_allclose(view.dsa_rect, original_rect)


def test_live_zoom_out_keeps_right_edge_when_rendered_column_count_is_unchanged(qtbot, app):
    view = DSAView(UserConfig(display_minutes=32.0 / 60.0), lambda _: None, lambda _: None)
    qtbot.addWidget(view)

    start_ts = 1_000.0
    for offset in range(int(120 / config.TIME_RESOLUTION)):
        psd = np.full(len(config.FREQ_BINS), float(offset + 1), dtype=np.float32)
        view.append(start_ts + offset * config.TIME_RESOLUTION, psd)

    view.jump_to_live()
    original_start = view.t0
    original_rect = view.dsa_rect.copy()

    view.apply_zoom(32.05 / 60.0)

    assert view.live_mode is True
    assert view.t0 == pytest.approx(original_start)
    np.testing.assert_allclose(view.dsa_rect, original_rect)


def test_render_grid_can_use_subsecond_time_resolution(qtbot, app):
    view = DSAView(UserConfig(display_minutes=0.5), lambda _: None, lambda _: None)
    qtbot.addWidget(view)

    actual_res, n_columns = view._render_grid(view._visible_width_sec())

    assert actual_res == pytest.approx(config.TIME_RESOLUTION)
    assert n_columns == int(view._visible_width_sec() / config.TIME_RESOLUTION)


def test_dragging_near_live_edge_does_not_snap_back_to_live(qtbot, app):
    view = DSAView(UserConfig(display_minutes=32.0 / 60.0), lambda _: None, lambda _: None)
    qtbot.addWidget(view)

    start_ts = 1_000.0
    for offset in range(int(120 / config.TIME_RESOLUTION)):
        psd = np.full(len(config.FREQ_BINS), float(offset + 1), dtype=np.float32)
        view.append(start_ts + offset * config.TIME_RESOLUTION, psd)

    view.jump_to_live()

    visible_width_sec = view._visible_width_sec()
    _, max_offset = view._pan_limits(visible_width_sec)
    view.live_mode = False
    view._dragging = True
    view._pan_sec = max_offset - 0.01

    view._sync_pan_window(visible_width_sec)

    assert view.live_mode is False
    assert view._pan_sec == pytest.approx(max_offset - 0.01)


def test_releasing_drag_at_live_edge_restores_live_mode(qtbot, app):
    view = DSAView(UserConfig(display_minutes=32.0 / 60.0), lambda _: None, lambda _: None)
    qtbot.addWidget(view)

    start_ts = 1_000.0
    for offset in range(int(120 / config.TIME_RESOLUTION)):
        psd = np.full(len(config.FREQ_BINS), float(offset + 1), dtype=np.float32)
        view.append(start_ts + offset * config.TIME_RESOLUTION, psd)

    view.jump_to_live()

    visible_width_sec = view._visible_width_sec()
    _, max_offset = view._pan_limits(visible_width_sec)
    view.live_mode = False
    view._dragging = True
    view._pan_sec = max_offset - 0.01

    release_event = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(1.0, 1.0),
        QPointF(1.0, 1.0),
        QPointF(1.0, 1.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    view.mouseReleaseEvent(release_event)

    assert view._dragging is False
    assert view.live_mode is True
    assert view._pan_sec == pytest.approx(max_offset)
