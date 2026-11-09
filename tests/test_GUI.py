import pytest
import numpy as np
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtGui import QCloseEvent
from main import DSAApplication, DSAView, PSDView, EEGView, SettingsDialog
from config import UserConfig
import config
import main as main_module

import os
import sys
import time

@pytest.fixture(scope="session")
def app():
    """Create the QApplication once for all tests."""
    test_app = QApplication.instance()
    if test_app is None:
        test_app = QApplication(sys.argv)
    return test_app


@pytest.fixture
def dsa_app(qtbot, app):
    """Create a DSAApplication instance for testing."""
    window = DSAApplication()
    qtbot.addWidget(window)
    window.show()
    return window


def test_initial_ui_state(dsa_app):
    """Verify that main views and widgets are initialized and visible as expected."""
    assert isinstance(dsa_app.dsa_view, DSAView)
    assert isinstance(dsa_app.psd_view, PSDView)
    assert isinstance(dsa_app.eeg_view, EEGView)
    # Initial visibility matches toggle defaults
    assert dsa_app.dsa_view.isVisible() is True
    assert dsa_app.psd_view.isVisible() is False
    assert dsa_app.eeg_view.isVisible() is True
    # TopBar widgets exist
    assert hasattr(dsa_app.topbar, "live_btn")
    assert hasattr(dsa_app.topbar, "zoom_slider")
    assert hasattr(dsa_app.topbar, "calibrate_btn")
    assert dsa_app.topbar.zoom_slider.maximum() == dsa_app.topbar.ZOOM_SLIDER_MAX


def test_zoom_slider_has_fine_control_when_zoomed_in(dsa_app):
    max_value = dsa_app.topbar.zoom_slider.maximum()

    dsa_app.topbar.zoom_slider.setValue(max_value - 1)
    almost_min_zoom = dsa_app.dsa_view.display_minutes

    dsa_app.topbar.zoom_slider.setValue(max_value)
    min_zoom = dsa_app.dsa_view.display_minutes

    assert min_zoom == pytest.approx(config.DISPLAY_MINUTES_BOUNDS[0])
    assert almost_min_zoom > min_zoom
    assert almost_min_zoom - min_zoom < 0.01


def test_menu_toggle_views(qtbot, dsa_app):
    """Toggle view visibility via menu actions."""
    dsa_app.action_show_psd.trigger()
    assert dsa_app.psd_view.isVisible() is True

    dsa_app.action_show_dsa.trigger()
    assert dsa_app.dsa_view.isVisible() is False

    dsa_app.action_show_eeg.trigger()
    assert dsa_app.eeg_view.isVisible() is False

    # Toggle back
    dsa_app.action_show_dsa.trigger()
    assert dsa_app.dsa_view.isVisible() is True




def test_settings_dialog_sliders_apply(qtbot, dsa_app):
    """Verify SettingsDialog can apply changes and trigger config callback."""
    dialog = SettingsDialog(dsa_app.user_config, dsa_app._on_config_change)
    qtbot.addWidget(dialog)
    dialog.show()

    # Modify a slider
    window_slider, scale = dialog.sliders["Window (s)"]
    old_value = window_slider.value()
    new_value = old_value + int(scale)
    window_slider.setValue(new_value)

    # Simulate Apply click
    dialog._apply()
    # Config updated
    assert dsa_app.user_config.window_sec == new_value


def test_eeg_view_append_sample(qtbot):
    """Appending a sample adds it to pending queue and eventually renders."""
    conf = UserConfig()
    view = EEGView(conf)
    qtbot.addWidget(view)
    val = 42.0
    view.append_sample(val)
    assert len(view._pending) > 0
    view._pending[0] = (time.perf_counter() - 1.0, val)
    view._render_frame_stable()
    assert view.history[-1] == val


def test_multitaper_menu_action_updates_user_config(dsa_app):
    dsa_app.action_multitaper.trigger()

    assert dsa_app.user_config.use_multitaper is False
    assert dsa_app.player.user_config.use_multitaper is False


def test_file_samples_are_ignored_when_eeg_view_is_hidden(dsa_app, monkeypatch):
    monkeypatch.setattr(dsa_app.player, "is_playing", True)
    dsa_app.eeg_view.hide()

    dsa_app._on_file_samples([1.0, 2.0, 3.0])

    assert len(dsa_app.eeg_view._pending) == 0


def test_menu_has_no_load_data_from_time_entry(dsa_app):
    texts = [
        action.text()
        for menu_action in dsa_app.menuBar().actions()
        if menu_action.menu() is not None
        for action in menu_action.menu().actions()
    ]

    assert "Load EEG File (CSV)" in texts
    assert "Load Data from Time" not in texts


def test_on_config_change_updates_psd_view_ranges(dsa_app):
    dsa_app.psd_view.update(np.ones(len(config.FREQ_BINS), dtype=np.float32))
    new_config = dsa_app.user_config.update(max_freq_hz=30, psd_db_min=-10, psd_db_max=25)

    dsa_app._on_config_change(new_config)

    x_range, y_range = dsa_app.psd_view.viewRange()
    assert dsa_app.psd_view.user_config == new_config
    assert x_range[0] == pytest.approx(config.LOWEST_FREQ_HZ)
    assert x_range[1] == pytest.approx(30)
    assert y_range[0] == pytest.approx(-15)
    assert y_range[1] == pytest.approx(30)


@pytest.mark.parametrize(
    ("reply", "expected_clear_calls"),
    [
        (QMessageBox.StandardButton.No, []),
        (QMessageBox.StandardButton.Yes, ["dsa", "eeg"]),
    ],
)
def test_confirm_clear_data_respects_confirmation_reply(dsa_app, monkeypatch, reply, expected_clear_calls):
    clear_calls = []

    monkeypatch.setattr(dsa_app, "_show_message", lambda *args, **kwargs: reply)
    monkeypatch.setattr(dsa_app.dsa_view, "clear_data", lambda: clear_calls.append("dsa"))
    monkeypatch.setattr(dsa_app.eeg_view, "clear_data", lambda: clear_calls.append("eeg"))

    dsa_app._confirm_clear_data()

    assert clear_calls == expected_clear_calls


def test_eeg_view_stops_render_timer_when_closed(qtbot):
    view = EEGView(UserConfig())
    qtbot.addWidget(view)
    view.show()

    assert view._timer.isActive() is True

    view.close()

    assert view._timer.isActive() is False


def test_close_event_stops_playback_and_timer(dsa_app, tmp_path):
    path = tmp_path / "eeg.csv"
    _write_test_eeg_csv(path, seconds=3)
    assert dsa_app._play_eeg_file(str(path)) is True

    event = QCloseEvent()
    dsa_app.closeEvent(event)

    assert event.isAccepted() is True
    assert dsa_app.player.is_playing is False
    assert dsa_app.status_timer.isActive() is False


def _write_test_eeg_csv(path, seconds=10):
    rate = config.SAMPLE_RATE_HZ
    t = np.arange(int(seconds * rate)) / rate
    values = 50 * np.sin(2 * np.pi * 10 * t)
    lines = []
    for ti, vi in zip(t, values):
        sec = 3600.0 + ti
        lines.append(f"{int(sec // 3600):02d}:{int(sec % 3600 // 60):02d}:{sec % 60:07.4f},{vi:.3f}")
    path.write_text("\n".join(lines) + "\n")


def test_load_eeg_csv_file_fills_dsa_and_clear_empties_it(dsa_app, tmp_path):
    path = tmp_path / "eeg.csv"
    _write_test_eeg_csv(path)

    assert dsa_app._load_eeg_file(str(path)) is True
    assert dsa_app.dsa_view.dsa_buffer.t0 is not None

    dsa_app._clear_data()
    assert dsa_app.dsa_view.dsa_buffer.t0 is None
    assert dsa_app.windowTitle() == main_module.WINDOW_TITLE


def test_play_eeg_csv_file_emits_dsa_columns(dsa_app, tmp_path, qtbot):
    path = tmp_path / "eeg.csv"
    _write_test_eeg_csv(path, seconds=3)

    assert dsa_app._play_eeg_file(str(path)) is True
    assert "Playing eeg.csv" in dsa_app.windowTitle()

    qtbot.waitUntil(lambda: dsa_app.dsa_view.dsa_buffer.t0 is not None, timeout=6000)
    qtbot.waitUntil(lambda: not dsa_app.player.is_playing, timeout=6000)
    assert "Playback finished" in dsa_app.windowTitle()


@pytest.mark.parametrize(("choice", "expected"), [("load", ["load"]), ("play", ["play"]), (None, [])])
def test_open_eeg_file_asks_whether_to_load_or_play(dsa_app, monkeypatch, choice, expected):
    calls = []
    monkeypatch.setattr(dsa_app, "_ask_load_or_play", lambda path: choice)
    monkeypatch.setattr(dsa_app, "_load_eeg_file", lambda path: calls.append("load"))
    monkeypatch.setattr(dsa_app, "_play_eeg_file", lambda path: calls.append("play"))
    monkeypatch.setattr(dsa_app, "_remember_eeg_dir", lambda path: None)

    dsa_app._open_eeg_file("eeg.csv")

    assert calls == expected


def test_dropping_csv_on_dsa_view_emits_file_path(dsa_app, tmp_path, qtbot, monkeypatch):
    from PySide6.QtCore import QMimeData, QPointF, QUrl, Qt
    from PySide6.QtGui import QDropEvent

    # The drop opens the modal Load/Play dialog, which would block the test.
    monkeypatch.setattr(dsa_app, "_ask_load_or_play", lambda p: None)
    path = tmp_path / "eeg.csv"
    path.write_text("20:12:06.2613,-82\n")
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    event = QDropEvent(QPointF(10, 10), Qt.DropAction.CopyAction, mime,
                       Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)

    with qtbot.waitSignal(dsa_app.dsa_view.eeg_file_dropped, timeout=1000) as blocker:
        dsa_app.dsa_view.dropEvent(event)

    assert len(blocker.args) == 1
    assert os.path.samefile(blocker.args[0], path)


def test_load_eeg_file_reports_errors(dsa_app, tmp_path, monkeypatch):
    path = tmp_path / "eeg.csv"
    path.write_text("garbage\n")
    shown = []
    monkeypatch.setattr(dsa_app, "_show_message", lambda title, text: shown.append(title))

    assert dsa_app._load_eeg_file(str(path)) is False
    assert dsa_app._file_name is None
    assert shown == ["Load Error"]


def test_open_eeg_file_remembers_folder_for_next_file_dialog(dsa_app, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    dsa_app._settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    path = tmp_path / "eeg.csv"
    path.write_text("20:12:06.2613,-82\n")
    monkeypatch.setattr(dsa_app, "_ask_load_or_play", lambda p: "load")
    monkeypatch.setattr(dsa_app, "_load_eeg_file", lambda p: True)

    dsa_app._open_eeg_file(str(path))

    assert os.path.samefile(dsa_app._eeg_start_dir(), tmp_path)


def test_load_single_column_file_asks_for_sample_rate(dsa_app, tmp_path, monkeypatch):
    path = tmp_path / "eeg.csv"
    rate = config.SAMPLE_RATE_HZ
    values = 50 * np.sin(2 * np.pi * 10 * np.arange(10 * rate) / rate)
    path.write_text("\n".join(f"{v:.3f}" for v in values) + "\n")
    asked = []

    def fake_get_double(parent, title, label, value, *args):
        asked.append(value)
        return value, True

    monkeypatch.setattr(main_module.QInputDialog, "getDouble", fake_get_double)

    assert dsa_app._load_eeg_file(str(path)) is True
    assert asked == [400.0]


def test_cancelling_sample_rate_prompt_aborts_play(dsa_app, tmp_path, monkeypatch):
    path = tmp_path / "eeg.csv"
    path.write_text("1\n2\n3\n")
    monkeypatch.setattr(main_module.QInputDialog, "getDouble", lambda *a: (400.0, False))

    assert dsa_app._play_eeg_file(str(path)) is False
    assert dsa_app._file_name is None


def test_load_eeg_file_shorter_than_window_shows_message(dsa_app, tmp_path, monkeypatch):
    path = tmp_path / "eeg.csv"
    _write_test_eeg_csv(path, seconds=1)
    shown = []
    monkeypatch.setattr(dsa_app, "_show_message", lambda title, text: shown.append(title))

    assert dsa_app._load_eeg_file(str(path)) is False
    assert dsa_app._file_name is None
    assert shown == ["Load EEG File"]


def test_load_eeg_file_zooms_to_recording_length(dsa_app, tmp_path, monkeypatch):
    path = tmp_path / "eeg.csv"
    _write_test_eeg_csv(path, seconds=90)
    zooms = []
    monkeypatch.setattr(dsa_app, "_on_zoom_change", zooms.append)

    assert dsa_app._load_eeg_file(str(path)) is True

    assert zooms == [pytest.approx(1.5, abs=0.01)]
    assert "File: eeg.csv" in dsa_app.windowTitle()


def test_play_eeg_file_reports_errors(dsa_app, tmp_path, monkeypatch):
    path = tmp_path / "eeg.csv"
    path.write_text("garbage\n")
    shown = []
    monkeypatch.setattr(dsa_app, "_show_message", lambda title, text: shown.append(title))

    assert dsa_app._play_eeg_file(str(path)) is False
    assert dsa_app._file_name is None
    assert dsa_app.player.is_playing is False
    assert shown == ["Load Error"]


def test_file_samples_are_shown_only_while_playing(dsa_app, tmp_path, monkeypatch):
    appended = []
    monkeypatch.setattr(dsa_app.eeg_view, "append_sample", appended.append)
    path = tmp_path / "eeg.csv"
    _write_test_eeg_csv(path, seconds=3)

    assert dsa_app._play_eeg_file(str(path)) is True
    dsa_app._on_file_samples([3.0])
    assert appended == [3.0]

    dsa_app._clear_data()
    dsa_app._on_file_samples([4.0])
    assert appended == [3.0]


def test_clear_data_stops_playback_and_ignores_late_file_columns(dsa_app, tmp_path):
    path = tmp_path / "eeg.csv"
    _write_test_eeg_csv(path, seconds=3)
    assert dsa_app._play_eeg_file(str(path)) is True

    dsa_app._clear_data()

    assert dsa_app.player.is_playing is False
    assert dsa_app.windowTitle() == main_module.WINDOW_TITLE
    dsa_app._on_file_dsa_column(time.time(), np.ones(len(config.FREQ_BINS)), 1)
    assert dsa_app.dsa_view.dsa_buffer.t0 is None


@pytest.mark.parametrize("suffix", [".pdf", ".svg", ".png"])
def test_screenshot_saves_file_of_views_only(dsa_app, tmp_path, monkeypatch, suffix):
    import plotdsa.ui.topbar as topbar_module

    monkeypatch.setattr(topbar_module.TopBar, "_show_screenshot_feedback", lambda *a: None)
    path = str(tmp_path / f"shot{suffix}")

    assert dsa_app.topbar._take_screenshot(path) == path
    assert os.path.getsize(path) > 0

    views = [v for v in (dsa_app.dsa_view, dsa_app.psd_view, dsa_app.eeg_view) if v.isVisible()]
    top = min(v.mapTo(dsa_app, v.rect().topLeft()).y() for v in views)
    assert top >= dsa_app.topbar.mapTo(dsa_app, dsa_app.topbar.rect().bottomLeft()).y()
    if suffix == ".svg":
        width = max(v.width() for v in views)
        height = sum(v.height() for v in views)
        assert f'viewBox="0 0 {width} {height}"' in open(path, encoding="utf-8").read(2000)
    if suffix == ".png":
        from PySide6.QtGui import QImage

        image = QImage(path)
        scale = topbar_module.SCREENSHOT_PNG_SCALE
        assert image.width() == max(v.width() for v in views) * scale
        assert image.height() == sum(v.height() for v in views) * scale
        # No white rows: the background is painted edge to edge.
        assert image.pixelColor(image.width() // 2, image.height() - 1).lightness() < 128


def test_screenshot_cancelled_dialog_saves_nothing(dsa_app, monkeypatch):
    import plotdsa.ui.topbar as topbar_module

    monkeypatch.setattr(topbar_module.QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))

    assert dsa_app.topbar._take_screenshot() is None


@pytest.mark.parametrize("selected_filter, suffix", [("SVG (*.svg)", ".svg"), ("PNG (*.png)", ".png")])
def test_screenshot_dialog_adds_extension_from_selected_type(dsa_app, tmp_path, monkeypatch, selected_filter, suffix):
    import plotdsa.ui.topbar as topbar_module

    target = str(tmp_path / "shot")
    monkeypatch.setattr(topbar_module.QFileDialog, "getSaveFileName", lambda *a, **k: (target, selected_filter))

    assert dsa_app.topbar._ask_screenshot_path() == target + suffix


def test_pdf_screenshot_stores_dsa_image_lossless(dsa_app, tmp_path, monkeypatch):
    import plotdsa.ui.topbar as topbar_module

    monkeypatch.setattr(topbar_module.TopBar, "_show_screenshot_feedback", lambda *a: None)
    eeg = tmp_path / "eeg.csv"
    _write_test_eeg_csv(eeg)
    assert dsa_app._load_eeg_file(str(eeg)) is True
    path = str(tmp_path / "shot.pdf")

    assert dsa_app.topbar._take_screenshot(path) == path
    raw = open(path, "rb").read()
    assert b"/Subtype /Image" in raw
    assert b"/DCTDecode" not in raw  # JPEG would blur the DSA cells


def test_playback_bar_only_visible_while_playing_a_file(dsa_app, tmp_path):
    path = tmp_path / "eeg.csv"
    _write_test_eeg_csv(path, seconds=10)
    assert dsa_app.playback_bar.isVisible() is False

    assert dsa_app._play_eeg_file(str(path)) is True
    assert dsa_app.playback_bar.isVisible() is True
    assert dsa_app.playback_bar.time_label.text() == "00:00 / 00:10"

    assert dsa_app._load_eeg_file(str(path)) is True
    assert dsa_app.playback_bar.isVisible() is False


def test_playback_bar_pauses_and_resumes_player(dsa_app, tmp_path):
    path = tmp_path / "eeg.csv"
    _write_test_eeg_csv(path, seconds=10)
    dsa_app._play_eeg_file(str(path))

    dsa_app.playback_bar.play_btn.click()
    assert dsa_app.player.is_playing is False
    assert "Paused" in dsa_app.windowTitle()

    dsa_app.playback_bar.play_btn.click()
    assert dsa_app.player.is_playing is True
    assert "Playing" in dsa_app.windowTitle()


def test_dragging_overview_rectangle_seeks_and_rebuilds_dsa(dsa_app, tmp_path):
    path = tmp_path / "eeg.csv"
    _write_test_eeg_csv(path, seconds=30)
    dsa_app._play_eeg_file(str(path))
    dsa_app.playback_bar.play_btn.click()  # pause so the position stays put
    overview = dsa_app.playback_bar.overview

    overview.region.setRegion((10.0, 20.0))  # same signal as finishing a drag
    assert dsa_app.player.position == pytest.approx(20.0)
    assert dsa_app.playback_bar.time_label.text() == "00:20 / 00:30"
    assert dsa_app.dsa_view.dsa_buffer.t0 is not None

    overview.region.setRegion((-5.0, 5.0))
    assert dsa_app.player.position == pytest.approx(5.0)


def test_overview_shows_whole_dsa_and_rectangle_spans_visible_dsa(dsa_app, tmp_path):
    path = tmp_path / "eeg.csv"
    _write_test_eeg_csv(path, seconds=30)
    dsa_app._play_eeg_file(str(path))
    dsa_app.playback_bar.play_btn.click()
    overview = dsa_app.playback_bar.overview

    image = overview.image.image
    assert image.shape[1] == len(dsa_app.dsa_view.freq_bins)
    assert np.isfinite(image[-1]).all()  # precomputed up to the end of the recording

    dsa_app._on_zoom_change(2.0)
    left, right = overview.region.getRegion()
    assert right == pytest.approx(dsa_app.player.position)
    assert right - left == pytest.approx(120.0)


def test_overview_rectangle_follows_panned_dsa(dsa_app, tmp_path):
    path = tmp_path / "eeg.csv"
    _write_test_eeg_csv(path, seconds=30)
    dsa_app._play_eeg_file(str(path))
    dsa_app.playback_bar.play_btn.click()
    dsa_app._seek_playback(30.0)
    dsa_app._on_zoom_change(0.25)
    overview = dsa_app.playback_bar.overview

    dsa_app.dsa_view.pan(-0.5)
    left, right = overview.region.getRegion()
    view_end = dsa_app.dsa_view.t0 + 15.0 - dsa_app._file_t0
    assert right == pytest.approx(view_end)
    assert right < dsa_app.player.position - 5.0
    assert right - left == pytest.approx(15.0)
    assert overview.future_mask.rect().left() == pytest.approx(dsa_app.player.position)

    dsa_app.dsa_view.jump_to_live()
    assert overview.region.getRegion()[1] == pytest.approx(dsa_app.player.position)


def test_overview_rectangle_keeps_size_when_dragging_short_dsa(dsa_app, tmp_path):
    path = tmp_path / "eeg.csv"
    _write_test_eeg_csv(path, seconds=30)
    dsa_app._play_eeg_file(str(path))
    dsa_app.playback_bar.play_btn.click()
    dsa_app._seek_playback(20.0)
    dsa_app._on_zoom_change(1.0)  # 60 s window, only 20 s of DSA: black space on the right
    overview = dsa_app.playback_bar.overview
    before = overview.region.getRegion()

    dsa_app.dsa_view._dragging = True  # mid-drag the view stays out of live mode
    dsa_app.dsa_view.live_mode = False
    dsa_app.dsa_view.update()

    assert overview.region.getRegion() == pytest.approx(before)
    assert before[1] == pytest.approx(dsa_app.player.position)


def test_playback_bar_speed_buttons_change_player_and_eeg_speed(dsa_app, tmp_path):
    path = tmp_path / "eeg.csv"
    _write_test_eeg_csv(path, seconds=10)
    dsa_app._play_eeg_file(str(path))
    bar = dsa_app.playback_bar
    assert bar.speed_label.text() == "1×"

    bar.faster_btn.click()
    bar.faster_btn.click()
    assert dsa_app.player.speed == 4.0
    assert dsa_app.eeg_view._sample_period == pytest.approx(1.0 / (4 * config.SAMPLE_RATE_HZ))
    assert bar.speed_label.text() == "4×"

    for _ in range(10):
        bar.slower_btn.click()
    assert dsa_app.player.speed == 0.25
    assert bar.slower_btn.isEnabled() is False
    assert bar.speed_label.text() == "0.25×"

    dsa_app._play_eeg_file(str(path))  # a new file starts at normal speed
    assert dsa_app.player.speed == 1.0


def test_overview_hides_future_while_playing_and_shows_whole_dsa_when_paused(dsa_app, tmp_path):
    path = tmp_path / "eeg.csv"
    _write_test_eeg_csv(path, seconds=30)
    dsa_app._play_eeg_file(str(path))
    overview = dsa_app.playback_bar.overview
    mask = overview.future_mask

    # Playing: everything from the playhead to the end is covered.
    assert overview.isVisible() is True
    assert mask.isVisible() is True
    assert mask.rect().left() == pytest.approx(dsa_app.player.position)
    assert mask.rect().right() == pytest.approx(dsa_app.player.duration)

    # Paused: the whole DSA, including future data, is shown.
    dsa_app.playback_bar.play_btn.click()
    assert dsa_app.player.is_playing is False
    assert mask.isVisible() is False
    dsa_app._seek_playback(12.0)
    assert mask.isVisible() is False

    # Playing again: the future is covered from the new playhead on.
    dsa_app.playback_bar.play_btn.click()
    assert mask.isVisible() is True
    assert mask.rect().left() == pytest.approx(dsa_app.player.position, abs=0.5)
    assert mask.rect().right() == pytest.approx(30.0)

    # Finished: nothing left to hide, whole DSA shown.
    dsa_app._on_playback_finished()
    assert mask.isVisible() is False


def _wheel(widget, delta_y):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    pos = QPointF(widget.width() / 2, widget.height() / 2)
    event = QWheelEvent(pos, QPointF(widget.mapToGlobal(pos.toPoint())), QPoint(0, 0), QPoint(0, delta_y),
                        Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    widget.wheelEvent(event)
    return event


def test_mouse_wheel_zooms_dsa_and_syncs_zoom_slider(dsa_app):
    start = dsa_app.dsa_view.display_minutes
    slider_start = dsa_app.topbar.zoom_slider.value()

    event = _wheel(dsa_app.dsa_view, 120)
    assert event.isAccepted()
    assert dsa_app.dsa_view.display_minutes == pytest.approx(start / dsa_app.dsa_view.WHEEL_ZOOM_STEP)
    assert dsa_app.topbar.zoom_slider.value() != slider_start

    _wheel(dsa_app.dsa_view, -120)
    assert dsa_app.dsa_view.display_minutes == pytest.approx(start)

    for _ in range(50):
        _wheel(dsa_app.dsa_view, 120)
    assert dsa_app.dsa_view.display_minutes == pytest.approx(config.DISPLAY_MINUTES_BOUNDS[0])


def test_mouse_wheel_steps_eeg_y_range_presets(dsa_app):
    options = config.EEG_Y_RANGE_OPTIONS
    view = dsa_app.eeg_view
    assert view.user_config.eeg_uv_range_max == options[0]

    _wheel(view, 120)  # already at the smallest range
    assert dsa_app.user_config.eeg_uv_range_max == options[0]

    _wheel(view, -60)  # half a notch (trackpad) does nothing yet
    assert dsa_app.user_config.eeg_uv_range_max == options[0]
    event = _wheel(view, -60)
    assert event.isAccepted()
    assert dsa_app.user_config.eeg_uv_range_max == options[1]
    assert view.getViewBox().viewRange()[1] == pytest.approx([-options[1], options[1]])

    for _ in range(5):
        _wheel(view, -120)
    assert dsa_app.user_config.eeg_uv_range_max == options[-1]

    _wheel(view, 120)
    assert dsa_app.user_config.eeg_uv_range_max == options[-2]


def test_mouse_wheel_on_playback_timeline_zooms_dsa(dsa_app):
    start = dsa_app.dsa_view.display_minutes
    event = _wheel(dsa_app.playback_bar.overview, 120)
    assert event.isAccepted()
    assert dsa_app.dsa_view.display_minutes == pytest.approx(start / dsa_app.dsa_view.WHEEL_ZOOM_STEP)
    assert dsa_app.playback_bar._window_width == pytest.approx(dsa_app.dsa_view.display_minutes * 60.0)


def _write_multi_channel_csv(path, n_channels, seconds=10):
    rate = config.SAMPLE_RATE_HZ
    t = np.arange(int(seconds * rate)) / rate
    lines = ["time," + ",".join(f"EEG{ch + 1}" for ch in range(n_channels))]
    for ti in t:
        values = [50 * np.sin(2 * np.pi * (8 + ch) * ti) for ch in range(n_channels)]
        lines.append(f"{ti:.4f}," + ",".join(f"{v:.3f}" for v in values))
    path.write_text("\n".join(lines) + "\n")


def _grid_position(grid, view):
    layout = grid.layout()
    row, col, _, _ = layout.getItemPosition(layout.indexOf(view))
    return row, col


def test_load_multi_channel_file_shows_dsas_above_eegs_in_a_grid(dsa_app, tmp_path, qtbot):
    path = tmp_path / "eeg.csv"
    _write_multi_channel_csv(path, 3)

    assert dsa_app._load_eeg_file(str(path)) is True

    dsa_views = [dsa_app.dsa_view] + dsa_app.extra_dsa_views
    eeg_views = [dsa_app.eeg_view] + dsa_app.extra_eeg_views
    assert len(dsa_views) == len(eeg_views) == 3
    # 3 channels: 2 columns, 2 rows in each grid.
    assert [_grid_position(dsa_app.dsa_grid, v) for v in dsa_views] == [(0, 0), (0, 1), (1, 0)]
    assert [_grid_position(dsa_app.eeg_grid, v) for v in eeg_views] == [(0, 0), (0, 1), (1, 0)]
    qtbot.waitUntil(lambda: all(v.isVisible() for v in dsa_views + eeg_views), timeout=2000)
    assert dsa_app.dsa_grid.geometry().bottom() < dsa_app.eeg_grid.geometry().top()
    assert [v.plot.titleLabel.text for v in dsa_views] == ["EEG1", "EEG2", "EEG3"]
    assert all(v.dsa_buffer.t0 is not None for v in dsa_views)

    # Zoom and pan apply to every DSA.
    dsa_app._on_zoom_change(0.05)
    dsa_app.dsa_view.pan(-0.5)
    for view in dsa_app.extra_dsa_views:
        assert view.display_minutes == pytest.approx(0.05)
        assert view.live_mode is False
        assert view.t0 == pytest.approx(dsa_app.dsa_view.t0)

    dsa_app.action_show_eeg.trigger()
    assert not any(v.isVisible() for v in eeg_views)
    dsa_app.action_show_eeg.trigger()

    dsa_app._clear_data()
    assert dsa_app.extra_dsa_views == [] and dsa_app.extra_eeg_views == []
    assert dsa_app.dsa_view.minimumHeight() == config.MIN_DSA_HEIGHT


def test_play_multi_channel_file_feeds_every_channel(dsa_app, tmp_path, qtbot):
    path = tmp_path / "eeg.csv"
    _write_multi_channel_csv(path, 2, seconds=4)

    assert dsa_app._play_eeg_file(str(path)) is True
    dsa_app._toggle_playback()  # pause
    dsa_app._seek_playback(3.5)

    assert len(dsa_app.extra_dsa_views) == 1
    assert dsa_app.extra_dsa_views[0].dsa_buffer.t0 is not None
    dsa_app._toggle_playback()  # resume
    qtbot.waitUntil(lambda: len(dsa_app.extra_eeg_views[0]._pending) > 0
                    or dsa_app.extra_eeg_views[0].display_head >= 0, timeout=3000)
