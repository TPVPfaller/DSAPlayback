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
def dsa_app(qtbot, app, monkeypatch):
    """Create a DSAApplication instance for testing."""
    class DummyWorker:
        def __init__(self, user_config):
            self.user_config = user_config

        def apply_config(self, user_config):
            self.user_config = user_config

        def stop(self):
            pass

    class DummyThread:
        def __init__(self):
            self.running = False

        def quit(self):
            pass

        def isRunning(self):
            return self.running

    def fake_init_worker(self):
        self.thread = DummyThread()
        self.worker = DummyWorker(self.user_config)

    monkeypatch.setattr(DSAApplication, "_init_worker", fake_init_worker)

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
    assert dsa_app.worker.user_config.use_multitaper is False


def test_on_new_samples_is_ignored_when_eeg_view_is_hidden(dsa_app):
    dsa_app.eeg_view.hide()
    before = dsa_app._last_data_receive_time

    dsa_app._on_new_samples([1.0, 2.0, 3.0])

    assert len(dsa_app.eeg_view._pending) == 0
    assert dsa_app._last_data_receive_time >= before


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


def test_load_data_uses_fractional_duration_to_append_expected_dsa_steps(dsa_app, monkeypatch):
    psd = np.ones(len(dsa_app.dsa_view.freq_bins), dtype=np.float32)
    appended = []

    monkeypatch.setattr(
        "main.Output.load_psd_from_time",
        lambda start_time_dt: [(1000.0, 3 * config.TIME_RESOLUTION, psd)]
    )
    monkeypatch.setattr(dsa_app.dsa_view, "append", lambda ts, loaded_psd: appended.append((ts, loaded_psd)))

    dsa_app._load_data_from_time(time.time())

    assert [ts for ts, _ in appended] == pytest.approx(
        [1000.0 + i * config.TIME_RESOLUTION for i in range(3)]
    )


def test_load_data_rounds_short_duration_up_to_one_dsa_step(dsa_app, monkeypatch):
    psd = np.ones(len(dsa_app.dsa_view.freq_bins), dtype=np.float32)
    appended = []

    monkeypatch.setattr(
        "main.Output.load_psd_from_time",
        lambda start_time_dt: [(1000.0, config.TIME_RESOLUTION / 3.0, psd)]
    )
    monkeypatch.setattr(dsa_app.dsa_view, "append", lambda ts, loaded_psd: appended.append((ts, loaded_psd)))

    dsa_app._load_data_from_time(time.time())

    assert len(appended) == 1
    assert appended[0][0] == pytest.approx(1000.0)
    np.testing.assert_array_equal(appended[0][1], psd)


def test_load_data_from_time_does_not_clear_eeg_view(dsa_app, monkeypatch):
    psd = np.ones(len(dsa_app.dsa_view.freq_bins), dtype=np.float32)

    monkeypatch.setattr(
        "main.Output.load_psd_from_time",
        lambda start_time_dt: [(1000.0, 0.1, psd)]
    )

    clear_calls = []
    monkeypatch.setattr(dsa_app.eeg_view, "clear_data", lambda: clear_calls.append("cleared"))
    monkeypatch.setattr(dsa_app.dsa_view, "append", lambda ts, loaded_psd: None)

    dsa_app._load_data_from_time(time.time())

    assert clear_calls == []


def test_load_data_from_time_reports_errors_without_clearing_existing_dsa(dsa_app, monkeypatch):
    clear_calls = []
    shown_messages = []

    def raise_load_error(start_time_dt):
        raise RuntimeError("broken csv")

    monkeypatch.setattr("main.Output.load_psd_from_time", raise_load_error)
    monkeypatch.setattr(dsa_app.dsa_view, "clear_data", lambda: clear_calls.append("cleared"))
    monkeypatch.setattr(dsa_app, "_show_message", lambda title, text: shown_messages.append((title, text)))

    dsa_app._load_data_from_time(time.time())

    assert clear_calls == []
    assert shown_messages == [("Load Error", "Failed to load data: broken csv")]


def test_load_data_from_time_keeps_existing_dsa_when_no_rows_found(dsa_app, monkeypatch):
    clear_calls = []

    monkeypatch.setattr("main.Output.load_psd_from_time", lambda start_time_dt: [])
    monkeypatch.setattr(dsa_app.dsa_view, "clear_data", lambda: clear_calls.append("cleared"))
    monkeypatch.setattr(dsa_app, "_show_message", lambda *args, **kwargs: None)

    dsa_app._load_data_from_time(time.time())

    assert clear_calls == []


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


def test_close_event_waits_asynchronously_for_running_worker(dsa_app):
    class ClosingWorker:
        def __init__(self):
            self.stop_called = False

        def stop(self):
            self.stop_called = True

    class ClosingThread:
        def __init__(self):
            self.running = True
            self.quit_called = False

        def quit(self):
            self.quit_called = True

        def isRunning(self):
            return self.running

    dsa_app.worker = ClosingWorker()
    dsa_app.thread = ClosingThread()

    first_event = QCloseEvent()
    dsa_app.closeEvent(first_event)

    assert dsa_app.worker.stop_called is True
    assert dsa_app.thread.quit_called is True
    assert first_event.isAccepted() is False
    assert dsa_app.status_timer.isActive() is False

    dsa_app.thread.running = False
    second_event = QCloseEvent()
    dsa_app.closeEvent(second_event)

    assert second_event.isAccepted() is True


def _write_test_eeg_csv(path, seconds=10):
    rate = config.SAMPLE_RATE_HZ
    t = np.arange(int(seconds * rate)) / rate
    values = 50 * np.sin(2 * np.pi * 10 * t)
    lines = []
    for ti, vi in zip(t, values):
        sec = 3600.0 + ti
        lines.append(f"{int(sec // 3600):02d}:{int(sec % 3600 // 60):02d}:{sec % 60:07.4f},{vi:.3f}")
    path.write_text("\n".join(lines) + "\n")


def test_load_eeg_csv_file_fills_dsa_and_blocks_live_data(dsa_app, tmp_path):
    path = tmp_path / "eeg.csv"
    _write_test_eeg_csv(path)

    assert dsa_app._load_eeg_file(str(path)) is True

    assert dsa_app._file_mode is True
    assert dsa_app.dsa_view.dsa_buffer.t0 is not None
    filled = len(dsa_app.dsa_view.dsa_buffer.buffers[config.TIME_RESOLUTION]["data"])
    dsa_app._on_new_dsa_column(time.time(), np.ones(len(config.FREQ_BINS)), 1)
    assert len(dsa_app.dsa_view.dsa_buffer.buffers[config.TIME_RESOLUTION]["data"]) == filled

    dsa_app._return_to_live()
    assert dsa_app._file_mode is False
    assert dsa_app.dsa_view.dsa_buffer.t0 is None


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


def test_dropping_csv_on_dsa_view_emits_file_path(dsa_app, tmp_path, qtbot):
    from PySide6.QtCore import QMimeData, QPointF, QUrl, Qt
    from PySide6.QtGui import QDropEvent

    path = tmp_path / "eeg.csv"
    path.write_text("20:12:06.2613,-82\n")
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    event = QDropEvent(QPointF(10, 10), Qt.DropAction.CopyAction, mime,
                       Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)

    with qtbot.waitSignal(dsa_app.dsa_view.eeg_file_dropped, timeout=1000) as blocker:
        dsa_app.dsa_view.dropEvent(event)

    assert blocker.args == [str(path)]


def test_load_eeg_file_reports_errors_and_stays_live(dsa_app, tmp_path, monkeypatch):
    path = tmp_path / "eeg.csv"
    path.write_text("garbage\n")
    shown = []
    monkeypatch.setattr(dsa_app, "_show_message", lambda title, text: shown.append(title))

    assert dsa_app._load_eeg_file(str(path)) is False
    assert dsa_app._file_mode is False
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


def test_loaded_and_played_eeg_files_are_not_saved_to_dsa_data(dsa_app, tmp_path, monkeypatch):
    saved = []
    monkeypatch.setattr("main.Output.save_psd_to_csv", lambda *args: saved.append(args))
    path = tmp_path / "eeg.csv"
    _write_test_eeg_csv(path, seconds=5)

    assert dsa_app._load_eeg_file(str(path)) is True
    assert dsa_app._play_eeg_file(str(path)) is True
    dsa_app.player._emit_chunk(0, len(dsa_app.player._timestamps))

    assert dsa_app.dsa_view.dsa_buffer.t0 is not None
    assert saved == []


def test_load_eeg_file_shorter_than_window_shows_message_and_stays_live(dsa_app, tmp_path, monkeypatch):
    path = tmp_path / "eeg.csv"
    _write_test_eeg_csv(path, seconds=1)
    shown = []
    monkeypatch.setattr(dsa_app, "_show_message", lambda title, text: shown.append(title))

    assert dsa_app._load_eeg_file(str(path)) is False
    assert dsa_app._file_mode is False
    assert shown == ["Load EEG File"]


def test_load_eeg_file_zooms_to_recording_length(dsa_app, tmp_path, monkeypatch):
    path = tmp_path / "eeg.csv"
    _write_test_eeg_csv(path, seconds=90)
    zooms = []
    monkeypatch.setattr(dsa_app, "_on_zoom_change", zooms.append)

    assert dsa_app._load_eeg_file(str(path)) is True

    assert zooms == [pytest.approx(1.5, abs=0.01)]
    assert "File: eeg.csv" in dsa_app.windowTitle()


def test_play_eeg_file_reports_errors_and_stays_live(dsa_app, tmp_path, monkeypatch):
    path = tmp_path / "eeg.csv"
    path.write_text("garbage\n")
    shown = []
    monkeypatch.setattr(dsa_app, "_show_message", lambda title, text: shown.append(title))

    assert dsa_app._play_eeg_file(str(path)) is False
    assert dsa_app._file_mode is False
    assert dsa_app.player.is_playing is False
    assert shown == ["Load Error"]


def test_file_mode_routes_eeg_samples_from_file_only(dsa_app, monkeypatch):
    appended = []
    monkeypatch.setattr(dsa_app.eeg_view, "append_sample", appended.append)

    dsa_app._enter_file_mode("eeg.csv")
    dsa_app._on_new_samples([1.0, 2.0])
    dsa_app._on_file_samples([3.0])
    assert appended == [3.0]

    dsa_app._return_to_live()
    dsa_app._on_file_samples([4.0])
    dsa_app._on_new_samples([5.0])
    assert appended == [3.0, 5.0]


def test_return_to_live_stops_playback_and_ignores_late_file_columns(dsa_app, tmp_path):
    path = tmp_path / "eeg.csv"
    _write_test_eeg_csv(path, seconds=3)
    assert dsa_app._play_eeg_file(str(path)) is True

    dsa_app._return_to_live()

    assert dsa_app.player.is_playing is False
    assert dsa_app.windowTitle() == main_module.WINDOW_TITLE
    dsa_app._on_file_dsa_column(time.time(), np.ones(len(config.FREQ_BINS)), 1)
    assert dsa_app.dsa_view.dsa_buffer.t0 is None
