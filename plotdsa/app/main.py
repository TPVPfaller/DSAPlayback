"""
IEC 62304 – Class B

EEG Density Spectral Array Viewer
"""

import os
import sys
import ctypes

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QMessageBox,
    QLabel,
    QFileDialog,
    QInputDialog,
)
from PySide6.QtCore import QEvent, QPoint, QSettings, QSize, QTimer, Qt
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPalette, QPixmap, QPolygon
import qdarktheme
import pyqtgraph as pg

if __package__ in (None, ""):
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from plotdsa.io.eeg_file import read_eeg_csv, compute_dsa_columns, is_single_column
    from plotdsa.app.player import EEGFilePlayer
    from plotdsa.config import UserConfig
    from plotdsa.ui.dsa_settings import SettingsDialog
    from plotdsa.ui.playback_bar import PlaybackBar
    from plotdsa.ui.topbar import TopBar
    from plotdsa.ui.views import DSAView, PSDView, EEGView
    from plotdsa import config
else:
    from ..io.eeg_file import read_eeg_csv, compute_dsa_columns, is_single_column
    from .player import EEGFilePlayer
    from ..config import UserConfig
    from ..ui.dsa_settings import SettingsDialog
    from ..ui.playback_bar import PlaybackBar
    from ..ui.topbar import TopBar
    from ..ui.views import DSAView, PSDView, EEGView
    from .. import config


def _enable_windows_darkmode():
    if '-platform' not in sys.argv:
        sys.argv += ['-platform', 'windows:darkmode=2']


def _colorref_from_rgb(red: int, green: int, blue: int) -> int:
    return red | (green << 8) | (blue << 16)


def _apply_windows_dark_titlebar(hwnd: int, red: int, green: int, blue: int):
    if sys.platform != "win32":
        return

    DWMWA_USE_IMMERSIVE_DARK_MODE = 20
    DWMWA_CAPTION_COLOR = 35
    DWMWA_TEXT_COLOR = 36

    try:
        dwmapi = ctypes.windll.dwmapi
    except AttributeError:
        return

    dark_mode_enabled = ctypes.c_int(1)
    caption_color = ctypes.c_uint32(_colorref_from_rgb(red, green, blue))
    is_dark = (0.2126 * red + 0.7152 * green + 0.0722 * blue) < 140
    text_color = ctypes.c_uint32(
        _colorref_from_rgb(255, 255, 255) if is_dark else _colorref_from_rgb(0, 0, 0)
    )

    dwmapi.DwmSetWindowAttribute(
        hwnd,
        DWMWA_USE_IMMERSIVE_DARK_MODE,
        ctypes.byref(dark_mode_enabled),
        ctypes.sizeof(dark_mode_enabled),
    )
    dwmapi.DwmSetWindowAttribute(
        hwnd,
        DWMWA_CAPTION_COLOR,
        ctypes.byref(caption_color),
        ctypes.sizeof(caption_color),
    )
    dwmapi.DwmSetWindowAttribute(
        hwnd,
        DWMWA_TEXT_COLOR,
        ctypes.byref(text_color),
        ctypes.sizeof(text_color),
    )


WINDOW_TITLE = "EEG Density Spectral Array"
LAST_EEG_DIR_KEY = "last_eeg_file_dir"
DEFAULT_SINGLE_COLUMN_RATE_HZ = 400.0
EEG_FILE_DIR = "C:\\temp\\VSCaptureWave\\Entropy_Data"


def _create_play_icon(size=16):
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("white"))
    painter.drawPolygon(QPolygon([QPoint(3, 2), QPoint(14, 8), QPoint(3, 14)]))
    painter.end()
    return QIcon(pixmap)


class DSAApplication(QMainWindow):
    """Main application wiring together UI, EEG file playback and views."""

    def __init__(self):
        super().__init__()
        self._themed_window_ids = set()
        self._file_name = None
        self._settings = QSettings("PlotDSA", "PlotDSA")

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1000, 650)
        self.setMinimumSize(config.MIN_WINDOW_WIDTH, config.MIN_WINDOW_HEIGHT)

        self.user_config = UserConfig()
        self._init_ui()
        self._init_player()
        self._init_timers()
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, event):
        if (
                event.type() == QEvent.Type.Show
                and isinstance(obj, QWidget)
                and obj.isWindow()
        ):
            window_id = int(obj.winId())
            if window_id not in self._themed_window_ids:
                background = obj.palette().color(QPalette.ColorRole.Window)
                _apply_windows_dark_titlebar(
                    window_id,
                    background.red(),
                    background.green(),
                    background.blue(),
                )
                self._themed_window_ids.add(window_id)

        return super().eventFilter(obj, event)

    def _init_ui(self):
        self.dsa_view = DSAView(self.user_config, self._on_config_change, self._on_zoom_change)
        self.psd_view = PSDView(self.user_config)
        self.eeg_view = EEGView(self.user_config, self._on_config_change)

        self.topbar = TopBar(self.user_config, self._on_config_change, self._on_zoom_change, self.dsa_view.pan,
                             self.dsa_view.calibrate)
        self.topbar.screenshot_widgets = lambda: [self.dsa_view, self.psd_view, self.eeg_view]
        self.playback_bar = PlaybackBar()
        self.playback_bar.hide()  # only shown while a file is played
        self._create_menu()

        container = QWidget()
        self.layout = QVBoxLayout(container)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.disclaimer_label = QLabel("Nur zu Lehrzwecken", self)
        self.disclaimer_label.setStyleSheet("color: red; font-weight: bold; font-size: 22px;")
        self.disclaimer_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self.disclaimer_label.setFixedHeight(32)

        self.layout.addSpacing(23)
        self.layout.addWidget(self.topbar)
        self.layout.addWidget(self.playback_bar)
        self.layout.addWidget(self.dsa_view)
        self.layout.addWidget(self.psd_view)
        self.layout.addWidget(self.eeg_view)

        self.layout.setStretchFactor(self.topbar, 0)
        self.layout.setStretchFactor(self.playback_bar, 0)
        self.layout.setStretchFactor(self.dsa_view, 1)
        self.layout.setStretchFactor(self.psd_view, 1)
        self.layout.setStretchFactor(self.eeg_view, 1)

        self.setCentralWidget(container)
        self._position_disclaimer_label()

        self._sync_view_visibility()

        self.topbar.live_btn.clicked.connect(self.dsa_view.jump_to_live)
        self.dsa_view.eeg_file_dropped.connect(self._open_eeg_file)
        self.topbar.sync_slider(self.user_config.display_minutes)
        pg.setConfigOptions(antialias=False)

    def _position_disclaimer_label(self):
        self.disclaimer_label.setGeometry(
            0,
            self.menuBar().height() - 8,
            self.width(),
            self.disclaimer_label.height(),
        )
        self.disclaimer_label.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "disclaimer_label"):
            self._position_disclaimer_label()

    # ------------------ EEG file load / playback ------------------ #
    def _init_player(self):
        self.player = EEGFilePlayer(self.user_config, self)
        self.player.new_dsa_column.connect(self._on_file_dsa_column)
        self.player.new_samples.connect(self._on_file_samples)
        self.player.finished.connect(self._on_playback_finished)
        self.player.position_changed.connect(self.playback_bar.set_position)
        self.playback_bar.play_pause_clicked.connect(self._toggle_playback)
        self.playback_bar.seek_requested.connect(self._seek_playback)
        self.playback_bar.speed_changed.connect(self._set_playback_speed)

    def _on_open_eeg_file_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load EEG File", self._eeg_start_dir(), "CSV files (*.csv);;All files (*)"
        )
        if path:
            self._open_eeg_file(path)

    def _eeg_start_dir(self):
        """Folder of the last opened EEG file, else the default EEG data folder."""
        last_dir = self._settings.value(LAST_EEG_DIR_KEY, "", type=str)
        if last_dir and os.path.isdir(last_dir):
            return last_dir
        return EEG_FILE_DIR if os.path.isdir(EEG_FILE_DIR) else ""

    def _remember_eeg_dir(self, path):
        folder = os.path.dirname(os.path.abspath(path))
        if os.path.isdir(folder):
            self._settings.setValue(LAST_EEG_DIR_KEY, folder)

    def _open_eeg_file(self, path):
        choice = self._ask_load_or_play(path)
        if choice is not None:
            self._remember_eeg_dir(path)
        if choice == "load":
            self._load_eeg_file(path)
        elif choice == "play":
            self._play_eeg_file(path)

    def _ask_load_or_play(self, path):
        msg = QMessageBox(self)
        msg.setWindowTitle("Open EEG File")
        msg.setText(
            f"How do you want to open the file?\n\n"
        )
        load_button = msg.addButton("Load", QMessageBox.ButtonRole.AcceptRole)
        play_button = msg.addButton("Play", QMessageBox.ButtonRole.AcceptRole)
        play_button.setIcon(_create_play_icon())
        play_button.setIconSize(QSize(14, 14))
        msg.addButton(QMessageBox.StandardButton.Cancel)
        msg.setDefaultButton(load_button)
        self._style_message_box(msg)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked is load_button:
            return "load"
        if clicked is play_button:
            return "play"
        return None

    def _ask_sample_rate(self, path):
        """Return (ok, rate): asks for the sampling rate if the file has only a value column."""
        if not is_single_column(path):
            return True, None
        rate, ok = QInputDialog.getDouble(
            self,
            "Sampling Rate",
            "The file contains only one column.\nSampling rate (Hz):",
            DEFAULT_SINGLE_COLUMN_RATE_HZ,
            1.0,
            100000.0,
            0,  # whole Hz only, so no decimal separator is shown
        )
        return ok, rate

    def _load_eeg_file(self, path):
        ok, sample_rate_hz = self._ask_sample_rate(path)
        if not ok:
            return False
        self.player.stop()
        self.playback_bar.hide()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            timestamps, values = read_eeg_csv(path, sample_rate_hz=sample_rate_hz)
            columns = compute_dsa_columns(timestamps, values, self.user_config)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            self._show_message("Load Error", f"Failed to load EEG file: {e}")
            return False
        QApplication.restoreOverrideCursor()

        if not columns:
            self._show_message("Load EEG File", "The file is too short to compute a DSA.")
            return False

        self._enter_file_mode(path)
        for ts, psd, steps in columns:
            self._append_dsa_steps(ts, psd, steps)

        # Zoom so the whole recording is visible.
        min_minutes, max_minutes = config.DISPLAY_MINUTES_BOUNDS
        duration_minutes = (timestamps[-1] - timestamps[0]) / 60.0
        self._on_zoom_change(float(min(max(duration_minutes, min_minutes), max_minutes)))
        self.dsa_view.update()
        self.dsa_view.jump_to_live()
        self._set_title_status(f"File: {self._file_name}")
        return True

    def _play_eeg_file(self, path):
        ok, sample_rate_hz = self._ask_sample_rate(path)
        if not ok:
            return False
        self.player.stop()
        try:
            timestamps, values = read_eeg_csv(path, sample_rate_hz=sample_rate_hz)
        except Exception as e:
            self._show_message("Load Error", f"Failed to load EEG file: {e}")
            return False

        self._enter_file_mode(path)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.player.start(timestamps, values)  # precomputes the whole DSA
        finally:
            QApplication.restoreOverrideCursor()
        self.playback_bar.set_speed(1.0)
        self.playback_bar.set_duration(self.player.duration)
        self.playback_bar.set_window_width(self.dsa_view._visible_width_sec())
        self._update_overview()
        self.playback_bar.set_playing(True)
        self.playback_bar.show()
        self._set_title_status(f"Playing {self._file_name}")
        return True

    def _toggle_playback(self):
        if self.player.is_playing:
            self.player.pause()
            self._set_title_status(f"Paused {self._file_name}")
        else:
            self.player.resume()
            self._set_title_status(f"Playing {self._file_name}")
        self.playback_bar.set_playing(self.player.is_playing)

    def _update_overview(self):
        """Show the whole precomputed DSA in the playback bar, cropped like the main DSA."""
        n_freq = len(self.dsa_view.freq_bins)
        self.playback_bar.overview.set_dsa(
            self.player.column_starts / config.SAMPLE_RATE_HZ,
            [psd[:n_freq] for _, psd, _ in self.player.dsa_columns],
            self.player.duration,
            self.dsa_view.lut,
            (self.user_config.psd_db_min, self.user_config.psd_db_max),
        )

    def _set_playback_speed(self, speed):
        self.player.set_speed(speed)
        self.eeg_view.set_playback_speed(speed)

    def _seek_playback(self, seconds):
        """Jump to `seconds` and rebuild the DSA up to there; the EEG sweep restarts."""
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            columns = self.player.seek(seconds)
            self.dsa_view.clear_data()
            self.eeg_view.clear_data()
            for ts, psd, steps in columns:
                self._append_dsa_steps(ts, psd, steps)
            self.dsa_view.update()
            self.dsa_view.jump_to_live()
            if columns and self.psd_view.isVisible():
                self.psd_view.update(columns[-1][1])
        finally:
            QApplication.restoreOverrideCursor()

    def _enter_file_mode(self, path):
        self._file_name = os.path.basename(path)
        self.dsa_view.clear_data()
        self.eeg_view.clear_data()

    def _clear_data(self):
        self.player.stop()
        self.playback_bar.hide()
        self._file_name = None
        self.dsa_view.clear_data()
        self.eeg_view.clear_data()
        self._set_title_status(None)

    def _on_playback_finished(self):
        self.playback_bar.set_playing(False)
        self._set_title_status(f"Playback finished: {self._file_name}")

    def _set_title_status(self, status):
        self.setWindowTitle(f"{WINDOW_TITLE} - {status}" if status else WINDOW_TITLE)

    def _on_file_dsa_column(self, ts, psd, steps):
        if self.player.is_playing:
            self._show_dsa_column(ts, psd, steps)

    def _on_file_samples(self, samples):
        if not self.player.is_playing or not self.eeg_view.isVisible():
            return
        for value in samples:
            self.eeg_view.append_sample(value)

    def _append_dsa_steps(self, ts, psd, steps):
        for i in range(steps):
            self.dsa_view.append(ts + i * config.TIME_RESOLUTION, psd)

    def _show_message(self, title, text, buttons=QMessageBox.StandardButton.Ok, default_button=None):
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setStandardButtons(buttons)
        if default_button is not None:
            msg.setDefaultButton(default_button)
        self._style_message_box(msg)
        return msg.exec()

    def _style_message_box(self, msg):
        msg.setOption(QMessageBox.Option.DontUseNativeDialog, True)

        msg.setStyleSheet(f"""
            QLabel {{
                font-size: {config.FONT_SIZE}px;
            }}
            QPushButton {{
                min-height: 25px;
                padding: 8px 16px;
                border-radius: 10px;
                font-size: {config.FONT_SIZE}px;
                font-weight: 600;
            }}
            QPushButton[text="Yes"], QPushButton[text="OK"], QPushButton[text="Save and Close"], QPushButton[text="Load Data"], QPushButton[text="&Yes"], QPushButton[text="Load"], QPushButton[text="Play"] {{
                background-color: rgb(0, 120, 215);
                color: white;
                border: 1px solid rgb(0, 120, 215);
            }}
            QPushButton[text="Yes"]:pressed, QPushButton[text="OK"]:pressed, QPushButton[text="Save and Close"]:pressed, QPushButton[text="Load Data"]:pressed, QPushButton[text="&Yes"]:pressed, QPushButton[text="Load"]:pressed, QPushButton[text="Play"]:pressed {{
                background-color: rgb(0, 100, 180);
            }}
            QPushButton[text="No"], QPushButton[text="Cancel"], QPushButton[text="Reset to Defaults"], QPushButton[text="&No"] {{
                background-color: rgb(58, 58, 58);
                color: white;
                border: 1px solid rgb(78, 78, 78);
            }}
            QPushButton[text="No"]:pressed, QPushButton[text="Cancel"]:pressed, QPushButton[text="Reset to Defaults"]:pressed, QPushButton[text="&No"]:pressed {{
                background-color: rgb(92, 92, 92);
            }}
        """)

    def _init_timers(self):
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(250)

    def _create_menu(self):
        menu = self.menuBar().addMenu("&Menu")

        action_settings = QAction("DSA Settings", self)
        action_settings.triggered.connect(self._open_settings)
        menu.addAction(action_settings)

        menu.addSeparator()

        self.action_multitaper = QAction("Use Multitaper", self)
        self.action_multitaper.setCheckable(True)
        self.action_multitaper.setChecked(self.user_config.use_multitaper)
        self.action_multitaper.toggled.connect(self._toggle_multitaper)
        menu.addAction(self.action_multitaper)

        view_menu = self.menuBar().addMenu("&View")

        self.action_show_dsa = self._create_toggle_action(view_menu, "Show DSA", True, self.dsa_view)
        self.action_show_psd = self._create_toggle_action(view_menu, "Show PSD", False, self.psd_view)
        self.action_show_eeg = self._create_toggle_action(view_menu, "Show EEG", True, self.eeg_view)

        action_open_eeg = QAction("Load EEG File (CSV)", self)
        action_open_eeg.triggered.connect(self._on_open_eeg_file_clicked)
        view_menu.addAction(action_open_eeg)


        action_clear_data = QAction("Clear Data", self)
        action_clear_data.triggered.connect(self._confirm_clear_data)
        view_menu.addAction(action_clear_data)

        help_menu = self.menuBar().addMenu("&Help")
        action_info = QAction("Information", self)
        action_info.setShortcut("Ctrl+H")
        action_info.triggered.connect(self._show_information)
        help_menu.addAction(action_info)

    def _show_information(self):
        text = f"""
        <p style="font-size:12pt;">
            - Open an EEG recording (CSV) via the View menu or by dropping it onto the DSA
        </p>
        <p style="font-size:12pt;">
            - Viewer supports up to <b>{int(config.DISPLAY_MINUTES_BOUNDS[1] / 60)} hours</b> of EEG data
        </p>
        <p style="font-size:12pt;">
            - EEG files should be sampled at <b>{config.SAMPLE_RATE_HZ} Hz</b>
        </p>
        <p style="font-size:12pt;">
            - More info: <a href="https://github.com/TPVPfaller/PlotDSA">GitHub</a>
        </p>
        """
        self._show_message("Information", text)

    def _confirm_clear_data(self):
        reply = self._show_message(
            "Confirm data deletion",
            "Are you sure you want to delete all EEG/DSA data?\nThe data can be reloaded by opening the EEG file again.",
            buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            default_button=QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            # Also stops a playing EEG file.
            self._clear_data()

    def _toggle_multitaper(self, checked):
        new_config = self.user_config.update(use_multitaper=checked)
        self._on_config_change(new_config)

    def _create_toggle_action(self, menu, text, default, widget):
        action = QAction(text, self)
        action.setCheckable(True)
        action.setChecked(default)
        action.toggled.connect(widget.setVisible)
        menu.addAction(action)
        return action

    def _sync_view_visibility(self):
        self.dsa_view.setVisible(self.action_show_dsa.isChecked())
        self.psd_view.setVisible(self.action_show_psd.isChecked())
        self.eeg_view.setVisible(self.action_show_eeg.isChecked())

    def _open_settings(self):
        dialog = SettingsDialog(self.user_config, self._on_config_change, self)
        dialog.exec()

    def _show_dsa_column(self, ts, psd, steps):
        self._append_dsa_steps(ts, psd, steps)
        self.dsa_view.update()

        if self.psd_view.isVisible():
            self.psd_view.update(psd)

    def _update_status(self):
        self.topbar.update_jump_live_btn(self.dsa_view)

    def _on_zoom_change(self, display_minutes):
        self.topbar.sync_slider(display_minutes)
        self.topbar.update_jump_live_btn(self.dsa_view)
        self.dsa_view.apply_zoom(display_minutes)
        if hasattr(self, "playback_bar"):
            self.playback_bar.set_window_width(display_minutes * 60.0)

    def _on_config_change(self, new_config):
        self.user_config = new_config

        if hasattr(self, 'action_multitaper'):
            self.action_multitaper.blockSignals(True)
            self.action_multitaper.setChecked(new_config.use_multitaper)
            self.action_multitaper.blockSignals(False)

        self.topbar.apply_config(new_config)
        dsa_recomputed = hasattr(self, "player") and self.player.apply_config(new_config)
        self.dsa_view.apply_config(new_config)
        self.psd_view.apply_config(new_config)
        self.eeg_view.apply_config(new_config)
        if hasattr(self, "playback_bar") and self.playback_bar.isVisibleTo(self):
            self._update_overview()
            if dsa_recomputed:
                self._seek_playback(self.player.position)

        self._update_status()

    def closeEvent(self, event):
        if hasattr(self, "status_timer"):
            self.status_timer.stop()
        if hasattr(self, "player"):
            self.player.stop()
        event.accept()


def main():
    _enable_windows_darkmode()
    qdarktheme.enable_hi_dpi()

    app = QApplication(sys.argv)
    qdarktheme.setup_theme()

    win = DSAApplication()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
