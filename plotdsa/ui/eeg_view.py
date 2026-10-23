import time
from collections import deque

import numpy as np
import pyqtgraph as pg
from pyqtgraph import GridItem
from PySide6.QtCore import QEvent, QSize, Qt, QTimer
from scipy.signal import lfilter

from .. import config
from ..core.calculations import DSACalculator
from .views import create_settings_gear_icon, set_axis_label, set_uniform_left_axis_width


# 7.5 mm/sekunde 15 mm/sekunde eeg view skalieren mit application window. 5 microvolt pro millimeter. 27 zoll pc. einstellung in system settings
class EEGView(pg.PlotWidget):
    """Real-time circular EEG display with smooth sweep line + gap (optimized)."""

    RENDER_HZ = 20

    def __init__(self, user_config, on_config_change=None):
        super().__init__()
        self.user_config = user_config
        self.on_config_change = on_config_change

        self.getPlotItem().setContentsMargins(10, 10, 60, 8)
        set_axis_label(self.plotItem, "left", "EEG", units="\N{MICRO SIGN}V")
        set_axis_label(self.plotItem, "bottom", "Time", units="s")
        set_uniform_left_axis_width(self.plotItem)
        self.setMinimumHeight(config.MIN_EEG_HEIGHT)
        self.showGrid(x=False, y=False)
        self.grid = GridItem()
        self.addItem(self.grid)
        self.grid.setZValue(-1)
        self.grid.setTickSpacing(
            x=[1.0],
            y=[50.0],
        )
        self.grid.setTextPen(None)
        self.getAxis("bottom").setTickPen(None)
        self.getAxis("left").setTickPen(None)

        self.setMenuEnabled(False)
        self.setClipToView(True)
        self.setDownsampling(auto=False, ds=1, mode="subsample")
        self.setMouseEnabled(False, False)
        self.setInteractive(False)

        self.curve_a = self.plot(pen=pg.mkPen((0, 200, 255), width=1))
        self.missing_curve_a = self.plot(pen=pg.mkPen((255, 80, 80), width=2))
        self.missing_curve_a.setDownsampling(auto=False, ds=1, method="subsample")

        self.update_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("w", style=Qt.DashLine))
        self.addItem(self.update_line)
        self._init_settings_controls()

        self.N = int(config.EEG_VIEW_WINDOW_SEC * config.SAMPLE_RATE_HZ)
        self.display = np.full(self.N, np.nan, dtype=np.float32)
        self.display_head = -1
        self.history = deque(maxlen=5 * 60 * config.SAMPLE_RATE_HZ)

        self.x = np.linspace(
            0,
            config.EEG_VIEW_WINDOW_SEC,
            self.N,
            endpoint=False,
            dtype=np.float32,
        )

        self.gap_samples = int(0.05 * config.EEG_VIEW_WINDOW_SEC * config.SAMPLE_RATE_HZ)

        self._pending = deque()
        self._sample_period = 1.0 / config.SAMPLE_RATE_HZ
        self._last_rendered_head = -1
        self.seconds_visible = config.EEG_VIEW_WINDOW_SEC
        self._init_view_filter()

        self.setXRange(0, config.EEG_VIEW_WINDOW_SEC, padding=0)
        self._apply_y_range()

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.timeout.connect(self._render_frame_stable)
        self._timer.start(int(1000 / self.RENDER_HZ))

    def _init_settings_controls(self):
        from PySide6.QtWidgets import QApplication, QFrame, QLabel, QStyle, QToolButton, QVBoxLayout

        self.sweep_buttons = {}
        self.amplitude_buttons = {}
        self.is_paused = False
        self._app_event_filter_installed = False

        self.settings_button = QToolButton(self.viewport())
        self.settings_button.setCursor(Qt.PointingHandCursor)
        self.settings_button.setFixedSize(32, 32)
        self.settings_button.setIcon(create_settings_gear_icon())
        self.settings_button.setIconSize(QSize(18, 18))
        self.settings_button.setStyleSheet(
            """
            QToolButton {
                background-color: rgba(60, 60, 60, 170);
                border: 1px solid rgba(255, 255, 255, 45);
                border-radius: 16px;
            }
            QToolButton:pressed {
                background-color: rgba(100, 100, 100, 255);
            }
        """
        )
        self.settings_button.clicked.connect(self._toggle_settings_popup)

        self.pause_button = QToolButton(self.viewport())
        self.pause_button.setCursor(Qt.PointingHandCursor)
        self.pause_button.setCheckable(True)
        self.pause_button.setFixedSize(32, 32)
        self.pause_button.setStyleSheet(
            """
            QToolButton {
                background-color: rgba(60, 60, 60, 170);
                border: 1px solid rgba(255, 255, 255, 45);
                border-radius: 16px;
            }
            QToolButton:pressed {
                background-color: rgba(100, 100, 100, 255);
            }
            QToolButton:checked {
                background-color: rgba(0, 150, 220, 220);
            }
        """
        )
        self.pause_button.clicked.connect(self._toggle_pause)
        self._pause_icon = self.style().standardIcon(QStyle.SP_MediaPause)
        self._play_icon = self.style().standardIcon(QStyle.SP_MediaPlay)
        self._sync_pause_button()

        self.settings_popup = QFrame(self.viewport())
        self.settings_popup.setObjectName("eegSettingsPopup")
        self.settings_popup.hide()
        self.settings_popup.setStyleSheet(
            f"""
            QFrame#eegSettingsPopup {{
                background-color: rgba(28, 28, 28, 245);
                border: 1px solid rgba(255, 255, 255, 35);
                border-radius: 8px;
            }}
            QLabel {{
                color: white;
                font-size: {max(config.FONT_SIZE - 2, 8)}px;
                font-weight: 600;
            }}
        """
        )

        popup_layout = QVBoxLayout(self.settings_popup)
        popup_layout.setContentsMargins(10, 2, 10, 6)
        popup_layout.setSpacing(8)

        popup_layout.addWidget(QLabel("Sweep Speed"))
        popup_layout.addLayout(
            self._create_settings_button_row(
                config.EEG_SWEEP_SPEED_OPTIONS,
                self.sweep_buttons,
                self._set_eeg_sweep_speed,
            )
        )

        popup_layout.addWidget(QLabel("Y Range"))
        popup_layout.addLayout(
            self._create_settings_button_row(
                config.EEG_Y_RANGE_OPTIONS,
                self.amplitude_buttons,
                self._set_eeg_y_max,
            )
        )

        self._sync_sweep_buttons()
        self._sync_amplitude_buttons()
        self._update_settings_button_pos()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
            self._app_event_filter_installed = True

    def _create_settings_button_row(self, options, button_map, handler):
        from PySide6.QtWidgets import QHBoxLayout, QPushButton

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        for option in options:
            label = f"{option:g}" if isinstance(option, float) else str(option)
            button = QPushButton(label)
            button.setCheckable(True)
            button.setFixedSize(50, 32)
            button.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: rgba(60, 60, 60, 170);
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: {max(config.FONT_SIZE - 3, 8)}px;
                }}
                QPushButton:checked {{
                    background-color: rgba(0, 150, 220, 220);
                }}
            """
            )
            button.clicked.connect(lambda _, value=option: handler(value))
            layout.addWidget(button)
            button_map[option] = button

        return layout

    def _toggle_settings_popup(self):
        if self.settings_popup.isVisible():
            self.settings_popup.hide()
            return

        self._sync_sweep_buttons()
        self._sync_amplitude_buttons()
        self._position_settings_popup()
        self.settings_popup.show()
        self.settings_popup.raise_()

    def _position_settings_popup(self):
        self.settings_popup.adjustSize()
        popup_x = max(0, self.settings_button.x() - self.settings_popup.width() - 6)
        popup_y = max(0, self.settings_button.y())
        self.settings_popup.move(popup_x, popup_y)

    def _toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self._pending.clear()
        self._sync_pause_button()

    def _init_view_filter(self):
        self._view_filter = DSACalculator(self.user_config.window_sec)
        self._view_filter_b = np.asarray(self._view_filter.bp_b, dtype=np.float32)
        self._view_filter_settle_samples = max(0, (len(self._view_filter_b) - 1) // 2)
        self._reset_view_filter_state()

    def _reset_view_filter_state(self):
        self._view_filter_zi = np.zeros(len(self._view_filter_b) - 1, dtype=np.float32)
        self._view_filter_settle_remaining = self._view_filter_settle_samples

    def _sync_pause_button(self):
        self.pause_button.blockSignals(True)
        self.pause_button.setChecked(self.is_paused)
        self.pause_button.setIcon(self._play_icon if self.is_paused else self._pause_icon)
        self.pause_button.setIconSize(QSize(18, 18))
        self.pause_button.setToolTip("Resume EEG" if self.is_paused else "Pause EEG")
        self.pause_button.blockSignals(False)

    def _update_settings_button_pos(self):
        if not hasattr(self, "settings_button"):
            return

        x = max(0, self.viewport().width() - self.settings_button.width() - 12)
        self.settings_button.move(x, 8)
        if hasattr(self, "pause_button"):
            self.pause_button.move(x, self.settings_button.y() + self.settings_button.height() + 6)
        if self.settings_popup.isVisible():
            self._position_settings_popup()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress and self.settings_popup.isVisible():
            from PySide6.QtWidgets import QApplication

            target = QApplication.widgetAt(event.globalPosition().toPoint())
            if not self._settings_click_is_inside(target):
                self.settings_popup.hide()

        return super().eventFilter(obj, event)

    def _settings_click_is_inside(self, target):
        if target is None:
            return False

        return (
            target is self.settings_popup
            or target is self.settings_button
            or target is self.pause_button
            or self.settings_popup.isAncestorOf(target)
            or self.settings_button.isAncestorOf(target)
            or self.pause_button.isAncestorOf(target)
        )

    def _apply_y_range(self):
        max_uv = self.user_config.eeg_uv_range_max
        self.setYRange(-max_uv, max_uv, padding=0)

    def _sync_amplitude_buttons(self):
        selected = self.user_config.eeg_uv_range_max
        for amplitude, button in self.amplitude_buttons.items():
            button.blockSignals(True)
            button.setChecked(amplitude == selected)
            button.blockSignals(False)

    def _sync_sweep_buttons(self):
        selected = self.user_config.eeg_sweep_speed
        for speed, button in self.sweep_buttons.items():
            button.blockSignals(True)
            button.setChecked(speed == selected)
            button.blockSignals(False)

    def _set_eeg_y_max(self, max_uv):
        if max_uv == self.user_config.eeg_uv_range_max:
            self._sync_amplitude_buttons()
            return

        if self.on_config_change is None:
            self.user_config = self.user_config.update(eeg_uv_range_max=max_uv)
            self._apply_y_range()
            self._sync_amplitude_buttons()
            return

        self.on_config_change(self.user_config.update(eeg_uv_range_max=max_uv))

    def _set_eeg_sweep_speed(self, sweep_speed):
        if sweep_speed == self.user_config.eeg_sweep_speed:
            self._sync_sweep_buttons()
            return

        if self.on_config_change is None:
            self.user_config = self.user_config.update(eeg_sweep_speed=sweep_speed)
            self._sync_sweep_buttons()
            self._update_time_scale()
            return

        self.on_config_change(self.user_config.update(eeg_sweep_speed=sweep_speed))

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._update_time_scale)

    def _update_time_scale(self):
        if not hasattr(self, "plotItem"):
            return

        win = self.window()
        if win is None:
            return

        handle = win.windowHandle()
        if handle is None:
            return

        screen = handle.screen()
        dpi = screen.logicalDotsPerInch() if screen else 96

        px_per_sec = self.user_config.eeg_sweep_speed * dpi / 25.4

        try:
            view_box = self.getViewBox()
        except AttributeError:
            return
        if view_box is None:
            return

        width_px = view_box.sceneBoundingRect().width()
        if width_px <= 0:
            return

        seconds_visible = width_px / px_per_sec
        new_N = int(seconds_visible * config.SAMPLE_RATE_HZ)

        if new_N <= 10:
            return

        if new_N != self.N:
            if self.display_head >= 0:
                old_ratio = (self.display_head + 1) / self.N
            else:
                old_ratio = 0.0

            new_head = int(old_ratio * new_N) - 1
            new_head = np.clip(new_head, -1, new_N - 1)

            new_display = np.full(new_N, np.nan, dtype=np.float32)

            if self.history:
                h_list = list(self.history)
                h_arr = np.array(h_list, dtype=np.float32)
                count = min(len(h_arr), new_N)
                to_copy = h_arr[-count:]
                start_idx = (new_head - count + 1) % new_N

                if start_idx + count <= new_N:
                    new_display[start_idx : start_idx + count] = to_copy
                else:
                    first_part = new_N - start_idx
                    new_display[start_idx:] = to_copy[:first_part]
                    new_display[: count - first_part] = to_copy[first_part:]

            self.display = new_display
            self.display_head = new_head
            self.N = new_N
            self._last_rendered_head = -1
            self.gap_samples = int(0.05 * self.N)

        self.seconds_visible = seconds_visible
        self.setXRange(0, seconds_visible, padding=0)

        self.x = np.linspace(
            0,
            seconds_visible,
            self.N,
            endpoint=False,
            dtype=np.float32,
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_time_scale()
        self._update_settings_button_pos()

    def closeEvent(self, event):
        self._timer.stop()
        if getattr(self, "_app_event_filter_installed", False):
            from PySide6.QtWidgets import QApplication

            app = QApplication.instance()
            if app is not None:
                app.removeEventFilter(self)
            self._app_event_filter_installed = False
        super().closeEvent(event)

    def clear_data(self):
        self.history.clear()
        self.display.fill(np.nan)
        self.display_head = -1
        self._pending.clear()
        self._last_rendered_head = -1
        self._reset_view_filter_state()
        self.update_line.setPos(0)
        self.curve_a.clear()
        self.missing_curve_a.clear()

    def apply_config(self, user_config):
        self.user_config = user_config
        self._apply_y_range()
        self._sync_sweep_buttons()
        self._sync_amplitude_buttons()
        self._update_time_scale()

    def append_sample(self, val):
        if val is None or self.is_paused:
            return

        sample = float(val)
        if not np.isfinite(sample):
            sample = np.nan
            self._reset_view_filter_state()
        else:
            filtered_sample, self._view_filter_zi = lfilter(
                self._view_filter_b,
                [1.0],
                np.asarray([sample], dtype=np.float32),
                zi=self._view_filter_zi,
            )
            if self._view_filter_settle_remaining > 0:
                self._view_filter_settle_remaining -= 1
                sample = np.nan
            else:
                sample = float(filtered_sample[0])

        now = time.perf_counter()
        last = self._pending[-1][0] if self._pending else now
        scheduled = max(last + self._sample_period, now)
        self._pending.append((scheduled, sample))

    def _build_visible_curve_data(self, head):
        visible_mask = np.zeros(self.N, dtype=bool)
        gap_end = (head + self.gap_samples) % self.N
        if head < gap_end:
            visible_mask[gap_end:] = True
            visible_mask[: head + 1] = True
        else:
            visible_mask[gap_end: head + 1] = True

        visible_values = np.full(self.N, np.nan, dtype=np.float32)
        visible_values[visible_mask] = self.display[visible_mask]

        missing_mask = visible_mask & np.isnan(self.display)
        valid_values = np.array(visible_values, copy=True)
        valid_values[missing_mask] = np.nan
        return valid_values, missing_mask

    def _build_missing_curve_segments(self, missing_mask):
        sample_width = np.float32(self.seconds_visible / self.N)
        half_sample_width = np.float32(sample_width / 2.0)
        missing_x = []
        missing_y = []
        idx = 0

        while idx < self.N:
            if not missing_mask[idx]:
                idx += 1
                continue

            run_start = idx
            while idx < self.N and missing_mask[idx]:
                idx += 1
            run_end = idx - 1

            missing_x.append(max(0.0, self.x[run_start] - half_sample_width))
            missing_y.append(0.0)
            missing_x.append(min(self.seconds_visible, self.x[run_end] + half_sample_width))
            missing_y.append(0.0)
            missing_x.append(np.nan)
            missing_y.append(np.nan)

        if not missing_x:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        return (
            np.asarray(missing_x, dtype=np.float32),
            np.asarray(missing_y, dtype=np.float32),
        )

    def _render_frame_stable(self):
        if not self._pending and self.display_head < 0:
            return

        now = time.perf_counter()
        changed = False

        while self._pending and self._pending[0][0] <= now:
            _, v = self._pending.popleft()
            self.display_head = (self.display_head + 1) % self.N
            self.display[self.display_head] = v
            self.history.append(v)
            changed = True

        if self._pending:
            next_wall = self._pending[0][0]
            prev_wall = next_wall - self._sample_period
            frac = (now - prev_wall) / self._sample_period
            if frac < 0.0:
                frac = 0.0
            elif frac > 1.0:
                frac = 1.0
            interp_head = (self.display_head + frac) % self.N
        else:
            if self.display_head < 0:
                return
            interp_head = float(self.display_head)

        self.update_line.setPos((interp_head / self.N) * self.seconds_visible)

        head = int(interp_head) % self.N
        if not changed and head == self._last_rendered_head:
            return

        self._last_rendered_head = head
        valid_values, missing_mask = self._build_visible_curve_data(head)
        missing_x, missing_y = self._build_missing_curve_segments(missing_mask)
        self.curve_a.setData(self.x, valid_values, connect="finite")
        if len(missing_x) == 0:
            self.missing_curve_a.clear()
        else:
            self.missing_curve_a.setData(missing_x, missing_y, connect="finite")


__all__ = ["EEGView"]
