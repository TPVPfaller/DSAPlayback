import datetime
import math
import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph import ColorBarItem
from PySide6.QtCore import QEvent, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QFont

from .. import config
from ..core.buffers import DSABuffer
from .views import (
    AXIS_LABEL_STYLE,
    create_settings_gear_icon,
    create_stepper_icon,
    set_axis_label,
    set_uniform_left_axis_width,
    style_axis,
)


class FrequencyAxis(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_freq = None
        style_axis(self)

    def set_max_freq(self, max_f):
        self.max_freq = float(max_f)
        self.update()

    def tickValues(self, minVal, maxVal, size):
        ticks = super().tickValues(minVal, maxVal, size)

        if self.max_freq is None or not ticks:
            return ticks

        spacing = 5.0

        major_ticks = []
        start = math.ceil(minVal / spacing) * spacing

        v = start
        while v < maxVal:
            major_ticks.append(float(v))
            v += spacing

        major_ticks = [t for t in major_ticks if abs(t - self.max_freq) > spacing * 0.6]
        major_ticks.append(float(self.max_freq))
        major_ticks = sorted(set(t for t in major_ticks if math.isfinite(t)))

        ticks[0] = (spacing, major_ticks)
        return ticks


class DSAView(pg.GraphicsLayoutWidget):
    """Dynamic Spectrum Analysis display with live/pan and pinch-zoom support."""

    eeg_file_dropped = Signal(str)  # path of a raw EEG .csv file dropped onto the view

    def __init__(self, user_config, on_config_change, on_zoom_change):
        super().__init__()

        self.user_config = user_config
        self.on_config_change = on_config_change
        self.on_zoom_change = on_zoom_change
        self.dsa_buffer = DSABuffer()
        self.setMinimumHeight(config.MIN_DSA_HEIGHT)
        self._init_parameters()
        self._init_plot()
        self._init_colormap()
        self._init_colorbar()
        self._init_settings_controls()
        self._init_gestures()
        self.setAcceptDrops(True)
        self.update()

    def _init_parameters(self):
        self.live_mode = True
        self._last_render = time.time()
        self._last_levels = (self.user_config.psd_db_min, self.user_config.psd_db_max)
        self._zoom_factor = 1.0
        self.display_minutes = self.user_config.display_minutes
        self._pan_sec = 0.0
        self._min_zoom = 1.0
        self._max_zoom = 10.0

        self.freq_bins = config.FREQ_BINS[(config.FREQ_BINS <= self.user_config.max_freq_hz)]
        self._dragging = False
        self._last_mouse_pos = None
        self.t0 = datetime.datetime.now().timestamp()

    def _init_plot(self):
        self.time_axis = pg.DateAxisItem("bottom")
        self.freq_axis = FrequencyAxis("left")
        self.plot = self.addPlot(row=0, col=0, axisItems={"bottom": self.time_axis, "left": self.freq_axis})
        set_axis_label(self.plot, "left", "Frequency", units="Hz")
        style_axis(self.time_axis)
        set_uniform_left_axis_width(self.plot)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.showGrid(x=False, y=False)
        self.plot.invertY(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self._update_y_axis()
        self.plot.setContentsMargins(0, 0, 0, 0)
        self.image = pg.ImageItem(axisOrder="col-major", interpolation="linear")
        self.plot.addItem(self.image)

    def _update_y_axis(self):
        max_f = self.user_config.max_freq_hz
        self.plot.setYRange(config.LOWEST_FREQ_HZ, max_f, padding=0)
        self.freq_axis.set_max_freq(max_f)

    def _style_adjust_button(self, button):
        button.setFixedSize(36, 36)
        button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: rgba(60, 60, 60, 150);
                color: white;
                border-radius: 4px;
                font-weight: bold;
                font-size: {config.FONT_SIZE + 6}px;
            }}
            QPushButton:pressed {{
                background-color: rgba(100, 100, 100, 255);
            }}
        """
        )

    def _adjust_max_freq(self, delta):
        new_max_freq = self.user_config.max_freq_hz + delta
        min_f, max_f = config.MAX_FREQ_HZ_BOUNDS
        if min_f <= new_max_freq <= max_f:
            new_config = self.user_config.update(max_freq_hz=new_max_freq)
            self.on_config_change(new_config)

    def _init_colormap(self):
        colors = [
            (10, 10, 50),
            (20, 40, 120),
            (40, 120, 200),
            (80, 200, 220),
            (80, 220, 140),
            (200, 220, 80),
            (240, 160, 40),
            (240, 80, 40),
            (240, 0, 0),
        ]
        self.cmap = pg.ColorMap(np.linspace(0, 1, len(colors)), colors)
        self.lut = self.cmap.getLookupTable(nPts=256, mode="byte")
        self.image.setLookupTable(self.lut)

    def _init_colorbar(self):
        self.colorbar = ColorBarItem(
            values=(self.user_config.psd_db_min, self.user_config.psd_db_max),
            colorMap=self.cmap,
            interactive=False,
            colorMapMenu=False,
        )
        self.colorbar.setImageItem(self.image)
        self.colorbar.setContentsMargins(0, 24, 20, 24)
        self.addItem(self.colorbar, row=0, col=1)

        cb_axis = self.colorbar.getAxis("left")
        style_axis(cb_axis)
        cb_axis.setLabel("Power", units="dB", **AXIS_LABEL_STYLE)

        number_axis = self.colorbar.getAxis("right")
        style_axis(number_axis)

    def _init_settings_controls(self):
        from PySide6.QtWidgets import QApplication, QFrame, QLabel, QToolButton, QVBoxLayout

        self.dsa_value_labels = {}
        self.dsa_stepper_buttons = {}
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

        self.settings_popup = QFrame(self.viewport())
        self.settings_popup.setObjectName("dsaSettingsPopup")
        self.settings_popup.hide()
        self.settings_popup.setStyleSheet(
            f"""
            QFrame#dsaSettingsPopup {{
                background-color: rgba(28, 28, 28, 245);
                border: 1px solid rgba(255, 255, 255, 35);
                border-radius: 8px;
            }}
            QLabel {{
                color: white;
                font-size: {max(config.FONT_SIZE - 2, 8)}px;
                font-weight: 600;
            }}
            QPushButton {{
                background-color: rgba(60, 60, 60, 170);
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: {max(config.FONT_SIZE, 8)}px;
                padding: 0px;
                text-align: center;
            }}
            QPushButton[role="stepper"] {{
                padding: 0px;
                text-align: center;
            }}
            QPushButton:disabled {{
                background-color: rgba(45, 45, 45, 120);
                color: rgba(255, 255, 255, 80);
            }}
            QLabel[role="value"] {{
                border-radius: 4px;
                font-size: {max(config.FONT_SIZE - 1, 8)}px;
                font-weight: bold;
                padding: 2px 8px;
            }}
        """
        )

        popup_layout = QVBoxLayout(self.settings_popup)
        popup_layout.setContentsMargins(10, 4, 10, 4)
        popup_layout.setSpacing(8)
        popup_layout.addLayout(
            self._create_step_control_row(
                "Max Frequency",
                "max_freq_hz",
                lambda value: f"{value} Hz",
                lambda: self._adjust_max_freq(-1),
                lambda: self._adjust_max_freq(1),
            )
        )
        popup_layout.addLayout(
            self._create_step_control_row(
                "Max Power",
                "psd_db_max",
                lambda value: f"{value} dB",
                lambda: self._adjust_psd_level("psd_db_max", -1),
                lambda: self._adjust_psd_level("psd_db_max", 1),
            )
        )
        popup_layout.addLayout(
            self._create_step_control_row(
                "Min Power",
                "psd_db_min",
                lambda value: f"{value} dB",
                lambda: self._adjust_psd_level("psd_db_min", -1),
                lambda: self._adjust_psd_level("psd_db_min", 1),
            )
        )

        self._sync_settings_labels()
        self._update_settings_button_pos()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
            self._app_event_filter_installed = True

    def _create_step_control_row(self, title, field_name, formatter, minus_handler, plus_handler):
        from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        title_label = QLabel(title)
        row.addWidget(title_label)
        row.addStretch()

        stepper_font = QFont()
        stepper_font.setBold(True)
        stepper_font.setPixelSize(max(config.FONT_SIZE + 4, 12))

        minus_button = QPushButton("-")
        minus_button.setProperty("role", "stepper")
        minus_button.setFixedSize(45, 30)
        minus_button.setFont(stepper_font)
        minus_button.setText("")
        minus_button.setIcon(create_stepper_icon("minus"))
        minus_button.setIconSize(QSize(20, 20))
        minus_button.clicked.connect(minus_handler)
        row.addWidget(minus_button)

        value_label = QLabel()
        value_label.setProperty("role", "value")
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setFixedWidth(70)
        row.addWidget(value_label)
        self.dsa_value_labels[field_name] = (value_label, formatter)

        plus_button = QPushButton("+")
        plus_button.setProperty("role", "stepper")
        plus_button.setFixedSize(45, 30)
        plus_button.setFont(stepper_font)
        plus_button.setText("")
        plus_button.setIcon(create_stepper_icon("plus"))
        plus_button.setIconSize(QSize(20, 20))
        plus_button.clicked.connect(plus_handler)
        row.addWidget(plus_button)
        self.dsa_stepper_buttons[field_name] = (minus_button, plus_button)

        return row

    def _sync_settings_labels(self):
        for field_name, (label, formatter) in self.dsa_value_labels.items():
            label.setText(formatter(getattr(self.user_config, field_name)))
        self._sync_step_control_states()

    def _can_adjust_step_control(self, field_name, delta):
        current_value = getattr(self.user_config, field_name)
        if field_name == "max_freq_hz":
            min_bound, max_bound = config.MAX_FREQ_HZ_BOUNDS
        else:
            min_bound, max_bound = getattr(config, f"{field_name.upper()}_BOUNDS")
        new_value = current_value + delta

        if not (min_bound <= new_value <= max_bound):
            return False
        if field_name == "psd_db_min":
            return new_value < self.user_config.psd_db_max
        if field_name == "psd_db_max":
            return new_value > self.user_config.psd_db_min
        return True

    def _sync_step_control_states(self):
        for field_name, (minus_button, plus_button) in self.dsa_stepper_buttons.items():
            minus_button.setEnabled(self._can_adjust_step_control(field_name, -1))
            plus_button.setEnabled(self._can_adjust_step_control(field_name, 1))

    def _toggle_settings_popup(self):
        if self.settings_popup.isVisible():
            self.settings_popup.hide()
            return

        self._sync_settings_labels()
        self._position_settings_popup()
        self.settings_popup.show()
        self.settings_popup.raise_()

    def _position_settings_popup(self):
        self.settings_popup.adjustSize()
        popup_x = max(0, self.settings_button.x() - self.settings_popup.width() - 6)
        popup_y = max(0, self.settings_button.y())
        self.settings_popup.move(popup_x, popup_y)

    def _update_settings_button_pos(self):
        if not hasattr(self, "settings_button"):
            return

        x = max(0, self.viewport().width() - self.settings_button.width() - 12)
        self.settings_button.move(x, 8)
        if self.settings_popup.isVisible():
            self._position_settings_popup()

    def eventFilter(self, obj, event):
        settings_popup = getattr(self, "settings_popup", None)
        if settings_popup is not None and event.type() == QEvent.MouseButtonPress and settings_popup.isVisible():
            from PySide6.QtWidgets import QApplication

            target = QApplication.widgetAt(event.globalPosition().toPoint())
            if not self._settings_click_is_inside(target):
                settings_popup.hide()

        return super().eventFilter(obj, event)

    def _settings_click_is_inside(self, target):
        if target is None:
            return False

        return (
            target is self.settings_popup
            or target is self.settings_button
            or self.settings_popup.isAncestorOf(target)
            or self.settings_button.isAncestorOf(target)
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_settings_button_pos()

    def closeEvent(self, event):
        if getattr(self, "_app_event_filter_installed", False):
            from PySide6.QtWidgets import QApplication

            app = QApplication.instance()
            if app is not None:
                app.removeEventFilter(self)
            self._app_event_filter_installed = False
        super().closeEvent(event)

    def _adjust_psd_level(self, field_name, delta):
        current_value = getattr(self.user_config, field_name)
        min_bound, max_bound = getattr(config, f"{field_name.upper()}_BOUNDS")
        new_value = current_value + delta

        if not (min_bound <= new_value <= max_bound):
            return

        updated_fields = {field_name: new_value}
        if field_name == "psd_db_min" and new_value >= self.user_config.psd_db_max:
            return
        if field_name == "psd_db_max" and new_value <= self.user_config.psd_db_min:
            return

        self.on_config_change(self.user_config.update(**updated_fields))

    def _init_gestures(self):
        self.grabGesture(Qt.PinchGesture)

    def append(self, ts, psd):
        if psd is not None:
            self.dsa_buffer.append(ts, psd)

    def _visible_width_sec(self):
        return self.display_minutes * 60.0

    def _target_resolution(self, visible_width_sec, divisor):
        return max(config.TIME_RESOLUTION, visible_width_sec / divisor)

    def _render_grid(self, visible_width_sec, target_divisor=2160.0):
        target_resolution = self._target_resolution(visible_width_sec, target_divisor)
        actual_res = min(self.dsa_buffer.RESOLUTIONS, key=lambda x: abs(x - target_resolution))
        n_columns = max(1, int(visible_width_sec / actual_res))
        return actual_res, n_columns

    def _live_window_start(self, visible_width_sec):
        if self.dsa_buffer.t0 is None:
            return self.dsa_buffer.get_oldest_timestamp()

        actual_res, n_columns = self._render_grid(visible_width_sec)
        buf = self.dsa_buffer.buffers[actual_res]
        last_slot = buf["last_slot"]
        if last_slot is None:
            return self.dsa_buffer.get_oldest_timestamp()

        oldest_slot = self.dsa_buffer._get_oldest_slot(actual_res) or 0
        start_slot = max(oldest_slot, last_slot - n_columns + 1)
        return self.dsa_buffer.t0 + start_slot * actual_res

    def _pan_limits(self, visible_width_sec):
        min_offset = self.dsa_buffer.get_oldest_timestamp()
        max_offset = max(min_offset, self._live_window_start(visible_width_sec))
        return min_offset, max_offset

    def _current_pan_start(self, visible_width_sec):
        min_offset, max_offset = self._pan_limits(visible_width_sec)
        if self.live_mode:
            return max_offset
        return float(np.clip(self._pan_sec, min_offset, max_offset))

    def _sync_pan_window(self, visible_width_sec):
        min_offset, max_offset = self._pan_limits(visible_width_sec)

        if self.live_mode:
            self._pan_sec = max_offset
            return

        self._pan_sec = float(np.clip(self._pan_sec, min_offset, max_offset))
        if not self._dragging and self._pan_sec >= max_offset - 0.05:
            self.live_mode = True
            self._pan_sec = max_offset

    def _get_visible_dsa_data(self, target_divisor):
        visible_width_sec = self._visible_width_sec()
        self._sync_pan_window(visible_width_sec)

        self.t0, self.dsa_rect, actual_res = self.dsa_buffer.get_view_at(
            width=visible_width_sec,
            height=len(self.freq_bins),
            pan_sec=self._pan_sec,
            target_resolution=self._target_resolution(visible_width_sec, target_divisor),
        )
        return visible_width_sec, self._psd_to_db(self.dsa_rect), actual_res

    def _psd_to_db(self, data):
        data = np.ascontiguousarray(data)
        np.maximum(data, np.finfo(np.float32).eps, out=data)
        np.log10(data, out=data)
        data *= 10
        return data

    def update(self):
        self._last_render = time.time()
        visible_width_sec, data, actual_res = self._get_visible_dsa_data(2160.0)

        levels = (self.user_config.psd_db_min, self.user_config.psd_db_max)
        if levels != self._last_levels:
            self.image.setLevels(levels, update=False)
            self.colorbar.setLevels(levels)
            self._last_levels = levels

        self.image.setImage(data, nan_policy="omit", autoLevels=False)
        self.image.setRect((self.t0, config.LOWEST_FREQ_HZ, actual_res * data.shape[0], self.user_config.max_freq_hz))
        self.plot.setXRange(self.t0, self.t0 + visible_width_sec, padding=0)

    # Triggered by the calibrate button
    def calibrate(self):
        _, data, _ = self._get_visible_dsa_data(1000.0)
        data = data[~np.isnan(data)]
        if len(data) > 0:
            data.sort()
            n_1_percent = max(1, int(len(data) * 0.001))
            psd_db_min = np.mean(data[:n_1_percent])
            psd_db_max = np.mean(data[-n_1_percent:])
            psd_db_min = int(np.clip(int(psd_db_min), config.PSD_DB_MIN_BOUNDS[0], config.PSD_DB_MIN_BOUNDS[1]))
            psd_db_max = int(np.clip(int(psd_db_max), config.PSD_DB_MAX_BOUNDS[0], config.PSD_DB_MAX_BOUNDS[1]))
            new_config = self.user_config.update(psd_db_min=psd_db_min, psd_db_max=psd_db_max)
            self.on_config_change(new_config)

    def jump_to_live(self):
        self.live_mode = True
        if self.dsa_buffer.t0 is not None:
            self._pan_sec = self._current_pan_start(self._visible_width_sec())
            self.update()

    def pan(self, delta_percent):
        """External pan control"""
        if self.dsa_buffer.t0 is None:
            return

        if delta_percent == "live":
            self.jump_to_live()
            return

        visible_width_sec = self._visible_width_sec()
        current_start = self._current_pan_start(visible_width_sec)
        self.live_mode = False
        self._pan_sec = current_start + visible_width_sec * delta_percent

        self.update()

    def clear_data(self):
        """Delete all buffered data"""
        self.dsa_buffer = DSABuffer()
        self.live_mode = True
        self._pan_sec = 0.0
        self.update()

    @staticmethod
    def _dropped_csv_path(event):
        mime = event.mimeData()
        if mime is None or not mime.hasUrls():
            return None
        for url in mime.urls():
            if url.isLocalFile() and url.toLocalFile().lower().endswith(".csv"):
                return url.toLocalFile()
        return None

    def dragEnterEvent(self, event):
        if self._dropped_csv_path(event):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if self._dropped_csv_path(event):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        path = self._dropped_csv_path(event)
        if not path:
            super().dropEvent(event)
            return
        event.acceptProposedAction()
        # Defer so the drag source (e.g. Explorer) is released before the Load/Play dialog opens.
        QTimer.singleShot(0, lambda: self.eeg_file_dropped.emit(path))

    def mousePressEvent(self, ev):
        super().mousePressEvent(ev)
        if ev.isAccepted():
            return
        if ev.button() == Qt.LeftButton:
            self._dragging = True
            self._last_mouse_pos = ev.pos()
            ev.accept()

    def mouseMoveEvent(self, ev):
        if self.dsa_buffer.t0 is not None and self._dragging and self._last_mouse_pos is not None:
            delta = ev.pos() - self._last_mouse_pos
            self._last_mouse_pos = ev.pos()
            visible_width_sec = self._visible_width_sec()
            width_px = self.plot.width()
            dt = (delta.x() / width_px) * visible_width_sec if width_px else 0
            self._pan_sec = self._current_pan_start(visible_width_sec) - dt
            self.live_mode = False
            self.update()

            ev.accept()
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        if ev.isAccepted():
            return
        if self._dragging:
            self._dragging = False
            self._last_mouse_pos = None
            if self.dsa_buffer.t0 is not None:
                self.update()
            ev.accept()

    def event(self, ev):
        if ev.type() == QEvent.Gesture:
            return self.gestureEvent(ev)
        return super().event(ev)

    def gestureEvent(self, ev):
        pinch = ev.gesture(Qt.PinchGesture)
        if pinch:
            self.handlePinch(pinch)
            return True
        return False

    def handlePinch(self, pinch):
        if pinch.state() == Qt.GestureUpdated:
            new_minutes = self.display_minutes / pinch.scaleFactor()
            min_m, max_m = config.DISPLAY_MINUTES_BOUNDS
            new_minutes = np.clip(new_minutes, min_m, max_m)
            self.on_zoom_change(new_minutes)
            self.update()
        return True

    WHEEL_ZOOM_STEP = 1.25  # zoom factor per mouse-wheel notch

    def wheelEvent(self, ev):
        """Wheel up zooms in (shorter time span), wheel down zooms out."""
        notches = ev.angleDelta().y() / 120.0
        if notches:
            new_minutes = self.display_minutes / (self.WHEEL_ZOOM_STEP ** notches)
            min_m, max_m = config.DISPLAY_MINUTES_BOUNDS
            self.on_zoom_change(float(np.clip(new_minutes, min_m, max_m)))
        ev.accept()  # never let the wheel scroll the view

    def apply_zoom(self, new_minutes):
        if new_minutes is not None:
            old_width = self._visible_width_sec()
            displayed_start = self.t0
            old_res, old_columns = self._render_grid(old_width)
            new_width = new_minutes * 60.0
            new_res, new_columns = self._render_grid(new_width)
            self.display_minutes = new_minutes

            if self.dsa_buffer.t0 is not None:
                min_offset, max_offset = self._pan_limits(new_width)
                if self.live_mode:
                    self._pan_sec = max_offset
                elif old_res == new_res and old_columns == new_columns:
                    self._pan_sec = float(np.clip(displayed_start, min_offset, max_offset))
                else:
                    center = displayed_start + old_width / 2.0
                    self._pan_sec = float(np.clip(center - new_width / 2.0, min_offset, max_offset))
        self.update()

    def apply_config(self, new_config):
        old_config = self.user_config
        self.user_config = new_config
        if new_config.max_freq_hz != old_config.max_freq_hz:
            self.freq_bins = config.FREQ_BINS[(config.FREQ_BINS <= self.user_config.max_freq_hz)]
            self._update_y_axis()

        self._sync_settings_labels()
        self.update()


__all__ = ["DSAView", "FrequencyAxis"]
