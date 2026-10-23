from datetime import datetime as dt
import datetime
import sys

from PySide6.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QDialog,
    QPushButton,
    QHBoxLayout,
    QDialogButtonBox,
    QFrame,
    QSlider,
)
from PySide6.QtCore import Qt

if __package__ in (None, ""):
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from plotdsa import config
else:
    from .. import config


class TimeSelectionDialog(QDialog):
    PRESET_OFFSETS = (
        ("2 hours ago", datetime.timedelta(hours=2)),
        ("8 hours ago", datetime.timedelta(hours=8)),
        ("24 hours ago", datetime.timedelta(hours=24)),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Start Time")
        self.setMinimumSize(400, 250)

        self._reference_now = self._normalize_datetime(dt.now())
        self._selected_dt = self._default_datetime()
        self._building_ui = False

        self._build_ui()
        self._apply_datetime(self._selected_dt)

    def _build_ui(self):
        self._building_ui = True

        self.setStyleSheet(f"""
            QDialog {{
                font-size: {config.FONT_SIZE}px;
                color: white;
            }}
            QFrame#pickerSection {{
                border: 1px solid rgb(78, 78, 78);
                border-radius: 12px;
                background: rgb(40, 40, 40);
            }}
            QPushButton#presetButton {{
                background-color: rgb(58, 58, 58);
                color: white;
                border: 1px solid rgb(78, 78, 78);
                border-radius: 10px;
                padding: 8px 16px;
                font-size: {config.FONT_SIZE}px;
                text-align: center;
                min-height: 25px;
                font-weight: 600;
            }}
            QPushButton#presetButton:pressed {{
                background-color: rgb(92, 92, 92);
            }}
            QPushButton#presetButton:disabled {{
                background-color: rgb(48, 48, 48);
                color: rgb(150, 150, 150);
            }}
            QSlider {{
                background: transparent;
                min-height: 36px;
            }}
            QSlider::groove:horizontal {{
                height: 10px;
                border: 1px solid rgb(64, 69, 75);
                border-radius: 5px;
                background: rgb(110, 116, 124);
            }}
            QSlider::sub-page:horizontal {{
                height: 10px;
                border: 1px solid rgb(74, 79, 86);
                border-radius: 5px;
                background: rgb(92, 98, 106);
            }}
            QSlider::add-page:horizontal {{
                height: 10px;
                border: 1px solid rgb(64, 69, 75);
                border-radius: 5px;
                background: rgb(110, 116, 124);
            }}
            QSlider::handle:horizontal {{
                width: 26px;
                height: 26px;
                margin: -9px 0;
                border-radius: 13px;
                background: rgb(250, 252, 255);
                border: 2px solid rgb(92, 98, 106);
            }}
            QSlider::handle:horizontal:pressed {{
                background: rgb(228, 231, 236);
            }}
            QLabel#pickerPreview {{
                padding: 10px;
                font-size: {config.FONT_SIZE + 4}px;
                font-weight: 700;
                color: white;
            }}
            QDialogButtonBox QPushButton {{
                min-height: 25px;
                padding: 8px 16px;
                border-radius: 10px;
                font-size: {config.FONT_SIZE}px;
                font-weight: 600;
            }}
            QPushButton[text="Cancel"] {{
                background-color: rgb(58, 58, 58);
                color: white;
                border: 1px solid rgb(78, 78, 78);
            }}
            QPushButton[text="Cancel"]:pressed {{
                background-color: rgb(92, 92, 92);
            }}
            QPushButton[text="Load Data"] {{
                background-color: rgb(0, 120, 215);
                color: white;
                border: 1px solid rgb(0, 120, 215);
            }}
            QPushButton[text="Load Data"]:pressed {{
                background-color: rgb(0, 100, 180);
            }}
        """)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 5, 12, 12)
        root_layout.setSpacing(12)

        # Split presets into two rows if needed, but for touchscreen let's use a flow or just one big row/grid
        # Given 6 presets, let's do 2 rows of 3
        v_presets = QVBoxLayout()
        row1 = QHBoxLayout()
        for i, (label, offset) in enumerate(self.PRESET_OFFSETS):
            button = QPushButton(label)
            button.setObjectName("presetButton")
            button.clicked.connect(lambda _, delta=offset: self._apply_preset(delta))

            row1.addWidget(button)

        v_presets.addLayout(row1)
        root_layout.addLayout(v_presets)

        # Slider Section
        slider_frame = QFrame()
        slider_frame.setObjectName("pickerSection")
        slider_layout = QVBoxLayout(slider_frame)
        slider_layout.setContentsMargins(10, 10, 10, 10)
        slider_layout.setSpacing(0)
        self.preview_label = QLabel()
        self.preview_label.setObjectName("pickerPreview")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        slider_layout.addWidget(self.preview_label)

        self.ago_slider = QSlider(Qt.Orientation.Horizontal)
        self.ago_slider.setRange(0, 24)
        self.ago_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.ago_slider.setTickInterval(1)
        self.ago_slider.setSingleStep(1)
        self.ago_slider.setPageStep(4)
        self.ago_slider.valueChanged.connect(self._sync_from_slider)
        slider_layout.addWidget(self.ago_slider)

        root_layout.addWidget(slider_frame)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        button_box.setStyleSheet(f"""
            QDialogButtonBox QPushButton {{
                min-height: 25px;
                padding: 8px 16px;
                border-radius: 10px;
                font-size: {config.FONT_SIZE}px;
                font-weight: 600;
            }}
            QPushButton[text="Cancel"] {{
                background-color: rgb(58, 58, 58);
                color: white;
                border: 1px solid rgb(78, 78, 78);
            }}
            QPushButton[text="Cancel"]:pressed {{
                background-color: rgb(92, 92, 92);
            }}
            QPushButton[text="Load Data"] {{
                background-color: rgb(0, 120, 215);
                color: white;
                border: 1px solid rgb(0, 120, 215);
            }}
            QPushButton[text="Load Data"]:pressed {{
                background-color: rgb(0, 100, 180);
            }}
        """)
        self.load_button = button_box.button(QDialogButtonBox.StandardButton.Ok)
        self.load_button.setText("Load Data")
        self.load_button.setMinimumWidth(150)

        cancel_button = button_box.button(QDialogButtonBox.StandardButton.Cancel)
        cancel_button.setMinimumWidth(150)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        root_layout.addWidget(button_box)

        self._building_ui = False

    def _default_datetime(self):
        return self._reference_now - datetime.timedelta(hours=1)

    def _apply_preset(self, offset):
        self._apply_datetime(self._reference_now - offset)

    def _normalize_datetime(self, value):
        return value.replace(second=0, microsecond=0)

    def _hours_ago(self, selected_dt, now):
        return round((now - selected_dt).total_seconds() / 3600)

    def _set_hours_ago(self, hours_ago):
        self.ago_slider.blockSignals(True)
        self.ago_slider.setValue(hours_ago)
        self.ago_slider.blockSignals(False)

    def _apply_datetime(self, selected_dt):
        limit = self._reference_now - datetime.timedelta(hours=24)
        if selected_dt < limit:
            selected_dt = limit

        normalized = self._normalize_datetime(min(selected_dt, self._reference_now))
        self._selected_dt = normalized
        hours_ago = self._hours_ago(normalized, self._reference_now)
        self._set_hours_ago(hours_ago)

        self._refresh_preview(normalized)

    def _sync_from_slider(self, hours_ago):
        if self._building_ui:
            return

        selected_dt = self._reference_now - datetime.timedelta(hours=hours_ago)
        self._selected_dt = selected_dt
        self._refresh_preview(selected_dt)

    def _refresh_preview(self, selected_dt):
        delta = max(datetime.timedelta(), self._reference_now - selected_dt)
        total_minutes = int(delta.total_seconds() // 60)
        hours, minutes = divmod(total_minutes, 60)

        if total_minutes < 1:
            relative = "just now"
        elif hours:
            if minutes == 0:
                relative = f"{hours}h ago"
            else:
                relative = f"{hours}h {minutes:02d}m ago"
        else:
            relative = f"{minutes}m ago"

        self.preview_label.setText(selected_dt.strftime(f"%a, %d %b %H:%M  |  {relative}"))

    def selected_datetime(self):
        return self._selected_dt
