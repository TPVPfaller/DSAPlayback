import math

from PySide6.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QScrollArea, QHBoxLayout, QPushButton, QSlider,
    QLabel, QGridLayout, QFrame, QSizePolicy, QMessageBox, QDialogButtonBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics
from .. import config


class SettingsDialog(QDialog):
    def __init__(self, user_config, on_config_change, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DSA Settings")
        self.setMinimumSize(640, 170)

        self.user_config = user_config
        self.on_config_change = on_config_change

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(10, 0, 10, 10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        scroll.setStyleSheet("QScrollArea { background-color: transparent; }")
        container = QWidget()
        container.setStyleSheet("QWidget { background-color: transparent; }")
        grid = QGridLayout(container)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(3)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 0)

        self.sliders = {}
        row_idx = 0

        def add_slider(name, bounds, value, scale=1, unit: str = "", display_factor: float = 1.0,
                       decimals_override=None):
            nonlocal row_idx
            row_h = 44  # taller for touch

            name_label = QLabel(name)
            name_label.setStyleSheet(f"font-size: {config.FONT_SIZE}px;")
            name_label.setFixedHeight(row_h)
            name_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            name_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(int(bounds[0] * scale))
            slider.setMaximum(int(bounds[1] * scale))
            slider.setValue(int(value * scale))
            slider.setFixedHeight(row_h)
            slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            slider.setStyleSheet(f"""
                QSlider::groove:horizontal {{
                    height: 8px;
                    background: rgb(58, 58, 58);
                    border: 1px solid rgb(78, 78, 78);
                    border-radius: 4px;
                }}
                QSlider::sub-page:horizontal {{
                    background: rgb(92, 98, 106);
                    border: 1px solid rgb(74, 79, 86);
                    border-radius: 4px;
                }}
                QSlider::add-page:horizontal {{
                    height: 8px;
                    background: rgb(110, 116, 124);
                    border: 1px solid rgb(64, 69, 75);
                    border-radius: 4px;
                }}
                QSlider::handle:horizontal {{
                    width: 22px;
                    height: 22px;
                    margin: -8px 0;
                    border-radius: 11px;
                    background: rgb(250, 252, 255);
                    border: 2px solid rgb(92, 98, 106);
                }}
            """)

            decimals = decimals_override if decimals_override is not None else max(0, int(round(math.log10(scale))) if scale > 1 else 0)

            def fmt_number(x: float) -> str:
                return f"{x:.{decimals}f}"

            def fmt_with_unit(x: float) -> str:
                return f"{fmt_number(x)}{unit}"

            value_label = QLabel(fmt_with_unit(value * display_factor))
            value_label.setStyleSheet(f"font-size: {config.FONT_SIZE}px;")
            value_label.setFixedHeight(row_h)
            value_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            value_label.setAlignment(Qt.AlignLeft)

            fm = QFontMetrics(value_label.font())
            b0_txt = fmt_with_unit(bounds[0] * display_factor)
            b1_txt = fmt_with_unit(bounds[1] * display_factor)
            max_w = max(fm.horizontalAdvance(b0_txt), fm.horizontalAdvance(b1_txt)) + 25
            value_label.setFixedWidth(max_w)

            slider.valueChanged.connect(lambda val: value_label.setText(fmt_with_unit(val / scale * display_factor)))

            grid.addWidget(name_label, row_idx, 0, alignment=Qt.AlignVCenter | Qt.AlignRight)
            grid.addWidget(slider, row_idx, 1, alignment=Qt.AlignVCenter)
            grid.addWidget(value_label, row_idx, 2, alignment=Qt.AlignLeft)

            self.sliders[name] = (slider, scale)
            row_idx += 1

        # --- Add sliders ---
        add_slider("Window (s)", config.WINDOW_SEC_BOUNDS, user_config.window_sec, 1, unit=" s")
        add_slider("Window Overlap", config.WINDOW_OVERLAP_BOUNDS, user_config.window_overlap, 100, unit=" %", display_factor=100.0, decimals_override=0)

        # --- Buttons row ---
        main_layout.addSpacing(4)
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.RestoreDefaults |
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.Save
        )
        button_box.setStyleSheet(f"""
            QDialogButtonBox QPushButton {{
                min-height: 25px;
                padding: 8px 16px;
                border-radius: 10px;
                font-size: {config.FONT_SIZE}px;
                font-weight: 600;
            }}
            QPushButton[text="Reset to Defaults"], QPushButton[text="Cancel"] {{
                background-color: rgb(58, 58, 58);
                color: white;
                border: 1px solid rgb(78, 78, 78);
            }}
            QPushButton[text="Reset to Defaults"]:pressed, QPushButton[text="Cancel"]:pressed {{
                background-color: rgb(92, 92, 92);
            }}
            QPushButton[text="Save and Close"] {{
                background-color: rgb(0, 120, 215);
                color: white;
                border: 1px solid rgb(0, 120, 215);
            }}
            QPushButton[text="Save and Close"]:pressed {{
                background-color: rgb(0, 100, 180);
            }}
        """)

        reset_btn = button_box.button(QDialogButtonBox.StandardButton.RestoreDefaults)
        reset_btn.setText("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_to_defaults)

        cancel_btn = button_box.button(QDialogButtonBox.StandardButton.Cancel)

        save_btn = button_box.button(QDialogButtonBox.StandardButton.Save)
        save_btn.setText("Save and Close")

        button_box.accepted.connect(self._apply)
        button_box.rejected.connect(self.reject)

        grid.addWidget(button_box, row_idx, 0, 1, 3)

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def _reset_to_defaults(self):
        resp = self.parent()._show_message(
            "Reset Settings",
            "Reset all settings to default values?\n(Changes are not saved until you click 'Save and Close'.)",
            buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            default_button=QMessageBox.StandardButton.No
        )
        if resp != QMessageBox.StandardButton.Yes:
            return

        mapping = {
            "Window (s)": config.WINDOW_SEC,
            "Window Overlap": config.WINDOW_OVERLAP,
        }
        for name, (slider, scale) in self.sliders.items():
            if name in mapping:
                val = mapping[name]
                slider.blockSignals(True)
                slider.setValue(int(round(val * scale)))
                slider.blockSignals(False)
                slider.valueChanged.emit(slider.value())

    def _apply(self):
        try:
            proposed_window_sec = self.sliders["Window (s)"][0].value()
            proposed_window_overlap = self.sliders["Window Overlap"][0].value() / 100

            new_config = self.user_config.update(
                window_sec=proposed_window_sec,
                window_overlap=proposed_window_overlap,
            )
            self.on_config_change(new_config)
            self.accept()
        except ValueError as e:
            self.parent()._show_message(
                "Invalid Configuration",
                str(e),
                buttons=QMessageBox.StandardButton.Ok
            )

