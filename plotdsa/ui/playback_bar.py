from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QStyle, QWidget

from .. import config
from .dsa_overview import DSAOverview

SPEED_OPTIONS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)

BUTTON_STYLE = f"""
    QPushButton {{
        background-color: rgb(58, 58, 58);
        color: white;
        border: 1px solid rgb(78, 78, 78);
        border-radius: 6px;
        padding: 0px 10px;
        font-size: {config.FONT_SIZE}px;
    }}
    QPushButton:pressed {{
        background-color: rgb(92, 92, 92);
    }}
    QPushButton:disabled {{
        background-color: rgb(48, 48, 48);
    }}
"""


def format_time(seconds):
    seconds = int(max(seconds, 0))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


class PlaybackBar(QWidget):
    """Play/pause, playback speed and DSA overview (seek) controls for a playing EEG file."""

    play_pause_clicked = Signal()
    seek_requested = Signal(float)  # target position in seconds
    speed_changed = Signal(float)  # playback speed factor (1.0 = real time)

    BUTTON_HEIGHT = 30

    def __init__(self, parent=None):
        super().__init__(parent)
        self._duration = 0.0
        self._position = 0.0
        self._window_width = 0.0
        self.speed = 1.0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 4)
        layout.setSpacing(8)

        self.slower_btn = self._create_button(QStyle.StandardPixmap.SP_MediaSeekBackward)
        self.slower_btn.setToolTip("Slower")
        self.slower_btn.clicked.connect(lambda: self._step_speed(-1))
        layout.addWidget(self.slower_btn)

        self.play_btn = self._create_button(QStyle.StandardPixmap.SP_MediaPause)
        self.play_btn.setMinimumWidth(44)
        self.play_btn.clicked.connect(self.play_pause_clicked)
        layout.addWidget(self.play_btn)

        self.faster_btn = self._create_button(QStyle.StandardPixmap.SP_MediaSeekForward)
        self.faster_btn.setToolTip("Faster")
        self.faster_btn.clicked.connect(lambda: self._step_speed(1))
        layout.addWidget(self.faster_btn)

        self.speed_label = QLabel()
        self.speed_label.setStyleSheet(f"font-size: {config.FONT_SIZE}px;")
        self.speed_label.setMinimumWidth(48)
        self.speed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.speed_label)

        self.overview = DSAOverview()
        self.overview.seek_requested.connect(self.seek_requested)
        self.overview.position_previewed.connect(self._update_label)
        layout.addWidget(self.overview, 1)

        self.time_label = QLabel()
        self.time_label.setStyleSheet(f"font-size: {config.FONT_SIZE}px;")
        layout.addWidget(self.time_label)

        self.set_playing(False)
        self.set_duration(0.0)
        self.set_speed(1.0)

    def _create_button(self, pixmap, text=""):
        button = QPushButton(text)
        button.setIcon(self.style().standardIcon(pixmap))
        button.setIconSize(QSize(16, 16))
        button.setMinimumHeight(self.BUTTON_HEIGHT)
        button.setStyleSheet(BUTTON_STYLE)
        return button

    def set_playing(self, playing):
        icon = QStyle.StandardPixmap.SP_MediaPause if playing else QStyle.StandardPixmap.SP_MediaPlay
        self.play_btn.setIcon(self.style().standardIcon(icon))
        self.play_btn.setToolTip("Pause" if playing else "Play")
        # Hide upcoming data only while playing; paused, the whole recording is shown.
        self.overview.set_future_hidden(playing)

    def set_duration(self, seconds):
        self._duration = max(float(seconds), 0.0)
        self.set_position(0.0)

    def set_window_width(self, seconds):
        """Width of the overview rectangle: the time span visible in the main DSA."""
        self._window_width = max(float(seconds), 0.0)
        self.overview.set_window(self._position, self._window_width)

    def set_position(self, seconds):
        self._position = min(max(float(seconds), 0.0), self._duration)
        self.overview.set_window(self._position, self._window_width)
        if not self.overview.region.moving:
            self._update_label(self._position)

    def _update_label(self, seconds):
        self.time_label.setText(f"{format_time(seconds)} / {format_time(self._duration)}")

    def set_speed(self, speed):
        """Show `speed` and emit speed_changed."""
        self.speed = speed
        index = SPEED_OPTIONS.index(speed)
        self.slower_btn.setEnabled(index > 0)
        self.faster_btn.setEnabled(index < len(SPEED_OPTIONS) - 1)
        self.speed_label.setText(f"{speed:g}×")
        self.speed_changed.emit(speed)

    def _step_speed(self, step):
        index = SPEED_OPTIONS.index(self.speed) + step
        if 0 <= index < len(SPEED_OPTIONS):
            self.set_speed(SPEED_OPTIONS[index])
