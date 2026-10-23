import os
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QSlider,
    QLabel, QFrame, QGraphicsOpacityEffect
)
from PySide6.QtCore import (
    QByteArray, QPoint, QSize, Qt, QRectF, QTimer,
    QPropertyAnimation, QParallelAnimationGroup, QEasingCurve
)
from PySide6.QtGui import QColor, QFontMetrics, QIcon, QPainter, QPixmap, QPolygon
from PySide6.QtSvg import QSvgRenderer
from .. import config


RESET_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
    <path fill="#FFFFFF" d="M18,28A12,12,0,1,0,6,16v6.2L2.4,18.6,1,20l6,6,6-6-1.4-1.4L8,22.2V16H8A10,10,0,1,1,18,26Z"/>
</svg>"""

SCREENSHOT_SVG = """<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" 
fill="#e3e3e3"><path d="M800-560v-120H680v-80h120q33 0 56.5 23.5T880-680v120h-80Zm-720 0v-120q0-33 
23.5-56.5T160-760h120v80H160v120H80Zm600 360v-80h120v-120h80v120q0 
33-23.5 56.5T800-200H680Zm-520 0q-33 0-56.5-23.5T80-280v-120h80v120h120v80H160Z"/></svg>"""

SCREENSHOT_DIR = os.path.join(os.path.dirname(config.BASE_DIR), "Screenshots")

SUCCESS_COLOR = "#3FA672"
ERROR_COLOR = "#D9534F"

CHECK_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
    <circle cx="12" cy="12" r="11" fill="{color}"/>
    <path d="M7 12.4l3.3 3.3L17 9" fill="none" stroke="#FFFFFF" stroke-width="2.2"
          stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

ERROR_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
    <circle cx="12" cy="12" r="11" fill="{color}"/>
    <path d="M12 6.5v7" fill="none" stroke="#FFFFFF" stroke-width="2.4" stroke-linecap="round"/>
    <circle cx="12" cy="17.3" r="1.4" fill="#FFFFFF"/>
</svg>"""


def create_svg_icon(svg, size=18):
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()

    return QIcon(pixmap)


def create_reset_icon(size=18):
    return create_svg_icon(RESET_SVG, size)


def _svg_pixmap(svg, size, dpr=1.0):
    """Render an SVG to a crisp pixmap for the given device pixel ratio."""
    px = max(1, int(round(size * dpr)))
    pixmap = QPixmap(px, px)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    QSvgRenderer(QByteArray(svg.encode("utf-8"))).render(painter, QRectF(0, 0, px, px))
    painter.end()
    pixmap.setDevicePixelRatio(dpr)
    return pixmap


# ------------------ Screenshot feedback ------------------ #
class _ScreenshotFlash(QWidget):
    """Brief, low-intensity white flash over the window (camera shutter cue)."""

    PEAK_OPACITY = 0.08
    DURATION_MS = 1200

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._effect = QGraphicsOpacityEffect(self)
        self._effect.setOpacity(0.0)
        self.setGraphicsEffect(self._effect)

        self._anim = QPropertyAnimation(self._effect, b"opacity", self)
        self._anim.setDuration(self.DURATION_MS)
        self._anim.setStartValue(self.PEAK_OPACITY)
        self._anim.setEndValue(0.0)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.finished.connect(self.hide)
        self.hide()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        painter.end()

    def play(self):
        self._anim.stop()
        self.setGeometry(self.parentWidget().rect())
        self.raise_()
        self.show()
        self._anim.start()


class _ScreenshotToast(QFrame):
    """Small status card that slides in below the top bar and fades out."""

    ICON_SIZE = 26
    HOLD_MS = 2600
    ERROR_HOLD_MS = 4500
    SLIDE_PX = 8
    MAX_DETAIL_WIDTH = 460

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("screenshotToast")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 18, 10)
        layout.setSpacing(12)

        self.icon_label = QLabel()
        self.icon_label.setFixedSize(self.ICON_SIZE, self.ICON_SIZE)
        layout.addWidget(self.icon_label, 0, Qt.AlignVCenter)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)
        self.title_label = QLabel()
        self.detail_label = QLabel()
        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.detail_label)
        layout.addLayout(text_layout)

        self._effect = QGraphicsOpacityEffect(self)
        self._effect.setOpacity(0.0)
        self.setGraphicsEffect(self._effect)

        self._fade_in = QPropertyAnimation(self._effect, b"opacity", self)
        self._fade_in.setDuration(250)
        self._fade_in.setStartValue(0.0)
        self._fade_in.setEndValue(1.0)

        self._slide_in = QPropertyAnimation(self, b"pos", self)
        self._slide_in.setDuration(220)
        self._slide_in.setEasingCurve(QEasingCurve.OutCubic)

        self._show_group = QParallelAnimationGroup(self)
        self._show_group.addAnimation(self._fade_in)
        self._show_group.addAnimation(self._slide_in)

        self._fade_out = QPropertyAnimation(self._effect, b"opacity", self)
        self._fade_out.setDuration(320)
        self._fade_out.setEndValue(0.0)
        self._fade_out.setEasingCurve(QEasingCurve.InCubic)
        self._fade_out.finished.connect(self.hide)

        self._hold = QTimer(self)
        self._hold.setSingleShot(True)
        self._hold.timeout.connect(self._start_fade_out)
        self.hide()

    def _start_fade_out(self):
        self._fade_out.setStartValue(self._effect.opacity())
        self._fade_out.start()

    def show_message(self, success, title, detail, anchor_top_right):
        color = SUCCESS_COLOR if success else ERROR_COLOR
        svg = (CHECK_SVG if success else ERROR_SVG).format(color=color)
        self.icon_label.setPixmap(_svg_pixmap(svg, self.ICON_SIZE, self.devicePixelRatioF()))

        self.setStyleSheet(f"""
            #screenshotToast {{
                background-color: rgb(43, 43, 43);
                border: 1px solid rgb(78, 78, 78);
                border-left: 4px solid {color};
                border-radius: 8px;
            }}
            QLabel {{ background: transparent; border: none; }}
        """)
        self.title_label.setStyleSheet("color: white; font-size: 16px; font-weight: 600;")
        self.detail_label.setStyleSheet("color: rgb(165, 165, 165); font-size: 13px;")

        self.title_label.setText(title)
        self.detail_label.ensurePolished()
        metrics = QFontMetrics(self.detail_label.font())
        self.detail_label.setText(metrics.elidedText(detail, Qt.ElideMiddle, self.MAX_DETAIL_WIDTH))
        self.detail_label.setVisible(bool(detail))
        self.adjustSize()

        end = QPoint(anchor_top_right.x() - self.width(), anchor_top_right.y())
        start = end - QPoint(0, self.SLIDE_PX)

        self._hold.stop()
        self._show_group.stop()
        self._fade_out.stop()
        self._effect.setOpacity(0.0)
        self._slide_in.setStartValue(start)
        self._slide_in.setEndValue(end)

        self.move(start)
        self.raise_()
        self.show()
        self._show_group.start()
        self._hold.start(self.HOLD_MS if success else self.ERROR_HOLD_MS)


# ------------------ TopBar ------------------ #
class TopBar(QWidget):
    BUTTON_HEIGHT = 35
    ZOOM_SLIDER_MAX = 1000

    def __init__(self, user_config, on_config_change, on_zoom_change, on_pan, on_calibrate):
        super().__init__()
        self.user_config = user_config
        self.on_config_change = on_config_change
        self.on_zoom_change = on_zoom_change
        self.on_pan = on_pan
        self.on_calibrate = on_calibrate
        self._screenshot_flash = None
        self._screenshot_toast = None

        self.GAMMA = 2.5 # Zoom shape parameter

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 2)
        layout.setSpacing(10)

        self.left_btn = self._create_arrow_button("left")
        self.left_btn.clicked.connect(lambda: self._pan(-1))
        layout.addWidget(self.left_btn)

        self.right_btn = self._create_arrow_button("right")
        self.right_btn.clicked.connect(lambda: self._pan(1))
        layout.addWidget(self.right_btn)

        self.live_btn = self._create_button("\u25b6 Live")
        self.live_btn.setMinimumWidth(70)
        policy = self.live_btn.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.live_btn.setSizePolicy(policy)
        self.live_btn.hide()
        layout.addWidget(self.live_btn)

        self.zoom_label = QLabel("Zoom:")
        self.zoom_label.setStyleSheet(f"font-size: {config.FONT_SIZE}px;")
        layout.addWidget(self.zoom_label)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(1)
        self.zoom_slider.setMaximum(self.ZOOM_SLIDER_MAX)
        self.zoom_slider.setStyleSheet(f"""
            QSlider {{
                background: transparent;
                min-height: 28px;
            }}
            QSlider::groove:horizontal {{
                height: 8px;
                background: rgb(110, 116, 124);
                border: 1px solid rgb(64, 69, 75);
                border-radius: 4px;
            }}
            QSlider::sub-page:horizontal {{
                background: rgb(92, 98, 106);
                border: 1px solid rgb(74, 79, 86);
                border-radius: 4px;
            }}
            QSlider::add-page:horizontal {{
                background: rgb(110, 116, 124);
                border: 1px solid rgb(64, 69, 75);
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                width: 20px;
                height: 20px;
                margin: -7px 0;
                border-radius: 10px;
                background: rgb(250, 252, 255);
                border: 2px solid rgb(92, 98, 106);
            }}
            QSlider::handle:horizontal:pressed {{
                background: rgb(228, 231, 236);
            }}
        """)
        self.zoom_slider.valueChanged.connect(self._zoom_changed)
        layout.addWidget(self.zoom_slider)

        self.calibrate_btn = self._create_button("Calibrate")
        self.calibrate_btn.clicked.connect(self.on_calibrate)
        layout.addWidget(self.calibrate_btn)

        self.reset_btn = self._create_button()
        self.reset_btn.setIcon(create_reset_icon(size=22))
        self.reset_btn.setIconSize(QSize(22, 22))
        self.reset_btn.clicked.connect(self._reset_calibration)
        layout.addWidget(self.reset_btn)

        self.screenshot_btn = self._create_button()
        self.screenshot_btn.setIcon(create_svg_icon(SCREENSHOT_SVG, size=22))
        self.screenshot_btn.setIconSize(QSize(22, 22))
        self.screenshot_btn.setToolTip("Save screenshot")
        self.screenshot_btn.clicked.connect(lambda: self._take_screenshot())
        layout.addWidget(self.screenshot_btn)

    def _button_style(self, font_size=None, padding=None):
        resolved_font_size = config.FONT_SIZE if font_size is None else font_size
        resolved_padding = padding if padding is not None else "0px 12px"
        return f"""
            QPushButton {{
                background-color: rgb(58, 58, 58);
                color: white;
                border: 1px solid rgb(78, 78, 78);
                border-radius: 6px;
                padding: {resolved_padding};
                font-size: {resolved_font_size}px;
                text-align: center;
            }}
            QPushButton:pressed {{
                background-color: rgb(92, 92, 92);
            }}
            QPushButton:disabled {{
                background-color: rgb(48, 48, 48);
                color: rgb(150, 150, 150);
            }}
        """

    def _create_button(self, text="", font_size=None, padding=None, min_width=None):
        button = QPushButton(text)
        button.setMinimumHeight(self.BUTTON_HEIGHT)
        if min_width is not None:
            button.setMinimumWidth(min_width)
        button.setStyleSheet(self._button_style(font_size=font_size, padding=padding))
        return button

    def _create_arrow_button(self, direction):
        button = self._create_button("", padding="0px", min_width=42)
        button.setIcon(self._create_arrow_icon(direction))
        button.setIconSize(QSize(16, 16))
        return button

    def _create_arrow_icon(self, direction):
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("white"))

        if direction == "left":
            points = [QPoint(10, 3), QPoint(5, 8), QPoint(10, 13)]
        else:
            points = [QPoint(6, 3), QPoint(11, 8), QPoint(6, 13)]

        painter.drawPolygon(QPolygon(points))
        painter.end()
        return QIcon(pixmap)

    # ------------------ New actions ------------------ #
    def _reset_calibration(self):
        new_config = self.user_config.update(psd_db_min=config.PSD_DB_MIN, psd_db_max=config.PSD_DB_MAX)
        self.on_config_change(new_config)

    def _take_screenshot(self, directory=SCREENSHOT_DIR):
        """Save a PNG of the whole window and return its path (None on failure)."""
        filename = datetime.now().strftime("DSA_%Y-%m-%d_%H%M%S") + ".png"
        path = os.path.join(directory, filename)
        window = self.window()
        try:
            os.makedirs(directory, exist_ok=True)
            ok = window.grab().save(path, "PNG")  # grab before any overlay is shown
        except OSError:
            ok = False

        if ok:
            self._show_screenshot_feedback(window, True, "Screenshot saved", directory)
        else:
            self._show_screenshot_feedback(window, False, "Screenshot failed", f"Could not write to {directory}")
        return path if ok else None

    def _show_screenshot_feedback(self, window, success, title, detail):
        if self._screenshot_toast is None or self._screenshot_toast.parentWidget() is not window:
            self._screenshot_flash = _ScreenshotFlash(window)
            self._screenshot_toast = _ScreenshotToast(window)

        if success:
            self._screenshot_flash.play()
        anchor = self.mapTo(window, QPoint(self.width() - 8, self.height() + 6))
        self._screenshot_toast.show_message(success, title, detail, anchor)

    def _pan(self, direction: int):
        """direction: -1 = left, +1 = right"""
        step = 0.27 # percent of display width
        self.on_pan(direction * step)

    def _zoom_changed(self, value):
        min_minutes, max_minutes = config.DISPLAY_MINUTES_BOUNDS
        slider_span = self.zoom_slider.maximum() - self.zoom_slider.minimum()

        t = (value - self.zoom_slider.minimum()) / slider_span  # 0..1 linear

        # invert gamma effect for desired behavior
        t = 1.0 - (1.0 - t) ** self.GAMMA

        display_minutes = max_minutes - t * (max_minutes - min_minutes)
        self.on_zoom_change(display_minutes)

    def sync_slider(self, display_minutes):
        min_minutes, max_minutes = config.DISPLAY_MINUTES_BOUNDS

        t = (max_minutes - display_minutes) / (max_minutes - min_minutes)
        t = max(0.0, min(1.0, t))

        # inverse of forward curve
        t = 1.0 - (1.0 - t) ** (1.0 / self.GAMMA)

        slider_span = self.zoom_slider.maximum() - self.zoom_slider.minimum()
        val_zoom = self.zoom_slider.minimum() + slider_span * t

        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(int(round(val_zoom)))
        self.zoom_slider.blockSignals(False)

    def update_jump_live_btn(self, dsa_view):
        if dsa_view.dsa_buffer.t0 is None or dsa_view.live_mode:
            self.live_btn.hide()
        else:
            self.live_btn.show()

    def apply_config(self, user_config):
        self.user_config = user_config
