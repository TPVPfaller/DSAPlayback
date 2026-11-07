import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import QGraphicsRectItem

# Horizontal resolution of the overview image; enough for any screen width.
OVERVIEW_BINS = 2000


def overview_image(column_starts_sec, psds, duration, bins=OVERVIEW_BINS):
    """
    Resample DSA columns onto `bins` equal time bins over [0, duration].

    Each bin shows the latest column starting at or before its centre, so no
    values are averaged or invented. Bins before the first column are NaN.
    Returns a (bins, n_freq) array in dB.
    """
    psds = np.asarray(psds, dtype=np.float32)
    if len(psds) == 0 or duration <= 0:
        return np.full((1, 1), np.nan, dtype=np.float32)
    centres = (np.arange(bins) + 0.5) * (duration / bins)
    index = np.searchsorted(np.asarray(column_starts_sec), centres, side="right") - 1
    image = 10 * np.log10(np.maximum(psds[np.clip(index, 0, None)], np.finfo(np.float32).eps))
    image[index < 0] = np.nan
    return image


RECT_PEN = pg.mkPen("w", width=2)


class _RectRegion(pg.LinearRegionItem):
    """LinearRegionItem drawn as a closed rectangle (top and bottom edges added)."""

    def paint(self, p, *args):
        super().paint(p, *args)
        device_rect = p.transform().mapRect(self.boundingRect())
        p.save()
        p.resetTransform()
        p.setPen(RECT_PEN)
        # Inset by one pixel so the edges are not clipped at the plot border.
        top, bottom = device_rect.top() + 1, device_rect.bottom() - 1
        p.drawLine(device_rect.left(), top, device_rect.right(), top)
        p.drawLine(device_rect.left(), bottom, device_rect.right(), bottom)
        p.restore()


class DSAOverview(pg.PlotWidget):
    """
    Miniature of the whole DSA with a draggable rectangle marking the time
    span visible in the main DSA. The rectangle's right edge is the playhead.
    While playing, everything after the playhead is covered so the overview
    does not show what is about to be played; when paused it shows everything.
    """

    seek_requested = Signal(float)  # playhead position in seconds
    position_previewed = Signal(float)  # playhead while the rectangle is dragged

    def __init__(self, parent=None):
        super().__init__(parent, background="k")
        self._duration = 0.0
        self._width = 0.0
        self._position = 0.0

        plot = self.getPlotItem()
        plot.hideAxis("left")
        plot.hideAxis("bottom")
        plot.setContentsMargins(0, 0, 0, 0)
        plot.setMenuEnabled(False)
        plot.hideButtons()
        self.setMouseEnabled(False, False)
        self.setMinimumHeight(40)
        self.setMaximumHeight(40)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self.image = pg.ImageItem(axisOrder="col-major")
        self.addItem(self.image)

        self.future_mask = QGraphicsRectItem()
        self.future_mask.setPen(QPen(Qt.PenStyle.NoPen))
        self.future_mask.setBrush(QBrush(QColor("black")))
        self.future_mask.setZValue(5)
        self.addItem(self.future_mask)

        self.region = _RectRegion(
            values=(0, 0),
            brush=pg.mkBrush(255, 255, 255, 45),
            hoverBrush=pg.mkBrush(255, 255, 255, 70),
            pen=RECT_PEN,
        )
        # The rectangle can be slid but not resized; its width follows the DSA zoom.
        for line in self.region.lines:
            line.setMovable(False)
        self.region.setZValue(10)
        self.addItem(self.region)
        self.region.sigRegionChanged.connect(self._region_moving)
        self.region.sigRegionChangeFinished.connect(self._region_moved)
        self.scene().sigMouseClicked.connect(self._clicked)

    def set_dsa(self, column_starts_sec, psds, duration, lut, levels):
        self._duration = max(float(duration), 0.0)
        self.image.setLookupTable(lut)
        self.image.setImage(overview_image(column_starts_sec, psds, self._duration),
                            levels=levels, autoLevels=False)
        if self._duration > 0 and len(psds):
            self.image.setRect(0, 0, self._duration, 1)
        self.setXRange(0, max(self._duration, 1e-6), padding=0)
        self.setYRange(0, 1, padding=0)
        self._update_mask()

    def set_future_hidden(self, hidden):
        self.future_mask.setVisible(hidden)

    def set_levels(self, levels):
        self.image.setLevels(levels)

    def set_window(self, position, width):
        """Place the rectangle so it spans `width` seconds ending at `position`."""
        self._width = max(float(width), 0.0)
        self._position = float(position)
        self._update_mask()
        if self.region.moving:
            return  # the user is dragging it
        self.region.blockSignals(True)
        self.region.setRegion((position - self._width, position))
        self.region.blockSignals(False)

    def _update_mask(self):
        self.future_mask.setRect(QRectF(self._position, 0, max(self._duration - self._position, 0.0), 1))

    def _playhead(self):
        return min(max(self.region.getRegion()[1], 0.0), self._duration)

    def _region_moving(self):
        self.position_previewed.emit(self._playhead())

    def _region_moved(self):
        self.seek_requested.emit(self._playhead())

    def _clicked(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x = self.getPlotItem().vb.mapSceneToView(event.scenePos()).x()
        self.seek_requested.emit(min(max(x, 0.0), self._duration))
