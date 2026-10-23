import numpy as np
import pyqtgraph as pg

from .. import config
from .views import set_axis_label, set_uniform_left_axis_width


class PSDView(pg.PlotWidget):
    def __init__(self, user_config, on_config_change=None):
        super().__init__()
        self.user_config = user_config
        self.on_config_change = on_config_change
        self._last_psd = None

        set_axis_label(self.plotItem, "bottom", "Frequency", units="Hz")
        set_axis_label(self.plotItem, "left", "Power", units="dB")
        set_uniform_left_axis_width(self.plotItem)
        self.getPlotItem().setContentsMargins(10, 10, 20, 8)
        self.setMinimumHeight(config.MIN_PSD_HEIGHT)
        self.setMenuEnabled(False)
        self.showGrid(x=True, y=True)
        self.setMouseEnabled(x=False, y=False)

        self.curve = self.plot(pen=pg.mkPen("y", width=2), title="PSD")

        self.setInteractive(False)
        self.apply_config(user_config)

    def update(self, psd):
        self._last_psd = np.asarray(psd, dtype=np.float32)
        psd_db = 10 * np.log10(np.clip(psd, np.finfo(np.float32).eps, None))
        self.curve.setData(config.FREQ_BINS, psd_db)

    def apply_config(self, user_config):
        self.user_config = user_config
        self.setXRange(config.LOWEST_FREQ_HZ, user_config.max_freq_hz, padding=0)
        self.setYRange(user_config.psd_db_min - 5, user_config.psd_db_max + 5, padding=0)
        if self._last_psd is not None:
            self.update(self._last_psd)


__all__ = ["PSDView"]
