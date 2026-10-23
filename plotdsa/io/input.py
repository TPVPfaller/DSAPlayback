import os
import sys
from datetime import datetime as dt

import numpy as np
from pylsl import StreamInlet, resolve_byprop

from .. import config


# fix lsl for single file .exe
if getattr(sys, "frozen", False):
    base = sys._MEIPASS if hasattr(sys, "_MEIPASS") else os.path.dirname(sys.executable)
    os.environ["PYLSL_LIB"] = os.path.join(base, "pylsl", "lib", "lsl.dll")


class EEGStream:
    def __init__(self):
        self.receiving = False
        self._inlet = None
        self.connect()

    def connect(self):
        try:
            streams = resolve_byprop("name", config.LSL_STREAM_NAME, timeout=2)
            if streams:
                self._inlet = StreamInlet(streams[0])
                print(f"Connected to: {streams[0].name()} (uid: {streams[0].uid()})")
                self.receiving = True
            else:
                self.receiving = False
        except Exception as e:
            print(f"LSL Connection error: {e}")
            self.receiving = False

    def read_lsl_samples(self):
        if not self.receiving or self._inlet is None:
            return []

        samples = []
        while True:
            sample, _ = self._inlet.pull_sample(timeout=0)
            if sample is None:
                break
            try:
                timestamp, eeg_str = sample[0].split(",")
                value = float(eeg_str)
                timestamp = dt.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")

                if not np.isfinite(value):
                    print(f"Non-finite EEG value: {value} at {timestamp}")
                    value = np.nan

                samples.append((timestamp, value))
            except Exception as e:
                print("Invalid sample: ", e)
                continue

        return samples
