import time
import csv
import os
from datetime import datetime, timedelta
import math
from pylsl import StreamInfo, StreamOutlet

import sys
from pathlib import Path

# Add project root to sys.path for direct execution
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from .. import config
except (ImportError, ValueError):
    from plotdsa import config

# =========================
# Configuration
# =========================
DATA_DIR = r"..\..\Entropy_Data"

STREAM_NAME = "EEG_DATA"
STREAM_TYPE = "EEG"
CHANNEL_COUNT = 1
CHANNEL_FORMAT = "string"
SOURCE_ID = "EEG_DSA_ENTROPY_EMULATOR"

# Set your sample rate here
SAMPLE_RATE_HZ = config.SAMPLE_RATE_HZ
MISSING_BURST_EVERY_SAMPLES = SAMPLE_RATE_HZ * 4
MISSING_BURST_LENGTH = 1


def load_csv_column(filepath, i):
    """Load only the first column of CSV, skipping NaNs at the start"""
    values = []
    started = False
    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                value = float(row[i])
                if not math.isfinite(value) and not started:
                    continue  # Skip initial NaNs
                else:
                    started = True
                values.append(value)
            except Exception:
                continue
    return values


def inject_missing_value(idx, value):
    """Replace short deterministic bursts with NaN to exercise EEG gap handling."""
    if idx % MISSING_BURST_EVERY_SAMPLES < MISSING_BURST_LENGTH:
        return math.nan
    return value


def main():
    # Create LSL outlet
    info = StreamInfo(
        name=STREAM_NAME,
        type=STREAM_TYPE,
        channel_count=CHANNEL_COUNT,
        nominal_srate=SAMPLE_RATE_HZ,
        channel_format=CHANNEL_FORMAT,
        source_id=SOURCE_ID
    )
    outlet = StreamOutlet(info)
    print(f"LSL stream '{STREAM_NAME}' @ {SAMPLE_RATE_HZ} Hz")
    # Deterministic timestamping
    ts = datetime.now()
    while True:
        for i in range(1, 11):
            if i == 8:
                continue
            for j in range(1, 11):
                filepath = os.path.join(DATA_DIR, f"JSMF_00{i}_filtered_emergence.csv")
                if i == 10:
                    filepath = os.path.join(DATA_DIR, f"JSMF_0{i}_filtered_emergence.csv")
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"CSV file not found: {filepath}")

                values = load_csv_column(filepath, j)
                if not values:
                    raise RuntimeError("No valid samples loaded")
                print(
                    f"Streaming {len(values)} samples from JSMF_00{i}_filtered_emergence.csv column {j} "
                    f"with {MISSING_BURST_LENGTH} missing samples every {MISSING_BURST_EVERY_SAMPLES} samples..."
                )

                interval = 1.0 / SAMPLE_RATE_HZ
                samples = []

                for idx, value in enumerate(values):
                    ts += timedelta(seconds=interval)
                    #emitted_value = inject_missing_value(idx, value)

                    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S.%f")
                    samples.append(f"{ts_str},{value}")

                    # Push to LSL
                    if idx % SAMPLE_RATE_HZ == 0:
                        time.sleep(0.999)
                        for s in samples:
                            outlet.push_sample([s])
                        samples = []

                print("All samples streamed.")


if __name__ == "__main__":
    main()
