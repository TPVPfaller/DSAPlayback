import csv
import glob
import os
import threading
from datetime import datetime as dt, timedelta

import numpy as np

from .. import config


class Output:
    _state_lock = threading.Lock()
    _file_states = {}
    _last_cleanup_date = None

    @staticmethod
    def _build_filename(base_dir, start_time=None):
        if start_time is None:
            start_time = dt.now()
        else:
            start_time = dt.fromtimestamp(start_time)
        ts = start_time.strftime("%Y-%m-%d")
        filename = f"dsa_{ts}.csv"
        return os.path.join(base_dir, filename)

    @staticmethod
    def _header_row():
        return ["timestamp", "duration"] + [f"f_{f:.1f}_Hz" for f in config.FREQ_BINS]

    @classmethod
    def _scan_file_state(cls, filepath):
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return {"has_data": False, "last_timestamp": None}

        with open(filepath, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)

            if not header:
                return {"has_data": False, "last_timestamp": None}
            if len(header) != 2 + len(config.FREQ_BINS):
                raise ValueError("Frequency bins mismatch.")

            last_timestamp = None
            for row in reader:
                if row:
                    last_timestamp = row[0]

        return {"has_data": True, "last_timestamp": last_timestamp}

    @classmethod
    def _get_file_state(cls, filepath):
        state = cls._file_states.get(filepath)
        if state is None:
            state = cls._scan_file_state(filepath)
            cls._file_states[filepath] = state
        return state

    @classmethod
    def _delete_stale_csv_files(cls):
        today = dt.now().date()
        if cls._last_cleanup_date == today:
            return

        cutoff_date = today - timedelta(days=1)
        pattern = os.path.join(config.BASE_DIR, "dsa_*.csv")
        for filepath in glob.glob(pattern):
            filename = os.path.basename(filepath)
            try:
                file_date = dt.strptime(filename[4:-4], "%Y-%m-%d").date()
            except ValueError:
                continue

            if file_date < cutoff_date:
                try:
                    os.remove(filepath)
                except OSError:
                    continue
                cls._file_states.pop(filepath, None)

        cls._last_cleanup_date = today

    @classmethod
    def _append_row(cls, filepath, state, row):
        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)
            if not state["has_data"]:
                writer.writerow(cls._header_row())
            writer.writerow(row)

        state["has_data"] = True
        state["last_timestamp"] = row[0]

    @classmethod
    def _rewrite_with_row(cls, filepath, row):
        existing_rows = []
        timestamp_exists = False

        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            with open(filepath, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header and len(header) != 2 + len(config.FREQ_BINS):
                    raise ValueError("Frequency bins mismatch.")

                for existing_row in reader:
                    if not existing_row:
                        continue
                    if existing_row[0] > row[0]:
                        continue
                    existing_rows.append(existing_row)
                    if existing_row[0] == row[0]:
                        timestamp_exists = True

        if timestamp_exists:
            cls._file_states[filepath] = {
                "has_data": True,
                "last_timestamp": existing_rows[-1][0] if existing_rows else None,
            }
            return

        existing_rows.append(row)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(cls._header_row())
            writer.writerows(existing_rows)

        cls._file_states[filepath] = {
            "has_data": True,
            "last_timestamp": row[0],
        }

    @staticmethod
    def load_psd_from_time(start_time_dt):
        threshold = start_time_dt
        loaded_data = []

        pattern = os.path.join(config.BASE_DIR, "dsa_*.csv")
        for filepath in sorted(glob.glob(pattern)):
            try:
                filename = os.path.basename(filepath)
                file_date = None
                try:
                    if filename.startswith("dsa_"):
                        file_date = dt.strptime(filename[4:-4], "%Y-%m-%d").date()
                except ValueError:
                    pass

                with open(filepath, "r", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if not header:
                        continue

                    has_duration = "duration" in header
                    psd_offset = 2 if has_duration else 1
                    if len(header) != psd_offset + len(config.FREQ_BINS):
                        continue

                    for row in reader:
                        if not row:
                            continue
                        if len(row) != psd_offset + len(config.FREQ_BINS):
                            continue
                        try:
                            ts = row[0]
                            try:
                                row_dt = dt.fromisoformat(ts)
                            except ValueError:
                                if file_date is None:
                                    continue
                                row_dt = dt.combine(
                                    file_date,
                                    dt.strptime(ts, "%H:%M:%S.%f").time()
                                )

                            if row_dt < threshold:
                                continue

                            duration = float(row[1]) if has_duration else config.TIME_RESOLUTION

                            psd = np.fromiter(
                                (float(v) if v and v.lower() != "nan" else np.nan for v in row[psd_offset:]),
                                dtype=np.float32
                            )
                            if psd.size != len(config.FREQ_BINS):
                                continue

                            loaded_data.append((row_dt.timestamp(), duration, psd))

                        except (ValueError, IndexError):
                            continue

            except Exception as e:
                print(f"Error loading {filepath}: {e}")

        loaded_data.sort(key=lambda x: x[0])
        return loaded_data

    @staticmethod
    def save_psd_to_csv(timestamp, duration, psd):
        psd = np.asarray(psd).ravel()

        try:
            ts_dt = (
                dt.now() if timestamp is None else
                timestamp if isinstance(timestamp, dt) else
                dt.fromtimestamp(timestamp) if isinstance(timestamp, (int, float)) else
                dt.fromisoformat(str(timestamp))
            )
        except ValueError:
            ts_dt = dt.now()

        ts_str = ts_dt.strftime("%H:%M:%S.%f")[:-3]
        filepath = Output._build_filename(config.BASE_DIR, ts_dt.timestamp())
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        row = [ts_str, f"{float(duration):g}"] + [np.round(x, 4) if np.isfinite(x) else "nan" for x in psd]

        with Output._state_lock:
            Output._delete_stale_csv_files()
            state = Output._get_file_state(filepath)
            last_timestamp = state["last_timestamp"]
            if last_timestamp is None:
                Output._append_row(filepath, state, row)
                return
            if ts_str == last_timestamp:
                return
            if ts_str > last_timestamp:
                Output._append_row(filepath, state, row)
                return
            Output._rewrite_with_row(filepath, row)
