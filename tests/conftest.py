import os
import sys
import types
from pathlib import Path

# Keep Qt headless for test runs.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


if "qdarktheme" not in sys.modules:
    qdarktheme = types.ModuleType("qdarktheme")
    qdarktheme.enable_hi_dpi = lambda: None
    qdarktheme.setup_theme = lambda *args, **kwargs: None
    sys.modules["qdarktheme"] = qdarktheme


if "pylsl" not in sys.modules:
    pylsl = types.ModuleType("pylsl")

    class StreamInlet:
        def __init__(self, stream):
            self.stream = stream

        def pull_sample(self, timeout=0):
            return None, None

    def resolve_byprop(*args, **kwargs):
        return []

    pylsl.StreamInlet = StreamInlet
    pylsl.resolve_byprop = resolve_byprop
    sys.modules["pylsl"] = pylsl
