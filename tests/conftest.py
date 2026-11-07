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
