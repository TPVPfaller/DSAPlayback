import sys

import plotdsa.app.main as _main
import plotdsa.config as _config
import plotdsa.core.buffers as _buffers
import plotdsa.core.calculations as _calculations
import plotdsa.ui.views as _views

sys.modules["config"] = _config
sys.modules["buffers"] = _buffers
sys.modules["calculations"] = _calculations
sys.modules["views"] = _views
sys.modules["main"] = _main
