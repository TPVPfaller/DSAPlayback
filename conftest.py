import sys
from types import ModuleType

import plotdsa.app.main as _main
import plotdsa.app.worker as _worker
import plotdsa.config as _config
import plotdsa.core.buffers as _buffers
import plotdsa.core.calculations as _calculations
import plotdsa.ui.views as _views
from plotdsa.io.input import EEGStream
from plotdsa.io.output import Output

sys.modules["config"] = _config
sys.modules["buffers"] = _buffers
sys.modules["calculations"] = _calculations
sys.modules["views"] = _views
sys.modules["main"] = _main
sys.modules["worker"] = _worker

input_output = ModuleType("input_output")
input_output.EEGStream = EEGStream
input_output.Output = Output
sys.modules["input_output"] = input_output
