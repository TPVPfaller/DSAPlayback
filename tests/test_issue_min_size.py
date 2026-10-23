import sys
import pytest
from PySide6.QtWidgets import QApplication
from main import DSAApplication
import config

@pytest.fixture(scope="session")
def app():
    """Create the QApplication once for all tests."""
    test_app = QApplication.instance()
    if test_app is None:
        test_app = QApplication(sys.argv)
    return test_app

@pytest.fixture
def dsa_app(qtbot, app, monkeypatch):
    """Create a DSAApplication instance for testing."""
    class DummyWorker:
        def __init__(self, user_config):
            self.user_config = user_config

        def apply_config(self, user_config):
            self.user_config = user_config

        def stop(self):
            pass

    class DummyThread:
        def quit(self):
            pass

        def wait(self):
            pass

    def fake_init_worker(self):
        self.thread = DummyThread()
        self.worker = DummyWorker(self.user_config)

    monkeypatch.setattr(DSAApplication, "_init_worker", fake_init_worker)

    window = DSAApplication()
    qtbot.addWidget(window)
    return window

def test_minimum_sizes(dsa_app):
    """Verify that DSA plot and main window have the correct minimum sizes."""
    # Check DSA plot minimum height
    assert dsa_app.dsa_view.minimumHeight() == config.MIN_DSA_HEIGHT
    
    # Check main window minimum size
    min_size = dsa_app.minimumSize()
    assert min_size.width() == config.MIN_WINDOW_WIDTH
    assert min_size.height() == config.MIN_WINDOW_HEIGHT
