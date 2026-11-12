import os

import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    application = QApplication.instance()
    if application is not None and not isinstance(application, QApplication):
        # CLI tests exercise their entry point in-process with QCoreApplication.
        application.shutdown()
        application = None
    return application or QApplication([])
