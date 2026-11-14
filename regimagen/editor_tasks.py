"""Background work whose results are applied by the editor on the GUI thread."""

from threading import Event

from PySide6.QtCore import QObject, QRunnable, Signal

from .codex import Cancelled


class _Signals(QObject):
    finished = Signal(object, object, str)


class EditorTask(QRunnable):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.cancelled = Event()
        self.signals = _Signals()

    def run(self):
        result, error = None, ""
        try:
            if not self.cancelled.is_set():
                result = self.operation(self.cancelled.is_set)
        except Cancelled:
            pass
        except Exception as exc:
            error = str(exc) or type(exc).__name__
        self.signals.finished.emit(self, result, error)
