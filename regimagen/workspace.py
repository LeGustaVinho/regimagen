"""Coordinate access to project artifacts across local processes."""

from contextlib import contextmanager
from pathlib import Path

from PySide6.QtCore import QLockFile


class WorkspaceBusy(RuntimeError):
    pass


def workspace_lock(root: Path) -> QLockFile:
    lock = QLockFile(str(root / "regimagen.lock"))
    lock.setStaleLockTime(0)
    return lock


@contextmanager
def exclusive_workspace(root: Path, *, timeout: int = 0):
    root.mkdir(parents=True, exist_ok=True)
    lock = workspace_lock(root)
    if not lock.tryLock(timeout):
        raise WorkspaceBusy("Another Regimagen process is using this workspace.")
    try:
        yield
    finally:
        lock.unlock()
