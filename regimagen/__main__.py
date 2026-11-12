"""Desktop application entry point."""

import sys
from pathlib import Path

from PySide6.QtCore import QLocale, QSettings, QStandardPaths
from PySide6.QtWidgets import QApplication, QMessageBox

from regimagen.app import Window, apply_style
from regimagen.automation import Operations
from regimagen.storage import Store
from regimagen.workspace import WorkspaceBusy, exclusive_workspace


def main() -> int:
    QLocale.setDefault(QLocale("en_US"))
    app = QApplication(sys.argv)
    app.setApplicationName("Regimagen")
    app.setApplicationDisplayName("Regimagen")
    apply_style(app)
    root = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation))
    try:
        with exclusive_workspace(root):
            Operations(root).require_idle()
            window = Window(Store(root), QSettings("Regimagen", "Regimagen"))
            window.show()
            return app.exec()
    except WorkspaceBusy as exc:
        QMessageBox.information(None, "Regimagen", str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
