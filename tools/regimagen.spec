from pathlib import Path

from PyInstaller.utils.hooks import copy_metadata


root = Path(SPECPATH).parent
package = root / "regimagen"
data = [(str(package / "assets"), "regimagen/assets")]
for dependency in ("PySide6-Essentials", "shiboken6", "numpy", "Pillow", "opencv-python-headless", "pypng"):
    data.extend(copy_metadata(dependency))

desktop = Analysis([str(package / "__main__.py")], pathex=[str(root)], datas=data)
command_line = Analysis([str(package / "cli.py")], pathex=[str(root)])
icon = str(root / "build" / "Regimagen.ico")

desktop_exe = EXE(
    PYZ(desktop.pure), desktop.scripts, [], exclude_binaries=True,
    name="Regimagen", console=False, icon=icon, upx=False,
)
command_line_exe = EXE(
    PYZ(command_line.pure), command_line.scripts, [], exclude_binaries=True,
    name="regimagen-cli", console=True, icon=icon, upx=False,
)

bundle = COLLECT(
    desktop_exe, command_line_exe, desktop.binaries, command_line.binaries,
    desktop.datas, command_line.datas, name="Regimagen", upx=False,
)
