# Third-party notices

The Windows package includes the components below. Their license terms apply to the respective components. License texts supplied by dependencies are preserved in `_internal` under their `.dist-info` directories. Additional texts are in `licenses/`.

## Qt for Python and Qt

This package uses the PySide6 Essentials and Shiboken6 bindings and Qt libraries from version 6.11.3 under LGPL-3.0-only. Copies of the GNU Lesser General Public License v3 and the GNU General Public License v3 are in `licenses/`. These upstream libraries are included as separate dynamic libraries.

The corresponding sources are available from the Qt Project.

- [PySide6 and Shiboken6 6.11.3 source archive](https://download.qt.io/official_releases/QtForPython/pyside6/PySide6-6.11.3-src/pyside-setup-everywhere-src-6.11.3.tar.xz)
- [Qt Base 6.11.3 source archive](https://download.qt.io/official_releases/qt/6.11/6.11.3/submodules/qtbase-everywhere-src-6.11.3.tar.xz)
- [Qt SVG 6.11.3 source archive](https://download.qt.io/official_releases/qt/6.11/6.11.3/submodules/qtsvg-everywhere-src-6.11.3.tar.xz)
- [All Qt 6.11.3 module sources, including image format plugins](https://download.qt.io/official_releases/qt/6.11/6.11.3/submodules/)

To use an interface-compatible Windows x64 build of the Qt libraries or bindings, close both Regimagen executables and replace the relevant files in the extracted `_internal/PySide6` or `_internal/shiboken6` directories. Keep the same layout for DLLs, Python extension modules, and plugins.

## Other bundled components

| Component | License or notice location |
| --- | --- |
| CPython | `licenses/CPython.txt` |
| PyInstaller bootloader | `licenses/PyInstaller.txt`, including its bootloader exception |
| NumPy, Pillow, and OpenCV | Dependency license files under `_internal` |
| PyPNG | `licenses/PyPNG.txt` |
| Inter font | `_internal/regimagen/assets/Inter-OFL.txt` |
