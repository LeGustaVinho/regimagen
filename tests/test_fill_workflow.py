import io
import time
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

import regimagen.app as app_module
from regimagen.codex import GeneratedImage, ImageTurnRejected
from regimagen.files import load_source
from regimagen.geometry import FRAME, Grid
from regimagen.imaging import read_tile
from regimagen.storage import Store

CONTEXT_WARNING = "Could not verify that ImageGen used the context references."
CONTEXT_ERROR = "ImageGen used different context references. Generate another variation."


def wait_for_task(window):
    deadline = time.monotonic() + 10
    while window.task is not None and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    assert window.task is None


def lasso(editor, points):
    canvas = editor._canvas
    vertices = [canvas.mapFromScene(QPointF(x, y)) for x, y in points]
    QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=vertices[0])
    for vertex in vertices[1:]:
        QTest.mouseMove(canvas.viewport(), vertex)
    QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, pos=vertices[-1])


def test_desktop_applies_then_reopens_and_reduces_fill_without_generation(tmp_path, monkeypatch, qt_app):
    store = Store(tmp_path / "workspace")
    original = tmp_path / "original.png"
    Image.new("RGB", (192, 160), (32, 52, 61)).save(original)
    calls = []

    class Generator:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def generate(self, prompt, references, *, verify_references):
            calls.append(prompt)
            assert verify_references
            assert len(references) == 3
            if len(calls) == 2:
                raise ImageTurnRejected(CONTEXT_ERROR)
            with Image.open(references[0]) as reference:
                assert reference.size == (1024, 1024)
            output = io.BytesIO()
            Image.new("RGB", (96, 96), (197, 113, 43)).save(output, format="PNG")
            return GeneratedImage(output.getvalue(), "fill-test", references_verified=None)

    generate = app_module.generate_fill
    def generate_with_fixture(*args, **kwargs):
        return generate(*args, **kwargs, client_factory=Generator)

    monkeypatch.setattr(app_module, "generate_fill", generate_with_fixture)
    monkeypatch.setattr(app_module.Window, "_codex_resolver", lambda _self: lambda: Path("mock-codex.exe"))
    window = app_module.Window(store, QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    window.show()
    try:
        window._import(original)
        wait_for_task(window)
        assert window.stack.currentWidget() is window.editor
        window.editor.reconstruct_button.click()
        wait_for_task(window)
        base = window.source
        window.comparison.fill_button.click()
        wait_for_task(window)
        editor = window.fill.editor
        qt_app.processEvents()
        lasso(editor, [(38, 32), (155, 32), (155, 129), (38, 129)])
        editor._prompt.setText("Replace the middle with copper")
        editor._generate.click()
        wait_for_task(window)
        assert len(calls) == 1 and editor._apply.isEnabled()
        assert CONTEXT_WARNING in editor._status.text()

        editor._generate.click()
        wait_for_task(window)
        assert len(calls) == 2 and editor._apply.isEnabled()
        assert editor.selected_variant == 0 and editor._chooser.count() == 1
        assert CONTEXT_ERROR in editor._status.text()

        editor._apply.click()
        wait_for_task(window)
        assert "Saved image version" in window.task_status.text() and "Recent" in window.task_status.text()
        first = window.source
        first_fill = store.get_fill(first)
        assert first_fill.base.id == base.id
        assert first_fill.metadata["references_verified"] is None
        first_pixels = np.asarray(load_source(first.path))
        assert not np.array_equal(first_pixels[80, 130], first_pixels[0, 0])

        window.version_button.menu().actions()[0].trigger()
        busy_at_reopen = []
        window.task.succeeded.connect(lambda _: busy_at_reopen.append(not window.fill.editor._prompt.isEnabled()))
        wait_for_task(window)
        editor = window.fill.editor
        assert busy_at_reopen == [True]
        assert "enlarged to fit" in editor._status.text()
        assert CONTEXT_WARNING in editor._status.text()
        assert not editor._apply.isEnabled()
        window._apply_fill()
        assert window.task is None and store.fill_sources() == [first, base]

        editor._prompt.setText("A different prompt")
        editor._prompt.setText("Replace the middle with copper")
        assert editor.selected_variant == 0 and not editor._apply.isEnabled()
        qt_app.processEvents()
        next(button for button in editor._selection_tools.buttons() if button.text() == "Subtract").click()
        lasso(editor, [(100, 20), (167, 20), (167, 142), (100, 142)])
        wait_for_task(window)
        assert len(calls) == 2 and editor._apply.isEnabled()
        assert CONTEXT_WARNING in editor._status.text()
        editor._apply.click()
        wait_for_task(window)
        second = window.source
        assert second.id != first.id
        saved = store.get_fill(second)
        assert saved.base.id == base.id and saved.derived_from == first.id
        before = np.asarray(load_source(base.path))
        after = np.asarray(load_source(second.path))
        np.testing.assert_array_equal(before[:, 102:], after[:, 102:])
        np.testing.assert_array_equal(first_pixels[80, 66], after[80, 66])
        assert window.history.count() >= 4

        job = store.ensure_job(base, Grid(2, 2))
        tile = job.folder / "sample-tile.png"
        tile.parent.mkdir(exist_ok=True)
        Image.new("RGB", (FRAME, FRAME), "blue").save(tile)
        warnings = []
        monkeypatch.setattr(QMessageBox, "information", lambda _parent, _title, msg: warnings.append(msg))
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args, **_options: (str(tile), ""))
        window._export_fill()
        assert window.task is None and "managed files" in warnings[0]
        assert tuple(read_tile(tile)[0, 0]) == (0, 0, 255)

        exported = tmp_path / "exported.png"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args, **_options: (str(exported), ""))
        window._export_fill()
        wait_for_task(window)
        assert exported.read_bytes() == second.path.read_bytes()
    finally:
        if window.task is not None:
            window.task.cancelled.set()
            wait_for_task(window)
        window.close()


def test_desktop_reopens_and_reduces_legacy_fill_without_reference_metadata(tmp_path, monkeypatch, qt_app):
    store = Store(tmp_path / "workspace")
    image = Image.new("RGB", (192, 160), (32, 52, 61))
    base = store.import_source(image, "original.png")
    bounds = (38, 32, 155, 129)
    mask = Image.new("L", image.size)
    mask.paste(255, bounds)
    filled = image.copy()
    copper = (197, 113, 43)
    filled.paste(copper, bounds)
    saved = store.save_fill(
        base, filled, mask, Image.new("RGB", (160, 160), copper),
        prompt="Replace the middle with copper", edit_alpha=False,
        metadata={"crop": [0, 0, 160, 160], "shift": [0, 0], "color": [0, 0, 0], "feather": 1},
        base_sha256=sha256(base.path.read_bytes()).hexdigest(),
    )
    generation_calls = []
    monkeypatch.setattr(app_module, "generate_fill", lambda *_args, **_kwargs: generation_calls.append(True))
    window = app_module.Window(store, QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    window.show()
    try:
        saved_index = next(
            index for index in range(window.history.count()) if window.history.itemData(index) == saved
        )
        window.history.setCurrentIndex(saved_index)
        wait_for_task(window)
        assert window.source == saved
        window.version_button.menu().actions()[0].trigger()
        wait_for_task(window)
        editor = window.fill.editor
        assert editor.selected_variant == 0 and not editor._apply.isEnabled()
        assert CONTEXT_WARNING not in editor._status.text()

        qt_app.processEvents()
        next(button for button in editor._selection_tools.buttons() if button.text() == "Subtract").click()
        lasso(editor, [(100, 20), (167, 20), (167, 142), (100, 142)])
        wait_for_task(window)
        assert editor._apply.isEnabled() and CONTEXT_WARNING not in editor._status.text()
        editor._apply.click()
        wait_for_task(window)

        assert not generation_calls
        assert store.get_fill(window.source).derived_from == saved.id
        output = load_source(window.source.path)
        assert output.getpixel((70, 70)) == copper
        assert output.getpixel((130, 70)) == image.getpixel((130, 70))
    finally:
        if window.task is not None:
            window.task.cancelled.set()
            wait_for_task(window)
        window.close()
