import time
from pathlib import Path

from PIL import Image, ImageChops
from PySide6.QtCore import QPointF, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

import regimagen.app as desktop
from regimagen.codex import Cancelled, CodexError
from regimagen.document import Document, render
from regimagen.fill import FillCandidate
from regimagen.project import load_project
from regimagen.remove_detection import RemovalTarget
from regimagen.storage import Store


def _wait_for_task(window):
    deadline = time.monotonic() + 10
    while window.task is not None and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    assert window.task is None


def _draw_rectangle(editor, start, end):
    editor.tool.setCurrentIndex(editor.tool.findData("rectangle"))
    viewport = editor.canvas.viewport()
    first, last = (editor.canvas.mapFromScene(QPointF(*point)) for point in (start, end))
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=first)
    QTest.mouseMove(viewport, last)
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=last)


def test_remove_reviews_targets_then_applies_and_saves_a_replacement_layer(qt_app, tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    window = desktop.Window(Store(tmp_path / "workspace"), settings)
    window.show()
    base = Image.new("RGBA", (96, 96))
    base.paste((65, 49, 32, 255), (12, 12, 88, 88))
    document = Document.from_image(base, "Test canvas")
    calls = []

    def detect(_base, selection, _executable, *, cancelled):
        assert not cancelled.is_set()
        assert selection.getpixel((24, 24))
        assert not selection.getpixel((36, 24))
        calls.append("detect")
        return (
            RemovalTarget("Antenna", Image.new("L", (22, 18), 255), (18, 19)),
            RemovalTarget("Housing", Image.new("L", (11, 18), 255), (38, 19),
                          selected=False, protected=True),
        )

    def generate(reference, core, footprint, labels, _executable, *, cancelled, prepare):
        calls.append("generate")
        assert prepare is (calls.count("generate") > 1)
        assert labels == ("Antenna",)
        assert reference.mode == "RGBA"
        assert core.getpixel((36, 24)) == 255
        assert footprint.getpixel((39, 24)) == 0
        if calls.count("generate") == 2:
            raise CodexError("Try another variation.")
        if calls.count("generate") == 3:
            assert cancelled.wait(5)
            raise Cancelled("Removal canceled.")
        return FillCandidate(Image.new("RGBA", reference.size), {
            "crop": (0, 0, 96, 96), "shift": (0, 0), "color": (0, 0, 0), "feather": 1,
        })

    monkeypatch.setattr(window, "_codex_resolver", lambda: lambda: Path("mock-codex.exe"))
    monkeypatch.setattr(desktop, "segment_removal_lasso", lambda *args, **kwargs: ())
    monkeypatch.setattr(desktop, "detect_removal", detect)
    monkeypatch.setattr(desktop, "generate_remove", generate)
    try:
        window._show_document(document)
        qt_app.processEvents()
        editor = window.editor
        _draw_rectangle(editor, (20, 20), (30, 30))
        editor.remove_button.click()
        _wait_for_task(window)
        remove = window.layer_fill.editor
        assert not remove._generate.isEnabled()
        remove._detect.click()
        _wait_for_task(window)
        assert calls == ["detect"]
        assert remove._generate.isEnabled() and not remove._apply.isEnabled()
        remove._generate.click()
        _wait_for_task(window)
        assert remove.selected_variant == 0
        assert editor.document is document

        remove._generate.click()
        _wait_for_task(window)
        assert remove.selected_variant == 0 and remove._apply.isEnabled()
        remove._generate.click()
        remove._cancel.click()
        _wait_for_task(window)
        assert remove.selected_variant == 0 and remove._apply.isEnabled()
        assert calls == ["detect", "generate", "generate", "generate"]

        _, footprint, _ = remove.removal_inputs()
        remove._apply.click()
        assert window.layer_fill is None and window.stack.currentWidget() is editor
        assert len(editor.document.layers) == 2
        applied = editor.document
        assert applied.layers[-1].name == "Generative Remove"
        assert applied.layers[-1].coverage is not None
        rendered = render(applied)
        assert rendered.getpixel((36, 24)) == (0, 0, 0, 0)
        assert rendered.getpixel((39, 24)) == base.getpixel((39, 24))
        for band in ImageChops.difference(render(document), rendered).split():
            assert ImageChops.multiply(band, ImageChops.invert(footprint)).getbbox() is None

        editor.undo.undo()
        assert editor.document is document
        editor.undo.redo()
        editor.layers.currentItem().setCheckState(Qt.CheckState.Unchecked)
        assert render(editor.document).tobytes() == render(document).tobytes()
        editor.undo.undo()
        destination = tmp_path / "removed.regimagen"
        monkeypatch.setattr(window, "_save_destination", lambda *_: destination)
        window.save_as_action.trigger()
        _wait_for_task(window)
        reopened = load_project(destination)
        assert reopened.layers[-1].coverage is not None
        assert render(reopened).tobytes() == rendered.tobytes()
        exported = tmp_path / "removed.png"
        monkeypatch.setattr(window, "_save_destination", lambda *_: exported)
        window.export_action.trigger()
        _wait_for_task(window)
        with Image.open(exported) as output:
            assert output.mode == "RGBA"
            assert output.tobytes() == rendered.tobytes()
    finally:
        if window.task is not None:
            window.task.cancelled.set()
            _wait_for_task(window)
        monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Discard)
        window.close()
        qt_app.processEvents()


def test_remove_click_clears_an_isolated_sprite_without_codex(qt_app, tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    window = desktop.Window(Store(tmp_path / "workspace"), settings)
    window.show()
    base = Image.new("RGBA", (96, 96))
    base.paste((136, 31, 22, 255), (20, 20, 40, 40))
    base.paste((58, 128, 88, 255), (54, 45, 84, 82))
    clicks = []

    def segment(_base, point, *, cancelled):
        assert not cancelled.is_set()
        clicks.append(point)
        return (RemovalTarget("Clicked object", Image.new("L", (20, 20), 255), (20, 20)),)

    def unexpected_codex():
        raise AssertionError("Local selection and isolated transparency do not require Codex.")

    monkeypatch.setattr(desktop, "segment_removal_click", segment)
    monkeypatch.setattr(window, "_codex_resolver", lambda: unexpected_codex)
    try:
        window._show_document(Document.from_image(base, "Transparent objects"))
        window.editor.remove_button.click()
        _wait_for_task(window)
        remove = window.layer_fill.editor
        target = remove._canvas.mapFromScene(QPointF(30, 30))
        QTest.mouseClick(remove._canvas.viewport(), Qt.MouseButton.LeftButton, pos=target)
        _wait_for_task(window)
        assert len(clicks) == 1 and all(abs(value - 30) <= 1 for value in clicks[0])
        assert remove._generate.isEnabled() and remove.selected_variant is None
        remove._generate.click()
        _wait_for_task(window)
        assert remove._apply.isEnabled()
        remove._apply.click()
        expected = base.copy()
        expected.paste((0, 0, 0, 0), (20, 20, 40, 40))
        assert render(window.editor.document).tobytes() == expected.tobytes()
    finally:
        if window.task is not None:
            window.task.cancelled.set()
            _wait_for_task(window)
        monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Discard)
        window.close()
        qt_app.processEvents()
