import time
from pathlib import Path
from threading import Event

import pytest
from PIL import Image, ImageChops
from PySide6.QtCore import QPointF, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

import regimagen.app as desktop
from regimagen.codex import Cancelled, CodexError
from regimagen.document import Document, render
from regimagen.document_fill import removed_object_layers
from regimagen.fill import FillCandidate
from regimagen.project import load_project
from regimagen.remove_detection import RemovalTarget
from regimagen.storage import Store


def _wait_for_task(window):
    deadline = time.monotonic() + 10
    while (window.task is not None or window.editor.edits_pending) and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    assert window.task is None and not window.editor.edits_pending


def _draw_rectangle(editor, start, end):
    editor.tool.setCurrentIndex(editor.tool.findData("rectangle"))
    viewport = editor.canvas.viewport()
    first, last = (editor.canvas.mapFromScene(QPointF(*point)) for point in (start, end))
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=first)
    QTest.mouseMove(viewport, last)
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=last)


def test_remove_reviews_then_applies_background_and_objects_in_one_edit(qt_app, tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    window = desktop.Window(Store(tmp_path / "workspace"), settings)
    window.show()
    base = Image.new("RGBA", (96, 96))
    base.paste((65, 49, 32, 255), (12, 12, 88, 88))
    document = Document.from_image(base, "Test canvas")
    calls = []
    generating = Event()

    def detect(_base, selection, _executable, *, point, cancelled):
        assert point is None and not cancelled.is_set()
        assert selection.getpixel((24, 24))
        assert not selection.getpixel((36, 24))
        calls.append("detect")
        return (
            RemovalTarget("Antenna", Image.new("L", (22, 18), 255), (18, 19)),
            RemovalTarget("Housing", Image.new("L", (11, 18), 255), (38, 19),
                          selected=False, protected=True),
        )

    def generate(reference, core, footprint, labels, _executable, *, cancelled, prepare, direct):
        calls.append("generate")
        assert prepare is (calls.count("generate") > 1)
        assert labels == ("Antenna",) and not direct
        assert reference.mode == "RGBA"
        assert core.getpixel((36, 24)) == 255
        assert footprint.getpixel((39, 24)) == 0
        if calls.count("generate") == 2:
            raise CodexError("Try another variation.")
        if calls.count("generate") == 3:
            generating.set()
            assert cancelled.wait(5)
            raise Cancelled("Removal canceled.")
        return FillCandidate(Image.new("RGBA", reference.size), {
            "crop": (0, 0, 96, 96), "shift": (0, 0), "color": (0, 0, 0), "feather": 1,
        })

    def extract(*args, **kwargs):
        calls.append("cutouts")
        return removed_object_layers(*args, **kwargs)

    monkeypatch.setattr(window, "_codex_resolver", lambda: lambda: Path("mock-codex.exe"))
    monkeypatch.setattr(desktop, "detect_removal", detect)
    monkeypatch.setattr(desktop, "generate_remove", generate)
    monkeypatch.setattr(desktop, "removed_object_layers", extract)
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
        assert window.task_status.text() == "Selecting with Codex…"
        _wait_for_task(window)
        assert calls == ["detect"]
        assert remove._generate.isEnabled() and not remove._apply.isEnabled()
        remove._generate.click()
        _wait_for_task(window)
        assert remove.selected_variant == 0
        assert editor.document is document
        assert remove._background.isChecked()
        variant = window.layer_fill.variants[0]
        assert variant.background.compose(base).getpixel((36, 24)) == (0, 0, 0, 0)
        assert variant.preview.compose(base).getpixel((36, 24)) == base.getpixel((36, 24))
        remove._before.click()
        assert remove._before.isChecked() and remove._removal_outline.isVisible()
        remove._after.click()
        assert remove._after.isChecked() and not remove._removal_outline.isVisible()
        remove._background.click()
        assert remove._background.isChecked() and not remove._removal_outline.isVisible()

        remove._generate.click()
        _wait_for_task(window)
        assert remove.selected_variant == 0 and remove._apply.isEnabled()
        remove._generate.click()
        assert generating.wait(3)
        remove._cancel.click()
        _wait_for_task(window)
        assert remove.selected_variant == 0 and remove._apply.isEnabled()
        assert calls == ["detect", "cutouts", "generate", "generate", "generate"]

        footprint = remove.removal_inputs().footprint
        history_index = editor.undo.index()
        remove._apply.click()
        _wait_for_task(window)
        assert window.task_status.text() == "Applied Generative Remove and 1 object layer. Undo with Ctrl+Z."
        assert window.layer_fill is None and window.stack.currentWidget() is editor
        applied = editor.document
        assert editor.undo.index() == history_index + 1
        assert len(applied.layers) == 3
        background, cutout = applied.layers[-2:]
        assert background.name == "Generative Remove" and background.coverage is not None
        assert cutout.name == "Antenna" and cutout.coverage is None and cutout.visible
        assert (cutout.x, cutout.y) == (18, 19)
        assert editor.active_layer is cutout
        rendered = render(applied)
        assert rendered.getpixel((36, 24)) == base.getpixel((36, 24))
        assert rendered.getpixel((39, 24)) == base.getpixel((39, 24))
        for band in ImageChops.difference(render(document), rendered).split():
            assert ImageChops.multiply(band, ImageChops.invert(footprint)).getbbox() is None

        editor.undo.undo()
        assert editor.document is document
        editor.undo.redo()
        assert editor.active_layer is cutout
        editor.layers.currentItem().setCheckState(Qt.CheckState.Unchecked)
        assert render(editor.document).tobytes() == variant.background.compose(base).tobytes()
        editor.undo.undo()
        destination = tmp_path / "removed.regimagen"
        monkeypatch.setattr(window, "_save_destination", lambda *_: destination)
        window.save_as_action.trigger()
        _wait_for_task(window)
        reopened = load_project(destination)
        assert reopened.layers[-2].coverage is not None
        assert reopened.layers[-1].coverage is None and reopened.layers[-1].visible
        assert reopened.layers[-1].image.tobytes() == cutout.image.tobytes()
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


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_direct_lasso_preserves_unselected_objects_and_reuses_cutouts(qt_app, tmp_path, monkeypatch, mode):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    window = desktop.Window(Store(tmp_path / "workspace"), settings)
    window.show()
    backdrop = (90, 120, 70)
    base = Image.new(mode, (96, 96), backdrop if mode == "RGB" else (0, 0, 0, 0))
    subject = (20, 20, 40, 40)
    shadow = (40, 45, 49, 56)
    neighbor = (41, 26, 43, 42) if mode == "RGBA" else (51, 26, 60, 42)
    base.paste((136, 31, 22) if mode == "RGB" else (136, 31, 22, 255), subject)
    base.paste((24, 179, 216) if mode == "RGB" else (24, 179, 216, 255), neighbor)
    if mode == "RGB":
        base.paste((59, 79, 46), shadow)

    calls = []
    checks = []

    def detect(reference, selection, executable, *, direct, cancelled):
        assert reference.tobytes() == base.tobytes() and executable == Path("mock-codex.exe")
        assert direct and selection.getpixel((18, 18)) and selection.getpixel((neighbor[0], 32))
        calls.append("detect")
        return (RemovalTarget("Sprite", Image.new("L", (20, 20), 255), subject[:2], selected=len(calls) > 1),)

    def check_effects(reference, targets, executable, *, cancelled, lasso):
        assert reference.tobytes() == base.tobytes() and executable == Path("mock-codex.exe")
        assert targets[0].selected and not cancelled.is_set()
        assert lasso.getpixel((42, 45)) and not lasso.getpixel((42, 53))
        checks.append(mode)
        if mode == "RGBA":
            return targets
        return (
            *targets,
            RemovalTarget("Cast shadow", Image.new("L", (9, 11), 255), shadow[:2], parent=0),
            RemovalTarget("Possible reflection", Image.new("L", (3, 3), 255), (58, 43), selected=False, parent=0),
        )

    def generate(_reference, core, footprint, labels, _executable, *, direct, **_options):
        assert mode == "RGB" and direct and labels == ("Sprite", "Cast shadow")
        assert core.getpixel((24, 24)) and core.getpixel((42, 53))
        assert not footprint.getpixel((55, 32))
        # Simulate a generated crop that omits the unrelated sprite.
        return FillCandidate(Image.new("RGB", base.size, backdrop), {
            "crop": (0, 0, 96, 96), "shift": (0, 0), "color": (0, 0, 0), "feather": 1,
        })

    if mode == "RGB":
        monkeypatch.setattr(desktop, "generate_remove", generate)
    monkeypatch.setattr(window, "_codex_resolver", lambda: lambda: Path("mock-codex.exe"))
    monkeypatch.setattr(desktop, "detect_removal", detect)
    monkeypatch.setattr(desktop, "detect_removal_effects", check_effects)
    try:
        window._show_document(Document.from_image(base, "Direct Lasso"))
        window.editor.remove_button.click()
        _wait_for_task(window)
        remove = window.layer_fill.editor
        remove._shapes.setCurrentIndex(remove._shapes.findData("direct-lasso"))
        canvas = remove._canvas
        path = [canvas.mapFromScene(QPointF(x, y)) for x, y in [(16, 16), (69, 16), (69, 49), (16, 49)]]
        QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=path[0])
        for point in path[1:]:
            QTest.mouseMove(canvas.viewport(), point)
        QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, pos=path[-1])
        assert window.task is None and calls == []
        assert remove._generate.isEnabled() and remove.selected_variant is None
        drawn_area = remove.detection_mask.point([0] + [255] * 255)
        assert remove.removal_inputs().core.tobytes() == drawn_area.tobytes()
        remove._generate.click()
        _wait_for_task(window)
        assert calls == ["detect"] and checks == [] and remove.selected_variant is None
        assert "uncertain object" in remove._status.text()
        assert remove._redetect.isEnabled() and remove._generate.isEnabled()
        assert remove.removal_inputs().core.tobytes() == drawn_area.tobytes()

        remove._generate.click()
        _wait_for_task(window)
        assert calls == ["detect", "detect"] and remove._apply.isEnabled()
        assert checks == [mode]
        first = window.layer_fill.variants[0]
        assert ("could not confirm every shadow" in first.background.warning) is (mode == "RGB")
        remove._generate.click()
        _wait_for_task(window)
        assert calls == ["detect", "detect"] and remove.selected_variant == 1
        assert checks == [mode]
        second = window.layer_fill.variants[1]
        assert first.document.layers[-1] is second.document.layers[-1]
        assert remove._background.isChecked()
        remove._apply.click()
        _wait_for_task(window)
        cutout = window.editor.active_layer
        assert cutout.name == "Sprite" and cutout.image.size == (20, 20)
        after = base.convert("RGBA")
        if mode == "RGB":
            after.paste((*backdrop, 255), shadow)
        assert render(window.editor.document).tobytes() == after.tobytes()
        window.editor.layers.currentItem().setCheckState(Qt.CheckState.Unchecked)
        expected = after.copy()
        expected.paste((*backdrop, 255) if mode == "RGB" else (0, 0, 0, 0), subject)
        assert render(window.editor.document).tobytes() == expected.tobytes()
    finally:
        if window.task is not None:
            window.task.cancelled.set()
            _wait_for_task(window)
        monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Discard)
        window.close()
        qt_app.processEvents()


@pytest.mark.parametrize("previous_variant", [False, True])
def test_cancel_rejects_a_removal_result_already_queued_for_publication(qt_app, tmp_path, monkeypatch, previous_variant):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    window = desktop.Window(Store(tmp_path / "workspace"), settings)
    window.show()
    base = Image.new("RGBA", (96, 96))
    base.paste((136, 31, 22, 255), (20, 20, 40, 40))
    document = Document.from_image(base, "Sprite")
    build_variant = desktop.removal_variant
    rendered = Event()
    release = Event()

    def finish(*args, **kwargs):
        variant = build_variant(*args, **kwargs)
        rendered.set()
        assert release.wait(5)
        return variant

    try:
        window._show_document(document)
        window.editor.remove_button.click()
        _wait_for_task(window)
        remove = window.layer_fill.editor
        remove.set_targets((RemovalTarget("Sprite", Image.new("L", (20, 20), 255), (20, 20)),), detailed=True)
        if previous_variant:
            remove._generate.click()
            _wait_for_task(window)
            assert remove._apply.isEnabled()
        previous = tuple(window.layer_fill.variants)
        monkeypatch.setattr(desktop, "removal_variant", finish)
        remove._generate.click()
        assert rendered.wait(5)
        # Queue the result before delivering the Cancel click on the GUI thread.
        release.set()
        assert window.task.wait(5000)
        remove._cancel.click()
        _wait_for_task(window)

        assert tuple(window.layer_fill.variants) == previous
        assert remove.selected_variant == (0 if previous_variant else None)
        assert remove._apply.isEnabled() is previous_variant
        assert window.editor.document is document
    finally:
        release.set()
        if window.task is not None:
            window.task.cancelled.set()
            _wait_for_task(window)
        window.close()
        qt_app.processEvents()
