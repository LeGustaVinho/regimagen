from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QFocusEvent, QPainterPath, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import regimagen.editor as editor_module
import regimagen.editor_preview as preview_module
from regimagen.codex import Cancelled
from regimagen.document import Document, Layer, render
from regimagen.editor import Editor
from regimagen.editor_preview import PreviewRegion, PreviewTiles, exact_preview, interactive_preview, interactive_region
from regimagen.geometry import Rect
from regimagen.viewer import qimage


def wait_for(condition):
    for _ in range(400):
        QApplication.processEvents()
        if condition():
            return
        QTest.qWait(5)
    assert condition()


@pytest.fixture
def editor(qt_app):
    editor = Editor()
    editor.resize(900, 600)
    editor.show()
    yield editor
    editor.close()
    qt_app.processEvents()


def test_preview_reuses_position_independent_tiles_and_preserves_fill_alpha(qt_app):
    size = 530, 300
    base = Layer("Base", Image.new("RGBA", size, "white"))
    pixels = Image.new("RGBA", size, "white")
    pixels.paste((110, 70, 25, 0), (30, 30, 300, 80))
    pixels.paste((90, 120, 200, 130), (30, 100, 300, 160))
    fill = Layer("Fill", pixels, coverage=Image.new("L", size, 255), mask=Image.new("L", size, 255))
    document = Document("Preview", *size, (base, fill))
    region = interactive_region(document.bounds, 1)
    cache = PreviewTiles()
    available = lambda: False

    first = interactive_preview(document, region, cache, available)
    assert first.pixelColor(256, 50).getRgb() == (0, 0, 0, 0)
    assert first.pixelColor(300, 50).getRgb() == (255, 255, 255, 255)
    tile = cache.tile(fill, 1, 0, 0, available)
    changes = {"x": -2, "opacity": 50}
    second = interactive_preview(document, region, cache, available, (fill.id, changes))
    expected = render(document.update(replace(fill, **changes)))
    assert cache.tile(fill, 1, 0, 0, available) is tile
    assert second.pixelColor(310, 50).getRgb() == (255, 255, 255, 255)
    for point in ((255, 50), (256, 50), (130, 130)):
        assert max(abs(a - b) for a, b in zip(second.pixelColor(*point).getRgb(), expected.getpixel(point))) <= 3

    limited = PreviewTiles(max_bytes=600_000)
    interactive_preview(document, region, limited, available)
    assert 0 < limited.bytes <= limited.max_bytes
    assert interactive_region(document.bounds, 0.26).step == 4


def test_exact_preview_matches_serial_stripes_with_transparency(qt_app, monkeypatch):
    size = (2160, 1190)
    base = Image.new("RGBA", size, (250, 160, 20, 210))
    base.paste((20, 150, 210, 255), (23, 850, 1900, 923))
    pixels = Image.new("RGBA", (2000, 1050), (50, 90, 120, 240))
    pixels.paste((80, 32, 52, 0), (30, 600, 1540, 960))
    coverage = Image.new("L", pixels.size, 255)
    coverage.paste(130, (0, 290, 1650, 980))
    document = Document("Stripes", *size, (Layer("Base", base), Layer("Fill", pixels, -9, 15, coverage=coverage)))
    region = PreviewRegion(Rect(7, 19, 2131, 1176), (711, 395))
    monkeypatch.setattr(preview_module, "process_cpu_count", lambda: 1)
    serial = exact_preview(document, region, lambda: False)
    monkeypatch.setattr(preview_module, "process_cpu_count", lambda: 4)
    assert exact_preview(document, region, lambda: False) == serial
    area = region.area
    native = PreviewRegion(area, (area.width, area.height))
    assert exact_preview(document, native, lambda: False) == qimage(np.asarray(render(document, area)))
    with pytest.raises(Cancelled):
        exact_preview(document, region, lambda: True)


def test_drag_displays_in_flight_frames_and_cancels_old_gestures(editor, monkeypatch):
    base = Layer("Base", Image.new("RGBA", (180, 100), "red"))
    moving = Layer("Moving", Image.new("RGBA", (20, 20), "blue"), 20, 20)
    document = Document("Drag", 180, 100, (base, moving))
    editor.set_document(document)
    preview = editor._preview
    wait_for(lambda: preview.shown is not None)
    requests = []
    first_gate, following_gate = Event(), Event()
    original_render = preview_module.interactive_preview

    def paused(doc, region, cache, cancelled, draft=None):
        if draft is not None:
            requests.append(draft[1])
            assert (first_gate if len(requests) == 1 else following_gate).wait(3)
        return original_render(doc, region, cache, lambda: False, draft)

    monkeypatch.setattr(preview_module, "interactive_preview", paused)
    canvas = editor.canvas
    viewport = canvas.viewport()
    start, first, latest, end = (canvas.mapFromScene(QPointF(x, 24)) for x in (24, 44, 74, 84))
    before = preview._displayed
    try:
        QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(viewport, first)
        wait_for(lambda: len(requests) == 1)
        QTest.mouseMove(viewport, latest)
        assert editor.document is document
        first_gate.set()
        wait_for(lambda: preview._displayed is not before and len(requests) >= 2)
        assert editor.canvas._image.pixmap().toImage().pixelColor(45, 24).name() == "#0000ff"
    finally:
        first_gate.set()
        following_gate.set()
    assert [request["x"] for request in requests] == [40, 70]
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=end)
    assert editor.active_layer.x == 80 and editor.undo.count() == 1
    wait_for(lambda: preview.shown is not None and preview.shown[0] is editor.document)
    exact = QPixmap.fromImage(preview_module._premultiplied(render(editor.document)))
    assert editor.canvas._image.pixmap().toImage() == exact.toImage()

    committed = editor.document
    item = editor.layers.currentItem()
    for cancel in (
        lambda: QTest.keyClick(canvas, Qt.Key.Key_Escape),
        lambda: QApplication.sendEvent(canvas, QFocusEvent(QEvent.Type.FocusOut)),
    ):
        QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(viewport, first)
        cancel()
        QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=end)
        assert editor.document is committed and preview.draft is None and editor.undo.count() == 1

    editor.undo.undo()
    assert editor.document is document and editor.layers.currentItem() is item
    editor.undo.redo()
    assert editor.document is committed and editor.layers.currentItem() is item

    replacement = Document.from_image(Image.new("RGBA", (110, 80), "green"), "Other")
    wait_for(lambda: preview._interactive is None)
    following_gate.clear()
    submitted = len(requests)
    try:
        QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(viewport, first)
        wait_for(lambda: len(requests) > submitted)
        editor.set_document(replacement)
        QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=end)
    finally:
        following_gate.set()
    wait_for(lambda: preview.shown is not None and preview.shown[0] is replacement)
    assert not editor.undo.canUndo() and preview.draft is None
    assert editor.canvas._image.pixmap().toImage().pixelColor(0, 0).name() == "#008000"


def test_pan_uses_overview_and_metadata_reuses_the_preview(editor):
    original = Document.from_image(Image.new("RGBA", (1800, 1000), "red"), "Overview")
    editor.set_document(original)
    preview = editor._preview
    wait_for(lambda: preview.shown is not None and preview._overview is not None)
    item = editor.layers.currentItem()
    frame = preview._displayed
    editor._update_layer("Rename layer", name="New name")
    editor.locked.setChecked(True)
    assert editor.layers.currentItem() is item and item.text() == "New name"
    assert preview._displayed is frame and preview.shown[0] is editor.document

    canvas = editor.canvas
    canvas.actual_size()
    canvas.centerOn(0, 0)
    canvas.moved.emit()
    wait_for(lambda: preview.shown is not None and preview.shown[1] != original.bounds)
    canvas.centerOn(original.width, original.height)
    canvas.moved.emit()
    assert preview._displayed is preview._overview and preview.shown is None
    wait_for(lambda: preview.shown is not None and preview.shown[1].right == original.width)


def test_old_selection_overlays_can_be_evicted_without_losing_undo(editor, monkeypatch):
    monkeypatch.setattr(editor_module, "_SELECTION_CACHE_BYTES", 14_000)
    editor.set_document(Document.from_image(Image.new("RGBA", (160, 120)), "Selection"))
    first = QPainterPath()
    first.addRect(10, 10, 50, 45)
    second = QPainterPath()
    second.addRect(65, 35, 55, 50)
    editor._select_pixels(first)
    wait_for(lambda: not editor.edits_pending)
    original = editor.selection
    editor._select_pixels(second)
    wait_for(lambda: not editor.edits_pending)
    assert editor._selection_bytes == 55 * 50 * 4
    editor.undo.undo()
    assert editor.selection is original
    wait_for(lambda: editor._overlay_task is None)
    assert not editor.canvas._selection.pixmap().isNull()
    assert editor.canvas._selection.pixmap().size().toTuple() == (50, 45)
