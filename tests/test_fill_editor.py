import pytest
from PIL import Image
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest

from regimagen.fill import FillPreview
from regimagen.fill_editor import FillEditor


@pytest.fixture
def make_editor(qt_app):
    editors = []

    def create(base, **kwargs):
        editor = FillEditor(base, **kwargs)
        editor.resize(640, 480)
        editor.show()
        qt_app.processEvents()
        editors.append(editor)
        return editor

    yield create
    for editor in editors:
        editor.close()
    qt_app.processEvents()


def drag_lasso(canvas, points):
    button = Qt.MouseButton.LeftButton
    points = [canvas.mapFromScene(QPointF(x, y)) for x, y in points]
    QTest.mousePress(canvas.viewport(), button, pos=points[0])
    for point in points[1:]:
        QTest.mouseMove(canvas.viewport(), point)
    QTest.mouseRelease(canvas.viewport(), button, pos=points[-1])


def test_lasso_keeps_image_coordinates_with_zoom_and_clips_to_canvas(make_editor):
    editor = make_editor(Image.new("RGB", (96, 80), "black"))
    canvas = editor._canvas
    drag_lasso(canvas, [(4, 4), (32, 4), (32, 30), (4, 30)])
    assert editor.mask.getpixel((10, 12)) == 255
    assert editor.mask.getpixel((44, 36)) == 0

    canvas.actual_size()
    canvas.zoom(2.5)
    next(button for button in editor._selection_tools.buttons() if button.text() == "Add").click()
    drag_lasso(canvas, [(52, 48), (102, 48), (102, 70), (52, 70)])
    assert editor.mask.getpixel((10, 12)) == 255
    assert editor.mask.getpixel((95, 56)) == 255
    assert editor.mask.size == (96, 80)

    next(button for button in editor._selection_tools.buttons() if button.text() == "Subtract").click()
    drag_lasso(canvas, [(60, 42), (75, 42), (75, 77), (60, 77)])
    assert editor.mask.getpixel((65, 60)) == 0
    assert editor.mask.getpixel((95, 56)) == 255


def test_variant_preview_replaces_alpha_and_invalidates_on_edit(make_editor):
    base = Image.new("RGBA", (80, 60), "red")
    editor = make_editor(base, mask=Image.new("L", base.size, 255), prompt="  Cut a hole  ")
    assert editor.prompt == "Cut a hole"
    assert editor.edit_alpha
    bounds = (20, 12, 63, 48)
    hole = Image.new("RGBA", (43, 36))
    editor.add_variant(FillPreview(hole, bounds[:2], "Review the edge."))
    editor.add_variant(FillPreview(base.crop(bounds), bounds[:2]))
    assert editor.selected_variant == 1
    assert editor._status.text() == "Variation ready. Use Before and After to compare."

    editor._chooser.setCurrentIndex(0)
    assert editor._status.text() == "Review the edge."
    canvas = editor._canvas
    center = canvas.mapFromScene(QPointF(40, 30))
    pixels = canvas.viewport().grab().toImage()
    after = pixels.pixelColor(center)
    assert after.alpha() == 255 and after.red() != 255
    assert pixels.pixelColor(canvas.mapFromScene(QPointF(10, 30))).red() == 255
    editor._before.click()
    before = canvas.viewport().grab().toImage().pixelColor(center)
    assert before.red() > before.green() + 100

    editor.set_busy(True, generating=True)
    canceled = []
    editor.cancel_requested.connect(lambda: canceled.append(True))
    editor._cancel.click()
    assert canceled and not editor._apply.isEnabled()
    editor.set_busy(False)

    observed = []
    editor.changed.connect(lambda: observed.append(editor.selected_variant))
    editor._prompt.setText("Cut a hole ")
    assert editor.selected_variant == 0
    editor._alpha.setChecked(False)
    assert observed == [None]
    assert not editor.edit_alpha and not editor._apply.isEnabled()


def test_generation_requires_visible_pixels_when_transparency_is_locked(make_editor):
    base = Image.new("RGBA", (80, 60))
    base.paste((255, 0, 0, 255), (50, 20, 75, 50))
    mask = Image.new("L", base.size)
    mask.paste(255, (5, 5, 30, 30))
    editor = make_editor(base, mask=mask, edit_alpha=False)

    assert not editor._generate.isEnabled()
    assert "turn on Edit Transparency" in editor._status.text()
    editor._alpha.setChecked(True)
    assert editor._generate.isEnabled()
    editor._alpha.setChecked(False)
    drag_lasso(editor._canvas, [(52, 22), (70, 22), (70, 42), (52, 42)])
    assert editor._generate.isEnabled()
