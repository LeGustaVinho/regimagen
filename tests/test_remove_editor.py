from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QPainterPath
from PySide6.QtTest import QSignalSpy, QTest

from regimagen.fill import FillPreview
from regimagen.remove_detection import RemovalTarget
from regimagen.remove_editor import RemoveEditor, removal_masks


def test_removal_masks_keep_deselected_objects_and_their_effects():
    size = (110, 80)
    targets = (
        RemovalTarget("first target", Image.new("L", (40, 35), 255), (4, 4)),
        RemovalTarget("second target", Image.new("L", (10, 11), 255), (43, 20)),
        RemovalTarget("second target's shadow", Image.new("L", (8, 7), 255), (40, 32), parent=1),
        RemovalTarget("foreground", Image.new("L", (5, 35), 255), (18, 4), protected=True),
    )
    core, footprint, labels = removal_masks(size, targets, {0, 2})
    assert labels == ("first target",)
    assert core.getpixel((10, 10)) == 255
    assert core.getpixel((20, 17)) == footprint.getpixel((20, 17)) == 0
    assert core.getpixel((41, 35)) == footprint.getpixel((41, 35)) == 0
    assert core.getpixel((43, 23)) == footprint.getpixel((43, 23)) == 0
    assert core.getpixel((3, 10)) == 0 and footprint.getpixel((3, 10)) == 255


def test_keeping_a_shadow_does_not_cut_into_its_selected_object():
    size = (80, 80)
    targets = (
        RemovalTarget("object", Image.new("L", (20, 20), 255), (20, 20)),
        RemovalTarget("shadow", Image.new("L", (30, 10), 255), (18, 35), selected=False, parent=0),
    )
    excluded = Image.new("L", size)
    excluded.putpixel((24, 39), 255)
    core, footprint, labels = removal_masks(size, targets, excluded=excluded)

    assert labels == ("object",)
    assert core.histogram()[255] == 399
    assert core.getpixel((24, 38)) == 255
    assert footprint.getpixel((24, 39)) == 0  # Explicit manual protection still wins.
    assert footprint.getpixel((24, 40)) == 0  # The exposed shadow is kept.


def test_remove_review_keeps_lasso_and_allows_manual_correction(qt_app):
    image = Image.new("RGB", (96, 80), "black")
    lasso = Image.new("L", image.size)
    lasso.paste(255, (12, 12, 23, 31))
    editor = RemoveEditor(image, mask=lasso)
    editor.show()
    qt_app.processEvents()
    assert editor._detect.isEnabled()
    assert not editor._generate.isVisible()

    object_mask = Image.new("L", (32, 28), 255)
    editor.set_targets((RemovalTarget("Object", object_mask, (12, 12)),))
    qt_app.processEvents()
    assert editor._generate.isEnabled()
    assert editor.detection_mask.getpixel((33, 25)) == 0
    assert editor.removal_inputs()[1].getpixel((33, 25)) == 255
    assert "Amber extends beyond" in editor._status.text()
    assert not editor._target_list.isVisible()
    editor._refine.click()
    assert editor._target_list.isVisible()

    editor._target_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert not editor._generate.isEnabled()
    editor._target_list.item(0).setCheckState(Qt.CheckState.Checked)
    editor._mode_buttons["subtract"].click()
    protected = QPainterPath()
    protected.addRect(QRectF(26, 20, 10, 10))
    editor._apply_lasso(protected)
    assert editor.removal_inputs()[1].getpixel((30, 25)) == 0

    semantic = QSignalSpy(editor.semantic_detection_requested)
    editor._detect_effects.click()
    assert semantic.count() == 1
    editor.set_busy(True, detecting=True)
    assert editor._cancel.isVisible()
    assert editor._status.isHidden()
    editor.set_targets((RemovalTarget("Confirmed object", object_mask, (12, 12)),), detailed=True)
    editor.set_busy(False)
    assert editor.removal_inputs()[1].getpixel((30, 25)) == 0
    assert editor.detection_mask.tobytes() == lasso.tobytes()

    editor.add_variant(FillPreview(image, (0, 0)))
    editor._before.click()
    contour = editor._removal_outline.path()
    cancel = QSignalSpy(editor.cancel_requested)
    editor.set_busy(True, detecting=True)
    editor._cancel.click()
    assert cancel.count() == 1
    editor.set_message("Object detection canceled.")
    editor.set_busy(False)
    assert editor._removal_outline.isVisible() and editor._removal_outline.path() == contour
    assert editor.selected_variant == 0 and editor._apply.isEnabled()
    assert "Object detection canceled" in editor._status.text()

    editor.set_busy(True, detecting=True)
    editor.set_targets(())
    editor.set_busy(False)
    assert editor._removal_outline.isVisible() and editor._removal_outline.path() == contour
    assert editor.selected_variant == 0 and editor._apply.isEnabled()
    assert "No additional objects or effects detected" in editor._status.text()

    editor._change_selection.click()
    assert editor.detection_mask.tobytes() == lasso.tobytes()
    assert editor.removal_inputs() is None
    editor.set_targets(())
    assert "No objects detected" in editor._status.text()
    editor._manual.click()
    assert editor._generate.isEnabled()
    assert editor.removal_inputs()[0].tobytes() == lasso.tobytes()
    editor.close()


def test_click_requests_detection_in_image_coordinates_and_waits_for_remove(qt_app):
    image = Image.new("RGBA", (96, 80))
    image.paste((200, 60, 0, 255), (20, 20, 40, 40))
    editor = RemoveEditor(image)
    editor.resize(640, 480)
    editor.show()
    qt_app.processEvents()
    canvas = editor._canvas
    detection = QSignalSpy(editor.detect_requested)
    removal = QSignalSpy(editor.generate_requested)
    button = Qt.MouseButton.LeftButton

    QTest.mouseClick(canvas.viewport(), button, pos=canvas.mapFromScene(QPointF(12.5, 12.5)))
    assert detection.count() == 0
    assert editor.detection_points == ((12, 12),)
    assert "Select a visible object" in editor._status.text()

    canvas.actual_size()
    canvas.zoom(2.5)
    QTest.mouseClick(canvas.viewport(), button, pos=canvas.mapFromScene(QPointF(28.5, 26.5)))
    assert detection.count() == 1
    assert editor.detection_points == ((28, 26),)
    assert editor.detection_mask.getbbox() is None
    assert editor._click_marker.isVisible()

    editor.set_targets((RemovalTarget("Object", Image.new("L", (20, 20), 255), (20, 20)),))
    assert editor._removal_outline.isVisible() and not editor._removal_outline.path().isEmpty()
    assert not editor._target_list.isVisible() and not editor._click_marker.isVisible()
    assert not editor._has_extension
    assert removal.count() == 0
    editor._generate.click()
    assert removal.count() == 1
    editor._change_selection.click()
    assert editor.detection_points == ((28, 26),) and editor._click_marker.isVisible()
    editor.close()


def test_new_lasso_opens_details_for_overlapping_objects_and_optional_effect(qt_app):
    editor = RemoveEditor(Image.new("RGB", (96, 80), "black"))
    editor.resize(640, 480)
    editor.show()
    qt_app.processEvents()
    editor._shapes.setCurrentIndex(editor._shapes.findData("lasso"))
    canvas = editor._canvas
    detection = QSignalSpy(editor.detect_requested)
    points = [canvas.mapFromScene(QPointF(x, y)) for x, y in [(12, 12), (42, 12), (42, 38), (12, 38)]]
    QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=points[0])
    for point in points[1:]:
        QTest.mouseMove(canvas.viewport(), point)
    QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, pos=points[-1])
    assert detection.count() == 1 and not editor.detection_points
    assert editor.detection_mask.getpixel((22, 22)) == 255

    editor.set_targets((
        RemovalTarget("Main object", Image.new("L", (25, 24), 255), (12, 12)),
        RemovalTarget("Other object", Image.new("L", (20, 20), 255), (36, 18), selected=False),
        RemovalTarget("Shadow", Image.new("L", (10, 10), 255), (18, 38), selected=False, parent=0),
        RemovalTarget("Foreground", Image.new("L", (4, 15), 255), (18, 16), protected=True),
    ))
    assert editor._target_list.isVisible()
    assert editor._mode_buttons["subtract"].isVisible()
    foreground = editor._target_list.item(3)
    assert foreground.text() == "Keep Foreground"
    assert not foreground.flags() & Qt.ItemFlag.ItemIsUserCheckable
    assert editor.removal_inputs()[1].getpixel((19, 20)) == 0
    assert editor.removal_inputs()[1].getpixel((20, 42)) == 0
    editor._target_list.item(2).setCheckState(Qt.CheckState.Checked)
    assert editor.removal_inputs()[1].getpixel((20, 42)) == 255
    editor._target_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert not editor._target_list.item(2).flags() & Qt.ItemFlag.ItemIsEnabled
    assert not editor._generate.isEnabled()
    editor.close()
