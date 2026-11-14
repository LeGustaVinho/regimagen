import pytest
from PIL import Image, ImageChops
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
    inputs = removal_masks(size, targets, {0, 2})
    core, footprint = inputs.core, inputs.footprint
    assert inputs.labels == ("first target",)
    assert inputs.objects == targets[:1]
    assert core.getpixel((10, 10)) == 255
    assert core.getpixel((20, 17)) == footprint.getpixel((20, 17)) == 0
    assert core.getpixel((41, 35)) == footprint.getpixel((41, 35)) == 0
    assert core.getpixel((43, 23)) == footprint.getpixel((43, 23)) == 0
    assert core.getpixel((3, 10)) == 0 and footprint.getpixel((3, 10)) == 255


@pytest.mark.parametrize("confirmed", [True, False])
def test_keeping_a_shadow_does_not_cut_into_its_selected_object(confirmed):
    size = (80, 80)
    targets = (
        RemovalTarget("object", Image.new("L", (20, 20), 255), (20, 20)),
        RemovalTarget("shadow", Image.new("L", (30, 10), 255), (18, 35), selected=confirmed, parent=0),
    )
    excluded = Image.new("L", size)
    excluded.putpixel((24, 39), 255)
    inputs = removal_masks(size, targets, {0}, excluded=excluded)
    core, footprint = inputs.core, inputs.footprint

    assert inputs.labels == ("object",)
    assert core.histogram()[255] == 399
    assert core.getpixel((24, 38)) == 255
    assert footprint.getpixel((24, 39)) == 0  # Explicit manual protection still wins.
    assert core.getpixel((24, 40)) == 0
    assert footprint.getpixel((24, 40)) == (0 if confirmed else 255)


def test_large_opaque_removal_reserves_a_blending_band_with_protection():
    size = (360, 360)
    manual = Image.new("L", size)
    manual.paste(255, (84, 84, 268, 280))
    protected = Image.new("L", size)
    protected.paste(255, (274, 130, 282, 230))

    inputs = removal_masks(size, (), manual=manual, excluded=protected)
    alpha = removal_masks(size, (), manual=manual, excluded=protected, opaque=False)
    core, footprint = inputs.core, inputs.footprint

    assert core.getpixel((66, 160)) == 0
    assert footprint.getpixel((66, 160)) == 255
    assert alpha.footprint.getpixel((66, 160)) == 0
    assert footprint.getpixel((276, 160)) == 0


def test_direct_lasso_bounds_objects_and_extends_the_blend_around_confirmed_effects():
    source = Image.new("RGB", (120, 100), (110, 90, 45))
    lasso = Image.new("L", source.size)
    lasso.paste(255, (20, 20, 76, 80))
    excluded = Image.new("L", source.size)
    excluded.putpixel((26, 32), 255)
    excluded.putpixel((82, 56), 255)
    targets = (
        RemovalTarget("Object", Image.new("L", (35, 25), 255), (11, 25)),
        RemovalTarget("Shadow", Image.new("L", (48, 15), 255), (40, 45), parent=0),
        RemovalTarget("Shadow behind object", Image.new("L", (8, 9), 255), (15, 30), parent=0),
        RemovalTarget("Foreground", Image.new("L", (5, 15), 255), (39, 29), protected=True),
        RemovalTarget("Protected plant", Image.new("L", (5, 8), 255), (83, 47), protected=True),
    )
    direct = removal_masks(source.size, (), manual=lasso, excluded=excluded)
    inputs = direct.with_detected_objects(source, targets)

    assert inputs.direct and inputs.objects == targets[:1]
    assert inputs.labels == ("Object", "Shadow", "Shadow behind object")
    assert inputs.core.getpixel((22, 32)) == inputs.core.getpixel((56, 50)) == 255
    assert inputs.core.getpixel((16, 32)) == inputs.core.getpixel((26, 32)) == 0
    assert inputs.footprint.getpixel((40, 35)) == inputs.footprint.getpixel((82, 32)) == 0
    assert inputs.core.getpixel((80, 56)) == 255
    for point in ((16, 32), (82, 56), (85, 51), (100, 56)):
        assert inputs.footprint.getpixel(point) == 0
    assert inputs.core.getpixel((88, 56)) == 0 and inputs.footprint.getpixel((88, 56)) == 255


def test_subtracting_the_last_visible_part_of_a_caster_disables_its_effect(qt_app):
    source = Image.new("RGBA", (90, 85))
    source.paste((175, 70, 32, 255), (25, 25, 37, 39))
    source.paste((70, 70, 70, 100), (38, 38, 51, 49))
    source.paste((24, 179, 216, 255), (54, 24, 70, 44))
    source.putpixel((70, 30), (60, 210, 42, 141))
    guide = Image.new("L", source.size)
    guide.paste(255, (18, 16, 75, 55))
    # The object contour extends into clear space, so only the painted pixels qualify.
    caster = RemovalTarget("Caster", Image.new("L", (18, 17), 255), (20, 20))
    effect = RemovalTarget("Shadow", Image.new("L", (16, 11), 255), (35, 38), parent=0)
    other = RemovalTarget("Other", Image.new("L", (16, 20), 255), (54, 24))
    editor = RemoveEditor(source, mask=guide)
    editor.set_targets((caster, effect, other), detailed=True)
    editor._mode_buttons["subtract"].click()
    path = QPainterPath()
    path.addRect(QRectF(25, 25, 12, 14))
    editor._apply_lasso(path)

    inputs = editor.removal_inputs()
    assert inputs.labels == ("Other",) and inputs.objects == (other,)
    assert inputs.core.getpixel((42, 43)) == inputs.footprint.getpixel((42, 43)) == 0
    assert inputs.footprint.getpixel((70, 30)) == 0
    assert not editor._target_list.item(1).flags() & Qt.ItemFlag.ItemIsEnabled
    editor._target_list.item(2).setCheckState(Qt.CheckState.Unchecked)
    assert editor.removal_inputs() is None and not editor._generate.isEnabled()
    editor.close()


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
    assert editor.removal_inputs().footprint.getpixel((33, 25)) == 255
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
    assert editor.removal_inputs().footprint.getpixel((30, 25)) == 0

    retry = QSignalSpy(editor.detect_requested)
    editor._redetect.click()
    assert retry.count() == 1
    editor.set_busy(True, detecting=True)
    assert editor._cancel.isVisible()
    assert editor._status.isHidden()
    editor.set_targets((RemovalTarget("Confirmed object", object_mask, (12, 12)),), detailed=True)
    editor.set_busy(False)
    assert editor.removal_inputs().footprint.getpixel((30, 25)) == 0
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
    assert "Codex found no replacement selection" in editor._status.text()

    editor._change_selection.click()
    assert editor.detection_mask.tobytes() == lasso.tobytes()
    assert editor.removal_inputs() is None
    editor.set_targets(())
    assert "Codex found no object" in editor._status.text()
    editor._manual.click()
    assert editor._generate.isEnabled()
    assert editor.removal_inputs().core.tobytes() == lasso.tobytes()
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
    assert editor.removal_inputs().footprint.getpixel((19, 20)) == 0
    assert editor.removal_inputs().footprint.getpixel((20, 42)) == 0
    editor._target_list.item(2).setCheckState(Qt.CheckState.Checked)
    assert editor.removal_inputs().footprint.getpixel((20, 42)) == 255
    editor._target_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert not editor._target_list.item(2).flags() & Qt.ItemFlag.ItemIsEnabled
    assert not editor._generate.isEnabled()
    editor.close()


def test_add_extends_the_focused_item_and_retracing_keeps_only_global_exclusions(qt_app):
    image = Image.new("RGB", (80, 70))
    lasso = Image.new("L", image.size)
    lasso.paste(255, (12, 10, 44, 50))
    editor = RemoveEditor(image, mask=lasso)
    targets = (
        RemovalTarget("Balance", Image.new("L", (10, 12), 255), (15, 14)),
        RemovalTarget("Shadow", Image.new("L", (6, 7), 255), (30, 20), parent=0),
        RemovalTarget("Keep", Image.new("L", (4, 4), 255), (16, 32), selected=False, protected=True),
    )
    editor.set_targets(targets, detailed=True)

    def stroke(x, y, width=4, height=4):
        path = QPainterPath()
        path.addRect(QRectF(x, y, width, height))
        editor._apply_lasso(path)

    editor._target_list.setCurrentRow(0)
    stroke(16, 27, 4, 10)
    stroke(34, 31)
    editor._target_list.setCurrentRow(1)
    stroke(40, 36)
    editor._mode_buttons["subtract"].click()
    stroke(17, 15)
    editor._target_list.setCurrentRow(0)
    editor._mode_buttons["add"].click()
    stroke(16, 14, 6, 6)
    inputs = editor.removal_inputs()
    assert inputs.core.getpixel((17, 29)) == 255
    assert inputs.core.getpixel((17, 33)) == 0
    assert inputs.core.getpixel((18, 16)) == 0
    assert inputs.core.getpixel((41, 37)) == 255
    (subject,) = inputs.objects
    x, y = subject.origin
    assert subject.mask.getpixel((17 - x, 29 - y)) == 255
    assert not (x <= 41 < x + subject.mask.width and y <= 37 < y + subject.mask.height)

    editor._target_list.item(1).setCheckState(Qt.CheckState.Unchecked)
    assert editor.removal_inputs().footprint.getpixel((41, 37)) == 0
    editor.set_targets(targets, detailed=True)
    retraced = editor.removal_inputs()
    assert retraced.core.getpixel((17, 29)) == 0
    assert retraced.core.getpixel((18, 16)) == 0
    assert retraced.core.getpixel((41, 37)) == 0
    editor._manual.click()
    assert editor.removal_inputs().direct and editor.removal_inputs().core.getpixel((18, 16)) == 0
    editor.close()
