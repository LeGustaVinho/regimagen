import time
from pathlib import Path
from threading import Event

import pytest
from PIL import Image
from PySide6.QtCore import QPointF, QSettings, QTimer, Qt
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox, QSpinBox

import regimagen.app as desktop
import regimagen.editor as editor_module
import regimagen.jobs as jobs
from regimagen.codex import AuthRequired, Cancelled, QuotaPaused
from regimagen.document import render
from regimagen.fill import FillCandidate
from regimagen.geometry import FRAME, Cell, Grid
from regimagen.project import load_project
from regimagen.storage import Store


def wait_for(predicate, timeout=12):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    assert predicate()


def idle(window):
    wait_for(lambda: window.task is None)


def gesture(view, *points):
    positions = [view.mapFromScene(QPointF(*point)) for point in points]
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=positions[0])
    for position in positions[1:]:
        QTest.mouseMove(view.viewport(), position)
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=positions[-1])


def choose_tool(editor, name):
    editor.tool.setCurrentIndex(editor.tool.findData(name))


def action(editor, name):
    return next(item for item in editor.findChildren(QAction) if item.text() == name)


@pytest.fixture
def window(tmp_path, monkeypatch, qt_app):
    store = Store(tmp_path / "workspace")
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    window = desktop.Window(store, settings)
    window.show()
    yield window
    if window.task is not None:
        window.task.cancelled.set()
        idle(window)
    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Discard)
    window.close()
    qt_app.processEvents()


def test_open_edit_save_and_reopen_without_codex(window, tmp_path, monkeypatch):
    source = tmp_path / "base.v2.png"
    Image.new("RGBA", (92, 76), "red").save(source)
    overlay = tmp_path / "overlay.webp"
    Image.new("RGBA", (28, 26), "blue").save(overlay, lossless=True)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_, **_options: (str(source), ""))
    window.open_action.trigger()
    idle(window)
    editor = window.editor
    assert window.stack.currentWidget() is editor
    assert editor.document.size == (92, 76) and len(editor.document.layers) == 1
    assert window.store.list_sources() == []

    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_, **_options: (str(overlay), ""))
    action(editor, "Import image…").trigger()
    idle(window)
    assert [layer.name for layer in editor.document.layers] == ["base.v2.png", "overlay.webp"]
    gesture(editor.canvas, (8, 8), (14, 12), (23, 20))
    assert (editor.active_layer.x, editor.active_layer.y) == (15, 12)
    editor.opacity.setSliderDown(True)
    editor.opacity.setValue(75)
    editor.opacity.setValue(60)
    before_opacity_commit = editor.undo.count()
    editor.opacity.setSliderDown(False)
    assert editor.active_layer.opacity == 60 and editor.undo.count() == before_opacity_commit + 1
    editor.undo.undo()
    assert editor.active_layer.opacity == 100
    editor.undo.redo()
    assert editor.active_layer.opacity == 60

    action(editor, "Move down").trigger()
    assert editor.active_layer is editor.document.layers[0]
    action(editor, "Move up").trigger()
    assert editor.active_layer is editor.document.layers[-1]
    editor.layers.currentItem().setCheckState(Qt.CheckState.Unchecked)
    assert not editor.active_layer.visible
    editor.undo.undo()
    assert editor.active_layer.visible
    action(editor, "Duplicate").trigger()
    assert editor.active_layer.image is editor.document.layers[-2].image
    action(editor, "Delete").trigger()
    assert len(editor.document.layers) == 2

    expected = render(editor.document)
    wait_for(lambda: editor._shown is not None and editor._shown[0] is editor.document)
    display = editor.canvas._image.pixmap().toImage()
    assert display.pixelColor(24, 22).getRgb() == expected.getpixel((24, 22))
    export = tmp_path / "result.png"
    project = tmp_path / "composite.regimagen"
    def destination(title, filename, _formats):
        if title == "Save project":
            assert filename == "base.v2.regimagen"
            return project
        assert filename == "base.v2.png"
        return export

    monkeypatch.setattr(window, "_save_destination", destination)
    window.export_action.trigger()
    idle(window)
    with Image.open(export) as output:
        assert output.tobytes() == expected.tobytes()
    assert editor.dirty

    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Cancel)
    window._import(overlay)
    assert window.task is None and editor.document.size == (92, 76)
    assert not window.close()
    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Save)
    window._import(overlay)
    idle(window)
    assert editor.document.size == (28, 26) and project.is_file()
    moved = project.rename(tmp_path / "moved.regimagen")
    window._import(moved)
    idle(window)
    assert editor.document.id == load_project(moved).id
    assert not editor.dirty and not editor.undo.canUndo()
    assert render(editor.document).tobytes() == expected.tobytes()


def test_canvas_selection_masks_brush_and_transforms(window, monkeypatch):
    monkeypatch.setattr(desktop, "dimensions", lambda *_: (100, 80))
    window.new_action.trigger()
    editor = window.editor
    assert editor.document.layers == () and render(editor.document).getbbox() is None
    action(editor, "Transparent layer").trigger()
    choose_tool(editor, "brush")
    editor.brush_size.setValue(12)
    before = editor.undo.count()
    gesture(editor.canvas, (16, 20), (21, 25), (30, 25))
    assert editor.undo.count() == before + 1
    assert editor.active_layer.image.getpixel((25, 25))[3] == 255
    choose_tool(editor, "eraser")
    gesture(editor.canvas, (25, 25))
    assert editor.active_layer.image.getpixel((25, 25))[3] == 0
    editor.undo.undo()
    assert editor.active_layer.image.getpixel((25, 25))[3] == 255

    editor.locked.setChecked(True)
    assert not action(editor, "Delete").isEnabled()
    choose_tool(editor, "move")
    gesture(editor.canvas, (20, 25), (35, 35))
    assert editor.active_layer.x == 0
    editor.locked.setChecked(False)
    choose_tool(editor, "rectangle")
    gesture(editor.canvas, (10, 10), (36, 38))
    assert editor.selection.getpixel((25, 25)) == 255
    editor.selection_mode.setCurrentIndex(editor.selection_mode.findData("add"))
    gesture(editor.canvas, (50, 30), (70, 55))
    choose_tool(editor, "lasso")
    editor.selection_mode.setCurrentIndex(editor.selection_mode.findData("subtract"))
    gesture(editor.canvas, (16, 17), (24, 17), (24, 35), (16, 35))
    assert editor.selection.getpixel((20, 24)) == 0 and editor.selection.getpixel((60, 40)) == 255

    action(editor, "Add layer mask").trigger()
    assert render(editor.document).getpixel((20, 24))[3] == 0
    editor.clear_button.click()
    editor.edit_mask.setChecked(True)
    choose_tool(editor, "brush")
    gesture(editor.canvas, (20, 24))
    assert render(editor.document).getpixel((20, 24))[3] == 255
    action(editor, "Rotate 90° clockwise").trigger()
    assert editor.active_layer.image.size == editor.active_layer.mask.size == (80, 100)
    action(editor, "Flip horizontally").trigger()
    monkeypatch.setattr(editor_module, "dimensions", lambda *_, preserve_aspect: (40, 50) if preserve_aspect else None)
    action(editor, "Resize…").trigger()
    assert editor.active_layer.image.size == editor.active_layer.mask.size == (40, 50)

    choose_tool(editor, "rectangle")
    editor.selection_mode.setCurrentIndex(0)
    gesture(editor.canvas, (5, 10), (30, 35))
    left, top, right, bottom = editor.selection.getbbox()
    asset = editor.active_layer.image
    editor.crop_button.click()
    assert editor.document.size == (right - left, bottom - top)
    assert (editor.active_layer.x, editor.active_layer.y) == (-left, -top)
    assert editor.active_layer.image is asset
    editor.undo.undo()
    assert editor.document.size == (100, 80) and editor.selection.getbbox() == (left, top, right, bottom)


def test_undo_crop_repaints_when_previous_preview_is_still_cached(window):
    editor = window.editor
    original = desktop.Document.from_image(Image.new("RGB", (300, 180), "red"), "Crop")
    window._show_document(original)
    choose_tool(editor, "rectangle")
    wait_for(lambda: editor._shown is not None and editor._shown[0] is original)
    gesture(editor.canvas, (40, 30), (160, 120))
    editor.crop_button.click()
    assert editor.document.size != original.size
    editor.undo.undo()
    assert editor.document is original
    wait_for(lambda: not editor.canvas._image.pixmap().isNull(), timeout=2)
    assert editor.canvas._image.pixmap().toImage().pixelColor(10, 10).getRgb() == (255, 0, 0, 255)


def test_undo_and_redo_cancel_a_stroke_before_switching_layers(window):
    editor = window.editor
    original = desktop.Document.from_image(Image.new("RGB", (100, 80), "red"), "Base")
    window._show_document(original)
    editor.add_image(Image.new("RGBA", original.size), "Overlay")
    with_overlay = editor.document
    choose_tool(editor, "brush")
    editor.brush_size.setValue(12)
    wait_for(lambda: editor._shown is not None and editor._shown[0] is with_overlay)

    canvas = editor.canvas
    point = canvas.mapFromScene(QPointF(20, 20))
    for key, restored in ((Qt.Key.Key_Z, original), (Qt.Key.Key_Y, with_overlay)):
        QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=point)
        assert editor._start is not None
        QTest.keyClick(canvas, key, Qt.KeyboardModifier.ControlModifier)
        assert editor.document is restored
        QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, pos=point)
        assert editor.document is restored
        assert editor.undo.count() == 1
        assert editor.undo.canRedo() == (restored is original)
        assert render(editor.document).getpixel((20, 20)) == (255, 0, 0, 255)


def test_color_adjustments_preview_cancel_and_undo(window):
    editor = window.editor
    original = desktop.Document.from_image(Image.new("RGBA", (40, 30), (100, 45, 20, 200)), "Paint")
    window._show_document(original)
    previews = []

    def adjust(accept):
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, QDialog)
        brightness, contrast, saturation = dialog.findChildren(QSpinBox)
        brightness.setValue(20)
        contrast.setValue(10)
        saturation.setValue(-30)
        previews.append(render(editor._draft).getpixel((12, 12)))
        dialog.accept() if accept else dialog.reject()

    QTimer.singleShot(0, lambda: adjust(False))
    action(editor, "Color adjustments…").trigger()
    assert editor.document is original and editor._draft is None and not editor.undo.canUndo()
    QTimer.singleShot(0, lambda: adjust(True))
    action(editor, "Color adjustments…").trigger()
    assert editor.active_layer.adjustments == (20, 10, -30)
    assert previews[0] == previews[1] == render(editor.document).getpixel((12, 12))
    assert previews[0][:3] != (100, 45, 20) and previews[0][3] == 200
    editor.undo.undo()
    assert editor.document is original


def test_fill_destination_preview_commit_and_cancellation(window, monkeypatch):
    editor = window.editor
    window._show_document(desktop.Document.from_image(Image.new("RGB", (96, 80), "red"), "base"))
    editor.add_image(Image.new("RGB", (36, 30), "blue"), "Active")
    gesture(editor.canvas, (5, 5), (22, 17))
    active_id = editor.active_layer.id
    editor.add_image(Image.new("RGB", (15, 13), "green"), "Overlay")
    gesture(editor.canvas, (3, 3), (30, 23))
    editor.layers.setCurrentRow(1)
    choose_tool(editor, "rectangle")
    gesture(editor.canvas, (8, 8), (45, 36))
    before = editor.document
    before_pixels = render(before)
    calls = []

    def generate(base, mask, prompt, edit_alpha, _exe, *, cancelled):
        calls.append((base.size, base.getpixel((0, 0)), mask.getbbox(), edit_alpha))
        if prompt == "Cancel":
            assert cancelled.wait(5)
            raise Cancelled("Generation canceled.")
        if prompt == "Sign in":
            raise AuthRequired("Sign in required.")
        if prompt == "Quota":
            raise QuotaPaused(2_000_000_000)
        side = max(base.size)
        return FillCandidate(Image.new("RGBA", (side, side)), {
            "crop": (0, 0, side, side), "shift": (0, 0), "color": (0, 0, 0), "feather": 1,
        })

    monkeypatch.setattr(window, "_codex_resolver", lambda: lambda: Path("mock-codex.exe"))
    monkeypatch.setattr(desktop, "generate_fill", generate)
    editor.fill_button.click()
    idle(window)
    fill = window.layer_fill.editor
    assert not fill.to_active_layer
    fill._alpha.setChecked(True)
    fill._generate.click()
    idle(window)
    assert calls[0][:2] == ((96, 80), (255, 0, 0, 255))
    assert fill._canvas._image.pixmap().toImage().pixelColor(22, 24).alpha() == 0

    fill._destination.setCurrentIndex(1)
    assert fill.selected_variant is None
    fill._generate.click()
    idle(window)
    assert calls[1][:2] == ((36, 30), (0, 0, 255, 255))
    assert calls[1][2][:2] == (0, 0)
    variant = window.layer_fill.variants[0]
    assert variant.preview.compose(before_pixels).tobytes() == render(variant.document).tobytes()
    assert render(variant.document).getpixel((22, 24)) == (255, 0, 0, 255)
    assert render(variant.document).getpixel((31, 23)) == (0, 128, 0, 255)
    fill._apply.click()
    idle(window)
    assert window.task_status.text() == "Applied to the active layer “Active”. Undo with Ctrl+Z."
    assert editor.active_layer.id == active_id and len(editor.document.layers) == 3
    assert render(editor.document).tobytes() == render(variant.document).tobytes()
    window.activateWindow()
    QApplication.processEvents()
    assert editor.canvas.hasFocus()
    QTest.keyClick(editor.canvas, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert editor.document is before
    assert not window.task_status.text()

    editor.fill_button.click()
    idle(window)
    fill = window.layer_fill.editor
    fill._alpha.setChecked(True)
    fill._prompt.setText("Cancel")
    fill._generate.click()
    assert window.task_status.text() == "Generating variation…"
    fill._cancel.click()
    assert window.task_status.text() == "Cancelling…" and not fill._cancel.isEnabled()
    idle(window)
    assert window.task_status.text() == "Generation canceled." and window.task_progress.isHidden()
    assert fill.selected_variant is None and editor.document is before
    fill._prompt.setText("Sign in")
    fill._generate.click()
    idle(window)
    assert "Run codex login" in window.task_status.text()
    fill._prompt.setText("Quota")
    fill._generate.click()
    idle(window)
    assert window.task_status.text().startswith("Codex usage limit reached. Try again after ")
    fill._discard.click()
    assert window.stack.currentWidget() is editor and editor.document is before
    editor.fill_button.click()
    idle(window)
    fill = window.layer_fill.editor
    fill._alpha.setChecked(True)
    fill._generate.click()
    idle(window)
    fill._apply.click()
    idle(window)
    assert window.task_status.text() == "Applied to a new layer “Generative Fill”. Undo with Ctrl+Z."
    assert len(editor.document.layers) == 4 and editor.active_layer.coverage is not None
    assert render(editor.document).getpixel((22, 24)) == (0, 0, 0, 0)
    editor.undo.undo()
    assert editor.document is before


def test_reconstruction_resumes_snapshot_and_opens_result(window, tmp_path):
    source_file = tmp_path / "wide.png"
    Image.new("RGB", (640, 16), "red").save(source_file)
    window._import(source_file)
    idle(window)
    editor = window.editor
    document_id = editor.document.id
    editor.reconstruct_button.click()
    idle(window)
    source = window.source
    job = window.store.ensure_job(source, Grid(3, 2))
    for cell in job.layout.cells:
        attempt = f"tile-{cell.column}-{cell.row}"
        window.store.begin_tile(job, cell, attempt)
        Image.new("RGB", (FRAME, FRAME), "navy").save(job.folder / f"{attempt}.png")
        window.store.finish_tile(job, cell, attempt, "mock", None, {})

    window.back_button.click()
    editor.reconstruct_button.click()
    idle(window)
    assert window.source == source and window.job == job
    assert window.edit_result.isEnabled()
    window.back_button.click()
    editor.opacity.setValue(60)
    editor.reconstruct_button.click()
    idle(window)
    assert window.source.id != source.id and window.job is None
    window.back_button.click()
    editor.undo.undo()
    editor.reconstruct_button.click()
    idle(window)
    assert window.source == source and window.job == job
    window.edit_result.click()
    idle(window)
    assert window.stack.currentWidget() is editor
    assert editor.document.id != document_id and editor.document.size == job.layout.size
    assert editor.active_layer.image.getpixel((100, 20)) == (0, 0, 128, 255)


def test_task_feedback_for_open_save_and_failed_export(window, tmp_path, monkeypatch):
    project = tmp_path / "Project.regimagen"
    document = desktop.Document.from_image(Image.new("RGBA", (24, 24)), "Base.png")
    desktop.save_project(document, project)
    release = Event()

    def slow_load(path, *, cancelled):
        assert release.wait(5)
        return load_project(path, cancelled=cancelled)

    monkeypatch.setattr(desktop, "load_project", slow_load)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_, **_options: (str(project), ""))
    window.open_action.trigger()
    try:
        assert window.task_status.isVisible() and window.task_status.text() == "Opening project…"
        assert window.task_progress.isVisible() and window.task_progress.maximum() == 0
        assert not window.empty_button.isEnabled()
    finally:
        release.set()
    idle(window)
    assert window.task_status.text() == "Opened Project.regimagen" and window.task_progress.isHidden()
    window.save_action.trigger()
    assert window.task_status.text() == "Saving project…"
    idle(window)
    assert window.task_status.text() == "Saved Project.regimagen" and window.open_folder.isVisible()
    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(Path(url.toLocalFile())) or True)
    window.open_folder.click()
    assert opened == [tmp_path]

    window.editor.reconstruct_button.click()
    idle(window)
    window.generate.click()
    idle(window)
    assert window.export.isEnabled()
    destination = tmp_path / "reconstruction.png"
    monkeypatch.setattr(QFileDialog, "exec", lambda _dialog: 1)
    monkeypatch.setattr(QFileDialog, "selectedFiles", lambda _dialog: [str(destination)])
    release.clear()
    export = desktop.export_png

    def broken_export(*_args, progress, **_kwargs):
        progress(3, 5)
        assert release.wait(5)
        raise PermissionError("The destination is read-only.")

    monkeypatch.setattr(desktop, "export_png", broken_export)
    window.export.click()
    try:
        wait_for(lambda: window.task_status.text() == "Exporting PNG · 60%")
        assert window.task_progress.value() == 3 and window.task_progress.maximum() == 5
        assert window.status.text() == "4/4 ready"
    finally:
        release.set()
    idle(window)
    assert "Export failed." in window.task_status.text() and "choose another folder" in window.task_status.text()
    assert window.task_progress.isHidden() and window.open_folder.isHidden()
    assert window.export.isEnabled() and not destination.exists()
    release.clear()

    def canceled_export(*_args, progress, cancelled, **_kwargs):
        progress(2, 5)
        assert release.wait(5)
        assert cancelled()
        raise Cancelled("Export canceled.")

    monkeypatch.setattr(desktop, "export_png", canceled_export)
    window.export.click()
    try:
        wait_for(lambda: window.task_status.text() == "Exporting PNG · 40%")
        assert window.cancel_task.isEnabled() and window.cancel_task.isVisible()
        window.cancel_task.click()
        assert window.task_status.text() == "Cancelling…" and not window.cancel_task.isEnabled()
    finally:
        release.set()
    idle(window)
    assert window.task_status.text() == "Export canceled."
    assert window.cancel_task.isHidden() and not destination.exists()

    monkeypatch.setattr(desktop, "export_png", export)
    window.export.click()
    idle(window)
    assert window.task_status.text() == "Exported reconstruction.png" and window.open_folder.isVisible()
    assert destination.is_file()

    destination = next(iter(window.store.snapshot(window.job).current_artifacts.values()))
    saved_tile = destination.read_bytes()
    warnings = []
    monkeypatch.setattr(QMessageBox, "information", lambda _parent, _title, message: warnings.append(message))
    window.export.click()
    assert window.task is None and warnings and "managed files" in warnings[0]
    assert destination.read_bytes() == saved_tile


def test_preview_and_regeneration_report_the_requested_tiles(window, monkeypatch):
    document = desktop.Document.from_image(Image.new("RGB", (32, 32), "red"), "Base.png")
    window._show_document(document)
    window.editor.reconstruct_button.click()
    idle(window)
    job = window.store.ensure_job(window.source, Grid(4, 4))
    window._show_grid(job.grid, job)
    preview = job.layout.preview()

    def save_tile(cell, attempt):
        window.store.begin_tile(job, cell, attempt)
        Image.new("RGB", (FRAME, FRAME), "navy").save(job.folder / f"{attempt}.png")
        window.store.finish_tile(job, cell, attempt, "mock", None, {})

    save_tile(preview[0], "cached")
    release = Event()
    requests = []

    def generate(_store, _job, cells, _executable, *, on_change, **_kwargs):
        requests.append(tuple(cells))
        on_change(cells[0])
        assert release.wait(5)
        for index, cell in enumerate(cells):
            save_tile(cell, f"request-{len(requests)}-tile-{index}")
        on_change(None)
        return jobs.RunOutcome()

    monkeypatch.setattr(desktop, "run_tiles", generate)
    window.preview.click()
    try:
        wait_for(lambda: window.active is not None)
        assert window.task_status.text() == "Preview 2×2 · 1/4 tiles ready"
        assert window.task_progress.value() == 1 and window.task_progress.maximum() == 4
        assert window.status.text() == "1/16 ready"
        window.grid_map.picked.emit(Cell(0, 0))
        assert window.task_status.text() == "Preview 2×2 · 1/4 tiles ready"
    finally:
        release.set()
    idle(window)
    assert requests == [preview[1:]] and window.status.text() == "4/16 ready"
    assert window.task_status.text().startswith("Preview ready.")

    release.clear()
    window.grid_map.picked.emit(preview[0])
    window.retry.click()
    try:
        wait_for(lambda: window.active is not None)
        assert window.task_status.text() == f"Regenerating tile · {preview[0].label}"
        assert window.task_progress.value() == 0 and window.task_progress.maximum() == 1
    finally:
        release.set()
    idle(window)
    assert requests[-1] == (preview[0],)
    assert "Tile regenerated" in window.task_status.text()


def test_preview_recovery_keeps_the_requested_region_when_the_quota_time_is_unavailable(window, monkeypatch):
    document = desktop.Document.from_image(Image.new("RGB", (32, 32), "red"), "Base.png")
    window._show_document(document)
    window.editor.reconstruct_button.click()
    idle(window)
    window._show_grid(Grid(4, 4))
    preview = window.image_layout.preview()
    release = Event()
    requests = []

    def generate(store, job, cells, _executable, *, on_change, **_kwargs):
        requests.append(tuple(cells))
        on_change(cells[0])
        assert release.wait(5)
        store.set_notice(job, "Codex has reached its usage limit.", "unknown")
        return jobs.RunOutcome("quota_limited", quota_reset="unknown")

    monkeypatch.setattr(desktop, "run_tiles", generate)
    window.preview.click()
    try:
        wait_for(lambda: window.active is not None)
        window.grid_map.picked.emit(Cell(0, 0))
    finally:
        release.set()
    idle(window)
    assert window.task_progress.isHidden()
    assert "Resume when the limit resets." in window.task_status.text()
    assert preview[0].label.lower() in window.task_status.text()
    assert window.retry_task.isVisible() and window.retry_task.isEnabled()
    window.retry_task.click()
    idle(window)
    assert requests == [preview, preview]

    window._show_grid(window.grid, window.job)
    assert "Codex has reached its usage limit." in window.status.text()
    assert window.retry_task.isHidden()


@pytest.mark.parametrize(("chained", "conflict_scale"), [(False, None), (False, 2), (True, 2), (True, 3)])
def test_scale_batch_reports_stage_and_resumes_in_the_same_folder(window, tmp_path, monkeypatch, chained, conflict_scale):
    document = desktop.Document.from_image(Image.new("RGBA", (24, 24)), "Base.png")
    window._show_document(document)
    window.editor.reconstruct_button.click()
    idle(window)
    window.settings.setValue("scales", "2, 3, 4")
    window.settings.setValue("chainedScales", chained)
    monkeypatch.setattr(QDialog, "exec", lambda _dialog: 1)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_, **_options: str(tmp_path))
    run_tiles = jobs.run_tiles
    pause_once = Event()

    def generate(store, job, cells, executable, *, pause, **kwargs):
        if job.scale == 3 and not pause_once.is_set():
            assert pause.wait(5)
            pause_once.set()
        return run_tiles(store, job, cells, executable, pause=pause, **kwargs)

    monkeypatch.setattr(jobs, "run_tiles", generate)
    export_png = jobs.export_png
    exported_scales = []

    def export(layout, *args, **kwargs):
        exported_scales.append(layout.scale)
        return export_png(layout, *args, **kwargs)

    monkeypatch.setattr(jobs, "export_png", export)
    window.batch.click()
    wait_for(lambda: window.task_status.text().startswith("Scale 2/3 · 3×"))
    window.generate.click()
    assert window.task_status.text().startswith("Pausing") and not window.generate.isEnabled()
    idle(window)
    assert "Scale batch paused." in window.task_status.text() and "Exported 1/3 PNGs." in window.task_status.text()
    folder = window.output_folder
    assert folder is not None and (folder / "Base-2x.png").is_file()
    assert window.generate.text() == "Resume batch" and window.generate.isEnabled()
    if conflict_scale is not None:
        conflicting_file = folder / f"Base-{conflict_scale}x.png"
        Image.new("RGB", (24 * conflict_scale, 24 * conflict_scale), "blue").save(conflicting_file)
        edited = conflicting_file.read_bytes()
        window.generate.click()
        idle(window)
        assert conflicting_file.read_bytes() == edited
        assert conflicting_file.name in window.task_status.text() and "Move or rename" in window.task_status.text()
        assert window.output_folder == folder and window.open_folder.isVisible()
        assert window.generate.text() == "Resume batch" and window.generate.isEnabled()
        preserved_file = conflicting_file.rename(conflicting_file.with_suffix(".edited"))
    window.generate.click()
    idle(window)
    assert window.task_status.text() == "Scale batch complete. Exported 3/3 PNGs."
    assert window.output_folder == folder
    assert exported_scales == ([2, 2, 3, 4] if conflict_scale == 2 and not chained else [2, 3, 4])
    assert sorted(path.name for path in folder.glob("*.png")) == ["Base-2x.png", "Base-3x.png", "Base-4x.png"]
    if conflict_scale is not None:
        assert preserved_file.read_bytes() == edited


def test_canvas_hint_tracks_missing_hidden_and_locked_layers(window):
    window._show_document(desktop.Document("Untitled", 24, 24))
    editor = window.editor
    choose_tool(editor, "brush")
    assert editor.tool_hint.text() == "Add a layer to use Brush." and editor.tool_hint.isVisible()
    editor.add_image(Image.new("RGBA", (24, 24)), "Pixels")
    assert editor.tool_hint.isHidden()
    editor.layers.setCurrentRow(-1)
    assert "Select a layer" in editor.tool_hint.text()
    editor.layers.setCurrentRow(0)
    editor.layers.currentItem().setCheckState(Qt.CheckState.Unchecked)
    assert "Turn on visibility" in editor.tool_hint.text()
    editor.layers.currentItem().setCheckState(Qt.CheckState.Checked)
    editor.locked.setChecked(True)
    assert "Turn off Lock layer" in editor.tool_hint.text()
    choose_tool(editor, "pan")
    assert editor.tool_hint.isHidden()
    choose_tool(editor, "eraser")
    assert "Turn off Lock layer" in editor.tool_hint.text()
    editor.locked.setChecked(False)
    assert editor.tool_hint.isHidden()
