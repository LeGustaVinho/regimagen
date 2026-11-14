"""The layered document editor and its undoable canvas tools."""

import math
from dataclasses import dataclass, replace
from uuid import uuid4

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QRectF, QRunnable, QSignalBlocker, QThreadPool, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPixmap, QShortcut, QUndoCommand, QUndoStack
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QFrame, QHBoxLayout, QInputDialog,
    QLabel, QListWidget, QListWidgetItem, QMenu, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from .canvas import CanvasView, SELECTION_TOOLS, combine_selection, pixmap
from .document import Document, Layer, asset_thumbnail, box, paint, paint_mask, render, validate_size
from .geometry import MAX_PNG_DIMENSION, Rect
from .viewer import qimage


def _button(text, slot, *, tip=None):
    button = QPushButton(text)
    button.clicked.connect(slot)
    if tip:
        button.setToolTip(tip)
    return button


def dimensions(parent, title, size, *, preserve_aspect=False):
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    form = QFormLayout(dialog)
    width, height = QSpinBox(), QSpinBox()
    for label, field, value in (("Width (px)", width, size[0]), ("Height (px)", height, size[1])):
        field.setRange(1, MAX_PNG_DIMENSION)
        field.setValue(value)
        field.setKeyboardTracking(False)
        form.addRow(label, field)
    if preserve_aspect:
        link = QCheckBox("Keep aspect ratio")
        link.setChecked(True)
        form.addRow(link)

        def match(value, field, factor):
            if link.isChecked():
                with QSignalBlocker(field):
                    field.setValue(round(value * factor))

        width.valueChanged.connect(lambda value: match(value, height, size[1] / size[0]))
        height.valueChanged.connect(lambda value: match(value, width, size[0] / size[1]))
        link.toggled.connect(lambda checked: match(width.value(), height, size[1] / size[0]))
    error = QLabel()
    error.setWordWrap(True)
    form.addRow(error)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    form.addRow(buttons)

    def accept():
        try:
            validate_size(width.value(), height.value())
        except ValueError as exc:
            error.setText(str(exc))
        else:
            dialog.accept()

    buttons.accepted.connect(accept)
    buttons.rejected.connect(dialog.reject)
    return (width.value(), height.value()) if dialog.exec() else None


@dataclass(frozen=True)
class _State:
    document: Document
    active_id: str | None
    selection: Image.Image | None = None


class _Edit(QUndoCommand):
    def __init__(self, label, before, after, show):
        super().__init__(label)
        self.before, self.after, self.show = before, after, show

    def undo(self):
        self.show(self.before)

    def redo(self):
        self.show(self.after)


class _PreviewSignals(QObject):
    finished = Signal(object, object, str)


class _Preview(QRunnable):
    def __init__(self, request):
        super().__init__()
        self.request = request
        self.signals = _PreviewSignals()

    def run(self):
        document, area, size = self.request
        try:
            image = qimage(np.asarray(self._render(document, area, size)))
        except Exception as exc:
            self.signals.finished.emit(self.request, None, str(exc))
        else:
            self.signals.finished.emit(self.request, image, "")

    @staticmethod
    def _render(document, area, size):
        if (area.width, area.height) == size:
            return render(document, area)
        # Resample stripes of the full-resolution composite. Include the
        # Lanczos support around each stripe so their boundaries match.
        ratio = area.height / size[1]
        padding = math.ceil(3 * max(1, ratio))
        step = max(1, int(2_000_000 / (area.width * ratio)))
        result = Image.new("RGBA", size)
        for top in range(0, size[1], step):
            bottom = min(top + step, size[1])
            first = area.top + top * ratio
            last = area.top + bottom * ratio
            region = Rect(
                area.left, max(area.top, math.floor(first) - padding),
                area.right, min(area.bottom, math.ceil(last) + padding),
            )
            pixels = render(document, region)
            scaled = pixels.resize(
                (size[0], bottom - top), Image.Resampling.LANCZOS,
                box=(0, first - region.top, area.width, min(region.height, last - region.top)),
            )
            result.paste(scaled, (0, top))
        return result


class Editor(QWidget):
    changed = Signal()
    import_requested = Signal()
    reconstruct_requested = Signal()
    fill_requested = Signal()
    remove_requested = Signal()

    def __init__(self):
        super().__init__()
        self._state: _State | None = None
        self._saved: Document | None = None
        self._draft: Document | None = None
        self._busy = False
        self._points = []
        self._start = None
        self._ink = QColor("#8ce1c6")
        self._thumbnails = {}
        self._preview_running = False
        self._shown = None
        self.undo = QUndoStack(self)
        self.undo.setUndoLimit(50)
        self.canvas = CanvasView()
        self.canvas.setToolTip("Scroll to zoom. Hold Space or use the middle button to pan.")
        self.canvas.stroke_finished.connect(self._select_pixels)
        self.canvas.pointer_started.connect(self._pointer_start)
        self.canvas.pointer_moved.connect(self._pointer_move)
        self.canvas.pointer_finished.connect(self._pointer_end)
        self.canvas.gesture_canceled.connect(self._reset_gesture)
        self.canvas.moved.connect(self._queue_preview)
        self.canvas.resized.connect(self._queue_preview)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._render_preview)
        self._build()

    def _build(self):
        content = QVBoxLayout(self)
        content.setContentsMargins(0, 0, 0, 0)
        tools = QHBoxLayout()
        self.undo_button = _button("Undo", self.undo.undo, tip="Undo (Ctrl+Z)")
        self.redo_button = _button("Redo", self.undo.redo, tip="Redo (Ctrl+Y)")
        self.undo.canUndoChanged.connect(self.undo_button.setEnabled)
        self.undo.canRedoChanged.connect(self.redo_button.setEnabled)
        self.undo_button.setEnabled(False)
        self.redo_button.setEnabled(False)
        tools.addWidget(self.undo_button)
        tools.addWidget(self.redo_button)
        for key, slot in (("Ctrl+Z", self.undo.undo), ("Ctrl+Y", self.undo.redo), ("Ctrl+Shift+Z", self.undo.redo)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)
        self.tool = QComboBox()
        self.tool.setAccessibleName("Canvas tool")
        for label, value in (
            ("Move", "move"), ("Rectangle", "rectangle"), ("Lasso", "lasso"),
            ("Brush", "brush"), ("Eraser", "eraser"), ("Pan", "pan"),
        ):
            self.tool.addItem(label, value)
        self.tool.currentIndexChanged.connect(self._tools_changed)
        tools.addWidget(self.tool)
        self.selection_mode = QComboBox()
        self.selection_mode.setAccessibleName("Selection mode")
        for label, mode in (("New selection", "new"), ("Add", "add"), ("Subtract", "subtract")):
            self.selection_mode.addItem(label, mode)
        tools.addWidget(self.selection_mode)
        self.clear_button = _button("Clear", self._clear_selection, tip="Clear the pixel selection")
        tools.addWidget(self.clear_button)
        self.crop_button = _button("Crop", self._crop, tip="Crop the canvas to the selection")
        tools.addWidget(self.crop_button)
        tools.addStretch()
        tools.addWidget(_button("Fit", self.canvas.fit))
        tools.addWidget(_button("100%", self.canvas.actual_size, tip="Show pixels at actual size"))
        self.fill_button = _button("Generative Fill…", self.fill_requested)
        self.remove_button = _button("Generative Remove…", self.remove_requested)
        self.reconstruct_button = _button("Reconstruct…", self.reconstruct_requested)
        tools.addWidget(self.fill_button)
        tools.addWidget(self.remove_button)
        tools.addWidget(self.reconstruct_button)
        content.addLayout(tools)

        work = QHBoxLayout()
        work.addWidget(self.canvas, 1)
        work.addWidget(self._layer_panel())
        content.addLayout(work, 1)
        self.message = QLabel()
        self.message.setObjectName("muted")
        self.message.setTextFormat(Qt.TextFormat.PlainText)
        self.message.setWordWrap(True)
        self.message.hide()
        content.addWidget(self.message)
        self._tools_changed()

    def _layer_panel(self):
        panel = QFrame()
        panel.setObjectName("editorSidebar")
        panel.setFixedWidth(260)
        column = QVBoxLayout(panel)
        title = QLabel("Layers")
        title.setObjectName("section")
        column.addWidget(title)
        self.layers = QListWidget()
        self.layers.setAccessibleName("Layers, topmost first")
        self.layers.currentItemChanged.connect(self._choose_layer)
        self.layers.itemChanged.connect(self._visibility)
        self.layers.itemDoubleClicked.connect(lambda _item: self._rename())
        column.addWidget(self.layers, 1)

        actions = QHBoxLayout()
        add = QPushButton("Add layer")
        add_menu = QMenu(add)
        add_menu.addAction("Transparent layer", self._add_blank)
        add_menu.addAction("Import image…", self.import_requested.emit)
        add.setMenu(add_menu)
        actions.addWidget(add)
        layer_menu = QPushButton("Layer")
        menu = QMenu(layer_menu)
        self.rename = menu.addAction("Rename…", self._rename)
        self.duplicate = menu.addAction("Duplicate", self._duplicate)
        self.remove = menu.addAction("Delete", self._delete)
        menu.addSeparator()
        self.raise_layer = menu.addAction("Move up", lambda: self._reorder(1))
        self.lower_layer = menu.addAction("Move down", lambda: self._reorder(-1))
        menu.addSeparator()
        self.transforms = menu.addMenu("Transform")
        for label, transform in (
            ("Rotate 90° clockwise", Image.Transpose.ROTATE_270),
            ("Rotate 90° counterclockwise", Image.Transpose.ROTATE_90),
            ("Flip horizontally", Image.Transpose.FLIP_LEFT_RIGHT),
            ("Flip vertically", Image.Transpose.FLIP_TOP_BOTTOM),
        ):
            self.transforms.addAction(label, lambda _checked=False, name=label, op=transform: self._transform(name, op))
        self.transforms.addAction("Resize…", self._resize_layer)
        self.adjustments = menu.addAction("Color adjustments…", self._adjustments)
        layer_menu.setMenu(menu)
        actions.addWidget(layer_menu)
        column.addLayout(actions)

        opacity = QHBoxLayout()
        label = QLabel("Opacity")
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setAccessibleName("Layer opacity")
        self.opacity.valueChanged.connect(self._opacity_changed)
        self.opacity.sliderReleased.connect(self._commit_opacity)
        label.setBuddy(self.opacity)
        self.opacity_value = QLabel("100%")
        opacity.addWidget(label)
        opacity.addWidget(self.opacity, 1)
        opacity.addWidget(self.opacity_value)
        column.addLayout(opacity)
        self.locked = QCheckBox("Lock layer")
        self.locked.toggled.connect(lambda checked: self._update_layer("Lock layer", locked=checked))
        column.addWidget(self.locked)

        mask_row = QHBoxLayout()
        mask_button = QPushButton("Mask")
        mask_menu = QMenu(mask_button)
        self.add_mask = mask_menu.addAction("Add layer mask", self._add_mask)
        self.add_mask.setToolTip("Use the selection as a mask, or reveal the whole layer.")
        self.reveal = mask_menu.addAction("Reveal selection", lambda: self._edit_mask_selection(255))
        self.conceal = mask_menu.addAction("Hide selection", lambda: self._edit_mask_selection(0))
        self.remove_mask = mask_menu.addAction("Remove mask", lambda: self._update_layer("Remove mask", mask=None))
        mask_button.setMenu(mask_menu)
        mask_row.addWidget(mask_button)
        self.edit_mask = QCheckBox("Paint mask")
        self.edit_mask.setToolTip("Brush reveals. Eraser hides.")
        self.edit_mask.toggled.connect(self._tools_changed)
        mask_row.addWidget(self.edit_mask)
        column.addLayout(mask_row)

        self.brush_controls = QWidget()
        form = QFormLayout(self.brush_controls)
        form.setContentsMargins(0, 0, 0, 0)
        self.brush_size = QSpinBox()
        self.brush_size.setRange(1, 1024)
        self.brush_size.setValue(24)
        self.brush_size.setSuffix(" px")
        self.brush_size.valueChanged.connect(self._brush_preview)
        form.addRow("Brush size", self.brush_size)
        self.strength = QSpinBox()
        self.strength.setRange(1, 100)
        self.strength.setValue(100)
        self.strength.setSuffix("%")
        self.strength.valueChanged.connect(self._brush_preview)
        form.addRow("Strength", self.strength)
        self.color = _button("Brush color…", self._choose_color)
        form.addRow(self.color)
        column.addWidget(self.brush_controls)
        self._brush_preview()
        return panel

    @property
    def document(self) -> Document | None:
        return self._state.document if self._state else None

    @property
    def active_layer(self) -> Layer | None:
        return self.document.layer(self._state.active_id) if self._state else None

    @property
    def selection(self) -> Image.Image | None:
        return self._state.selection if self._state else None

    @property
    def dirty(self) -> bool:
        return self.document is not self._saved

    def set_document(self, document: Document):
        self.canvas.cancel_gesture()
        self.undo.clear()
        self._saved = document
        self._thumbnails.clear()
        self._shown = None
        self._state = None
        self.set_message("")
        top = document.layers[-1].id if document.layers else None
        self._restore(_State(document, top))

    def mark_saved(self, document: Document):
        self._saved = document
        self.changed.emit()

    def set_busy(self, busy: bool):
        self._busy = busy
        self.setEnabled(not busy)
        self._tools_changed()

    def set_message(self, text: str):
        self.message.setText(text)
        self.message.setVisible(bool(text))

    def add_image(self, image: Image.Image, name: str):
        self._insert(Layer(name, image.convert("RGBA")), "Import layer")

    def apply_fill(self, document: Document, layer_id: str, selection: Image.Image):
        layer = document.layer(layer_id)
        self._commit(layer.name if layer is not None and layer.coverage is not None else "Generative Fill",
                     _State(document, layer_id, selection))

    def apply_removal(self, document: Document, layer_id: str):
        self._commit("Generative Remove", _State(document, layer_id, None))

    def _commit(self, label: str, state: _State):
        self.undo.push(_Edit(label, self._state, state, self._restore))

    def _restore(self, state: _State):
        self.canvas.cancel_gesture()
        previous = self._state
        self._draft = None
        self._state = state
        if previous is None or previous.document.size != state.document.size:
            self._shown = None
            self.canvas.set_size(state.document.size)
        if previous is None or previous.selection is not state.selection:
            self.canvas.set_selection(state.selection, state.selection.getbbox() if state.selection is not None else None)
        if previous is None or previous.active_id != state.active_id:
            with QSignalBlocker(self.edit_mask):
                self.edit_mask.setChecked(False)
        if previous is None or previous.document.layers is not state.document.layers:
            self._list_layers()
        else:
            self._sync_current_item()
        self._update_layer_controls()
        self._queue_preview()
        self.changed.emit()

    def _list_layers(self):
        icons = {}
        with QSignalBlocker(self.layers):
            self.layers.clear()
            for layer in reversed(self.document.layers):
                key = id(layer.image)
                if key in self._thumbnails:
                    icon = self._thumbnails[key]
                else:
                    thumbnail = asset_thumbnail(layer.image, self._thumbnail_size(layer.image.size))
                    icon = QIcon(pixmap(thumbnail))
                icons[key] = icon
                item = QListWidgetItem(icon, layer.name)
                item.setData(Qt.ItemDataRole.UserRole, layer.id)
                item.setCheckState(Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked)
                kind = "Fill layer" if layer.coverage is not None else "Raster layer"
                item.setToolTip(f"{kind} · {layer.image.width} × {layer.image.height} px · ({layer.x}, {layer.y})")
                self.layers.addItem(item)
        self._thumbnails = icons
        self._sync_current_item()

    @staticmethod
    def _thumbnail_size(size):
        scale = min(1, 32 / max(size))
        return tuple(max(1, round(value * scale)) for value in size)

    def _sync_current_item(self):
        with QSignalBlocker(self.layers):
            for index in range(self.layers.count()):
                if self.layers.item(index).data(Qt.ItemDataRole.UserRole) == self._state.active_id:
                    self.layers.setCurrentRow(index)
                    return
            self.layers.setCurrentRow(-1)

    def _choose_layer(self, item, _previous):
        if self._state:
            self.canvas.cancel_gesture()
            layer_id = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
            self._state = replace(self._state, active_id=layer_id)
            self.edit_mask.setChecked(False)
            self._update_layer_controls()

    def _update_layer_controls(self):
        layer = self.active_layer
        editable = layer is not None and not layer.locked
        has_mask = layer is not None and layer.mask is not None
        row = self.layers.currentRow()
        with QSignalBlocker(self.locked), QSignalBlocker(self.opacity), QSignalBlocker(self.edit_mask):
            self.locked.setChecked(layer.locked if layer is not None else False)
            self.opacity.setValue(layer.opacity if layer is not None else 100)
            if not has_mask:
                self.edit_mask.setChecked(False)
        self.opacity_value.setText(f"{self.opacity.value()}%")
        for widget in (self.rename, self.duplicate, self.locked, self.opacity):
            widget.setEnabled(layer is not None)
        for widget in (self.remove, self.transforms, self.adjustments):
            widget.setEnabled(editable)
        self.raise_layer.setEnabled(row > 0)
        self.lower_layer.setEnabled(0 <= row < self.layers.count() - 1)
        self.add_mask.setEnabled(editable and not has_mask)
        self.remove_mask.setEnabled(editable and has_mask)
        self.reveal.setEnabled(editable and has_mask and self.selection is not None)
        self.conceal.setEnabled(self.reveal.isEnabled())
        self.edit_mask.setEnabled(editable and has_mask)
        self._tools_changed()

    def _visibility(self, item):
        layer = self.document.layer(item.data(Qt.ItemDataRole.UserRole))
        visible = item.checkState() == Qt.CheckState.Checked
        if visible != layer.visible:
            self._replace_layer("Show layer" if visible else "Hide layer", replace(layer, visible=visible))

    def _insert(self, layer: Layer, label: str):
        document = self.document.insert(layer, above=self.active_layer)
        self._commit(label, _State(document, layer.id, self.selection))

    def _add_blank(self):
        self._insert(Layer("Layer", Image.new("RGBA", self.document.size)), "Add layer")

    def _replace_layer(self, label, layer: Layer):
        self._commit(label, replace(self._state, document=self.document.update(layer)))

    def _update_layer(self, label, **changes):
        layer = self.active_layer
        if layer is not None and any(getattr(layer, key) != value for key, value in changes.items()):
            self._replace_layer(label, replace(layer, **changes))

    def _rename(self):
        if self.active_layer is None:
            return
        name, accepted = QInputDialog.getText(self, "Rename layer", "Name", text=self.active_layer.name)
        if accepted and name.strip():
            self._update_layer("Rename layer", name=name.strip())

    def _duplicate(self):
        layer = self.active_layer
        if layer is not None:
            self._insert(replace(layer, name=f"{layer.name} copy", id=uuid4().hex, locked=False), "Duplicate layer")

    def _delete(self):
        layer = self.active_layer
        if layer is None or layer.locked:
            return
        index = self.document.layers.index(layer)
        layers = self.document.layers[:index] + self.document.layers[index + 1:]
        active_id = layers[max(0, index - 1)].id if layers else None
        self._commit("Delete layer", _State(replace(self.document, layers=layers), active_id, self.selection))

    def _reorder(self, direction):
        layers = list(self.document.layers)
        layer = self.active_layer
        if layer is None:
            return
        index = layers.index(layer)
        destination = index + direction
        if 0 <= destination < len(layers):
            layers.insert(destination, layers.pop(index))
            self._commit("Reorder layer", replace(self._state, document=replace(self.document, layers=tuple(layers))))

    def _opacity_changed(self, value):
        self.opacity_value.setText(f"{value}%")
        if self.opacity.isSliderDown() and self.active_layer is not None:
            self._draft = self.document.update(replace(self.active_layer, opacity=value))
            self._queue_preview()
        else:
            self._commit_opacity()

    def _commit_opacity(self):
        self._draft = None
        self._update_layer("Layer opacity", opacity=self.opacity.value())
        self._queue_preview()

    def _select_pixels(self, path):
        selection = combine_selection(self.selection, path, self.document.size, self.selection_mode.currentData())
        if selection is not self.selection:
            if selection is not None and selection.getbbox() is None:
                selection = None
            self._commit("Select pixels", replace(self._state, selection=selection))

    def _clear_selection(self):
        if self.selection is not None:
            self._commit("Clear selection", replace(self._state, selection=None))

    def _crop(self):
        if self.selection is not None:
            area = Rect(*self.selection.getbbox())
            if area != self.document.bounds:
                self._commit("Crop canvas", replace(self._state, document=self.document.crop(area), selection=None))

    def _tools_changed(self, *_):
        tool = self.tool.currentData()
        self.canvas.set_tool(tool)
        layer = self.active_layer
        editable = layer is not None and layer.visible and not layer.locked
        editing_mask = self.edit_mask.isChecked()
        painting = tool in ("brush", "eraser")
        can_paint = editable and (layer.mask is not None if editing_mask else layer.coverage is None)
        self.canvas.set_drawing_enabled(
            not self._busy and (tool in SELECTION_TOOLS or (editable and tool == "move") or (painting and can_paint)),
        )
        self.selection_mode.setVisible(tool in SELECTION_TOOLS)
        self.clear_button.setEnabled(self.selection is not None)
        self.crop_button.setEnabled(self.selection is not None and self.selection.getbbox() != box(self.document.bounds))
        self.brush_controls.setVisible(painting)
        self.color.setEnabled(tool == "brush" and not editing_mask)
        self._brush_preview()

    def _pointer_start(self, point):
        self._start = point
        self._points = [(math.floor(point.x()), math.floor(point.y()))]

    def _move_layer(self, point):
        delta = point - self._start
        layer = self.active_layer
        return replace(layer, x=layer.x + round(delta.x()), y=layer.y + round(delta.y()))

    def _pointer_move(self, point):
        if self._start is None:
            return
        if self.tool.currentData() == "move":
            self._draft = self.document.update(self._move_layer(point))
            self._queue_preview()
        else:
            self._points.append((math.floor(point.x()), math.floor(point.y())))

    def _pointer_end(self, point):
        if self._start is None:
            return
        layer = self.active_layer
        tool = self.tool.currentData()
        if tool == "move":
            moved = self._move_layer(point)
            self._update_layer("Move layer", x=moved.x, y=moved.y)
        else:
            self._points.append((math.floor(point.x()), math.floor(point.y())))
            color = self._ink.getRgb()[:3] if tool == "brush" else None
            painted = paint(
                self.document, layer, self._points, size=self.brush_size.value(), strength=self.strength.value(),
                color=color, edit_mask=self.edit_mask.isChecked(), selection=self.selection,
            )
            if painted is not layer:
                label = "Paint mask" if self.edit_mask.isChecked() else ("Brush stroke" if color else "Erase stroke")
                self._replace_layer(label, painted)
        self._reset_gesture()

    def _reset_gesture(self):
        self._draft = self._start = None
        self._points.clear()
        self._queue_preview()

    def _brush_preview(self):
        color = QColor(self._ink)
        if self.tool.currentData() == "eraser":
            color = QColor("#aab4c5")
        elif self.edit_mask.isChecked():
            color = QColor("white")
        color.setAlpha(round(200 * self.strength.value() / 100))
        self.canvas.set_brush_preview(color, self.brush_size.value())

    def _choose_color(self):
        color = QColorDialog.getColor(self._ink, self, "Brush color")
        if color.isValid():
            self._ink = color
            self._brush_preview()

    def _add_mask(self):
        layer = self.active_layer
        mask = Image.new("L", layer.image.size, 255) if self.selection is None else self.selection.crop(box(layer.bounds))
        self._update_layer("Add layer mask", mask=mask)

    def _edit_mask_selection(self, value):
        layer = self.active_layer
        overlap = self.document.bounds.intersect(layer.bounds)
        if overlap is None:
            return
        coverage = self.selection.crop(box(overlap))
        if coverage.getbbox() is not None:
            mask = paint_mask(layer.mask, overlap.relative_to(layer.bounds), coverage, value)
            self._update_layer("Edit layer mask", mask=mask)

    def _transform(self, label, operation):
        layer = self.active_layer.transform(lambda image: image.transpose(operation))
        self._replace_layer(label, layer)

    def _resize_layer(self):
        layer = self.active_layer
        size = dimensions(self, "Resize layer", layer.image.size, preserve_aspect=True)
        if size is not None and size != layer.image.size:
            self._replace_layer("Resize layer", layer.resize(size))

    def _adjustments(self):
        layer = self.active_layer
        dialog = QDialog(self)
        dialog.setWindowTitle("Color adjustments")
        form = QFormLayout(dialog)
        fields = []
        for label, value in zip(("Brightness", "Contrast", "Saturation"), layer.adjustments):
            field = QSpinBox()
            field.setRange(-100, 100)
            field.setValue(value)
            form.addRow(label, field)
            fields.append(field)

        def preview():
            values = tuple(field.value() for field in fields)
            self._draft = self.document.update(replace(layer, adjustments=values))
            self._queue_preview()

        for field in fields:
            field.valueChanged.connect(preview)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        accepted = dialog.exec()
        self._draft = None
        if accepted:
            self._update_layer("Color adjustments", adjustments=tuple(field.value() for field in fields))
        self._queue_preview()

    def _preview_request(self):
        document = self._draft or self.document
        if document is None or not self.isVisible():
            return None
        visible = self.canvas.mapToScene(self.canvas.viewport().rect()).boundingRect()
        scale = self.canvas.transform().m11()
        margin = 48 / scale
        area = Rect(
            math.floor(visible.left() - margin), math.floor(visible.top() - margin),
            math.ceil(visible.right() + margin), math.ceil(visible.bottom() + margin),
        ).intersect(document.bounds)
        if area is None:
            return None
        scale = min(1, scale * self.canvas.devicePixelRatioF())
        size = max(1, round(area.width * scale)), max(1, round(area.height * scale))
        return document, area, size

    def _queue_preview(self):
        if not self._preview_running and not self._timer.isActive():
            self._timer.start()

    def _render_preview(self):
        request = self._preview_request()
        if request is None or request == self._shown:
            return
        self._preview_running = True
        worker = _Preview(request)
        worker.signals.finished.connect(self._preview_finished)
        QThreadPool.globalInstance().start(worker)

    def _preview_finished(self, request, image, error):
        self._preview_running = False
        if request != self._preview_request():
            self._queue_preview()
            return
        self._shown = request
        if error:
            self.set_message(error)
        else:
            _, area, _ = request
            self.canvas.set_image(QPixmap.fromImage(image), rect=QRectF(area.left, area.top, area.width, area.height))

    def showEvent(self, event):
        super().showEvent(event)
        self._queue_preview()
