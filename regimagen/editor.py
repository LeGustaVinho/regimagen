"""The layered document editor and its undoable canvas tools."""

import math
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from uuid import uuid4
from weakref import ref

from PIL import Image
from PySide6.QtCore import QSignalBlocker, QThreadPool, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QKeySequence, QShortcut, QUndoCommand, QUndoStack
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QFrame, QHBoxLayout, QInputDialog,
    QLabel, QListWidget, QListWidgetItem, QMenu, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from .canvas import CanvasView, SELECTION_TOOLS, combine_selection, pixmap, selection_overlay
from .document import Document, Layer, asset_thumbnail, box, paint, paint_mask, validate_size
from .editor_preview import EditorPreview
from .editor_tasks import EditorTask
from .geometry import MAX_PNG_DIMENSION, Rect

_SELECTION_CACHE_BYTES = 64 * 1024 * 1024


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


@dataclass(eq=False)
class _Selection:
    mask: Image.Image
    bounds: tuple[int, int, int, int]
    overlay: QImage | None

    @classmethod
    def prepare(cls, mask):
        bounds = mask.getbbox() if mask is not None else None
        return cls(mask, bounds, selection_overlay(mask, bounds)) if bounds is not None else None


@dataclass(frozen=True)
class _State:
    document: Document
    active_id: str | None
    selection: _Selection | None = None


@dataclass(frozen=True)
class _LocalEdit:
    label: str
    update: Callable[[_State], _State | None]
    background: bool = False
    keep_gesture: bool = False


class _Edit(QUndoCommand):
    def __init__(self, label, before, after, show, *, keep_gesture=False):
        super().__init__(label)
        self.before, self.after, self.show = before, after, show
        self.keep_gesture = keep_gesture

    def undo(self):
        self.show(self.before)

    def redo(self):
        self.show(self.after, keep_gesture=self.keep_gesture)
        self.keep_gesture = False


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
        self._busy = False
        self._points = []
        self._start = None
        self._ink = QColor("#8ce1c6")
        self._items = {}
        self._thumbnails = {}
        self._edits = deque()
        self._after_edits = deque()
        self._edit_task = None
        self._draining_edits = False
        self._overlay_task = None
        self._selection_cache = OrderedDict()
        self._selection_bytes = 0
        self.undo = QUndoStack(self)
        self.undo.setUndoLimit(50)
        self.canvas = CanvasView()
        self.canvas.setToolTip("Scroll to zoom. Hold Space or use the middle button to pan.")
        self.canvas.stroke_finished.connect(self._select_pixels)
        self.canvas.pointer_started.connect(self._pointer_start)
        self.canvas.pointer_moved.connect(self._pointer_move)
        self.canvas.pointer_finished.connect(self._pointer_end)
        self.canvas.gesture_canceled.connect(self._reset_gesture)
        self._preview = EditorPreview(self.canvas, self.set_message, self)
        self._build()

    def _build(self):
        content = QVBoxLayout(self)
        content.setContentsMargins(0, 0, 0, 0)
        tools = QHBoxLayout()
        self.undo_button = _button("Undo", lambda: self._history_step(self.undo.undo), tip="Undo (Ctrl+Z)")
        self.redo_button = _button("Redo", lambda: self._history_step(self.undo.redo), tip="Redo (Ctrl+Y)")
        self.undo.canUndoChanged.connect(self._sync_undo_button)
        self.undo.canRedoChanged.connect(self.redo_button.setEnabled)
        self.undo_button.setEnabled(False)
        self.redo_button.setEnabled(False)
        tools.addWidget(self.undo_button)
        tools.addWidget(self.redo_button)
        for key, slot in (("Ctrl+Z", self.undo.undo), ("Ctrl+Y", self.undo.redo), ("Ctrl+Shift+Z", self.undo.redo)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda operation=slot: self._history_step(operation))
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
        selection = self._state.selection if self._state else None
        return selection.mask if selection is not None else None

    @property
    def dirty(self) -> bool:
        return self.document is not self._saved

    @property
    def edits_pending(self) -> bool:
        return self._edit_task is not None or bool(self._edits)

    def after_edits(self, operation) -> bool:
        if self.edits_pending:
            self._after_edits.append(operation)
            return True
        return False

    def set_document(self, document: Document):
        self.canvas.cancel_gesture()
        if self._edit_task is not None:
            self._edit_task[0].cancelled.set()
            self._edit_task = None
        self._edits.clear()
        self._after_edits.clear()
        self.undo.clear()
        self._sync_undo_button()
        self._selection_cache.clear()
        self._selection_bytes = 0
        self._saved = document
        self._state = None
        self.set_message("")
        top = document.layers[-1].id if document.layers else None
        self._restore(_State(document, top))

    def mark_saved(self, document: Document):
        self._saved = document
        self.changed.emit()

    def set_busy(self, busy: bool):
        if self._busy != busy:
            self._busy = busy
            self.setEnabled(not busy)
            self._tools_changed()

    def set_message(self, text: str):
        self.message.setText(text)
        self.message.setVisible(bool(text))

    def add_image(self, image: Image.Image, name: str):
        self._insert("Import layer", lambda _state: Layer(name, image if image.mode == "RGBA" else image.convert("RGBA")))

    def apply_fill(self, document: Document, layer_id: str, selection: Image.Image):
        layer = document.layer(layer_id)
        label = layer.name if layer is not None and layer.coverage is not None else "Generative Fill"
        if selection is self.selection:
            self._submit(label, lambda state: _State(document, layer_id, state.selection))
        else:
            self._submit(
                label, lambda _state: _State(document, layer_id, _Selection.prepare(selection)), background=True,
            )

    def _submit(self, label, update, *, background=False, keep_gesture=False):
        if self._state is not None:
            self._edits.append(_LocalEdit(label, update, background, keep_gesture))
            self._drain_edits()

    def _history_step(self, operation):
        self.canvas.cancel_gesture()
        self._edits.append(operation)
        self._drain_edits()

    def _sync_undo_button(self):
        self.undo_button.setEnabled(self.undo.canUndo() or self.edits_pending)

    def _drain_edits(self):
        if self._edit_task is not None or self._draining_edits:
            return
        self._draining_edits = True
        try:
            while self._edits:
                edit = self._edits.popleft()
                if not isinstance(edit, _LocalEdit):
                    edit()
                    continue
                before = self._state
                if edit.background:
                    task = EditorTask(lambda _cancelled: edit.update(before))
                    self._edit_task = task, edit, before
                    task.signals.finished.connect(self._edit_finished, Qt.ConnectionType.QueuedConnection)
                    QThreadPool.globalInstance().start(task)
                    break
                self._commit(edit, edit.update(before))
        finally:
            self._draining_edits = False
            self._sync_undo_button()
        if not self.edits_pending:
            callbacks, self._after_edits = self._after_edits, deque()
            for operation in callbacks:
                if not self.after_edits(operation):
                    operation()

    def _edit_finished(self, task, state, error):
        if self._edit_task is None or self._edit_task[0] is not task:
            return
        _, edit, before = self._edit_task
        # Selecting another layer while this operation runs is independent of its result.
        if state is not None and state.active_id == before.active_id:
            state = replace(state, active_id=self._state.active_id)
        if error:
            self.set_message(error)
        else:
            self._commit(edit, state)
        self._edit_task = None
        self._drain_edits()

    def _commit(self, edit, state):
        if state is not None:
            self.undo.push(_Edit(edit.label, self._state, state, self._restore, keep_gesture=edit.keep_gesture))

    def _restore(self, state: _State, *, keep_gesture=False):
        previous = self._state
        keep_gesture = (
            keep_gesture and previous is not None
            and previous.active_id == state.active_id and previous.document.size == state.document.size
        )
        if not keep_gesture:
            self.canvas.cancel_gesture()
        self._state = state
        if keep_gesture and self._start is not None and self.tool.currentData() == "move":
            draft = self._preview.draft
            if draft is not None:
                before = previous.document.layer(state.active_id)
                layer = self.active_layer
                if before is not None and layer is not None:
                    self._preview.update_draft(
                        layer.id, x=draft[1]["x"] + layer.x - before.x, y=draft[1]["y"] + layer.y - before.y,
                    )
        self._preview.set_document(state.document, reset=previous is None, keep_draft=keep_gesture)
        if previous is None or previous.selection is not state.selection:
            self._show_selection(state.selection)
        if previous is None or previous.active_id != state.active_id:
            with QSignalBlocker(self.edit_mask):
                self.edit_mask.setChecked(False)
        if previous is None or previous.document.layers is not state.document.layers:
            self._sync_layers(previous.document.layers if previous is not None else ())
        else:
            self._sync_current_item()
        self._update_layer_controls()
        self.changed.emit()

    def _show_selection(self, selection):
        if self._overlay_task is not None:
            self._overlay_task[0].cancelled.set()
            self._overlay_task = None
        if selection is not None and selection.overlay is not None:
            self._display_overlay(selection, selection.overlay)
            return
        self.canvas.set_selection_overlay(None, None)
        if selection is not None:
            task = EditorTask(lambda _cancelled: selection_overlay(selection.mask, selection.bounds))
            self._overlay_task = task, selection
            task.signals.finished.connect(self._overlay_finished, Qt.ConnectionType.QueuedConnection)
            QThreadPool.globalInstance().start(task)

    def _overlay_finished(self, task, overlay, error):
        if self._overlay_task is None or self._overlay_task[0] is not task:
            return
        _, selection = self._overlay_task
        self._overlay_task = None
        if error:
            self.set_message(error)
        elif overlay is not None:
            self._display_overlay(selection, overlay)

    def _display_overlay(self, selection, overlay):
        # Retain derived overlays for recent history without keeping old masks alive.
        key = ref(selection)
        if key in self._selection_cache:
            self._selection_cache.move_to_end(key)
        else:
            for entry in tuple(self._selection_cache):
                if entry() is None:
                    self._selection_bytes -= self._selection_cache.pop(entry)
            size = overlay.sizeInBytes()
            while self._selection_cache and self._selection_bytes + size > _SELECTION_CACHE_BYTES:
                oldest, size_removed = self._selection_cache.popitem(last=False)
                self._selection_bytes -= size_removed
                old_selection = oldest()
                if old_selection is not None:
                    old_selection.overlay = None
            if size <= _SELECTION_CACHE_BYTES:
                selection.overlay = overlay
                self._selection_cache[key] = size
                self._selection_bytes += size
            else:
                selection.overlay = None
        self.canvas.set_selection_overlay(overlay, selection.bounds)

    def _sync_layers(self, previous_layers):
        previous = {layer.id: layer for layer in previous_layers}
        current = {layer.id: layer for layer in self.document.layers}
        icons = {}
        with QSignalBlocker(self.layers):
            for layer_id in self._items.keys() - current.keys():
                item = self._items.pop(layer_id)
                self.layers.takeItem(self.layers.row(item))
            for row, layer in enumerate(reversed(self.document.layers)):
                old = previous.get(layer.id)
                item = self._items.get(layer.id)
                if item is None:
                    item = QListWidgetItem()
                    item.setData(Qt.ItemDataRole.UserRole, layer.id)
                    self._items[layer.id] = item
                    self.layers.insertItem(row, item)
                elif self.layers.item(row) is not item:
                    self.layers.takeItem(self.layers.row(item))
                    self.layers.insertItem(row, item)
                key = id(layer.image)
                icon = icons.get(key) or self._thumbnails.get(key)
                if icon is None:
                    thumbnail = asset_thumbnail(layer.image, self._thumbnail_size(layer.image.size))
                    icon = QIcon(pixmap(thumbnail))
                icons[key] = icon
                if old is None or old.image is not layer.image:
                    item.setIcon(icon)
                if old is None or old.name != layer.name:
                    item.setText(layer.name)
                if old is None or old.visible != layer.visible:
                    item.setCheckState(Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked)
                if old is None or (old.bounds, old.coverage is None) != (layer.bounds, layer.coverage is None):
                    kind = "Fill layer" if layer.coverage is not None else "Raster layer"
                    item.setToolTip(f"{kind} · {layer.image.width} × {layer.image.height} px · ({layer.x}, {layer.y})")
        self._thumbnails = icons
        self._sync_current_item()

    @staticmethod
    def _thumbnail_size(size):
        scale = min(1, 32 / max(size))
        return tuple(max(1, round(value * scale)) for value in size)

    def _sync_current_item(self):
        item = self._items.get(self._state.active_id)
        if self.layers.currentItem() is not item:
            with QSignalBlocker(self.layers):
                self.layers.setCurrentItem(item)

    def _choose_layer(self, item, _previous):
        if self._state:
            self.canvas.cancel_gesture()
            self._preview.end_draft()
            layer_id = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
            self._state = replace(self._state, active_id=layer_id)
            self.edit_mask.setChecked(False)
            self._update_layer_controls()

    def _update_layer_controls(self):
        layer = self.active_layer
        editable = layer is not None and not layer.locked
        has_mask = layer is not None and layer.mask is not None
        row = self.layers.currentRow()
        draft = self._preview.draft
        with QSignalBlocker(self.locked), QSignalBlocker(self.opacity), QSignalBlocker(self.edit_mask):
            self.locked.setChecked(layer.locked if layer is not None else False)
            if draft is None or "opacity" not in draft[1]:
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
        if layer is not None and (self.edits_pending or visible != layer.visible):
            self._update_layer("Show layer" if visible else "Hide layer", layer_id=layer.id, visible=visible)

    def _insert(self, label, make_layer, *, background=False):
        above_id = self._state.active_id

        def insert(state):
            layer = make_layer(state)
            document = state.document.insert(layer, above=state.document.layer(above_id))
            return _State(document, layer.id, state.selection)

        self._submit(label, insert, background=background)

    def _add_blank(self):
        self._insert("Add layer", lambda state: Layer("Layer", Image.new("RGBA", state.document.size)), background=True)

    def _edit_layer(self, label, layer_id, update, *, background=False, keep_gesture=False):
        def apply(state):
            layer = state.document.layer(layer_id)
            if layer is None:
                return None
            updated = update(state, layer)
            if updated is not layer:
                return replace(state, document=state.document.update(updated))
            return None

        self._submit(label, apply, background=background, keep_gesture=keep_gesture)

    def _update_layer(self, label, *, layer_id=None, **changes):
        def update(_state, layer):
            return replace(layer, **changes) if any(getattr(layer, key) != value for key, value in changes.items()) else layer

        self._edit_layer(label, self._state.active_id if layer_id is None else layer_id, update)

    def _rename(self):
        if self.active_layer is None:
            return
        name, accepted = QInputDialog.getText(self, "Rename layer", "Name", text=self.active_layer.name)
        if accepted and name.strip():
            self._update_layer("Rename layer", name=name.strip())

    def _duplicate(self):
        layer = self.active_layer
        if layer is not None:
            def duplicate(state):
                original = state.document.layer(layer.id)
                if original is None:
                    return None
                copied = replace(original, name=f"{original.name} copy", id=uuid4().hex, locked=False)
                return _State(state.document.insert(copied, above=original), copied.id, state.selection)

            self._submit("Duplicate layer", duplicate)

    def _delete(self):
        layer_id = self._state.active_id

        def delete(state):
            layer = state.document.layer(layer_id)
            if layer is None or layer.locked:
                return None
            index = state.document.layers.index(layer)
            layers = state.document.layers[:index] + state.document.layers[index + 1:]
            active_id = state.active_id
            if active_id == layer_id:
                active_id = layers[max(0, index - 1)].id if layers else None
            return _State(replace(state.document, layers=layers), active_id, state.selection)

        self._submit("Delete layer", delete)

    def _reorder(self, direction):
        layer_id = self._state.active_id

        def reorder(state):
            layer = state.document.layer(layer_id)
            if layer is None:
                return None
            layers = list(state.document.layers)
            index = layers.index(layer)
            destination = index + direction
            if 0 <= destination < len(layers):
                layers.insert(destination, layers.pop(index))
                return replace(state, document=replace(state.document, layers=tuple(layers)))
            return None

        self._submit("Reorder layer", reorder)

    def _opacity_changed(self, value):
        self.opacity_value.setText(f"{value}%")
        if self.opacity.isSliderDown() and self.active_layer is not None:
            self._preview.update_draft(self.active_layer.id, opacity=value)
        else:
            self._commit_opacity()

    def _commit_opacity(self):
        layer, opacity = self.active_layer, self.opacity.value()
        changed = layer is not None and opacity != layer.opacity
        self._preview.end_draft(commit=changed)
        if layer is not None and (changed or self.edits_pending):
            self._update_layer("Layer opacity", opacity=opacity)

    def _select_pixels(self, path):
        mode = self.selection_mode.currentData()

        def select(state):
            previous = state.selection.mask if state.selection is not None else None
            mask = combine_selection(previous, path, state.document.size, mode)
            if mask is not previous:
                selection = _Selection.prepare(mask)
                if selection is not None or previous is not None:
                    return replace(state, selection=selection)
            return None

        self._submit("Select pixels", select, background=True, keep_gesture=True)

    def _clear_selection(self):
        self._submit("Clear selection", lambda state: replace(state, selection=None) if state.selection else None)

    def _crop(self):
        def crop(state):
            if state.selection is not None:
                area = Rect(*state.selection.bounds)
                if area != state.document.bounds:
                    return replace(state, document=state.document.crop(area), selection=None)
            return None

        self._submit("Crop canvas", crop)

    def _tools_changed(self, *_):
        tool = self.tool.currentData()
        self.canvas.set_tool(tool)
        layer = self.active_layer
        editable = layer is not None and layer.visible and not layer.locked
        editing_mask = self.edit_mask.isChecked()
        painting = tool in ("brush", "eraser")
        can_paint = editable and (layer.mask is not None if editing_mask else layer.coverage is None)
        can_draw = tool in SELECTION_TOOLS or (editable and tool == "move") or (painting and can_paint)
        self.canvas.set_drawing_enabled(not self._busy and can_draw)
        self.selection_mode.setVisible(tool in SELECTION_TOOLS)
        self.clear_button.setEnabled(self.selection is not None)
        selection = self._state.selection if self._state is not None else None
        self.crop_button.setEnabled(selection is not None and selection.bounds != box(self.document.bounds))
        self.brush_controls.setVisible(painting)
        self.color.setEnabled(tool == "brush" and not editing_mask)
        self._brush_preview()

    def _pointer_start(self, point):
        self._start = point
        if self.tool.currentData() == "move":
            self._preview.begin_draft()
        else:
            self._points = [(math.floor(point.x()), math.floor(point.y()))]

    def _layer_position(self, point):
        dx, dy = self._drag_offset(point)
        layer = self.active_layer
        return layer.x + dx, layer.y + dy

    def _drag_offset(self, point):
        delta = point - self._start
        return round(delta.x()), round(delta.y())

    def _pointer_move(self, point):
        if self._start is None:
            return
        if self.tool.currentData() == "move":
            x, y = self._layer_position(point)
            self._preview.update_draft(self.active_layer.id, x=x, y=y)
        else:
            self._points.append((math.floor(point.x()), math.floor(point.y())))

    def _pointer_end(self, point):
        if self._start is None:
            return
        layer = self.active_layer
        tool = self.tool.currentData()
        if tool == "move":
            dx, dy = self._drag_offset(point)
            self._start = None
            moved = bool(dx or dy)
            self._preview.end_draft(commit=moved)
            if moved:
                self._edit_layer(
                    "Move layer", layer.id, lambda _state, target: replace(target, x=target.x + dx, y=target.y + dy),
                    keep_gesture=True,
                )
        else:
            points, self._points = self._points, []
            points.append((math.floor(point.x()), math.floor(point.y())))
            self._start = None
            color = self._ink.getRgb()[:3] if tool == "brush" else None
            size, strength, edit_mask = self.brush_size.value(), self.strength.value(), self.edit_mask.isChecked()
            label = "Paint mask" if edit_mask else ("Brush stroke" if color is not None else "Erase stroke")

            def stroke(state, target):
                if target.locked or not target.visible or (target.mask is None if edit_mask else target.coverage is not None):
                    return target
                return paint(
                    state.document, target, points, size=size, strength=strength, color=color, edit_mask=edit_mask,
                    selection=state.selection.mask if state.selection is not None else None,
                )

            self._edit_layer(label, layer.id, stroke, background=True, keep_gesture=True)

    def _reset_gesture(self):
        self._start = None
        self._points.clear()
        self._preview.end_draft()

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
        def add_mask(state, layer):
            if layer.locked or layer.mask is not None:
                return layer
            selection = state.selection
            mask = selection.mask.crop(box(layer.bounds)) if selection else Image.new("L", layer.image.size, 255)
            return replace(layer, mask=mask)

        self._edit_layer("Add layer mask", self._state.active_id, add_mask, background=True)

    def _edit_mask_selection(self, value):
        def edit_mask(state, layer):
            overlap = state.document.bounds.intersect(layer.bounds)
            if overlap is None or state.selection is None or layer.mask is None or layer.locked:
                return layer
            coverage = state.selection.mask.crop(box(overlap))
            if coverage.getbbox() is None:
                return layer
            mask = paint_mask(layer.mask, overlap.relative_to(layer.bounds), coverage, value)
            return replace(layer, mask=mask) if mask is not layer.mask else layer

        self._edit_layer("Edit layer mask", self._state.active_id, edit_mask, background=True)

    def _transform(self, label, operation):
        self._edit_layer(
            label, self._state.active_id,
            lambda _state, layer: layer.transform(lambda image: image.transpose(operation)) if not layer.locked else layer,
            background=True,
        )

    def _resize_layer(self):
        layer = self.active_layer
        size = dimensions(self, "Resize layer", layer.image.size, preserve_aspect=True)
        if size is not None and size != layer.image.size:
            self._edit_layer(
                "Resize layer", layer.id, lambda _state, target: target.resize(size) if not target.locked else target,
                background=True,
            )

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
            self._preview.update_draft(layer.id, adjustments=values)

        for field in fields:
            field.valueChanged.connect(preview)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        accepted = dialog.exec()
        values = tuple(field.value() for field in fields)
        changed = bool(accepted and values != layer.adjustments)
        self._preview.end_draft(commit=changed)
        if changed:
            self._update_layer("Color adjustments", layer_id=layer.id, adjustments=values)

    def showEvent(self, event):
        super().showEvent(event)
        self._preview.resume()

    def hideEvent(self, event):
        self.canvas.cancel_gesture()
        self._preview.end_draft()
        self._preview.suspend()
        super().hideEvent(event)
