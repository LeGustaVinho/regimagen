"""Review a pattern's seam mask and repair variations."""

from dataclasses import dataclass, field

from PIL import Image, ImageChops
from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QPainterPath
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from .canvas import CanvasView, combine_selection
from .document import Document
from .editor_preview import EditorPreview
from .pattern import SeamWarning


@dataclass(frozen=True)
class PatternVariant:
    document: Document
    layer_id: str
    shifted: Document
    warnings: tuple[SeamWarning, ...]
    notes: tuple[str, ...]


@dataclass
class PatternSession:
    document: Document
    original: Image.Image
    shifted: Image.Image
    offset: tuple[int, int]
    editor: "PatternEditor"
    variants: list[PatternVariant] = field(default_factory=list)


class PatternEditor(QWidget):
    detect_requested = Signal()
    generate_requested = Signal()
    apply_requested = Signal()
    discard_requested = Signal()
    cancel_requested = Signal()
    changed = Signal()

    def __init__(self, original: Document, shifted: Document):
        super().__init__()
        self.mask = None
        self._size = shifted.size
        self._original = original
        self._shifted = shifted
        self._variants: list[PatternVariant] = []
        self._busy = False
        self._operation = "add"
        self.canvas = CanvasView()
        self.canvas.setToolTip("Scroll to zoom. Hold Space or use the middle button to pan.")
        self.canvas.stroke_finished.connect(self._edit_mask)
        self._preview = EditorPreview(self.canvas, self.set_message, self)
        content = QVBoxLayout(self)
        content.setContentsMargins(0, 0, 0, 0)
        toolbar = QHBoxLayout()
        self._shape = QComboBox()
        self._shape.setAccessibleName("Selection tool")
        self._shape.addItem("Lasso", "lasso")
        self._shape.addItem("Rectangle", "rectangle")
        self._shape.currentIndexChanged.connect(lambda: self.canvas.set_tool(self._shape.currentData()))
        toolbar.addWidget(self._shape)
        self._modes = QButtonGroup(self)
        for label, operation in (("Add", "add"), ("Subtract", "subtract")):
            button = self._button(toolbar, label,
                                  lambda _checked=False, mode=operation: self._set_operation(mode), checkable=True)
            button.setChecked(operation == self._operation)
            self._modes.addButton(button)
        toolbar.addStretch()
        self._repeated = self._button(toolbar, "Pattern Preview", self._show_view, checkable=True)
        self._button(toolbar, "Fit", self.canvas.fit)
        self._button(toolbar, "100%", self.canvas.actual_size)
        self.discard = self._button(toolbar, "Discard", self.discard_requested)
        content.addLayout(toolbar)
        content.addWidget(self.canvas, 1)

        actions = QHBoxLayout()
        self.detect = self._button(actions, "Detect Seams", self.detect_requested)
        self.generate = self._button(actions, "Generate", self.generate_requested)
        self.generate.setProperty("primary", True)
        self.cancel = self._button(actions, "Cancel", self.cancel_requested)
        actions.addStretch()
        compare = QButtonGroup(self)
        self.before = self._button(actions, "Before", self._show_view, checkable=True)
        self.after = self._button(actions, "After", self._show_view, checkable=True)
        self.before.setChecked(True)
        for button in (self.before, self.after):
            compare.addButton(button)
        self.chooser = QComboBox()
        self.chooser.setAccessibleName("Pattern variation")
        self.chooser.currentIndexChanged.connect(self._choose_variant)
        actions.addWidget(self.chooser)
        self.apply = self._button(actions, "Apply", self.apply_requested)
        content.addLayout(actions)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        content.addWidget(self.status)
        self.set_message("Detect the seams or draw a mask on the offset tile.")
        self._show_view()

    @staticmethod
    def _button(row, label, action, *, checkable=False):
        button = QPushButton(label)
        button.setCheckable(checkable)
        if checkable:
            button.setStyleSheet("QPushButton:checked { color: #d0fff2; background: #28504a; border-color: #75d8be; }")
        button.clicked.connect(action)
        row.addWidget(button)
        return button

    @property
    def selected_variant(self):
        index = self.chooser.currentIndex()
        return index if 0 <= index < len(self._variants) else None

    @property
    def can_generate(self):
        return self.mask is not None

    def _set_operation(self, operation):
        self._operation = operation

    def _edit_mask(self, path):
        self.set_mask(combine_selection(self.mask, path, self._size, self._operation))

    def set_mask(self, mask: Image.Image | None):
        if mask is self.mask or (
            mask is not None and self.mask is not None and ImageChops.difference(mask, self.mask).getbbox() is None
        ):
            return
        bounds = mask.getbbox() if mask is not None else None
        self.mask = mask if bounds else None
        self._variants.clear()
        with QSignalBlocker(self.chooser):
            self.chooser.clear()
        self.before.setChecked(True)
        self.canvas.set_selection(self.mask, bounds)
        self.changed.emit()
        self.set_message("Review the mask, then generate." if bounds else
                         "Draw a seam mask or run detection again.")
        self._show_view()

    def show_detection(self, mask):
        self.detect.setText("Detect Again")
        self.before.setChecked(True)
        self._repeated.setChecked(False)
        if mask.getbbox():
            self.set_mask(mask)
            self.set_message("Review the detected seams, then generate.")
        else:
            self.set_message("No seams detected. Draw a mask or run detection again.")
        self._show_view()

    def add_variant(self, variant):
        self._variants.append(variant)
        with QSignalBlocker(self.chooser):
            self.chooser.addItem(f"Variation {len(self._variants)}")
            self.chooser.setCurrentIndex(len(self._variants) - 1)
        self._repeated.setChecked(True)
        self._choose_variant(self.chooser.currentIndex())

    def _choose_variant(self, index):
        if index >= 0:
            self.after.setChecked(True)
            variant = self._variants[index]
            message = ("Amber marks possible remaining seams." if variant.warnings else
                       "Compare the repeat before applying.")
            self.set_message(" ".join((message, *variant.notes)))
        self._show_view()

    def _show_view(self):
        index = self.selected_variant
        variant = self._variants[index] if index is not None and self.after.isChecked() else None
        repeated = self._repeated.isChecked()
        if variant is None:
            document = self._original if repeated else self._shifted
        else:
            document = variant.document if repeated else variant.shifted
        self._preview.set_pattern_preview(repeated)
        self._preview.set_selection_visible(not repeated and variant is None)
        self._preview.set_document(document)
        guides = QPainterPath()
        if repeated and variant is not None:
            width, height = document.size
            for warning in variant.warnings:
                for edge in (0, width if warning.axis == "vertical" else height):
                    if warning.axis == "vertical":
                        guides.moveTo(edge, warning.start)
                        guides.lineTo(edge, warning.end)
                    else:
                        guides.moveTo(warning.start, edge)
                        guides.lineTo(warning.end, edge)
        self.canvas.set_guides(guides)
        self._refresh_controls()

    def set_message(self, message):
        self.status.setText(message)
        self.status.setVisible(bool(message))

    def set_busy(self, busy, *, cancellable=False, cancelling=False):
        self._busy = busy
        self.cancel.setVisible(busy and cancellable)
        self.cancel.setEnabled(not cancelling)
        self._refresh_controls()

    def _refresh_controls(self):
        variants = bool(self._variants)
        edit = not self._busy and self.before.isChecked() and not self._repeated.isChecked()
        self.canvas.set_drawing_enabled(edit)
        self._shape.setEnabled(edit)
        for button in self._modes.buttons():
            button.setEnabled(edit)
        self.detect.setEnabled(not self._busy)
        self.generate.setEnabled(not self._busy and self.can_generate)
        self.generate.setText("Generate Another" if variants else "Generate")
        self.discard.setEnabled(not self._busy)
        self.before.setEnabled(variants)
        self.after.setEnabled(variants)
        self.chooser.setVisible(len(self._variants) > 1)
        self.chooser.setEnabled(not self._busy)
        self.apply.setEnabled(variants and not self._busy)

    def showEvent(self, event):
        super().showEvent(event)
        self._preview.resume()

    def hideEvent(self, event):
        self.canvas.cancel_gesture()
        self._preview.suspend()
        super().hideEvent(event)
