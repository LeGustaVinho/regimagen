"""Selection and before/after previews for Generative Fill."""

from PIL import Image
from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)

from .canvas import CanvasView, combine_selection, pixmap
from .fill import TRANSPARENT_SELECTION, FillPreview, selection_has_visible_pixels


class FillEditor(QWidget):
    generate_requested = Signal()
    apply_requested = Signal()
    discard_requested = Signal()
    cancel_requested = Signal()
    changed = Signal(object)

    def __init__(
        self, base: Image.Image, *, mask: Image.Image | None = None, prompt: str = "",
        edit_alpha: bool | None = None, layered: bool = False,
        active_target: tuple[Image.Image, tuple[int, int, int, int]] | None = None,
    ):
        super().__init__()
        if mask is not None and mask.size != base.size:
            raise ValueError("The selection must match the image dimensions.")
        self.setStyleSheet("QPushButton:checked { color: #d0fff2; background: #28504a; border-color: #75d8be; }")
        self._source = base
        self._base = pixmap(base.convert("RGBA") if layered and base.mode == "RGB" else base)
        self._mask = mask.convert("L") if mask is not None else Image.new("L", base.size)
        self._bounds = self._mask.getbbox()
        self._visible_selection: bool | None = None
        self._target_selected = False
        self._active_target = active_target
        self._has_alpha = layered or base.mode == "RGBA"
        self._operation = "new"
        self._busy = False
        self._generating = False
        self._variants: list[FillPreview] = []
        self._message = ""

        content = QVBoxLayout(self)
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(10)
        tools = QHBoxLayout()
        self._selection_tools = QButtonGroup(self)
        self._mode_buttons: dict[str, QPushButton] = {}
        for operation, label, tip in (
            ("new", "New", "Draw a new selection"),
            ("add", "Add", "Add to the selection"),
            ("subtract", "Subtract", "Remove from the selection"),
        ):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setToolTip(tip)
            button.setChecked(operation == self._operation)
            button.clicked.connect(lambda _checked, mode=operation: self._select_operation(mode))
            self._selection_tools.addButton(button)
            self._mode_buttons[operation] = button
            tools.addWidget(button)
        tools.addStretch()

        self._canvas = CanvasView(self._base)
        self._canvas.setToolTip("Drag to select. Scroll to zoom. Hold Space or use the middle button to pan.")
        self._canvas.stroke_started.connect(self._show_before)
        self._canvas.stroke_finished.connect(self._apply_lasso)
        if layered:
            shapes = QComboBox()
            shapes.setAccessibleName("Selection tool")
            shapes.addItem("Lasso", "lasso")
            shapes.addItem("Rectangle", "rectangle")
            shapes.currentIndexChanged.connect(lambda: self._canvas.set_tool(shapes.currentData()))
            self._shapes = shapes
            tools.insertWidget(0, shapes)
        else:
            self._shapes = None
        for label, tip, action in (
            ("−", "Zoom out", lambda: self._canvas.zoom(1 / 1.2)),
            ("100%", "Show pixels at actual size", self._canvas.actual_size),
            ("+", "Zoom in", lambda: self._canvas.zoom(1.2)),
            ("Fit", "Fit to window", self._canvas.fit),
        ):
            button = QPushButton(label)
            button.setToolTip(tip)
            button.clicked.connect(action)
            tools.addWidget(button)
        self._discard = QPushButton("Discard")
        self._discard.clicked.connect(self.discard_requested)
        tools.addWidget(self._discard)
        content.addLayout(tools)
        content.addWidget(self._canvas, 1)

        self._controls = QWidget()
        controls = QVBoxLayout(self._controls)
        controls.setContentsMargins(0, 0, 0, 0)
        request = QHBoxLayout()
        self._destination = None
        if layered:
            self._destination = QComboBox()
            self._destination.setAccessibleName("Fill destination")
            self._destination.setToolTip(
                "New layer uses the visible composition. Active layer uses its own pixels."
                if active_target is not None else "Select a visible, unlocked raster layer to edit it directly."
            )
            self._destination.addItem("New layer", False)
            self._destination.addItem("Active layer", True)
            self._destination.model().item(1).setEnabled(active_target is not None)
            self._destination.currentIndexChanged.connect(self._draft_changed)
            request.addWidget(self._destination)
        self._prompt = QLineEdit(prompt)
        self._prompt_value = self._prompt.text().strip()
        self._prompt.setPlaceholderText("Describe the change, or leave blank to fill from context")
        self._prompt.setAccessibleName("Fill prompt")
        self._prompt.textChanged.connect(self._prompt_changed)
        request.addWidget(self._prompt, 1)
        self._alpha = QCheckBox("Edit Transparency")
        self._alpha.setChecked(base.mode == "RGBA" if edit_alpha is None else self._has_alpha and edit_alpha)
        self._alpha.setVisible(self._has_alpha)
        self._alpha.toggled.connect(self._draft_changed)
        request.addWidget(self._alpha)
        self._generate = QPushButton("Generate")
        self._generate.setProperty("primary", True)
        self._generate.clicked.connect(self.generate_requested)
        self._prompt.setMinimumHeight(self._generate.sizeHint().height())
        request.addWidget(self._generate)
        self._cancel = QPushButton("Cancel generation")
        self._cancel.clicked.connect(self.cancel_requested)
        request.addWidget(self._cancel)
        controls.addLayout(request)

        self._review = QWidget()
        review = QHBoxLayout(self._review)
        review.setContentsMargins(0, 0, 0, 0)
        self._before = QPushButton("Before")
        self._after = QPushButton("After")
        self._compare = QButtonGroup(self)
        for button in (self._before, self._after):
            button.setCheckable(True)
            button.clicked.connect(self._show_preview)
            self._compare.addButton(button)
            review.addWidget(button)
        self._before.setChecked(True)
        self._chooser = QComboBox()
        self._chooser.setAccessibleName("Generated variation")
        self._chooser.currentIndexChanged.connect(self._select_variant)
        review.addWidget(self._chooser)
        review.addStretch()
        self._apply = QPushButton("Apply")
        self._apply.setProperty("primary", True)
        self._apply.clicked.connect(self.apply_requested)
        review.addWidget(self._apply)
        controls.addWidget(self._review)
        content.addWidget(self._controls)

        self._status = QLabel()
        self._status.setWordWrap(True)
        self._status.setTextFormat(Qt.TextFormat.PlainText)
        content.addWidget(self._status)
        self._canvas.set_selection(self._mask, self._bounds)
        self._refresh()

    @property
    def mask(self) -> Image.Image:
        return self._mask

    @property
    def prompt(self) -> str:
        return self._prompt_value

    @property
    def edit_alpha(self) -> bool:
        return self._has_alpha and self._alpha.isChecked()

    @property
    def to_active_layer(self) -> bool:
        return self._destination is not None and bool(self._destination.currentData())

    def _reference(self) -> tuple[Image.Image, Image.Image]:
        if self.to_active_layer and self._active_target is not None:
            image, region = self._active_target
            return image, self._mask.crop(region)
        return self._source, self._mask

    def generation_input(self) -> tuple[Image.Image, Image.Image]:
        image, mask = self._reference()
        if self.edit_alpha and image.mode != "RGBA":
            image = image.convert("RGBA")
        return image, mask

    @property
    def can_generate(self) -> bool:
        if self._bounds is None:
            return False
        if self._visible_selection is None:
            image, mask = self._reference()
            bounds = mask.getbbox() if self.to_active_layer else self._bounds
            self._target_selected = bounds is not None
            self._visible_selection = self._target_selected and (
                image.mode == "RGB" or self.edit_alpha or selection_has_visible_pixels(image, mask, bounds)
            )
        return self._visible_selection

    @property
    def selected_variant(self) -> int | None:
        index = self._chooser.currentIndex()
        return index if 0 <= index < len(self._variants) else None

    @property
    def can_apply(self) -> bool:
        return self.selected_variant is not None and not self._chooser.currentData()

    def set_busy(self, busy: bool, *, generating: bool = False):
        self._busy = busy
        self._generating = busy and generating
        self._canvas.set_drawing_enabled(not busy)
        self._refresh()

    def set_message(self, message: str):
        self._message = message
        self._refresh()

    def clear_variants(self):
        self._variants.clear()
        with QSignalBlocker(self._chooser):
            self._chooser.clear()
        self._show_before()
        self._refresh()

    def add_variant(self, preview: FillPreview, *, already_applied: bool = False):
        self._message = ""
        self._variants.append(preview)
        count = len(self._variants)
        with QSignalBlocker(self._chooser):
            self._chooser.addItem("Saved variation" if already_applied else f"Variation {count}", already_applied)
            self._chooser.setCurrentIndex(count - 1)
        self._after.setChecked(True)
        self._show_preview()
        self._refresh()

    def _select_operation(self, operation: str):
        self._operation = operation

    def _apply_lasso(self, path: QPainterPath):
        mask = combine_selection(self._mask, path, self._mask.size, self._operation)
        if mask is self._mask:
            return
        reuse_variant = self.selected_variant if self._operation == "subtract" else None
        self._mask = mask
        self._bounds = mask.getbbox()
        self._canvas.set_selection(mask, self._bounds)
        self._draft_changed(reuse_variant=reuse_variant)

    def _draft_changed(self, *, reuse_variant: int | None = None):
        self._message = ""
        self._visible_selection = None
        self.clear_variants()
        self.changed.emit(reuse_variant)

    def _prompt_changed(self, text: str):
        value = text.strip()
        if value != self._prompt_value:
            self._prompt_value = value
            self._draft_changed()

    def _show_before(self):
        self._before.setChecked(True)
        self._show_preview()

    def _select_variant(self, index: int):
        if index >= 0:
            self._after.setChecked(True)
        self._show_preview()
        self._refresh()

    def _show_preview(self):
        index = self.selected_variant
        showing_after = self._after.isChecked() and index is not None
        display = self._base
        if showing_after:
            preview = self._variants[index]
            display = QPixmap(self._base)
            painter = QPainter(display)
            # Replace the base alpha as well so a cutout reveals the checkerboard.
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            painter.drawPixmap(*preview.origin, pixmap(preview.patch))
            painter.end()
        self._canvas.set_image(display, selection_visible=not showing_after)

    def _refresh(self):
        selected = self._bounds is not None
        can_generate = self.can_generate
        index = self.selected_variant
        self._controls.setVisible(selected)
        self._review.setVisible(index is not None)
        self._chooser.setVisible(len(self._variants) > 1)
        self._discard.setEnabled(not self._busy)
        self._prompt.setEnabled(not self._busy)
        self._alpha.setEnabled(not self._busy)
        for field in (self._destination, self._shapes):
            if field is not None:
                field.setEnabled(not self._busy)
        self._generate.setText("Generate another" if self._variants else "Generate")
        self._generate.setEnabled(can_generate and not self._busy)
        self._cancel.setVisible(self._generating)
        self._apply.setEnabled(self.can_apply and not self._busy)
        for button in self._selection_tools.buttons():
            button.setEnabled(not self._busy)
        message = self._message
        if not message:
            if not selected:
                message = "Draw around the area you want to fill."
            elif not self._target_selected:
                message = "Select an area inside the active layer."
            elif not can_generate:
                message = TRANSPARENT_SELECTION
            elif index is not None:
                warning = self._variants[index].warning
                message = warning if self.can_apply else f"This variation is already applied. {warning}".strip()
        self._status.setText(message)
        self._status.setVisible(bool(message))
