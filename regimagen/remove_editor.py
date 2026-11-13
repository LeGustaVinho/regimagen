"""Review detected objects and their editable removal masks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageChops
from PySide6.QtCore import QPointF, QSignalBlocker, Qt, Signal
from PySide6.QtGui import QColor, QPainterPath, QPen, QPixmap, QTransform
from PySide6.QtWidgets import QApplication, QGraphicsItem, QHBoxLayout, QListWidget, QListWidgetItem, QPushButton

from .canvas import combine_selection, pixmap
from .fill import selection_has_visible_pixels
from .fill_editor import FillEditor

if TYPE_CHECKING:
    from .remove_detection import RemovalTarget


_BINARY = [0] + [255] * 255


def _union_target(destination: Image.Image, target: RemovalTarget, *, excluding: RemovalTarget | None = None):
    left, top = target.origin
    area = (left, top, left + target.mask.width, top + target.mask.height)
    mask = target.mask
    if excluding is not None:
        x, y = excluding.origin
        overlap = excluding.mask.crop((left - x, top - y, area[2] - x, area[3] - y))
        mask = ImageChops.subtract(mask, overlap.point(_BINARY))
    destination.paste(ImageChops.lighter(destination.crop(area), mask), (left, top))


def removal_masks(
    size: tuple[int, int], targets: tuple[RemovalTarget, ...], checked: set[int] | None = None, *,
    manual: Image.Image | None = None, added: Image.Image | None = None, excluded: Image.Image | None = None,
) -> tuple[Image.Image, Image.Image, tuple[str, ...]] | None:
    """Return the fully replaced pixels, approved footprint, and confirmed labels."""
    if checked is None:
        checked = {index for index, target in enumerate(targets) if target.selected and not target.protected}
    core = manual.copy() if manual is not None else Image.new("L", size)
    protected = excluded.copy() if excluded is not None else Image.new("L", size)
    selected = []
    for index, target in enumerate(targets):
        parent = targets[target.parent] if target.parent is not None and target.parent in checked else None
        if target.protected or index not in checked or (target.parent is not None and parent is None):
            # A kept effect can only be preserved where its removed parent is not visible.
            _union_target(protected, target, excluding=None if target.protected else parent)
        else:
            _union_target(core, target)
            selected.append(target)
    if added is not None:
        core = ImageChops.lighter(core, added)
    protected = protected.point(_BINARY)
    core = ImageChops.subtract(core.point(_BINARY), protected)
    bounds = core.getbbox()
    if bounds is None:
        return None

    left, top, right, bottom = bounds
    feather = max(1, min(12, round(min(right - left, bottom - top) * 0.04)))
    area = (max(0, left - feather), max(0, top - feather),
            min(size[0], right + feather), min(size[1], bottom + feather))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * feather + 1, 2 * feather + 1))
    expanded = Image.fromarray(cv2.dilate(np.asarray(core.crop(area)), kernel))
    footprint = Image.new("L", size)
    footprint.paste(ImageChops.subtract(expanded, protected.crop(area)), area[:2])
    labels = []
    for target in selected:
        x, y = target.origin
        overlap = core.crop((x, y, x + target.mask.width, y + target.mask.height))
        if ImageChops.darker(overlap, target.mask).getbbox() is not None:
            labels.append(target.label)
    return core, footprint, tuple(labels)


def _outline(mask: Image.Image, origin: tuple[int, int] = (0, 0)) -> QPainterPath:
    bounds = mask.getbbox()
    path = QPainterPath()
    if bounds is None:
        return path
    left, top = origin[0] + bounds[0], origin[1] + bounds[1]
    pixels = mask.crop(bounds)
    width, height = pixels.size
    pixels.thumbnail((2048, 2048), Image.Resampling.BILINEAR)
    scale_x, scale_y = width / pixels.width, height / pixels.height
    contours, _ = cv2.findContours(np.asarray(pixels), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points = contour.reshape(-1, 2)
        x, y = points[0]
        if len(points) == 1:
            path.addRect(left + x * scale_x, top + y * scale_y, scale_x, scale_y)
            continue
        path.moveTo(left + x * scale_x, top + y * scale_y)
        for x, y in points[1:]:
            path.lineTo(left + x * scale_x, top + y * scale_y)
        path.closeSubpath()
    return path


class RemoveEditor(FillEditor):
    detect_requested = Signal()
    semantic_detection_requested = Signal()

    def __init__(self, base: Image.Image, *, mask: Image.Image | None = None):
        super().__init__(base, mask=mask, layered=True)
        self._original_mask: Image.Image | None = None
        self._targets: tuple[RemovalTarget, ...] = ()
        self._manual_mask: Image.Image | None = None
        self._added: Image.Image | None = None
        self._excluded: Image.Image | None = None
        self._inputs: tuple[Image.Image, Image.Image, tuple[str, ...]] | None = None
        self._detection_points: tuple[tuple[int, int], ...] = ()
        self._click_start: QPointF | None = None
        self._detailed = False
        self._has_extension = False
        self._prompt.hide()
        self._alpha.hide()
        self._destination.hide()
        self._cancel.setText("Cancel")

        actions = QHBoxLayout()
        self._detect = QPushButton("Detect objects")
        self._detect.setProperty("primary", True)
        self._detect.clicked.connect(self.detect_requested)
        actions.addWidget(self._detect)
        self._manual = QPushButton("Use selection manually")
        self._manual.clicked.connect(self._use_manual)
        actions.addWidget(self._manual)
        self._change_selection = QPushButton("Change selection")
        self._change_selection.clicked.connect(self._reset_detection)
        actions.addWidget(self._change_selection)
        self._refine = QPushButton("Refine selection")
        self._refine.clicked.connect(self._open_refinement)
        actions.addWidget(self._refine)
        self._detect_effects = QPushButton("Detect objects and effects")
        self._detect_effects.clicked.connect(self.semantic_detection_requested)
        actions.addWidget(self._detect_effects)
        actions.addStretch()

        self._target_list = QListWidget()
        self._target_list.setAccessibleName("Objects and effects to remove")
        self._target_list.setMaximumHeight(100)
        self._target_list.itemChanged.connect(self._targets_changed)
        self._target_list.currentItemChanged.connect(self._focus_target)
        self._controls.layout().insertLayout(0, actions)
        self._controls.layout().insertWidget(1, self._target_list)

        scene = self._canvas.scene()
        self._extension = scene.addPixmap(QPixmap())
        self._extension.setZValue(1.1)
        lasso_pen = QPen(QColor("#edf1f7"), 1.4, Qt.PenStyle.DashLine)
        lasso_pen.setCosmetic(True)
        self._lasso_outline = scene.addPath(QPainterPath(), lasso_pen)
        self._lasso_outline.setZValue(1.2)
        removal_pen = QPen(QColor("#8ce1c6"), 2)
        removal_pen.setCosmetic(True)
        self._removal_outline = scene.addPath(QPainterPath(), removal_pen)
        self._removal_outline.setZValue(1.25)
        self._target_outline = scene.addPath(QPainterPath())
        self._target_outline.setZValue(1.3)
        self._click_marker = scene.addEllipse(-4, -4, 8, 8, lasso_pen)
        self._click_marker.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self._click_marker.setZValue(1.3)

        self._shapes.setAccessibleName("Object selection tool")
        with QSignalBlocker(self._shapes):
            self._shapes.clear()
            self._shapes.addItem("Click", "click")
            self._shapes.addItem("Lasso", "lasso")
            self._shapes.setCurrentIndex(1 if self._bounds else 0)
        self._shapes.currentIndexChanged.connect(self._change_selection_tool)
        self._canvas.set_tool(self._shapes.currentData())
        self._canvas.setToolTip("Click an object or choose Lasso to draw around it. Scroll to zoom. Hold Space or use the middle button to pan.")
        self._canvas.pointer_started.connect(self._begin_click)
        self._canvas.pointer_finished.connect(self._finish_click)
        self._show_before()
        self._refresh()

    @property
    def detection_mask(self) -> Image.Image:
        return self._original_mask if self._original_mask is not None else self.mask

    @property
    def detection_points(self) -> tuple[tuple[int, int], ...]:
        return self._detection_points

    @property
    def can_detect(self) -> bool:
        if self._detection_points:
            return self._source.mode == "RGB" or self._source.getpixel(self._detection_points[0])[-1] > 0
        return self._can_detect_lasso()

    @property
    def can_generate(self) -> bool:
        return getattr(self, "_inputs", None) is not None

    def removal_inputs(self) -> tuple[Image.Image, Image.Image, tuple[str, ...]] | None:
        return self._inputs

    def set_targets(self, targets: tuple[RemovalTarget, ...], *, detailed: bool = False):
        if not any(not target.protected for target in targets):
            if self._original_mask is not None:
                message = "No additional objects or effects detected. You can refine the marked area manually."
            elif self._detection_points:
                message = "No objects detected. Click a different spot or use Lasso to mark the object."
            else:
                message = "No objects detected. Adjust the lasso or use the selection manually."
            self.set_message(message)
            return
        requires_review = len(targets) > 1 or not targets[0].selected or targets[0].parent is not None
        self._start_review(targets, detailed=detailed or requires_review or self._detailed)

    def _can_detect_lasso(self) -> bool:
        bounds = self._bounds if self._original_mask is None else self._original_mask.getbbox()
        return bounds is not None and (
            self._source.mode == "RGB" or selection_has_visible_pixels(self._source, self.detection_mask, bounds)
        )

    def _change_selection_tool(self):
        self._detection_points = ()
        self._mask = Image.new("L", self._source.size)
        self._bounds = None
        self._canvas.set_selection(self._mask, None)
        self._draft_changed()

    def _begin_click(self, point: QPointF):
        self._click_start = point

    def _finish_click(self, point: QPointF):
        start, self._click_start = self._click_start, None
        if start is None or self._shapes.currentData() != "click" or self._original_mask is not None or self._busy:
            return
        distance = (self._canvas.mapFromScene(start) - self._canvas.mapFromScene(point)).manhattanLength()
        if distance >= QApplication.startDragDistance():
            self.set_message("Choose Lasso to draw around an object.")
            return
        x, y = int(start.x()), int(start.y())
        if not (0 <= x < self._source.width and 0 <= y < self._source.height):
            return
        self._detection_points = ((x, y),)
        self._click_marker.setPos(x + 0.5, y + 0.5)
        self._draft_changed()
        if self.can_detect:
            self.detect_requested.emit()

    def _use_manual(self):
        if self._can_detect_lasso():
            self._start_review((), detailed=True, manual=self.detection_mask)

    def _start_review(self, targets: tuple[RemovalTarget, ...], *, detailed: bool, manual: Image.Image | None = None):
        self._original_mask = self.detection_mask
        self._detailed = detailed
        self._targets, self._manual_mask = targets, manual
        self._canvas.set_tool("lasso")
        if detailed:
            self._select_operation("add")
            self._mode_buttons["add"].setChecked(True)
        self._lasso_outline.setPath(_outline(self._original_mask))
        with QSignalBlocker(self._target_list):
            self._target_list.clear()
            for index, target in enumerate(targets):
                item = QListWidgetItem(f"Keep {target.label}" if target.protected else target.label)
                item.setData(Qt.ItemDataRole.UserRole, index)
                if target.protected:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                    item.setToolTip("This object is protected from changes.")
                else:
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Checked if target.selected else Qt.CheckState.Unchecked)
                self._target_list.addItem(item)
            if self._target_list.count():
                self._target_list.setCurrentRow(0)
        self._targets_changed()

    def _open_refinement(self):
        self._detailed = True
        self._select_operation("add")
        self._mode_buttons["add"].setChecked(True)
        self._show_preview()
        self._refresh()

    def _reset_detection(self):
        self._mask = self.detection_mask
        self._bounds = self._mask.getbbox()
        self._original_mask = self._manual_mask = self._added = self._excluded = None
        self._targets = ()
        self._inputs = None
        self._detailed = False
        self._has_extension = False
        with QSignalBlocker(self._target_list):
            self._target_list.clear()
        self._select_operation("new")
        self._mode_buttons["new"].setChecked(True)
        self._canvas.set_tool(self._shapes.currentData())
        self._canvas.set_selection(self._mask, self._bounds)
        self._draft_changed()

    def _targets_changed(self, *_):
        items = [self._target_list.item(row) for row in range(self._target_list.count())]
        checked = {
            item.data(Qt.ItemDataRole.UserRole) for item in items
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable and item.checkState() == Qt.CheckState.Checked
        }
        with QSignalBlocker(self._target_list):
            for item in items:
                target = self._targets[item.data(Qt.ItemDataRole.UserRole)]
                enabled = target.parent is None or target.parent in checked
                flags = item.flags()
                item.setFlags(flags | Qt.ItemFlag.ItemIsEnabled if enabled else flags & ~Qt.ItemFlag.ItemIsEnabled)
        self._inputs = removal_masks(
            self._source.size, self._targets, checked, manual=self._manual_mask, added=self._added, excluded=self._excluded,
        )
        if self._inputs is not None and self._source.mode == "RGBA":
            core = self._inputs[0]
            bounds = core.getbbox()
            if bounds is not None and not selection_has_visible_pixels(self._source, core, bounds):
                self._inputs = None
        self._mask = self._inputs[1] if self._inputs is not None else Image.new("L", self._source.size)
        self._bounds = self._mask.getbbox()
        self._canvas.set_selection(self._mask, self._bounds)
        self._removal_outline.setPath(_outline(self._mask))
        self._show_extension()
        self._focus_target(self._target_list.currentItem())
        self._draft_changed()

    def _show_extension(self):
        if self._detection_points:
            self._has_extension = False
            self._extension.setPixmap(QPixmap())
            return
        extra = ImageChops.subtract(self._mask, self._original_mask.point(_BINARY))
        bounds = extra.getbbox()
        self._has_extension = bounds is not None
        if bounds is None:
            self._extension.setPixmap(QPixmap())
            return
        cropped = extra.crop(bounds)
        width, height = cropped.size
        cropped.thumbnail((4096, 4096), Image.Resampling.BILINEAR)
        overlay = Image.new("RGBA", cropped.size, (255, 168, 67, 0))
        overlay.putalpha(cropped.point([round(value * 110 / 255) for value in range(256)]))
        self._extension.setPixmap(pixmap(overlay))
        self._extension.setPos(*bounds[:2])
        self._extension.setTransform(QTransform.fromScale(width / cropped.width, height / cropped.height))

    def _focus_target(self, item, *_):
        if item is None:
            self._target_outline.setPath(QPainterPath())
            return
        target = self._targets[item.data(Qt.ItemDataRole.UserRole)]
        pen = QPen(QColor("#ffe6a7" if target.protected else "#edf1f7"), 1.5, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self._target_outline.setPen(pen)
        self._target_outline.setPath(_outline(target.mask, target.origin))

    def _apply_lasso(self, path: QPainterPath):
        if self._original_mask is None:
            previous = self._mask
            super()._apply_lasso(path)
            if self._mask is not previous and self.can_detect:
                self.detect_requested.emit()
            return
        if not self._detailed:
            return
        stroke = combine_selection(None, path, self._source.size, "new")
        if stroke is None:
            return
        if self._operation == "subtract":
            self._excluded = stroke if self._excluded is None else ImageChops.lighter(self._excluded, stroke)
            if self._added is not None:
                self._added = ImageChops.subtract(self._added, stroke.point(_BINARY))
        else:
            self._added = stroke if self._added is None else ImageChops.lighter(self._added, stroke)
            if self._excluded is not None:
                self._excluded = ImageChops.subtract(self._excluded, stroke.point(_BINARY))
        self._targets_changed()

    def _show_preview(self):
        super()._show_preview()
        if not hasattr(self, "_extension"):
            return
        visible = self._original_mask is not None and not (self._after.isChecked() and self.selected_variant is not None)
        for overlay in (self._extension, self._lasso_outline, self._removal_outline):
            overlay.setVisible(visible)
        self._target_outline.setVisible(visible and self._detailed)
        self._click_marker.setVisible(self._original_mask is None and bool(self._detection_points))

    def set_busy(self, busy: bool, *, generating: bool = False, detecting: bool = False, cancelling: bool = False):
        super().set_busy(busy, generating=generating or detecting, cancelling=cancelling)

    def _refresh(self):
        super()._refresh()
        if not hasattr(self, "_detect"):
            return
        reviewing = self._original_mask is not None
        selected = self._bounds is not None or bool(self._detection_points)
        can_detect = self.can_detect
        show_manual = not reviewing and self._shapes.currentData() == "lasso"
        self._controls.setVisible(selected or reviewing or self._generating)
        self._target_list.setVisible(reviewing and self._detailed and self._target_list.count() > 0)
        self._target_list.setEnabled(not self._busy)
        self._detect.setVisible(not reviewing)
        self._detect.setEnabled(not self._busy and not reviewing and can_detect)
        self._manual.setVisible(show_manual)
        self._manual.setEnabled(not self._busy and show_manual and can_detect)
        self._change_selection.setVisible(reviewing)
        self._change_selection.setEnabled(not self._busy)
        self._refine.setVisible(reviewing and not self._detailed)
        self._refine.setEnabled(not self._busy)
        self._detect_effects.setVisible(reviewing and self._detailed and bool(self._targets))
        self._detect_effects.setEnabled(not self._busy and can_detect)
        self._shapes.setVisible(not reviewing)
        self._mode_buttons["new"].hide()
        for mode in ("add", "subtract"):
            self._mode_buttons[mode].setVisible(reviewing and self._detailed)
        self._canvas.set_drawing_enabled(not self._busy and (not reviewing or self._detailed))
        self._generate.setVisible(reviewing)
        self._generate.setText("Generate another" if self._variants else "Remove")
        if self._busy or self._message:
            return
        variant = self.selected_variant
        if variant is not None:
            message = self._variant_message(variant)
        elif reviewing:
            if not self.can_generate:
                message = "Choose at least one object or add an area to remove."
            elif self._has_extension:
                message = "Review the marked area. Amber extends beyond your lasso."
            else:
                message = "Review the marked area before removing."
        elif selected and not can_detect:
            message = "Select a visible object to remove."
        elif self._shapes.currentData() == "click":
            message = "Click the object you want to remove, or choose Lasso to draw around it."
        else:
            message = "Draw around the objects you want to remove."
        self._status.setText(message)
        self._status.setVisible(bool(message))
