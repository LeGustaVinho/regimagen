"""Review detected objects and their editable removal masks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageChops
from PySide6.QtCore import QPointF, QSignalBlocker, Qt, Signal
from PySide6.QtGui import QColor, QPainterPath, QPen, QPixmap, QTransform
from PySide6.QtWidgets import QApplication, QGraphicsItem, QHBoxLayout, QListWidget, QListWidgetItem, QPushButton

from .canvas import combine_selection, pixmap
from .fill import FillPreview, removal_transition_width, selection_has_visible_pixels
from .fill_editor import FillEditor

if TYPE_CHECKING:
    from .remove_detection import RemovalTarget


_BINARY = [0] + [255] * 255


@dataclass(frozen=True)
class RemovalInputs:
    core: Image.Image
    footprint: Image.Image
    labels: tuple[str, ...]
    objects: tuple[RemovalTarget, ...]
    direct: bool
    warning: str = ""
    excluded: Image.Image | None = None

    def with_detected_objects(self, source: Image.Image, targets: tuple[RemovalTarget, ...]) -> RemovalInputs | None:
        approved = removal_masks(
            source.size, targets, object_limit=self.core, excluded=self.excluded,
            source=source, opaque=source.mode == "RGB",
        )
        if approved is None:
            return None
        objects = {id(target) for target in approved.objects}
        unresolved = ImageChops.invert(approved.core)
        if self.excluded is not None:
            unresolved = ImageChops.subtract(unresolved, self.excluded)
        uncertain = any(
            target.parent is not None and not target.selected and not target.protected
            and id(targets[target.parent]) in objects
            and ImageChops.darker(unresolved.crop(_target_bounds(target)), target.mask).getbbox() is not None
            for target in targets
        )
        warning = "Codex could not confirm every shadow or reflection. Check the background." if uncertain else ""
        return replace(approved, direct=True, warning=warning)


def _target_bounds(target: RemovalTarget) -> tuple[int, int, int, int]:
    left, top = target.origin
    return left, top, left + target.mask.width, top + target.mask.height


def _target_coverage(target: RemovalTarget, area: tuple[int, int, int, int]) -> Image.Image:
    left, top, right, bottom = area
    x, y = target.origin
    return target.mask.crop((left - x, top - y, right - x, bottom - y)).point(_BINARY)


def _union_target(destination: Image.Image, target: RemovalTarget, *, excluding: Image.Image | None = None):
    area = _target_bounds(target)
    mask = target.mask
    if excluding is not None:
        mask = ImageChops.subtract(mask, excluding)
    destination.paste(ImageChops.lighter(destination.crop(area), mask), target.origin)


def _expand_mask(mask: Image.Image, feather: int) -> Image.Image:
    pixels = np.asarray(mask)
    if feather > 12:
        distance = cv2.distanceTransform((pixels == 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5)
        return Image.fromarray((distance <= feather).astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * feather + 1, 2 * feather + 1))
    return Image.fromarray(cv2.dilate(pixels, kernel))


def removal_masks(
    size: tuple[int, int], targets: tuple[RemovalTarget, ...], checked: set[int] | None = None, *,
    manual: Image.Image | None = None, excluded: Image.Image | None = None,
    opaque: bool = True, object_limit: Image.Image | None = None, source: Image.Image | None = None,
) -> RemovalInputs | None:
    """Keep the approved object masks alongside the background edit area."""
    if checked is None:
        checked = {index for index, target in enumerate(targets) if target.selected and not target.protected}
    protected = excluded.copy() if excluded is not None else Image.new("L", size)
    pending_effects = {}
    for index, target in enumerate(targets):
        parent = target.parent
        if target.protected or (parent is None and index not in checked):
            _union_target(protected, target)
        elif parent is None and object_limit is not None:
            # An effect can extend past the lasso. Preserve parts of the object excluded by it.
            _union_target(protected, target, excluding=object_limit.crop(_target_bounds(target)))
        elif parent is not None:
            if parent not in checked or targets[parent].protected:
                _union_target(protected, target)
            elif index in checked:
                pending_effects[index] = target
            elif target.selected or not opaque:
                # An unchecked effect can be preserved outside its caster.
                _union_target(protected, target, excluding=_target_coverage(targets[parent], _target_bounds(target)))
    protected = protected.point(_BINARY)
    core = ImageChops.subtract(manual.point(_BINARY), protected) if manual is not None else Image.new("L", size)
    objects = {}
    for index, target in enumerate(targets):
        if target.protected or target.parent is not None or index not in checked:
            continue
        area = _target_bounds(target)
        mask = ImageChops.subtract(target.mask.point(_BINARY), protected.crop(area))
        visible = (ImageChops.darker(mask, source.crop(area).getchannel("A"))
                   if source is not None and source.mode == "RGBA" else mask)
        if visible.getbbox() is not None:
            _union_target(core, replace(target, mask=mask))
            objects[index] = target

    if not objects and manual is None:
        return None

    effects = {index: target for index, target in pending_effects.items() if target.parent in objects}
    for index, target in pending_effects.items():
        if index not in effects:
            # A fully excluded caster leaves its shadow in the scene.
            _union_target(protected, target, excluding=core.crop(_target_bounds(target)))
    for target in effects.values():
        _union_target(core, target)
    protected = protected.point(_BINARY)
    core = ImageChops.subtract(core.point(_BINARY), protected)
    bounds = core.getbbox()
    if bounds is None:
        return None

    left, top, right, bottom = bounds
    # Keep the smaller analysis neighborhood for transparency removal, where
    # nearby visible components determine whether an object can simply clear.
    feather = (removal_transition_width(core.crop(bounds)) if opaque
               else max(1, min(12, round(min(right - left, bottom - top) * 0.04))))
    area = (max(0, left - feather), max(0, top - feather),
            min(size[0], right + feather), min(size[1], bottom + feather))
    expanded = _expand_mask(core.crop(area), feather)
    footprint = Image.new("L", size)
    blocked = protected.crop(area)
    if object_limit is not None:
        allowed = object_limit.crop(area)
        if effects:
            effect_mask = Image.new("L", allowed.size)
            for target in effects.values():
                x, y = target.origin
                _union_target(effect_mask, replace(target, origin=(x - area[0], y - area[1])))
            effect_mask = ImageChops.darker(effect_mask, core.crop(area))
            allowed = ImageChops.lighter(allowed, _expand_mask(effect_mask, feather))
        blocked = ImageChops.lighter(blocked, ImageChops.invert(allowed))
    if not opaque and source is not None and source.mode == "RGBA":
        # The blend can reach clear pixels. Keep visible pixels outside the approved shapes.
        outside_objects = ImageChops.invert(core.crop(area))
        visible = source.crop(area).getchannel("A").point(_BINARY)
        blocked = ImageChops.lighter(blocked, ImageChops.darker(outside_objects, visible))
    footprint.paste(ImageChops.subtract(expanded, blocked), area[:2])
    labels = []
    for index, target in enumerate(targets):
        if index in objects:
            labels.append(target.label)
        elif index in effects and ImageChops.darker(core.crop(_target_bounds(target)), target.mask).getbbox() is not None:
            labels.append(target.label)
    return RemovalInputs(core, footprint, tuple(labels), tuple(objects.values()), manual is not None, excluded=excluded)


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

    def __init__(self, base: Image.Image, *, mask: Image.Image | None = None):
        super().__init__(base, mask=mask, layered=True)
        self._original_mask: Image.Image | None = None
        self._targets: tuple[RemovalTarget, ...] = ()
        self._manual_mask: Image.Image | None = None
        self._excluded: Image.Image | None = None
        self._inputs: RemovalInputs | None = None
        self._detection_points: tuple[tuple[int, int], ...] = ()
        self._click_start: QPointF | None = None
        self._detailed = False
        self._has_extension = False
        self._background_variants: dict[int, FillPreview] = {}
        self._background = QPushButton("Background")
        self._background.setCheckable(True)
        self._background.clicked.connect(self._show_preview)
        self._compare.addButton(self._background)
        self._review.layout().insertWidget(1, self._background)
        self._prompt.hide()
        self._alpha.hide()
        self._destination.hide()
        self._cancel.setText("Cancel")

        actions = QHBoxLayout()
        self._detect = QPushButton("Select with Codex")
        self._detect.setProperty("primary", True)
        self._detect.clicked.connect(self.detect_requested)
        actions.addWidget(self._detect)
        self._manual = QPushButton("Use Lasso directly")
        self._manual.clicked.connect(self._use_manual)
        actions.addWidget(self._manual)
        self._change_selection = QPushButton("Change selection")
        self._change_selection.clicked.connect(self._reset_detection)
        actions.addWidget(self._change_selection)
        self._refine = QPushButton("Refine selection")
        self._refine.clicked.connect(self._open_refinement)
        actions.addWidget(self._refine)
        self._redetect = QPushButton("Regenerate selection")
        self._redetect.clicked.connect(self.detect_requested)
        actions.addWidget(self._redetect)
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
            self._shapes.addItem("Direct Lasso", "direct-lasso")
            self._shapes.setCurrentIndex(1 if self._bounds else 0)
        self._shapes.currentIndexChanged.connect(self._change_selection_tool)
        self._canvas.set_tool(self._shapes.currentData())
        self._canvas.setToolTip(
            "Click or draw a lasso to ask Codex for a selection. Direct Lasso uses your mark for the cutout "
            "and can repair associated shadows or reflections outside it. Scroll to zoom."
        )
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

    def removal_inputs(self) -> RemovalInputs | None:
        return self._inputs

    def set_targets(self, targets: tuple[RemovalTarget, ...], *, detailed: bool = False):
        if not any(target.parent is None and not target.protected for target in targets):
            if self._original_mask is not None:
                message = "Codex found no replacement selection. You can keep editing the current area."
            elif self._detection_points:
                message = "No objects detected. Click a different spot or use Lasso to mark the object."
            else:
                message = "Codex found no object. Adjust the lasso or use it directly."
            self.set_message(message)
            return
        if self._manual_mask is not None and self._inputs is not None:
            # Keep the adjusted Direct Lasso as the guide when moving to object review.
            self._original_mask = self._inputs.core
            with QSignalBlocker(self._shapes):
                self._shapes.setCurrentIndex(self._shapes.findData("lasso"))
        requires_review = len(targets) > 1 or not targets[0].selected or targets[0].parent is not None
        self._start_review(targets, detailed=detailed or requires_review or self._detailed)

    def _can_detect_lasso(self) -> bool:
        bounds = self._bounds if self._original_mask is None else self._original_mask.getbbox()
        return bounds is not None and (
            self._source.mode == "RGB" or selection_has_visible_pixels(self._source, self.detection_mask, bounds)
        )

    def _change_selection_tool(self):
        self._canvas.set_tool("click" if self._shapes.currentData() == "click" else "lasso")
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
            if targets:
                first_checked = next((i for i, target in enumerate(targets) if target.selected and not target.protected), 0)
                self._target_list.setCurrentRow(first_checked)
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
        self._original_mask = self._manual_mask = self._excluded = None
        self._targets = ()
        self._inputs = None
        self._detailed = False
        self._has_extension = False
        with QSignalBlocker(self._target_list):
            self._target_list.clear()
        self._select_operation("new")
        self._mode_buttons["new"].setChecked(True)
        self._canvas.set_tool("click" if self._shapes.currentData() == "click" else "lasso")
        self._canvas.set_selection(self._mask, self._bounds)
        self._draft_changed()

    def _targets_changed(self, *_):
        items = [self._target_list.item(row) for row in range(self._target_list.count())]
        checked = {
            item.data(Qt.ItemDataRole.UserRole) for item in items
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable and item.checkState() == Qt.CheckState.Checked
        }
        self._inputs = removal_masks(
            self._source.size, self._targets, checked, manual=self._manual_mask, excluded=self._excluded,
            opaque=self._source.mode == "RGB", source=self._source,
        )
        active_objects = {id(target) for target in self._inputs.objects} if self._inputs is not None else set()
        with QSignalBlocker(self._target_list):
            for item in items:
                target = self._targets[item.data(Qt.ItemDataRole.UserRole)]
                enabled = target.parent is None or id(self._targets[target.parent]) in active_objects
                flags = item.flags()
                item.setFlags(flags | Qt.ItemFlag.ItemIsEnabled if enabled else flags & ~Qt.ItemFlag.ItemIsEnabled)
        if self._inputs is not None and self._source.mode == "RGBA":
            core = self._inputs.core
            bounds = core.getbbox()
            if bounds is not None and not selection_has_visible_pixels(self._source, core, bounds):
                self._inputs = None
        self._mask = self._inputs.footprint if self._inputs is not None else Image.new("L", self._source.size)
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
            self._refresh()
            return
        target = self._targets[item.data(Qt.ItemDataRole.UserRole)]
        pen = QPen(QColor("#ffe6a7" if target.protected else "#edf1f7"), 1.5, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self._target_outline.setPen(pen)
        self._target_outline.setPath(_outline(target.mask, target.origin))
        self._refresh()

    def _focused_checked_target(self) -> int | None:
        item = self._target_list.currentItem()
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable
        if item is not None and item.flags() & flags == flags and item.checkState() == Qt.CheckState.Checked:
            return item.data(Qt.ItemDataRole.UserRole)
        return None

    def _apply_lasso(self, path: QPainterPath):
        if self._original_mask is None:
            previous = self._mask
            super()._apply_lasso(path)
            if self._mask is not previous and self.can_detect:
                if self._shapes.currentData() == "direct-lasso":
                    self._use_manual()
                else:
                    self.detect_requested.emit()
            return
        if not self._detailed or self._busy:
            return
        target_index = self._focused_checked_target()
        if self._operation == "add" and self._manual_mask is None and target_index is None:
            return
        stroke = combine_selection(None, path, self._source.size, "new")
        if stroke is None:
            return
        if self._operation == "subtract":
            self._excluded = stroke if self._excluded is None else ImageChops.lighter(self._excluded, stroke)
        elif self._manual_mask is not None:
            self._manual_mask = ImageChops.lighter(self._manual_mask, stroke)
        else:
            assert target_index is not None
            target = self._targets[target_index]
            x, y = target.origin
            left, top, right, bottom = stroke.getbbox()
            area = min(left, x), min(top, y), max(right, x + target.mask.width), max(bottom, y + target.mask.height)
            extended = stroke.crop(area)
            _union_target(extended, replace(target, origin=(x - area[0], y - area[1])))
            updated = replace(target, mask=extended, origin=area[:2])
            self._targets = self._targets[:target_index] + (updated,) + self._targets[target_index + 1:]
        self._targets_changed()

    def add_removal_variant(self, after: FillPreview, background: FillPreview):
        self._background_variants[len(self._variants)] = background
        self.add_variant(after)

    def clear_variants(self):
        self._background_variants.clear()
        super().clear_variants()

    def _select_generated_view(self):
        if self.selected_variant in self._background_variants:
            self._background.setChecked(True)
        else:
            super()._select_generated_view()

    def _show_preview(self):
        index = self.selected_variant
        background = self._background_variants.get(index) if index is not None else None
        if background is not None and self._background.isChecked():
            self._display_preview(background)
        else:
            super()._show_preview()
        if not hasattr(self, "_extension"):
            return
        visible = self._original_mask is not None and (self._before.isChecked() or index is None)
        for overlay in (self._extension, self._lasso_outline, self._removal_outline):
            overlay.setVisible(visible)
        self._target_outline.setVisible(visible and self._detailed)
        self._click_marker.setVisible(self._original_mask is None and bool(self._detection_points))

    def set_busy(self, busy: bool, *, generating: bool = False, detecting: bool = False, cancelling: bool = False):
        super().set_busy(busy, generating=generating or detecting, cancelling=cancelling)

    def _select_operation(self, operation: str):
        super()._select_operation(operation)
        self._refresh()

    def _refresh(self):
        super()._refresh()
        if not hasattr(self, "_detect"):
            return
        reviewing = self._original_mask is not None
        selected = self._bounds is not None or bool(self._detection_points)
        can_detect = self.can_detect
        tool = self._shapes.currentData()
        show_detect = not reviewing and tool != "direct-lasso"
        show_manual = tool == "lasso" and self._manual_mask is None
        self._controls.setVisible(selected or reviewing or self._generating)
        self._target_list.setVisible(reviewing and self._detailed and self._target_list.count() > 0)
        self._target_list.setEnabled(not self._busy)
        self._detect.setVisible(show_detect)
        self._detect.setEnabled(not self._busy and show_detect and can_detect)
        self._manual.setVisible(show_manual)
        self._manual.setEnabled(not self._busy and show_manual and self._can_detect_lasso())
        self._change_selection.setVisible(reviewing)
        self._change_selection.setEnabled(not self._busy)
        self._refine.setVisible(reviewing and not self._detailed)
        self._refine.setEnabled(not self._busy)
        direct = self._manual_mask is not None
        self._redetect.setText("Review with Lasso" if direct else "Regenerate selection")
        self._redetect.setVisible(reviewing and self._detailed and (direct or bool(self._targets)))
        self._redetect.setEnabled(not self._busy and (self.can_generate if direct else can_detect))
        self._shapes.setVisible(not reviewing)
        self._mode_buttons["new"].hide()
        for mode in ("add", "subtract"):
            self._mode_buttons[mode].setVisible(reviewing and self._detailed)
        self._mode_buttons["add"].setEnabled(not self._busy and (direct or self._focused_checked_target() is not None))
        can_refine = self._detailed and (self._operation == "subtract" or self._mode_buttons["add"].isEnabled())
        self._canvas.set_drawing_enabled(not self._busy and (not reviewing or can_refine))
        self._background.setEnabled(self.selected_variant in self._background_variants)
        self._generate.setVisible(reviewing)
        self._generate.setText("Generate another" if self._variants else "Remove")
        if self._busy or self._message:
            return
        variant = self.selected_variant
        if variant is not None:
            message = self._variants[variant].warning or "Review the background. Apply keeps each object visible on its own layer."
        elif reviewing:
            if not self.can_generate:
                message = "Choose at least one object or add an area to remove."
            elif self._has_extension:
                message = "Review the marked area. Amber extends beyond your lasso."
            else:
                message = "Review the marked area before removing."
        elif selected and not can_detect:
            message = "Select a visible object to remove."
        elif tool == "click":
            message = "Click an object to have Codex trace its selection."
        elif tool == "direct-lasso":
            message = "Draw around the objects you want to remove."
        else:
            message = "Draw around the objects you want Codex to trace."
        self._status.setText(message)
        self._status.setVisible(bool(message))
