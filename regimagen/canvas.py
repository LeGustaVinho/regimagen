"""Canvas navigation, pointer gestures, and selection overlays."""

import math

import numpy as np
from PIL import Image, ImageChops
from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QTransform

from .viewer import ImageView, qimage

SELECTION_TOOLS = ("rectangle", "lasso")


def pixmap(image: Image.Image) -> QPixmap:
    return QPixmap.fromImage(qimage(np.asarray(image)), Qt.ImageConversionFlag.NoOpaqueDetection)


def combine_selection(mask: Image.Image | None, path: QPainterPath, size: tuple[int, int], mode: str):
    bounds = path.boundingRect().intersected(QRectF(0, 0, *size))
    if bounds.isEmpty():
        return mask
    left, top = math.floor(bounds.left()), math.floor(bounds.top())
    width = math.ceil(bounds.right()) - left
    height = math.ceil(bounds.bottom()) - top
    raster = QImage(width, height, QImage.Format.Format_RGBA8888_Premultiplied)
    raster.fill(Qt.GlobalColor.transparent)
    painter = QPainter(raster)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.translate(-left, -top)
    painter.fillPath(path, Qt.GlobalColor.white)
    painter.end()
    rows = np.frombuffer(raster.constBits(), dtype=np.uint8).reshape(height, raster.bytesPerLine())
    stroke = Image.fromarray(rows[:, 3:width * 4:4].copy())
    if stroke.getbbox() is None or (mask is None and mode == "subtract"):
        return mask

    if mode == "new" or mask is None:
        result = Image.new("L", size)
        result.paste(stroke, (left, top))
        if mask is not None and ImageChops.difference(result, mask).getbbox() is None:
            return mask
        return result

    old = mask.crop((left, top, left + width, top + height))
    combined = ImageChops.lighter(old, stroke) if mode == "add" else ImageChops.multiply(old, ImageChops.invert(stroke))
    if ImageChops.difference(old, combined).getbbox() is None:
        return mask
    result = mask.copy()
    result.paste(combined, (left, top))
    return result


def selection_overlay(mask: Image.Image, bounds: tuple[int, int, int, int]) -> QImage:
    cropped = mask.crop(bounds)
    cropped.thumbnail((4096, 4096), Image.Resampling.BILINEAR)
    coverage = np.asarray(cropped)
    overlay = np.empty((*coverage.shape, 4), dtype=np.uint8)
    overlay[..., :3] = (117, 216, 190)
    overlay[..., 3] = (coverage.astype(np.uint16) * 82 // 255).astype(np.uint8)
    return qimage(overlay).convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)


class CanvasView(ImageView):
    stroke_started = Signal()
    stroke_finished = Signal(QPainterPath)
    pointer_started = Signal(QPointF)
    pointer_moved = Signal(QPointF)
    pointer_finished = Signal(QPointF)
    gesture_canceled = Signal()

    def __init__(self, base: QPixmap | None = None):
        super().__init__()
        self.setDragMode(self.DragMode.NoDrag)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._image = self.scene().addPixmap(QPixmap())
        self._image.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._image.setShapeMode(self._image.ShapeMode.BoundingRectShape)
        self._selection = self.scene().addPixmap(QPixmap())
        self._selection.setZValue(1)
        self._selection.setShapeMode(self._selection.ShapeMode.BoundingRectShape)
        self._stroke_item = self.scene().addPath(QPainterPath())
        self._stroke_item.setZValue(2)
        self._outline = QPen(QColor("#d0fff2"), 1, Qt.PenStyle.DashLine)
        self._outline.setCosmetic(True)
        self._brush_pen = self._outline
        self._checker = self._checker_texture()
        self._start: QPointF | None = None
        self._stroke: QPainterPath | None = None
        self._stroke_timer = QTimer(self, singleShot=True, interval=16, timerType=Qt.TimerType.PreciseTimer)
        self._stroke_timer.timeout.connect(self._show_stroke)
        self._pan_position = None
        self._pan_button = Qt.MouseButton.NoButton
        self._space = False
        self._drawing_enabled = True
        self._fitted = True
        self._tool = "lasso"
        self.resized.connect(self._fit_if_needed)
        self._update_cursor()
        if base is not None:
            self.set_size((base.width(), base.height()))
            self.set_image(base)

    @staticmethod
    def _checker_texture() -> QPixmap:
        texture = QPixmap(20, 20)
        texture.fill(QColor("#53565f"))
        painter = QPainter(texture)
        painter.fillRect(0, 0, 10, 10, QColor("#70747d"))
        painter.fillRect(10, 10, 10, 10, QColor("#70747d"))
        painter.end()
        return texture

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)
        brush = QBrush(self._checker)
        scale = self.transform().m11()
        brush.setTransform(QTransform.fromScale(1 / scale, 1 / scale))
        painter.fillRect(rect.intersected(self.sceneRect()), brush)

    def set_size(self, size: tuple[int, int]):
        self.setSceneRect(0, 0, *size)
        self.set_image(QPixmap())
        self.fit()
        QTimer.singleShot(0, self._fit_if_needed)

    def set_image(self, image: QPixmap, *, selection_visible=True, rect: QRectF | None = None):
        # One pixmap leaves the checkerboard visible through real transparency.
        self._image.setPixmap(image)
        self._image.setPos(rect.topLeft() if rect is not None else QPointF())
        transform = QTransform()
        if rect is not None and not image.isNull():
            transform.scale(rect.width() / image.width(), rect.height() / image.height())
        self._image.setTransform(transform)
        self._selection.setVisible(selection_visible)

    def set_selection(self, mask: Image.Image | None, bounds: tuple[int, int, int, int] | None):
        self.set_selection_overlay(selection_overlay(mask, bounds) if bounds is not None else None, bounds)

    def set_selection_overlay(self, overlay: QImage | None, bounds: tuple[int, int, int, int] | None):
        if bounds is None:
            self._selection.setPixmap(QPixmap())
            return
        left, top, right, bottom = bounds
        self._selection.setPixmap(QPixmap.fromImage(overlay, Qt.ImageConversionFlag.NoOpaqueDetection))
        self._selection.setPos(left, top)
        self._selection.setTransform(QTransform.fromScale(
            (right - left) / overlay.width(), (bottom - top) / overlay.height(),
        ))

    def set_tool(self, tool: str):
        if self._tool != tool:
            self.cancel_gesture()
            self._tool = tool
        self._update_cursor()

    def set_brush_preview(self, color: QColor, size: int):
        self._brush_pen = QPen(color, size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)

    def set_drawing_enabled(self, enabled: bool):
        self._drawing_enabled = enabled
        if not enabled:
            self.cancel_gesture()
        self._update_cursor()

    def fit(self):
        self._fitted = True
        self.camera(self.fit_scale(), self.sceneRect().center())
        self.moved.emit()

    def actual_size(self):
        self._fitted = False
        self.camera(1 / self.devicePixelRatioF(), self.center())
        self.moved.emit()

    def zoom(self, factor: float, position=None):
        self._fitted = False
        super().zoom(factor, position)

    def _fit_if_needed(self):
        if self._fitted:
            self.fit()

    def _update_cursor(self):
        if self._pan_position is not None:
            cursor = Qt.CursorShape.ClosedHandCursor
        elif self._space or self._tool == "pan":
            cursor = Qt.CursorShape.OpenHandCursor
        elif self._drawing_enabled:
            cursor = Qt.CursorShape.SizeAllCursor if self._tool == "move" else Qt.CursorShape.CrossCursor
        else:
            cursor = Qt.CursorShape.ArrowCursor
        self.viewport().setCursor(cursor)

    def _scene_point(self, position: QPointF) -> QPointF:
        inverse, _ = self.viewportTransform().inverted()
        return inverse.map(position)

    def _clear_gesture(self):
        self._start = self._stroke = None
        self._stroke_timer.stop()
        if not self._stroke_item.path().isEmpty():
            self._stroke_item.setPath(QPainterPath())

    def cancel_gesture(self):
        if self._start is not None:
            self._clear_gesture()
            self.gesture_canceled.emit()

    def mousePressEvent(self, event):
        button = event.button()
        if button == Qt.MouseButton.MiddleButton or (
            (self._space or self._tool == "pan") and button == Qt.MouseButton.LeftButton
        ):
            self.cancel_gesture()
            self._fitted = False
            self._pan_position = event.position().toPoint()
            self._pan_button = button
            self._update_cursor()
            event.accept()
            return
        point = self._scene_point(event.position())
        if self._drawing_enabled and button == Qt.MouseButton.LeftButton and self.sceneRect().contains(point):
            self.setFocus()
            self._start = point
            self._stroke = QPainterPath(point) if self._tool != "move" else None
            if self._tool in SELECTION_TOOLS:
                self._stroke_item.setPen(self._outline)
                self.stroke_started.emit()
            else:
                self._stroke_item.setPen(self._brush_pen)
                self.pointer_started.emit(point)
            event.accept()
            return
        super().mousePressEvent(event)

    def _update_stroke(self, point: QPointF):
        if self._stroke is None:
            return
        if self._tool == "rectangle":
            self._stroke = QPainterPath()
            self._stroke.addRect(QRectF(self._start, point).normalized())
        else:
            self._stroke.lineTo(point)
        if not self._stroke_timer.isActive():
            self._stroke_timer.start()

    def _show_stroke(self):
        if self._stroke is not None:
            self._stroke_item.setPath(self._stroke)

    def mouseMoveEvent(self, event):
        if self._pan_position is not None:
            position = event.position().toPoint()
            delta = position - self._pan_position
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._pan_position = position
            self.moved.emit()
            event.accept()
            return
        if self._start is not None:
            point = self._scene_point(event.position())
            self._update_stroke(point)
            if self._tool not in SELECTION_TOOLS:
                self.pointer_moved.emit(point)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._pan_position is not None and event.button() == self._pan_button:
            self._pan_position = None
            self._update_cursor()
            event.accept()
            return
        if self._start is not None and event.button() == Qt.MouseButton.LeftButton:
            point = self._scene_point(event.position())
            self._update_stroke(point)
            path = self._stroke
            self._clear_gesture()
            if self._tool in SELECTION_TOOLS:
                path.closeSubpath()
                if path.elementCount() > 3:
                    self.stroke_finished.emit(path)
            else:
                self.pointer_finished.emit(point)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self._space = True
            self._update_cursor()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and self._start is not None:
            self.cancel_gesture()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space = False
            self._update_cursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        self._space = False
        self._pan_position = None
        self.cancel_gesture()
        self._update_cursor()
        super().focusOutEvent(event)
