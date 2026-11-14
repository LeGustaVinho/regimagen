"""Side-by-side viewer with synchronized navigation and full-resolution compositing."""

import math
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, QThreadPool, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QTransform
from PySide6.QtWidgets import (
    QFrame, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QPushButton, QSplitter,
    QVBoxLayout, QWidget,
)

from .composite import Compositor
from .geometry import Cell, Layout, Rect
from .storage import Snapshot


def qimage(pixels: np.ndarray) -> QImage:
    contiguous = np.ascontiguousarray(pixels)
    height, width, channels = contiguous.shape
    pixel_format = QImage.Format.Format_RGBA8888 if channels == 4 else QImage.Format.Format_RGB888
    return QImage(
        contiguous.data, width, height, contiguous.strides[0], pixel_format,
    ).copy()


class ImageView(QGraphicsView):
    moved = Signal()
    resized = Signal()

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setBackgroundBrush(QColor("#11131a"))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setAcceptDrops(False)

    def center(self) -> QPointF:
        return self.mapToScene(self.viewport().rect().center())

    def camera(self, scale: float, center: QPointF):
        self.setTransform(QTransform.fromScale(scale, scale))
        self.centerOn(center)

    def fit_scale(self) -> float:
        rect = self.sceneRect()
        if rect.isEmpty():
            return 1.0
        return min(
            max(1, self.viewport().width() - 24) / rect.width(),
            max(1, self.viewport().height() - 24) / rect.height(),
        )

    def zoom(self, factor: float, position=None):
        if self.sceneRect().isEmpty():
            return
        pos = position if position is not None else self.viewport().rect().center()
        anchor = self.mapToScene(pos)
        scale = max(self.minimum_scale(), min(8.0, self.transform().m11() * factor))
        self.setTransform(QTransform.fromScale(scale, scale))
        self.centerOn(self.center() + anchor - self.mapToScene(pos))
        self.moved.emit()

    def minimum_scale(self):
        return min(0.1, self.fit_scale())

    def wheelEvent(self, event):
        self.zoom(1.2 ** (event.angleDelta().y() / 120), event.position().toPoint())
        event.accept()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.moved.emit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()


class _DetailSignals(QObject):
    finished = Signal(int, object, object)


class _DetailTask(QRunnable):
    def __init__(
        self, token: int, layout: Layout, artifacts: dict[Cell, Path], area: Rect,
        alpha: Image.Image | None,
    ):
        super().__init__()
        self.token, self.layout, self.artifacts, self.area = token, layout, artifacts, area
        self.alpha = alpha
        self.signals = _DetailSignals()

    def run(self):
        rendered = QImage()
        try:
            compositor = Compositor(self.layout, self.artifacts, cache_size=8, alpha=self.alpha)
            rendered = qimage(compositor.render(self.area))
        except (OSError, ValueError):
            # The job may have replaced a tile while the preview was rendering.
            pass
        self.signals.finished.emit(self.token, self.area, rendered)


class Comparison(QWidget):
    def __init__(self):
        super().__init__()
        self.image_layout: Layout | None = None
        self.alpha: Image.Image | None = None
        self.artifacts: dict[Cell, Path] = {}
        self.thumbnails = {}
        self.highlights = []
        self.selected: Cell | None = None
        self.detail = None
        self.fitted = True
        self.detail_token = 0
        self.detail_running = False
        self.detail_pool = QThreadPool(self)
        self.detail_pool.setMaxThreadCount(1)
        self.detail_timer = QTimer(self)
        self.detail_timer.setSingleShot(True)
        self.detail_timer.timeout.connect(self._render_detail)
        self.fit_timer = QTimer(self)
        self.fit_timer.setSingleShot(True)
        self.fit_timer.timeout.connect(self.fit)

        self.before, self.after = ImageView(), ImageView()
        self.before.moved.connect(lambda: self._sync(self.before, self.after))
        self.after.moved.connect(lambda: self._sync(self.after, self.before))
        self.before.resized.connect(self._on_resize)
        self.after.resized.connect(self._on_resize)

        content = QVBoxLayout(self)
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(12)
        controls = QHBoxLayout()
        controls.addStretch()
        minus = QPushButton("−")
        minus.setToolTip("Zoom out")
        plus = QPushButton("+")
        plus.setToolTip("Zoom in")
        self.percent = QPushButton("100%")
        self.percent.setToolTip("Show pixels at actual size")
        fit = QPushButton("Fit to window")
        for control in (minus, plus, self.percent, fit):
            controls.addWidget(control)
        minus.clicked.connect(lambda: self.before.zoom(1 / 1.2))
        plus.clicked.connect(lambda: self.before.zoom(1.2))
        self.percent.clicked.connect(self.actual_size)
        fit.clicked.connect(self.fit)
        content.addLayout(controls)
        panels = QSplitter(Qt.Orientation.Horizontal)
        panels.setChildrenCollapsible(False)
        self.before_title = QLabel("Original")
        self.fill_button = QPushButton("Generative Fill…")
        self.fill_button.setToolTip("Select and edit an area of the source image at its original size.")
        panels.addWidget(self._panel(self.before_title, self.before, self.fill_button))
        panels.addWidget(self._panel(QLabel("Reconstruction"), self.after))
        panels.setSizes([500, 500])
        content.addWidget(panels, 1)

    @staticmethod
    def _panel(title: QLabel, view: ImageView, action: QPushButton | None = None) -> QWidget:
        panel = QFrame()
        panel.setObjectName("imagePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        title.setObjectName("imageTitle")
        header = QHBoxLayout()
        header.addWidget(title, 1)
        if action is not None:
            header.addWidget(action)
        layout.addLayout(header)
        layout.addWidget(view, 1)
        return panel

    def set_source(self, layout: Layout, original: QPixmap, alpha: Image.Image | None = None):
        self.detail_token += 1
        self.detail = None
        self.thumbnails.clear()
        self.highlights.clear()
        self.selected = None
        self.artifacts = {}
        self.image_layout = layout
        self.alpha = alpha
        self.before_title.setText(f"Input · {layout.input_scale}×" if layout.input_scale > 1 else "Original")
        for view in (self.before, self.after):
            view.scene().clear()
            view.setSceneRect(0, 0, *layout.size)
        item = self.before.scene().addPixmap(original)
        item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        item.setTransform(QTransform.fromScale(
            layout.size[0] / max(1, original.width()), layout.size[1] / max(1, original.height()),
        ))
        self.fit()
        self.fit_timer.start(0)

    def update_tiles(self, snapshot: Snapshot):
        if self.image_layout is None:
            return
        self.artifacts = snapshot.artifacts
        for cell, tile in snapshot.tiles.items():
            previous = self.thumbnails.get(cell)
            revision = None
            if tile.thumbnail:
                try:
                    revision = tile.thumbnail, tile.thumbnail.stat().st_mtime_ns
                except OSError:
                    pass
            if previous is not None and previous[0] == revision:
                continue
            if previous is not None:
                self.after.scene().removeItem(previous[1])
                del self.thumbnails[cell]
            if revision is None:
                continue
            pixmap = QPixmap(str(tile.thumbnail))
            if pixmap.isNull():
                continue
            core = self.image_layout.core(cell)
            item = self.after.scene().addPixmap(pixmap)
            item.setPos(core.left, core.top)
            item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
            item.setTransform(QTransform.fromScale(core.width / pixmap.width(), core.height / pixmap.height()))
            self.thumbnails[cell] = (revision, item)
        if self.detail is not None:
            self.after.scene().removeItem(self.detail)
            self.detail = None
        self._queue_detail()

    def select(self, cell: Cell | None, *, focus: bool = False):
        self.selected = cell
        for item in self.highlights:
            item.scene().removeItem(item)
        self.highlights.clear()
        if cell is None or self.image_layout is None:
            return
        core = self.image_layout.core(cell)
        rect = QRectF(core.left, core.top, core.width, core.height)
        pen = QPen(QColor("#75d8be"))
        pen.setCosmetic(True)
        pen.setWidth(2)
        for view in (self.before, self.after):
            item = view.scene().addRect(rect, pen)
            item.setZValue(4)
            self.highlights.append(item)
        if focus:
            self.fit_timer.stop()
            self.fitted = False
            scale = self.before.transform().m11()
            for view in (self.before, self.after):
                view.camera(scale, rect.center())
            self._queue_detail()

    def fit(self):
        if self.image_layout is None:
            return
        self.fitted = True
        scale = min(self.before.fit_scale(), self.after.fit_scale())
        self._camera(scale, self.before.sceneRect().center())

    def actual_size(self):
        if self.image_layout is not None:
            self.fit_timer.stop()
            self.fitted = False
            center = self.before.center()
            if self.selected is not None:
                core = self.image_layout.core(self.selected)
                center = QPointF((core.left + core.right) / 2, (core.top + core.bottom) / 2)
            self._camera(1 / self.before.devicePixelRatioF(), center)

    def _camera(self, scale: float, center: QPointF):
        for view in (self.before, self.after):
            view.camera(scale, center)
        self._show_percent()
        self._queue_detail()

    def _sync(self, source: ImageView, other: ImageView):
        self.fit_timer.stop()
        self.fitted = False
        other.camera(source.transform().m11(), source.center())
        self._show_percent()
        self._queue_detail()

    def _show_percent(self):
        factor = self.before.transform().m11() * self.before.devicePixelRatioF()
        self.percent.setText(f"{max(1, round(100 * factor))}%")

    def _on_resize(self):
        if self.fitted:
            self.fit_timer.start(0)
        else:
            self._queue_detail()

    def _queue_detail(self):
        self.detail_token += 1
        self.detail_timer.start(100)

    def _render_detail(self):
        if self.detail_running or self.image_layout is None or not self.artifacts:
            return
        scene = self.after.mapToScene(self.after.viewport().rect()).boundingRect()
        area = Rect(
            math.floor(scene.left()), math.floor(scene.top()),
            math.ceil(scene.right()), math.ceil(scene.bottom()),
        ).intersect(self.image_layout.bounds)
        if area is None or area.width * area.height > 2_500_000:
            return
        self.detail_running = True
        task = _DetailTask(self.detail_token, self.image_layout, self.artifacts, area, self.alpha)
        task.signals.finished.connect(self._show_detail)
        self.detail_pool.start(task)

    def _show_detail(self, token: int, area: Rect, image: QImage):
        self.detail_running = False
        if token != self.detail_token:
            self.detail_timer.start(0)
            return
        if image.isNull():
            return
        if self.detail is not None:
            self.after.scene().removeItem(self.detail)
        self.detail = self.after.scene().addPixmap(QPixmap.fromImage(image))
        self.detail.setPos(area.left, area.top)
        self.detail.setZValue(2)
