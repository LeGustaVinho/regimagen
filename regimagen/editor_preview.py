"""Viewport previews for the layered editor."""

import math
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from os import process_cpu_count
from time import monotonic
from weakref import ref

import numpy as np
from PIL import Image, ImageChops
from PySide6.QtCore import QObject, QRectF, QThreadPool, QTimer, Qt
from PySide6.QtGui import QImage, QPainter, QPixmap

from .codex import Cancelled
from .document import RASTER_TILE_SIZE, Document, Layer, _adjust, box, render
from .editor_tasks import EditorTask
from .geometry import Rect
from .viewer import qimage

PREVIEW_PIXELS = 1_000_000
PREVIEW_CACHE_BYTES = 128 * 1024 * 1024
_EXACT_STRIPE_PIXELS = 2_000_000
_FORMAT = QImage.Format.Format_ARGB32_Premultiplied
_MODE = QPainter.CompositionMode


@dataclass(frozen=True)
class PreviewRegion:
    area: Rect
    size: tuple[int, int]
    step: int = 1


def interactive_region(area: Rect, scale: float) -> PreviewRegion:
    step = 1
    while True:
        size = tuple((length + step - 1) // step for length in (area.width, area.height))
        if 1 / step <= min(1, scale) and math.prod(size) <= PREVIEW_PIXELS:
            return PreviewRegion(area, size, step)
        step *= 2


def exact_preview(document: Document, region: PreviewRegion, cancelled):
    area, size = region.area, region.size
    unscaled = (area.width, area.height) == size
    if unscaled and math.prod(size) <= _EXACT_STRIPE_PIXELS:
        return qimage(np.asarray(render(document, area, cancelled=cancelled)))

    # Preserve the Lanczos support at the edges of each resampled stripe.
    ratio = area.height / size[1]
    padding = 0 if unscaled else math.ceil(3 * max(1, ratio))
    step = max(1, int(_EXACT_STRIPE_PIXELS / (area.width * ratio)))
    tops = range(0, size[1], step)

    def render_stripe(top):
        if cancelled():
            raise Cancelled("Preview superseded.")
        bottom = min(top + step, size[1])
        first = area.top + top * ratio
        last = area.top + bottom * ratio
        stripe = Rect(
            area.left, max(area.top, math.floor(first) - padding),
            area.right, min(area.bottom, math.ceil(last) + padding),
        )
        pixels = render(document, stripe, cancelled=cancelled)
        if cancelled():
            raise Cancelled("Preview superseded.")
        if unscaled:
            return top, pixels
        return top, pixels.resize(
            (size[0], bottom - top), Image.Resampling.LANCZOS,
            box=(0, first - stripe.top, area.width, min(stripe.height, last - stripe.top)),
        )

    result = Image.new("RGBA", size)

    def paste_stripes(stripes):
        for top, pixels in stripes:
            if cancelled():
                raise Cancelled("Preview superseded.")
            result.paste(pixels, (0, top))

    workers = min(5, len(tops), process_cpu_count() or 1)
    if workers == 1:
        paste_stripes(map(render_stripe, tops))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            paste_stripes(pool.map(render_stripe, tops, buffersize=workers))
    return qimage(np.asarray(result))


def _surface(width, height):
    image = QImage(width, height, _FORMAT)
    if image.isNull():
        raise MemoryError("Could not allocate the editor preview.")
    image.fill(Qt.GlobalColor.transparent)
    return image


def _premultiplied(pixels: Image.Image):
    return qimage(np.asarray(pixels)).convertToFormat(_FORMAT)


@dataclass(frozen=True)
class _Tile:
    pixels: QImage
    coverage: QImage | None = None

    @property
    def bytes(self):
        return self.pixels.sizeInBytes() + (self.coverage.sizeInBytes() if self.coverage is not None else 0)


class PreviewTiles:
    """Worker-owned cache of premultiplied tiles, independent of layer position and opacity."""

    def __init__(self, max_bytes=PREVIEW_CACHE_BYTES):
        self.max_bytes = max_bytes
        self.bytes = 0
        self._tiles = OrderedDict()

    def tile(self, layer: Layer, step: int, x: int, y: int, cancelled, *, store=True):
        if cancelled():
            raise Cancelled("Preview superseded.")
        assets = tuple(asset for asset in (layer.image, layer.mask, layer.coverage) if asset is not None)
        key = id(layer.image), id(layer.mask), id(layer.coverage), step, x, y, layer.adjustments
        cached = self._tiles.get(key)
        if cached is not None:
            references, tile = cached
            if all(reference() is asset for reference, asset in zip(references, assets)):
                self._tiles.move_to_end(key)
                return tile
            self.bytes -= self._tiles.pop(key)[1].bytes

        if any(layer.adjustments):
            base = self.tile(replace(layer, adjustments=(0, 0, 0)), step, x, y, cancelled)
            rgba = base.pixels.convertToFormat(QImage.Format.Format_RGBA8888)
            pixels = Image.frombytes("RGBA", (rgba.width(), rgba.height()), rgba.constBits())
            tile = _Tile(_premultiplied(_adjust(pixels, layer.adjustments)), base.coverage)
        else:
            tile = self._prepare(layer, step, x, y, cancelled)

        if store and tile.bytes <= self.max_bytes:
            while self.bytes + tile.bytes > self.max_bytes:
                self.bytes -= self._tiles.popitem(last=False)[1][1].bytes
            self._tiles[key] = tuple(ref(asset) for asset in assets), tile
            self.bytes += tile.bytes
        return tile

    def _prepare(self, layer, step, x, y, cancelled):
        width = min(RASTER_TILE_SIZE * step, layer.image.width - x)
        height = min(RASTER_TILE_SIZE * step, layer.image.height - y)
        size = (width + step - 1) // step, (height + step - 1) // step
        # Bound the source reads even at a very coarse zoom. Intermediate tiles
        # from the recursion do not evict the cache for the displayed levels.
        if width * height <= 1_048_576:
            region = x, y, x + width, y + height
            pixels = layer.image.crop(region)
            mask = layer.mask.crop(region) if layer.mask is not None else None
            coverage = None
            if layer.coverage is not None:
                weight = layer.coverage.crop(region)
                if mask is not None:
                    weight = ImageChops.multiply(weight, mask)
                alpha = pixels.getchannel("A")
                # The replacement may erase the lower layer even with zero source alpha.
                pixels.putalpha(ImageChops.multiply(alpha, weight))
                coverage = Image.new("RGBA", pixels.size)
                coverage.putalpha(weight)
            elif mask is not None:
                pixels.putalpha(ImageChops.multiply(pixels.getchannel("A"), mask))
            tile = _Tile(_premultiplied(pixels), _premultiplied(coverage) if coverage is not None else None)
        else:
            half_step = step // 2
            span = RASTER_TILE_SIZE * half_step
            width_at_half = (width + half_step - 1) // half_step
            height_at_half = (height + half_step - 1) // half_step
            pixels = _surface(width_at_half, height_at_half)
            coverage = _surface(width_at_half, height_at_half) if layer.coverage is not None else None
            with QPainter(pixels) as image_painter, QPainter() as weight_painter:
                if coverage is not None:
                    weight_painter.begin(coverage)
                for top in range(y, y + height, span):
                    for left in range(x, x + width, span):
                        child = self.tile(layer, half_step, left, top, cancelled, store=False)
                        col, row = (left - x) // half_step, (top - y) // half_step
                        image_painter.drawImage(col, row, child.pixels)
                        if coverage is not None:
                            weight_painter.drawImage(col, row, child.coverage)
            tile = _Tile(pixels, coverage)

        if step == 1:
            return tile
        return _Tile(
            tile.pixels.scaled(*size, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation),
            tile.coverage.scaled(*size, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
            if tile.coverage is not None else None,
        )


def _at_opacity(image, opacity):
    if opacity == 100:
        return image
    result = _surface(image.width(), image.height())
    with QPainter(result) as painter:
        painter.setOpacity(opacity / 100)
        painter.drawImage(0, 0, image)
    return result


def interactive_preview(document, region, tiles, cancelled, draft=None):
    area, step = region.area, region.step
    image = _surface(*region.size)
    span = RASTER_TILE_SIZE * step
    with QPainter(image) as painter:
        painter.scale(region.size[0] / area.width, region.size[1] / area.height)
        painter.translate(-area.left, -area.top)
        for layer in document.layers:
            if draft is not None and layer.id == draft[0]:
                layer = replace(layer, **draft[1])
            if not layer.visible or not layer.opacity:
                continue
            bounds = layer.bounds
            overlap = area.intersect(bounds)
            if overlap is None:
                continue
            local = overlap.relative_to(bounds)
            for y in range(local.top // span * span, local.bottom, span):
                for x in range(local.left // span * span, local.right, span):
                    tile = tiles.tile(layer, step, x, y, cancelled)
                    destination = QRectF(
                        layer.x + x, layer.y + y, min(span, bounds.width - x), min(span, bounds.height - y),
                    )
                    if tile.coverage is None:
                        painter.setCompositionMode(_MODE.CompositionMode_SourceOver)
                        painter.setOpacity(layer.opacity / 100)
                        painter.drawImage(destination, tile.pixels)
                    else:
                        # Scale both inputs before Plus, which otherwise clips the
                        # unscaled sum before applying the painter's opacity.
                        painter.setOpacity(1)
                        painter.setCompositionMode(_MODE.CompositionMode_DestinationOut)
                        painter.drawImage(destination, _at_opacity(tile.coverage, layer.opacity))
                        painter.setCompositionMode(_MODE.CompositionMode_Plus)
                        painter.drawImage(destination, _at_opacity(tile.pixels, layer.opacity))
    return image


def _same_composition(first: Document, second: Document):
    return first.size == second.size and len(first.layers) == len(second.layers) and all(
        a is b or (
            a.image is b.image and a.mask is b.mask and a.coverage is b.coverage
            and (a.x, a.y, a.opacity, a.visible, a.adjustments) == (b.x, b.y, b.opacity, b.visible, b.adjustments)
        )
        for a, b in zip(first.layers, second.layers)
    )


@dataclass(frozen=True)
class _Request:
    epoch: int
    region: PreviewRegion
    overview: bool = False


@dataclass(frozen=True)
class _Frame:
    area: Rect
    pixmap: QPixmap
    exact_size: tuple[int, int] | None = None


class EditorPreview(QObject):
    def __init__(self, canvas, show_error, parent):
        super().__init__(parent)
        self.canvas = canvas
        self.show_error = show_error
        self.document = self.shown = self.draft = None
        self._tiles = PreviewTiles()
        self._epoch = 0
        self._gesturing = False
        self._displayed = self._overview = self._before_draft = None
        self._interactive = self._exact = None
        self._interactive_due = self._exact_due = False
        self._last_frame_at = float("-inf")
        self._frame_timer = QTimer(self, singleShot=True, timerType=Qt.TimerType.PreciseTimer)
        self._frame_timer.timeout.connect(self._start_frame)
        self._exact_timer = QTimer(self, singleShot=True)
        self._exact_timer.timeout.connect(self._exact_ready)
        canvas.moved.connect(self._camera_changed)
        canvas.resized.connect(self._camera_changed)

    def set_document(self, document, *, reset=False, keep_draft=False):
        previous = self.document
        self.document = document
        if not reset and previous is not None and (previous is document or _same_composition(previous, document)):
            if self.shown is not None:
                self.shown = (document, *self.shown[1:])
            if not keep_draft:
                self.end_draft()
            return

        self._supersede()
        self.shown = self._overview = self._before_draft = None
        if not keep_draft:
            self.draft = None
            self._gesturing = False
        if reset or previous is None or previous.size != document.size:
            self._displayed = None
            self.canvas.set_size(document.size)
        self._queue_frame()
        self._schedule_exact(0)

    def begin_draft(self):
        if not self._gesturing:
            self._gesturing = True
            self._before_draft = self._displayed
            self._supersede()

    def update_draft(self, layer_id, **changes):
        draft = (layer_id, changes)
        if draft != self.draft:
            self.begin_draft()
            self.draft = draft
            self._queue_frame()

    def end_draft(self, *, commit=False):
        if self._gesturing:
            self._supersede()
            self._gesturing = False
            self.draft = None
            frame, self._before_draft = self._before_draft, None
            if commit:
                self._interactive_due = False
                self._frame_timer.stop()
                return
            visible = self._visible()
            if frame is not None:
                if visible is not None and self._overview is not None and frame.area.intersect(visible) != visible:
                    frame = self._overview
                self._show(frame)
            self._queue_frame()
            self._schedule_exact(0)

    def resume(self):
        self._queue_frame()
        self._schedule_exact(0)

    def suspend(self):
        self._supersede()
        self._interactive_due = False
        self._frame_timer.stop()

    def _supersede(self):
        self._epoch += 1
        if self._interactive is not None:
            self._interactive[0].cancelled.set()
        self._stop_exact()

    def _visible(self, margin=0):
        if self.document is None or not self.canvas.isVisible():
            return None
        visible = self.canvas.mapToScene(self.canvas.viewport().rect()).boundingRect()
        return Rect(
            math.floor(visible.left() - margin), math.floor(visible.top() - margin),
            math.ceil(visible.right() + margin), math.ceil(visible.bottom() + margin),
        ).intersect(self.document.bounds)

    def _viewport(self):
        zoom = self.canvas.transform().m11()
        return self._visible(48 / zoom), min(1, zoom * self.canvas.devicePixelRatioF())

    def _camera_changed(self):
        if self._overview is not None and not self._gesturing:
            visible = self._visible()
            if visible is not None and (
                self._displayed is None or self._displayed.area.intersect(visible) != visible
            ):
                self._show(self._overview)
        if self._interactive is not None and self._interactive[1].overview:
            self._interactive[0].cancelled.set()
        self._queue_frame()
        self._schedule_exact(100)

    def _queue_frame(self):
        self._interactive_due = True
        self._schedule_frame()

    def _schedule_frame(self):
        if self._interactive is None and not self._frame_timer.isActive() and self.canvas.isVisible():
            delay = math.ceil(max(0, 16 - (monotonic() - self._last_frame_at) * 1000))
            self._frame_timer.start(delay)

    def _start_frame(self):
        if self._interactive is not None:
            return
        area, scale = self._viewport()
        if area is None:
            return
        overview = not self._interactive_due
        self._interactive_due = False
        if overview:
            if self._overview is not None or self._gesturing:
                return
            area = self.document.bounds
            scale = self.canvas.fit_scale() * self.canvas.devicePixelRatioF()
        region = interactive_region(area, scale)
        request = _Request(self._epoch, region, overview)
        document, draft = self.document, self.draft
        tiles = self._tiles
        task = EditorTask(lambda cancelled: interactive_preview(document, region, tiles, cancelled, draft))
        self._interactive = task, request
        self._last_frame_at = monotonic()
        task.signals.finished.connect(self._frame_finished, Qt.ConnectionType.QueuedConnection)
        QThreadPool.globalInstance().start(task)

    def _frame_finished(self, task, image, error):
        _, request = self._interactive
        self._interactive = None
        if self._accept(task, request):
            if error:
                self.show_error(error)
            elif image is not None:
                frame = _Frame(request.region.area, QPixmap.fromImage(image))
                if not self._gesturing and frame.area == self.document.bounds:
                    self._overview = frame
                if not request.overview and (self._gesturing or self.shown != self._exact_target()):
                    visible = self._visible()
                    if visible is not None and frame.area.intersect(visible) == visible:
                        self._show(frame)
        if self._interactive_due or (not error and self._overview is None and not self._gesturing):
            self._schedule_frame()

    def _schedule_exact(self, delay):
        self._stop_exact()
        if not self._gesturing and self.canvas.isVisible():
            self._exact_timer.start(delay)

    def _stop_exact(self):
        self._exact_timer.stop()
        self._exact_due = False
        if self._exact is not None:
            self._exact[0].cancelled.set()

    def _exact_target(self):
        area, scale = self._viewport()
        if area is None:
            return None
        size = max(1, round(area.width * scale)), max(1, round(area.height * scale))
        return self.document, area, size

    def _exact_ready(self):
        self._exact_due = True
        self._start_exact()

    def _start_exact(self):
        if not self._exact_due or self._exact is not None:
            return
        self._exact_due = False
        target = self._exact_target()
        if target is None or target == self.shown:
            return
        document, area, size = target
        region = PreviewRegion(area, size)
        task = EditorTask(lambda cancelled: exact_preview(document, region, cancelled))
        self._exact = task, _Request(self._epoch, region)
        task.signals.finished.connect(self._exact_finished, Qt.ConnectionType.QueuedConnection)
        QThreadPool.globalInstance().start(task)

    def _exact_finished(self, task, image, error):
        _, request = self._exact
        self._exact = None
        area, size = request.region.area, request.region.size
        if self._accept(task, request) and (self.document, area, size) == self._exact_target():
            if error:
                self.show_error(error)
            elif image is not None:
                self._show(_Frame(area, QPixmap.fromImage(image), size))
        self._start_exact()

    def _accept(self, task, request):
        return self.canvas.isVisible() and request.epoch == self._epoch and not task.cancelled.is_set()

    def _show(self, frame):
        if frame is not self._displayed:
            self._displayed = frame
            area = frame.area
            self.canvas.set_image(frame.pixmap, rect=QRectF(area.left, area.top, area.width, area.height))
        self.shown = (self.document, frame.area, frame.exact_size) if frame.exact_size is not None else None
