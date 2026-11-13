"""Layered raster documents, compositing, and pixel edits."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance

from .codex import Cancelled
from .files import SRGB_BYTES, SrgbPngWriter, atomic_output
from .geometry import MAX_PNG_DIMENSION, Rect

RASTER_TILE_SIZE = 256


def box(area: Rect) -> tuple[int, int, int, int]:
    return area.left, area.top, area.right, area.bottom


def validate_size(width: int, height: int):
    if any(type(value) is not int or not 1 <= value <= MAX_PNG_DIMENSION for value in (width, height)):
        raise ValueError("Width and height must be positive pixel dimensions.")
    if Image.MAX_IMAGE_PIXELS and width * height > 2 * Image.MAX_IMAGE_PIXELS:
        raise ValueError("These dimensions are too large for the image editor.")


def _tiles(area: Rect, size: tuple[int, int]):
    step = RASTER_TILE_SIZE
    width, height = size
    for top in range(area.top // step * step, area.bottom, step):
        for left in range(area.left // step * step, area.right, step):
            yield (left, top), Rect(left, top, min(left + step, width), min(top + step, height))


def _sample_grid(image: Image.Image, area: Rect, step: int) -> Image.Image:
    if step == 1:
        return image.crop(box(area))
    size = (area.width + step - 1) // step, (area.height + step - 1) // step
    # Map each output center to the center of the first source pixel in its block.
    return image.transform(size, Image.Transform.AFFINE, (
        step, 0, area.left - (step - 1) / 2,
        0, step, area.top - (step - 1) / 2,
    ), Image.Resampling.NEAREST)


@dataclass(frozen=True, eq=False)
class Raster:
    """An immutable source image with replacement tiles shared by edit history."""

    _base: Image.Image
    _tiles: dict[tuple[int, int], Image.Image]

    @property
    def size(self) -> tuple[int, int]:
        return self._base.size

    @property
    def width(self) -> int:
        return self._base.width

    @property
    def height(self) -> int:
        return self._base.height

    @property
    def mode(self) -> str:
        return self._base.mode

    def crop(self, region: tuple[int, int, int, int]) -> Image.Image:
        result = self._base.crop(region)
        area = Rect(*region)
        inside = area.intersect(Rect(0, 0, *self.size))
        if inside is not None:
            for origin, tile in _tiles(inside, self.size):
                pixels = self._tiles.get(origin)
                if pixels is not None:
                    overlap = inside.intersect(tile)
                    local = overlap.relative_to(tile)
                    result.paste(pixels.crop(box(local)), (overlap.left - area.left, overlap.top - area.top))
        return result

    def sample(self, area: Rect, step: int) -> Image.Image:
        """Take a strided sample from an in-bounds rectangle, including raster edits."""
        if step == 1:
            return self.crop(box(area))
        result = _sample_grid(self._base, area, step)
        grid = RASTER_TILE_SIZE
        left, top = area.left // grid, area.top // grid
        columns = (area.right - 1) // grid - left + 1
        rows = (area.bottom - 1) // grid - top + 1
        if len(self._tiles) < columns * rows:
            patches = self._tiles.items()
        else:
            patches = ((origin, self._tiles.get(origin)) for origin, _ in _tiles(area, self.size))
        for (x, y), pixels in patches:
            if pixels is None:
                continue
            overlap = area.intersect(Rect(x, y, x + pixels.width, y + pixels.height))
            if overlap is None:
                continue
            offset = overlap.relative_to(area)
            first_col, last_col = (offset.left + step - 1) // step, (offset.right + step - 1) // step
            first_row, last_row = (offset.top + step - 1) // step, (offset.bottom + step - 1) // step
            if first_col < last_col and first_row < last_row:
                sample = Rect(
                    area.left + first_col * step - x, area.top + first_row * step - y,
                    overlap.right - x, overlap.bottom - y,
                )
                result.paste(_sample_grid(pixels, sample, step), (first_col, first_row))
        return result

    def getpixel(self, point: tuple[int, int]):
        x, y = point
        x = x + self.width if x < 0 else x
        y = y + self.height if y < 0 else y
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError("Pixel index is outside the image.")
        left = x // RASTER_TILE_SIZE * RASTER_TILE_SIZE
        top = y // RASTER_TILE_SIZE * RASTER_TILE_SIZE
        tile = self._tiles.get((left, top))
        return self._base.getpixel((x, y)) if tile is None else tile.getpixel((x - left, y - top))


type PixelAsset = Image.Image | Raster


def sample_asset(asset: PixelAsset, area: Rect, step: int) -> Image.Image:
    """Sample the first pixel of each step-sized block in an in-bounds rectangle."""
    return asset.sample(area, step) if isinstance(asset, Raster) else _sample_grid(asset, area, step)


def asset_image(asset: PixelAsset) -> Image.Image:
    return asset.crop((0, 0, *asset.size)) if isinstance(asset, Raster) else asset


def asset_thumbnail(asset: PixelAsset, size: tuple[int, int]) -> Image.Image:
    if isinstance(asset, Image.Image):
        return asset.resize(size, Image.Resampling.NEAREST)
    width, height = size
    columns = [(2 * x + 1) * asset.width // (2 * width) for x in range(width)]
    rows = [(2 * y + 1) * asset.height // (2 * height) for y in range(height)]
    image = Image.new(asset.mode, size)
    image.putdata([asset.getpixel((x, y)) for y in rows for x in columns])
    return image


def _edit_pixels(
    asset: PixelAsset, area: Rect, edit: Callable[[Image.Image, Rect, Image.Image | None], Image.Image],
    *, coverage: Image.Image | None = None,
) -> PixelAsset:
    """Apply a local edit while keeping pixels shared with older versions."""
    inside = area.intersect(Rect(0, 0, *asset.size))
    if inside is None:
        return asset
    base, tiles = (asset._base, asset._tiles) if isinstance(asset, Raster) else (asset, {})
    updated = {}
    for origin, tile in _tiles(inside, asset.size):
        overlap = inside.intersect(tile)
        part = overlap.relative_to(area)
        weight = coverage.crop(box(part)) if coverage is not None else None
        if weight is not None and weight.getbbox() is None:
            continue
        local = overlap.relative_to(tile)
        stored = tiles.get(origin)
        previous = base.crop(box(overlap)) if stored is None else stored.crop(box(local))
        patch = edit(previous, part, weight)
        if patch == previous:
            continue
        changed = base.crop(box(tile)) if stored is None else stored.copy()
        changed.paste(patch, (local.left, local.top))
        updated[origin] = changed
    return Raster(base, tiles | updated) if updated else asset


def paste_pixels(asset: PixelAsset, patch: Image.Image, origin: tuple[int, int]) -> PixelAsset:
    left, top = origin
    area = Rect(left, top, left + patch.width, top + patch.height)
    return _edit_pixels(asset, area, lambda _previous, part, _weight: patch.crop(box(part)))


def paint_mask(mask: PixelAsset, area: Rect, coverage: Image.Image, value: int) -> PixelAsset:
    def edit(previous, _part, weight):
        patch = previous.copy()
        patch.paste(value, (0, 0, *patch.size), weight)
        return patch

    return _edit_pixels(mask, area, edit, coverage=coverage)


@dataclass(frozen=True, eq=False)
class Layer:
    # Treat pixel assets as immutable. History and duplicated layers share them.
    name: str
    image: PixelAsset
    x: int = 0
    y: int = 0
    opacity: int = 100
    visible: bool = True
    locked: bool = False
    mask: PixelAsset | None = None
    coverage: Image.Image | None = None
    adjustments: tuple[int, int, int] = (0, 0, 0)
    id: str = field(default_factory=lambda: uuid4().hex)

    @property
    def bounds(self) -> Rect:
        return Rect(self.x, self.y, self.x + self.image.width, self.y + self.image.height)

    def transform(self, operation: Callable[[Image.Image], Image.Image]) -> "Layer":
        return replace(
            self, image=operation(asset_image(self.image)),
            mask=operation(asset_image(self.mask)) if self.mask is not None else None,
            coverage=operation(self.coverage) if self.coverage is not None else None,
        )

    def resize(self, size: tuple[int, int]) -> "Layer":
        if self.image.size == size:
            return self
        if self.coverage is None:
            return self.transform(lambda image: image.resize(size, Image.Resampling.LANCZOS))
        image, coverage = _resize_replacement(asset_image(self.image), self.coverage, size)
        mask = asset_image(self.mask).resize(size, Image.Resampling.LANCZOS) if self.mask is not None else None
        return replace(self, image=image, mask=mask, coverage=coverage)


@dataclass(frozen=True, eq=False)
class Document:
    name: str
    width: int
    height: int
    layers: tuple[Layer, ...] = ()  # Bottom to top.
    id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self):
        validate_size(*self.size)

    @classmethod
    def from_image(cls, image: Image.Image, name: str) -> "Document":
        return cls(name, *image.size, (Layer(name, image.convert("RGBA")),))

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def bounds(self) -> Rect:
        return Rect(0, 0, *self.size)

    def layer(self, layer_id: str | None) -> Layer | None:
        return next((layer for layer in self.layers if layer.id == layer_id), None)

    def insert(self, layer: Layer, *, above: Layer | None = None) -> "Document":
        index = len(self.layers) if above is None else self.layers.index(above) + 1
        return replace(self, layers=self.layers[:index] + (layer,) + self.layers[index:])

    def update(self, layer: Layer) -> "Document":
        return replace(self, layers=tuple(layer if item.id == layer.id else item for item in self.layers))

    def crop(self, area: Rect) -> "Document":
        return replace(
            self, width=area.width, height=area.height,
            layers=tuple(replace(layer, x=layer.x - area.left, y=layer.y - area.top) for layer in self.layers),
        )


def _resize_replacement(image: Image.Image, coverage: Image.Image, size: tuple[int, int]):
    """Filter the replacement contribution and its coverage independently."""
    pixels = np.asarray(image)
    weight = np.asarray(coverage, dtype=np.float32) / 255
    covered_alpha = pixels[..., 3] * weight

    def sample(channel):
        return np.asarray(Image.fromarray(channel).resize(size, Image.Resampling.LANCZOS))

    def unpremultiply(numerator, denominator):
        result = np.zeros_like(numerator)
        np.divide(numerator, denominator, out=result, where=denominator > 0)
        return np.rint(np.clip(result, 0, 255)).astype(np.uint8)

    resized_weight = sample(weight)
    resized_alpha = sample(covered_alpha)
    result = np.empty((size[1], size[0], 4), dtype=np.uint8)
    result[..., 3] = unpremultiply(resized_alpha, resized_weight)
    # A transparent replacement still covers the lower layers. Filtering color
    # by both coverage and alpha excludes pixels outside the original selection.
    for channel in range(3):
        contribution = sample(pixels[..., channel] * covered_alpha)
        result[..., channel] = unpremultiply(contribution, resized_alpha)
    mask = np.rint(np.clip(resized_weight, 0, 1) * 255).astype(np.uint8)
    result[(mask == 0) | (result[..., 3] == 0)] = 0
    return Image.fromarray(result), Image.fromarray(mask)


def _adjust(image: Image.Image, values: tuple[int, int, int]) -> Image.Image:
    brightness, contrast, saturation = values
    if not any(values):
        return image
    rgb = image.convert("RGB")
    if brightness or contrast:
        # A fixed midpoint gives the same result for every rendered region.
        table = [
            min(255, max(0, round((value - 128) * (1 + contrast / 100) + 128 + 255 * brightness / 100)))
            for value in range(256)
        ]
        rgb = rgb.point(table * 3)
    if saturation:
        rgb = ImageEnhance.Color(rgb).enhance(1 + saturation / 100)
    rgb.putalpha(image.getchannel("A"))
    return rgb


def _replace_pixels(lower: Image.Image, upper: Image.Image, weight: Image.Image) -> Image.Image:
    histogram = weight.histogram()
    if histogram[0] + histogram[255] == weight.width * weight.height:
        result = Image.composite(upper, lower, weight)
        alpha = result.getchannel("A")
        if alpha.getextrema()[0] == 0:
            result.paste((0, 0, 0, 0), (0, 0, *result.size), alpha.point([255] + [0] * 255))
        return result

    low, high = np.asarray(lower), np.asarray(upper)
    mix = np.asarray(weight, dtype=np.float32) / 255
    low_alpha = low[..., 3] * (1 - mix)
    high_alpha = high[..., 3] * mix
    alpha = low_alpha + high_alpha
    # Dividing by infinity clears RGB wherever the rounded alpha is zero.
    divisor = np.where(alpha > 0.5, alpha, np.float32(np.inf))
    color = np.empty_like(alpha)
    contribution = np.empty_like(alpha)
    result = np.empty_like(low)
    for channel in range(3):
        np.multiply(low[..., channel], low_alpha, out=color)
        np.multiply(high[..., channel], high_alpha, out=contribution)
        color += contribution
        color /= divisor
        np.rint(color, out=color)
        result[..., channel] = color
    result[..., 3] = np.rint(alpha)
    return Image.fromarray(result)


def _render_band(document: Document, area: Rect) -> Image.Image:
    result = Image.new("RGBA", (area.width, area.height))
    for layer in document.layers:
        overlap = area.intersect(layer.bounds)
        if overlap is None or not layer.visible or layer.opacity == 0:
            continue
        region = box(overlap.relative_to(layer.bounds))
        target = overlap.relative_to(area)
        pixels = _adjust(layer.image.crop(region), layer.adjustments)
        origin = target.left, target.top
        if layer.coverage is None and layer.mask is None and layer.opacity == 100:
            result.alpha_composite(pixels, origin)
            continue

        weight = layer.coverage.crop(region) if layer.coverage is not None else pixels.getchannel("A")
        if layer.mask is not None:
            weight = ImageChops.multiply(weight, layer.mask.crop(region))
        if layer.opacity != 100:
            weight = weight.point([round(value * layer.opacity / 100) for value in range(256)])
        if layer.coverage is None:
            pixels.putalpha(weight)
            result.alpha_composite(pixels, origin)
        else:
            result.paste(_replace_pixels(result.crop(box(target)), pixels, weight), origin)
    return result


def _bands(document, area, cancelled, progress):
    # Keep temporary pixel buffers bounded even for wide reconstructions.
    rows = max(1, min(128, 500_000 // area.width))
    for top in range(area.top, area.bottom, rows):
        if cancelled():
            raise Cancelled("Rendering canceled.")
        bottom = min(top + rows, area.bottom)
        yield top - area.top, _render_band(document, Rect(area.left, top, area.right, bottom))
        progress(bottom - area.top, area.height)


def render(
    document: Document, area: Rect | None = None, *,
    cancelled: Callable[[], bool] = lambda: False, progress: Callable[[int, int], None] = lambda *_: None,
) -> Image.Image:
    area = document.bounds if area is None else area.intersect(document.bounds)
    if area is None:
        raise ValueError("The render area is outside the canvas.")
    result = Image.new("RGBA", (area.width, area.height))
    for top, band in _bands(document, area, cancelled, progress):
        result.paste(band, (0, top))
    return result


def export_document(
    document: Document, destination: Path, *, cancelled: Callable[[], bool] = lambda: False,
    progress: Callable[[int, int], None] = lambda *_: None, before_publish: Callable[[Path], None] | None = None,
):
    suffix = destination.suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".webp"):
        raise ValueError("Choose PNG, JPEG, or WebP.")

    def publish(path):
        if cancelled():
            raise Cancelled("Export canceled.")
        if before_publish is not None:
            before_publish(path)

    with atomic_output(destination, before_publish=publish) as stream:
        if suffix == ".png":
            def rows():
                for _, band in _bands(document, document.bounds, cancelled, progress):
                    yield from (row.tobytes() for row in np.asarray(band))

            writer = SrgbPngWriter(*document.size, greyscale=False, alpha=True, compression=3)
            writer.write(stream, rows())
        else:
            pixels = render(document, cancelled=cancelled, progress=progress)
            if suffix == ".webp":
                pixels.save(stream, format="WEBP", lossless=True, icc_profile=SRGB_BYTES)
            else:
                white = Image.new("RGB", pixels.size, "white")
                white.paste(pixels, mask=pixels.getchannel("A"))
                white.save(stream, format="JPEG", quality=95, icc_profile=SRGB_BYTES)


def paint(
    document: Document, layer: Layer, points: list[tuple[int, int]], *,
    size: int, strength: int, color: tuple[int, int, int] | None,
    edit_mask: bool, selection: Image.Image | None,
) -> Layer:
    if not points or strength == 0:
        return layer
    xs, ys = zip(*points)
    radius = size // 2
    stroke = Rect(min(xs) - radius, min(ys) - radius, max(xs) + radius + 1, max(ys) + radius + 1)
    canvas = document.bounds.intersect(layer.bounds)
    area = stroke.intersect(canvas) if canvas is not None else None
    if area is None:
        return layer

    coverage = Image.new("L", (area.width, area.height))
    draw = ImageDraw.Draw(coverage)
    local_points = [(x - area.left, y - area.top) for x, y in points]
    if len(local_points) > 1:
        draw.line(local_points, fill=255, width=size, joint="curve")
    if size == 1:
        draw.point(local_points, fill=255)
    else:
        for x, y in local_points:
            draw.ellipse((x - radius, y - radius, x - radius + size - 1, y - radius + size - 1), fill=255)
    if selection is not None:
        coverage = ImageChops.multiply(coverage, selection.crop(box(area)))
    if strength != 100:
        coverage = coverage.point([round(value * strength / 100) for value in range(256)])
    if coverage.getbbox() is None:
        return layer

    local = area.relative_to(layer.bounds)
    if edit_mask:
        mask = layer.mask if layer.mask is not None else Image.new("L", layer.image.size, 255)
        mask = paint_mask(mask, local, coverage, 255 if color is not None else 0)
        return layer if mask is layer.mask else replace(layer, mask=mask)

    def edit(previous, _part, weight):
        patch = previous.copy()
        if color is None:
            patch.putalpha(ImageChops.multiply(patch.getchannel("A"), ImageChops.invert(weight)))
        else:
            ink = Image.new("RGBA", patch.size, (*color, 255))
            ink.putalpha(weight)
            patch.alpha_composite(ink)
        return patch

    image = _edit_pixels(layer.image, local, edit, coverage=coverage)
    return layer if image is layer.image else replace(layer, image=image)
