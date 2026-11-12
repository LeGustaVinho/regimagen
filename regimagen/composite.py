"""Multiband compositing and streamed PNG export."""

import io
import json
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import png
from PIL import Image

from .codex import Cancelled
from .files import SRGB_BYTES, atomic_output
from .geometry import FRAME, Cell, Layout, Rect
from .imaging import read_tile
from .matte import SpriteMatte
from .seams import COLOR_FEATHER, SEAM_MARGIN, SHIFT_LIMIT, SeamPath, find_junction, find_seam, junction_region, seam_region

_levels = np.arange(256, dtype=np.float32) / 255
LINEAR_RGB = np.where(_levels <= 0.04045, _levels / 12.92, ((_levels + 0.055) / 1.055) ** 2.4)
BANDS = (COLOR_FEATHER, 5.0, 1.25)
BLUR_RADIUS = 24
STRIPE_HEIGHT = 64


@dataclass(frozen=True)
class _Border:
    distance: np.ndarray
    room: np.ndarray | None = None
    feather: np.ndarray | None = None
    shift: np.ndarray | None = None
    across_x: bool = False


class Compositor:
    def __init__(
        self, layout: Layout, artifacts: dict[Cell, Path], cache_size: int | None = None,
        *, alpha: Image.Image | None = None,
    ):
        self.layout = layout
        self.artifacts = {cell: artifacts[cell] for cell in layout.cells
                          if cell in artifacts and layout.core(cell).intersect(layout.bounds) is not None}
        # An L-mode mask can be used when only alpha coverage is available.
        self.matte = SpriteMatte(alpha, layout.size) if alpha is not None else None
        self.cache_size = cache_size or 2 * layout.grid.columns + 2
        self.cache: OrderedDict[Cell, np.ndarray] = OrderedDict()
        self.seams: dict[tuple[Cell, Cell], SeamPath] = {}
        self.junctions: dict[Cell, tuple[int | None, int | None]] = {}

    def _tile(self, cell: Cell) -> np.ndarray:
        if cell not in self.cache:
            self.cache[cell] = read_tile(self.artifacts[cell])
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(cell)
        return self.cache[cell]

    def _seam(self, first: Cell, second: Cell) -> SeamPath:
        key = first, second
        if key not in self.seams:
            area = seam_region(self.layout, first, second)
            coverage = self.matte.coverage(area) if self.matte is not None else None
            across_x = first.row == second.row
            following = Cell(second.column, second.row + 1) if across_x else Cell(second.column + 1, second.row)
            junctions = self._junction(second), self._junction(following)
            self.seams[key] = find_seam(
                self.layout, first, second, self._tile(first), self._tile(second), coverage, junctions,
            )
        return self.seams[key]

    def _junction(self, corner: Cell) -> tuple[int | None, int | None]:
        if corner not in self.junctions:
            cells = (Cell(corner.column - 1, corner.row - 1), Cell(corner.column, corner.row - 1),
                     Cell(corner.column - 1, corner.row), corner)
            point = None, None
            margin = max(self.layout.band(SEAM_MARGIN))
            if all(cell in self.artifacts for cell in cells) and all(
                min(self.layout.core(cell).width, self.layout.core(cell).height) > 2 * (margin + 4) + 48
                for cell in cells
            ):
                area = junction_region(self.layout, corner)
                patches = [self._tile(cell)[area.relative_to(self.layout.frame(cell)).slices] for cell in cells]
                coverage = self.matte.coverage(area) if self.matte is not None else None
                point = find_junction(patches, area, coverage)
            self.junctions[corner] = point
        return self.junctions[corner]

    def _borders(self, cell: Cell, core: Rect, area: Rect, *, padding: int = 0) -> list[_Border]:
        result = []
        x = np.arange(area.left, area.right, dtype=np.float32)[None, :] + 0.5
        y = np.arange(area.top, area.bottom, dtype=np.float32)[:, None] + 0.5
        support_x, support_y = self.layout.band(SEAM_MARGIN)
        fallback_x, fallback_y = self.layout.band()
        for neighbor in self.layout.neighbors(cell):
            across_x = neighbor.row == cell.row
            before = neighbor.column < cell.column if across_x else neighbor.row < cell.row
            sign = 1 if before else -1
            if across_x:
                nominal = core.left if before else core.right
                extent = area.left if before else area.right
                axis, support, fallback = x, support_x, fallback_x
                start, end = area.top, area.bottom
            else:
                nominal = core.top if before else core.bottom
                extent = area.top if before else area.bottom
                axis, support, fallback = y, support_y, fallback_y
                start, end = area.left, area.right
            if sign * (extent - nominal) >= support + padding:
                continue
            if neighbor not in self.artifacts:
                result.append(_Border(sign * (axis - nominal) + fallback))
                continue
            pair = (neighbor, cell) if before else (cell, neighbor)
            position, room, feather, shift = self._seam(*pair).sample(start, end)
            if across_x:
                position, room, feather, shift = (value[:, None] for value in (position, room, feather, shift))
            result.append(_Border(sign * (axis - position), room, feather, shift * (0.5 * sign), across_x))
        return result

    def render(self, area: Rect) -> np.ndarray:
        if area.intersect(self.layout.bounds) != area:
            raise ValueError("The compositing area must be inside the image.")
        if self.matte is None:
            pixels, missing = self._render_rgb(area)
            pixels[missing] = (20, 22, 28)
            return pixels

        # Matte correction uses the same neighborhood for zoom and export.
        pad = self.matte.padding
        extended = area.expand(pad, pad).intersect(self.layout.bounds)
        assert extended is not None
        coverage = self.matte.coverage(extended)
        keep = area.relative_to(extended).slices
        if not np.any(coverage[keep]):
            return np.zeros((area.height, area.width, 4), dtype=np.uint8)
        pixels, missing = self._render_rgb(extended)
        self.matte.refine(extended, pixels, coverage, missing)
        coverage[missing] = 0
        pixels[coverage == 0] = 0
        return np.dstack((pixels[keep], coverage[keep]))

    def _render_rgb(self, area: Rect) -> tuple[np.ndarray, np.ndarray]:
        height, width = area.height, area.width
        sums = np.zeros((len(BANDS), height, width, 3), dtype=np.float32)
        weights = np.zeros((len(BANDS), height, width), dtype=np.float32)
        support = self.layout.band(SEAM_MARGIN)
        for cell in self.artifacts:
            core = self.layout.core(cell)
            shared = area.intersect(core.expand(*support))
            if shared is None:
                continue
            bands = self._frequencies(cell, shared)
            region = shared.relative_to(area).slices
            borders = self._borders(cell, core, shared)
            for index, (maximum, band) in enumerate(zip(BANDS, bands, strict=True)):
                blend = np.ones((shared.height, shared.width), dtype=np.float32)
                for border in borders:
                    if border.room is None:
                        blend *= border.distance >= 0
                    else:
                        half_band = np.minimum(border.room, border.feather if index == 0 else maximum)
                        phase = np.clip((border.distance + half_band) / (2 * half_band), 0, 1)
                        blend *= 0.5 - 0.5 * np.cos(np.pi * phase)
                sums[index][region] += band * blend[:, :, None]
                weights[index][region] += blend

        denominator = weights[:, :, :, None]
        np.divide(sums, denominator, out=sums, where=denominator > 0)
        linear = np.clip(sums.sum(axis=0), 0, 1)
        srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
        pixels = np.rint(srgb * 255).astype(np.uint8)
        return pixels, weights[0] == 0

    def _frequencies(self, cell: Cell, area: Rect) -> tuple[np.ndarray, ...]:
        frame = self.layout.frame(cell)
        region = area.relative_to(frame)
        # Include pixels outside the requested stripe or zoom region for filtering.
        local_frame = Rect(0, 0, FRAME, FRAME)
        padded = region.expand(BLUR_RADIUS, BLUR_RADIUS).intersect(local_frame)
        assert padded is not None
        patch = self._registered_patch(cell, frame, padded, local_frame)
        keep = region.relative_to(padded).slices
        low = cv2.GaussianBlur(patch, (49, 49), 8)[keep]
        smooth = cv2.GaussianBlur(patch, (13, 13), 2)[keep]
        return low, smooth - low, patch[keep] - smooth

    def _registered_patch(self, cell: Cell, frame: Rect, padded: Rect, local_frame: Rect) -> np.ndarray:
        area = Rect(frame.left + padded.left, frame.top + padded.top,
                    frame.left + padded.right, frame.top + padded.bottom)
        borders = [border for border in self._borders(cell, self.layout.core(cell), area, padding=31)
                   if border.shift is not None and np.any(border.shift)]
        tile = self._tile(cell)
        if not borders:
            return LINEAR_RGB[tile[padded.slices]]

        margin = SHIFT_LIMIT // 2 + 2
        samples = padded.expand(margin, margin).intersect(local_frame)
        assert samples is not None
        map_x, map_y = np.meshgrid(np.arange(padded.left, padded.right, dtype=np.float32),
                                   np.arange(padded.top, padded.bottom, dtype=np.float32))
        for border in borders:
            phase = np.clip((np.abs(border.distance) - 3) / 28, 0, 1)
            envelope = 0.5 + 0.5 * np.cos(np.pi * phase)
            coordinates = map_y if border.across_x else map_x
            coordinates += envelope * border.shift
        map_x -= samples.left
        map_y -= samples.top
        return cv2.remap(LINEAR_RGB[tile[samples.slices]], map_x, map_y, cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REFLECT_101)


class _SrgbWriter(png.Writer):
    def write_preamble(self, stream):
        super().write_preamble(stream)
        png.write_chunk(stream, b"sRGB", b"\x00")


def render_part(layout: Layout, cell: Cell, artifact: Path, *, alpha: Image.Image | None = None):
    core = layout.core(cell)
    area = core.expand(*layout.band()).intersect(layout.bounds)
    assert area is not None
    image = Image.fromarray(Compositor(layout, {cell: artifact}, alpha=alpha).render(area))
    position = {
        "x": area.left, "y": area.top, "width": area.width, "height": area.height,
        "core": {"x": core.left, "y": core.top, "width": core.width, "height": core.height},
    }
    return image, position


def export_parts(
    layout: Layout,
    artifacts: dict[Cell, Path],
    destination: Path,
    *,
    progress: Callable[[int, int], None] = lambda *_: None,
    cancelled: Callable[[], bool] = lambda: False,
    alpha: Image.Image | None = None,
    overwrite: bool = True,
):
    if not artifacts:
        raise ValueError("Generate at least one tile before exporting.")
    width, height = layout.size
    manifest = {"width": width, "height": height, "tiles": []}
    with atomic_output(destination, overwrite=overwrite) as stream, ZipFile(stream, "w") as archive:
        for index, (cell, artifact) in enumerate(artifacts.items(), start=1):
            if cancelled():
                raise Cancelled("Export canceled.")
            name = f"row-{cell.row + 1:02d}_column-{cell.column + 1:02d}.png"
            image, position = render_part(layout, cell, artifact, alpha=alpha)
            data = io.BytesIO()
            image.save(data, format="PNG", icc_profile=SRGB_BYTES, compress_level=3)
            archive.writestr(name, data.getvalue())
            manifest["tiles"].append({"file": name, **position})
            progress(index, len(artifacts))
        archive.writestr("positions.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        if cancelled():
            raise Cancelled("Export canceled.")


def export_png(
    layout: Layout,
    artifacts: dict[Cell, Path],
    destination: Path,
    *,
    progress: Callable[[int, int], None] = lambda *_: None,
    cancelled: Callable[[], bool] = lambda: False,
    alpha: Image.Image | None = None,
    overwrite: bool = True,
):
    if set(artifacts) != set(layout.cells):
        raise ValueError("Complete all tiles in the grid before exporting.")
    width, height = layout.size
    compositor = Compositor(layout, artifacts, alpha=alpha)

    def rows() -> Iterator[bytes]:
        for top in range(0, height, STRIPE_HEIGHT):
            if cancelled():
                raise Cancelled("Export canceled.")
            bottom = min(top + STRIPE_HEIGHT, height)
            band = compositor.render(Rect(0, top, width, bottom))
            yield from (row.tobytes() for row in band)
            progress(bottom, height)

    with atomic_output(destination, overwrite=overwrite) as stream:
        _SrgbWriter(width, height, greyscale=False, alpha=alpha is not None, compression=3).write(stream, rows())
        if cancelled():
            raise Cancelled("Export canceled.")
