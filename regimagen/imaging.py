"""ImageGen reference preparation, tile alignment, and seam checks."""

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .files import save_png
from .geometry import FRAME, Cell, Layout, Rect
from .seams import seam_region

SHIFT_LIMIT = 16

EDIT_PROMPT = (
    "Use the native image generation tool exactly once to EDIT image 1. Use the attached "
    "images as input to the image generation tool in their original order. Image 1 is the "
    "exact square crop to reconstruct. Image 2 is guidance only: its left panel shows the "
    "area around the crop and its right panel shows the full composition. Produce one "
    "opaque square image, ideally 1024 by 1024 pixels, matching the exact field of view, "
    "object positions, contours, perspective, lighting, palette and medium of image 1. "
    "Add fine detail that belongs to the existing materials and style. Preserve the exact "
    "position and continuity of lines that leave the crop. Keep existing text, object "
    "counts and silhouettes. The output should depict image 1 only, edge to edge."
)
REPAIR_PROMPT = (
    " Image 3 is a seam guide for the same square, with already reconstructed neighbor "
    "strips in their exact positions. Continue existing parts across each guided side with "
    "the same shape, thickness, lighting and texture. Where strips meet, use image 1 as "
    "the geometric reference. Preserve complete features that cross the crop."
)


def prepare_frame(source: np.ndarray, layout: Layout, cell: Cell) -> np.ndarray:
    box = layout.frame(cell)
    width, height = layout.size
    x = (np.arange(box.left, box.right, dtype=np.float32) + 0.5) * layout.source_width / width
    y = (np.arange(box.top, box.bottom, dtype=np.float32) + 0.5) * layout.source_height / height
    map_x, map_y = np.meshgrid(x - 0.5, y - 0.5)
    return cv2.remap(source, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)


def context_guide(source: Image.Image, layout: Layout, cell: Cell) -> Image.Image:
    left = layout.core(Cell(max(0, cell.column - 1), max(0, cell.row - 1)))
    right = layout.core(Cell(
        min(layout.grid.columns - 1, cell.column + 1), min(layout.grid.rows - 1, cell.row + 1),
    ))
    width, height = layout.size
    local = source.crop((
        left.left * source.width // width, left.top * source.height // height,
        math.ceil(right.right * source.width / width), math.ceil(right.bottom * source.height / height),
    ))
    guide = Image.new("RGB", (768, 384), (41, 42, 46))
    for x, image in ((0, local), (384, source)):
        thumb = ImageOps.contain(image, (376, 376), Image.Resampling.LANCZOS)
        guide.paste(thumb, (x + (384 - thumb.width) // 2, (384 - thumb.height) // 2))
    return guide


def overlap(layout: Layout, cell: Cell, neighbor: Cell) -> Rect | None:
    band = layout.band()
    first = layout.core(cell).expand(*band).intersect(layout.bounds)
    second = layout.core(neighbor).expand(*band).intersect(layout.bounds)
    return first.intersect(second) if first and second else None


def write_references(
    folder: Path, source: Image.Image, frame: np.ndarray, layout: Layout, cell: Cell,
    neighbors: dict[Cell, Path],
) -> tuple[Path, ...]:
    target, context = folder / "tile.png", folder / "context.png"
    save_png(Image.fromarray(frame), target)
    save_png(context_guide(source, layout, cell), context)
    if not neighbors:
        return target, context
    border = frame.copy()
    frame_box = layout.frame(cell)
    nearest = np.full((FRAME, FRAME), FRAME, dtype=np.int16)
    for neighbor, artifact in neighbors.items():
        across_x = neighbor.row == cell.row
        before = neighbor.column < cell.column if across_x else neighbor.row < cell.row
        first, second = (neighbor, cell) if before else (cell, neighbor)
        shared = seam_region(layout, first, second).intersect(layout.bounds)
        if shared is None:
            continue
        region = shared.relative_to(frame_box).slices
        core = layout.core(first)
        edge = core.right if across_x else core.bottom
        start, end = (shared.left, shared.right) if across_x else (shared.top, shared.bottom)
        distance = np.abs(2 * np.arange(start - edge, end - edge, dtype=np.int16) + 1)
        distance = distance[None, :] if across_x else distance[:, None]
        use = distance < nearest[region]
        border[region][use] = read_tile(artifact)[shared.relative_to(layout.frame(neighbor)).slices][use]
        np.minimum(nearest[region], distance, out=nearest[region])
    repair = folder / "seams.png"
    save_png(Image.fromarray(border), repair)
    return target, context, repair


def read_tile(path: Path) -> np.ndarray:
    with Image.open(path, formats=("PNG",)) as image:
        if image.size != (FRAME, FRAME) or image.mode != "RGB":
            raise ValueError("The saved tile has an invalid format.")
        return np.asarray(image).copy()


def _low_luminance(rgb: np.ndarray, sigma: float) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)


def align_tile(
    generated: Image.Image, reference: np.ndarray, layout: Layout, cell: Cell,
) -> tuple[np.ndarray, str | None]:
    pixels = np.asarray(generated)
    frame = layout.frame(cell)
    core = layout.core(cell).relative_to(frame)
    sigma = max(1.6, min(8.0, layout.size[0] / layout.source_width / 1.5))
    target = _low_luminance(reference, sigma)[core.slices]
    dx, dy = 0, 0
    warnings: list[str] = []
    if target.std() >= 4:
        search = _low_luminance(pixels, sigma)[core.expand(SHIFT_LIMIT, SHIFT_LIMIT).slices]
        scores = cv2.matchTemplate(search, target, cv2.TM_CCOEFF_NORMED)
        _, score, _, (match_x, match_y) = cv2.minMaxLoc(scores)
        if score >= 0.82 and score - scores[SHIFT_LIMIT, SHIFT_LIMIT] >= 0.0015:
            dx, dy = match_x - SHIFT_LIMIT, match_y - SHIFT_LIMIT
        if score < 0.82 or max(abs(dx), abs(dy)) == SHIFT_LIMIT:
            warnings.append("Check alignment with the original.")
    if dx or dy:
        pixels = cv2.warpAffine(
            pixels, np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32), (FRAME, FRAME),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE,
        )
    sampled_reference = reference[core.slices][::4, ::4]
    sampled_result = pixels[core.slices][::4, ::4]
    difference = sampled_reference.astype(np.int16) - sampled_result.astype(np.int16)
    correction = np.clip(np.rint(np.median(difference, axis=(0, 1))), -8, 8).astype(np.int16)
    corrected = np.clip(pixels.astype(np.int16) + correction, 0, 255).astype(np.uint8)
    if np.mean(np.abs(difference - correction)) > 22:
        warnings.append("Check the tile's colors and features.")
    return corrected, " ".join(warnings) or None


def seam_warning(
    layout: Layout, cell: Cell, neighbor: Cell, pixels: np.ndarray, other: np.ndarray,
) -> str | None:
    shared = overlap(layout, cell, neighbor)
    if shared is None:
        return None
    first = pixels[shared.relative_to(layout.frame(cell)).slices]
    second = other[shared.relative_to(layout.frame(neighbor)).slices]
    blur_a = cv2.GaussianBlur(first, (0, 0), 2)
    blur_b = cv2.GaussianBlur(second, (0, 0), 2)
    if np.mean(cv2.absdiff(blur_a, blur_b)) > 17:
        return "Colors or shapes differ visibly in the overlap."

    vertical = cell.row == neighbor.row
    sigma = max(2.0, min(8.0, layout.size[0] / layout.source_width / 2))
    edge_a = _crossing_edges(first, sigma, vertical)
    edge_b = _crossing_edges(second, sigma, vertical)
    core_a, core_b = layout.core(cell), layout.core(neighbor)
    # The transition should preserve contours that cross the center line.
    if vertical:
        center = max(core_a.left, core_b.left) - shared.left
        region = slice(None), slice(max(0, center - 8), center + 8)
    else:
        center = max(core_a.top, core_b.top) - shared.top
        region = slice(max(0, center - 8), center + 8), slice(None)
    count = np.count_nonzero(edge_a[region]) + np.count_nonzero(edge_b[region])
    if count >= 12:
        kernel = np.ones((13, 13), dtype=np.uint8)
        unpaired = (
            np.count_nonzero((edge_a & ~cv2.dilate(edge_b, kernel))[region])
            + np.count_nonzero((edge_b & ~cv2.dilate(edge_a, kernel))[region])
        )
        if unpaired / count > 0.55:
            return "Check whether contours line up across the overlap."
    return None


def _crossing_edges(pixels: np.ndarray, sigma: float, vertical: bool) -> np.ndarray:
    gray = cv2.GaussianBlur(cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY), (0, 0), sigma)
    edges = cv2.Canny(gray, 24, 56, L2gradient=True)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    tangent, normal = (dy, dx) if vertical else (dx, dy)
    crosses = np.abs(tangent) >= 0.75 * np.abs(normal)
    return np.where(crosses, edges, 0).astype(np.uint8)
