"""Find seam paths through overlapping tiles."""

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import MARGIN, Cell, Layout, Rect

SEAM_MARGIN = MARGIN - 4
COLOR_FEATHER = 22.0
SHIFT_LIMIT = 6


@dataclass(frozen=True)
class SeamPath:
    start: int
    position: np.ndarray
    room: np.ndarray
    color_feather: np.ndarray
    shift: np.ndarray

    def sample(self, start: int, end: int) -> tuple[np.ndarray, ...]:
        index = np.clip(np.arange(start, end) - self.start, 0, len(self.position) - 1)
        return self.position[index], self.room[index], self.color_feather[index], self.shift[index]


def seam_region(layout: Layout, first: Cell, second: Cell) -> Rect:
    core = layout.core(first)
    horizontal, vertical = layout.band(SEAM_MARGIN)
    if first.row == second.row:
        return Rect(core.right - horizontal, core.top, core.right + horizontal, core.bottom)
    return Rect(core.left, core.bottom - vertical, core.right, core.bottom + vertical)


def _cost(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = first.astype(np.float32), second.astype(np.float32)
    color_error = np.mean(np.abs(a - b), axis=2)
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    ax, ay = (cv2.Sobel(a, cv2.CV_32F, dx, dy, scale=0.125) for dx, dy in ((1, 0), (0, 1)))
    bx, by = (cv2.Sobel(b, cv2.CV_32F, dx, dy, scale=0.125) for dx, dy in ((1, 0), (0, 1)))
    edges = np.maximum(cv2.magnitude(ax, ay), cv2.magnitude(bx, by))
    mismatch = cv2.GaussianBlur(cv2.magnitude(ax - bx, ay - by), (0, 0), 2.0)
    cost = cv2.GaussianBlur(color_error, (0, 0), 2.0) + 0.9 * mismatch
    cost += 0.65 * cv2.GaussianBlur(edges, (0, 0), 3.0)
    return cost, mismatch


def _minimum_path(
    cost: np.ndarray, start_guard: int, end_guard: int, start_anchor: int, end_anchor: int,
) -> np.ndarray:
    height, width = cost.shape
    center = width // 2
    # Anchor the seam to the grid near four-tile intersections.
    indices = np.arange(width)
    allowed = (indices >= 3) & (indices < width - 3)
    allowed[center] = True
    cost[:, ~allowed] = np.inf
    cost[:start_guard, indices != start_anchor] = np.inf
    if end_guard:
        cost[-end_guard:, indices != end_anchor] = np.inf

    distance = (indices - center) / max(center, 1)
    cost += (2 * distance ** 2).astype(np.float32)
    parents = np.empty((height, width), dtype=np.int8)
    previous = cost[0]
    neighbors = np.empty((3, width), dtype=np.float32)
    neighbors[1, 0] = neighbors[2, -1] = np.inf
    for row in range(1, height):
        neighbors[0] = previous
        neighbors[1, 1:] = previous[:-1] + 1.5
        neighbors[2, :-1] = previous[1:] + 1.5
        choice = np.argmin(neighbors, axis=0)
        parents[row] = np.take((0, -1, 1), choice)
        previous = cost[row] + neighbors[choice, indices]

    path = np.empty(height, dtype=np.float32)
    column = int(np.argmin(previous))
    for row in range(height - 1, 0, -1):
        path[row] = column
        column += int(parents[row, column])
    path[0] = column
    # Subpixel transitions reduce jagged edges along diagonal paths.
    return cv2.GaussianBlur(path[:, None], (1, 9), 1.5).ravel()


def _match_shift(template: np.ndarray, search: np.ndarray) -> int | None:
    if template.std() < 2.5:
        return None
    scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED).ravel()
    peak = int(np.argmax(scores))
    others = scores[np.abs(np.arange(len(scores)) - peak) > 1]
    if scores[peak] < 0.88 or scores[peak] - others.max() < 0.05:
        return None
    if peak != SHIFT_LIMIT and scores[peak] - scores[SHIFT_LIMIT] < 0.10:
        return None
    return peak - SHIFT_LIMIT


def _tangent_shifts(
    first: np.ndarray, second: np.ndarray, path: np.ndarray, start_guard: int, end_guard: int,
) -> np.ndarray:
    height, width = first.shape[:2]
    radius, step = 20, 16
    guard = radius + 2 * SHIFT_LIMIT
    start, stop = start_guard + guard, height - end_guard - guard
    if stop <= start or width < 16:
        return np.zeros(height, dtype=np.float32)

    def crossing_gradient(patch: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0.9)
        return cv2.Sobel(blurred, cv2.CV_32F, 0, 1, scale=0.125)

    a, b = crossing_gradient(first), crossing_gradient(second)
    positions = np.arange(start, stop, step)
    shifts = np.zeros(len(positions), dtype=np.float32)
    for index, row in enumerate(positions):
        column = int(path[row])
        cols = slice(max(0, column - 12), min(width, column + 13))
        top, bottom = row - radius, row + radius + 1
        shift = _match_shift(a[top:bottom, cols], b[top - SHIFT_LIMIT:bottom + SHIFT_LIMIT, cols])
        if not shift:
            continue
        # Forward-backward matching avoids shifting textures that were reconstructed differently.
        top, bottom = top + shift, bottom + shift
        reverse = _match_shift(b[top:bottom, cols], a[top - SHIFT_LIMIT:bottom + SHIFT_LIMIT, cols])
        if reverse is not None and abs(shift + reverse) <= 1:
            shifts[index] = shift
    offsets = np.interp(np.arange(height), positions, shifts, left=0, right=0).astype(np.float32)
    return cv2.GaussianBlur(offsets[:, None], (1, 25), 5).ravel()


def find_seam(
    layout: Layout, first: Cell, second: Cell, a: np.ndarray, b: np.ndarray,
    coverage: np.ndarray | None,
    junctions: tuple[tuple[int | None, int | None], tuple[int | None, int | None]],
) -> SeamPath:
    region = seam_region(layout, first, second)
    left_right = first.row == second.row
    patch_a, patch_b = a[region.relative_to(layout.frame(first)).slices], b[region.relative_to(layout.frame(second)).slices]
    cost, mismatch = _cost(patch_a, patch_b)
    if coverage is not None:
        cost *= 0.15 + 0.85 * coverage.astype(np.float32) / 255
    if not left_right:
        cost = cost.T.copy()
        mismatch = mismatch.T
        patch_a, patch_b = patch_a.transpose(1, 0, 2), patch_b.transpose(1, 0, 2)
    start = region.top if left_right else region.left
    end = region.bottom if left_right else region.right
    origin = region.left if left_right else region.top
    extent = layout.size[1 if left_right else 0]
    guard = min(len(cost) // 2, layout.band(SEAM_MARGIN)[1 if left_right else 0] + 4)
    start_guard, end_guard = guard if start else 0, guard if end < extent else 0
    if not left_right:
        # The vertical branches are already anchored across the horizontal band.
        # The horizontal branches only need to meet where vertical weights still mix columns.
        start_x, end_x = (point[0] for point in junctions)
        if start_x is not None:
            start_guard = min(start_guard, abs(start_x - start) + int(COLOR_FEATHER) + 4)
        if end_x is not None:
            end_guard = min(end_guard, abs(end_x - end) + int(COLOR_FEATHER) + 4)
    center = cost.shape[1] // 2
    anchors = (point[0 if left_right else 1] for point in junctions)
    first_anchor, last_anchor = (center if anchor is None else anchor - origin for anchor in anchors)
    path = _minimum_path(cost, start_guard, end_guard, first_anchor, last_anchor)
    room = np.minimum(path, cost.shape[1] - path)
    nearby_edges = cv2.dilate(mismatch, np.ones((1, 21), dtype=np.uint8))
    local_error = nearby_edges[np.arange(len(path)), np.rint(path).astype(int)]
    local_error = cv2.GaussianBlur(local_error[:, None], (1, 21), 4).ravel()
    # Smooth exposure differences and blend less where contours disagree.
    feather = np.clip(COLOR_FEATHER / (1 + (local_error / 9) ** 2), 5, COLOR_FEATHER)
    feather[:start_guard] = COLOR_FEATHER
    if end_guard:
        feather[-end_guard:] = COLOR_FEATHER
    shifts = _tangent_shifts(patch_a, patch_b, path, start_guard, end_guard)
    return SeamPath(start, origin + path, room, feather, shifts)


def junction_region(layout: Layout, corner: Cell) -> Rect:
    core = layout.core(corner)
    x, y = layout.band(SEAM_MARGIN)
    return Rect(core.left - x, core.top - y, core.left + x, core.top + y)


def find_junction(patches: list[np.ndarray], area: Rect, coverage: np.ndarray | None) -> tuple[int, int]:
    top_left, top_right, bottom_left, bottom_right = patches
    visible = 1 if coverage is None else 0.15 + 0.85 * coverage.astype(np.float32) / 255

    def cumulative(first: np.ndarray, second: np.ndarray, axis: int) -> np.ndarray:
        cost = _cost(first, second)[0] * visible
        padding = ((1, 0), (0, 0)) if axis == 0 else ((0, 0), (1, 0))
        return np.pad(cost.cumsum(axis=axis), padding)

    north = cumulative(top_left, top_right, 0)
    south = cumulative(bottom_left, bottom_right, 0)
    west = cumulative(top_left, bottom_left, 1)
    east = cumulative(top_right, bottom_right, 1)

    def candidates(length: int) -> tuple[np.ndarray, np.ndarray]:
        limit = min(24, length // 4)
        offsets = np.arange(-limit, limit + 1)
        return length // 2 + offsets, 4 * (offsets / max(1, limit)) ** 2

    xs, x_bias = candidates(area.width)
    ys, y_bias = candidates(area.height)
    x, y = xs[None, :], ys[:, None]
    # Each branch uses the tile pair that contributes on its side of the intersection.
    vertical = (north[y, x] + south[-1, x] - south[y, x]) / area.height
    horizontal = (west[y, x] + east[y, -1] - east[y, x]) / area.width
    iy, ix = np.unravel_index(np.argmin(vertical + horizontal + x_bias + y_bias[:, None]), vertical.shape)
    return area.left + int(xs[ix]), area.top + int(ys[iy])
