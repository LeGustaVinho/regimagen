"""Refine semantic removal contours against the original pixels."""

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw


def _distance(mask: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5)


def refine_matte(image: Image.Image, contour: Image.Image, radius: int) -> Image.Image:
    """Fit a narrow trimap around a supplied silhouette and retain thin details."""
    original = np.asarray(contour) != 0
    pixels = np.asarray(image)
    inside, outside = _distance(original), _distance(~original)
    foreground = inside > radius
    background = outside > radius
    if not background.any():
        return contour
    band = ~foreground & ~background
    edge_colors = pixels[band]
    if edge_colors.size == 0 or np.ptp(edge_colors, axis=0).max() < 8:
        return contour

    labels = np.where(original, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    labels[foreground] = cv2.GC_FGD
    labels[background] = cv2.GC_BGD
    # Keep the center of fine structures near the semantic outline. They may
    # be too small to be represented in GrabCut's foreground color model.
    neighborhood = cv2.dilate(inside, np.ones((5, 5), np.uint8))
    fine = original & (neighborhood <= 2.5) & (inside >= neighborhood - 0.25)
    labels[fine] = cv2.GC_FGD
    models = [np.zeros((1, 65), np.float64) for _ in range(2)]
    fitted = original
    if foreground.any():
        try:
            cv2.grabCut(pixels, labels, None, *models, 2, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return contour
        fitted = labels % 2 != 0
    if not fitted.any() or fitted.all():
        return contour

    # Sample subpixel coverage only at the edge. Keep fully covered fine
    # strokes and their interiors intact.
    matte = cv2.GaussianBlur(fitted.astype(np.float32), (3, 3), 0.66)
    matte[fine | (_distance(fitted) >= 2)] = 1
    matte[background] = 0
    return Image.fromarray(np.rint(matte * 255).astype(np.uint8))


def clean_matte_colors(image: Image.Image, matte: Image.Image) -> Image.Image:
    """Borrow nearby interior color at semitransparent edges of an opaque source."""
    alpha = np.asarray(matte)
    rgb = np.asarray(image.convert("RGB"))
    edge = (alpha > 0) & (alpha < 255)
    if not edge.any():
        return image
    solid = (alpha == 255).astype(np.float32)
    support = cv2.GaussianBlur(solid, (7, 7), 1.3)
    color = cv2.GaussianBlur(rgb.astype(np.float32) * solid[..., None], (7, 7), 1.3)
    weight = ((255 - alpha[edge]) / 255 * np.minimum(1, support[edge] * 5))[:, None]
    mixed = rgb[edge] * (1 - weight) + color[edge] / np.maximum(support[edge, None], 0.001) * weight
    result = rgb.copy()
    result[edge] = np.rint(mixed).astype(np.uint8)
    return Image.fromarray(result).convert("RGBA")


def fit_wire(image: Image.Image, path: list[tuple[int, int]], width: float) -> Image.Image:
    """Follow a proposed thin stroke using the nearby cross-section contrast."""
    vertices = np.asarray(path, np.float32)
    delta = np.diff(vertices, axis=0)
    length = np.linalg.norm(delta, axis=1)
    keep = np.r_[True, length > 0]
    vertices = vertices[keep]
    if len(vertices) < 2:
        return Image.new("L", image.size)
    delta = np.diff(vertices, axis=0)
    length = np.linalg.norm(delta, axis=1)
    distance = np.r_[0, np.cumsum(length)]
    sample = np.linspace(0, distance[-1], max(2, min(8192, round(distance[-1] * 2))))
    x, y = (np.interp(sample, distance, vertices[:, axis]) for axis in (0, 1))
    nx, ny = -np.gradient(y), np.gradient(x)
    norm = np.maximum(np.hypot(nx, ny), 0.01)
    nx, ny = nx / norm, ny / norm
    radius = max(2, min(6, round(width * 1.5)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    color = np.asarray(image, dtype=np.float32)

    def sample_colors(offset):
        return cv2.remap(color, (x[:, None] + nx[:, None] * offset).astype(np.float32),
                         (y[:, None] + ny[:, None] * offset).astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    centers = sample_colors(offsets)
    cross = max(1.5, width)
    neighbors = (sample_colors(offsets - cross) + sample_colors(offsets + cross)) / 2
    difference = centers - neighbors
    contrast = np.linalg.norm(difference, axis=2)
    # Prefer a consistent ridge close to the supplied path.
    response = cv2.GaussianBlur(contrast, (1, 13), 2) - 2.5 * abs(offsets)
    best = np.argmax(response, axis=1)
    movement = cv2.GaussianBlur(offsets[best, None], (1, 13), 2)[:, 0]
    confident = response[np.arange(len(sample)), best] >= 10
    movement *= confident
    factor = 4
    result = Image.new("L", image.size)
    vertices = list(zip(x + nx * movement, y + ny * movement))
    for start in range(0, len(vertices) - 1, 256):
        segment = vertices[start:start + 257]
        xs, ys = zip(*segment)
        bounds = (max(0, int(min(xs)) - 8), max(0, int(min(ys)) - 8),
                  min(image.width, int(max(xs)) + 9), min(image.height, int(max(ys)) + 9))
        left, top, right, bottom = bounds
        points = [(round((a - left) * factor), round((b - top) * factor)) for a, b in segment]
        size = (right - left, bottom - top)
        matte = Image.new("L", (size[0] * factor, size[1] * factor))
        ImageDraw.Draw(matte).line(points, fill=255, width=max(3, round(width * factor)), joint="curve")
        local = matte.resize(size, Image.Resampling.LANCZOS).point([0] * 12 + list(range(12, 256)))
        result.paste(ImageChops.lighter(result.crop(bounds), local), (left, top))
    return result


def refine_occlusion(
    image: Image.Image, contour: Image.Image, subject: Image.Image, radius: int, *,
    keep_points: tuple[tuple[int, int], ...] = (), object_points: tuple[tuple[int, int], ...] = (),
) -> Image.Image:
    """Fit the foreground using exposed occluder and object pixels as distinct seeds."""
    proposed = np.asarray(contour) != 0
    object_pixels = np.asarray(subject) >= 128
    inside = _distance(proposed)
    outside = _distance(~proposed)
    nearby = outside <= radius
    clear_foreground = proposed & ~object_pixels & (inside >= 2)
    labels = np.full(proposed.shape, cv2.GC_BGD, np.uint8)
    labels[nearby] = cv2.GC_PR_BGD
    labels[proposed] = cv2.GC_PR_FGD
    labels[clear_foreground] = cv2.GC_FGD
    # Adjacent subject colors supply background samples for the foreground
    # matte. They matter when foliage covers a small part of a wall or roof.
    subject_only = object_pixels & ~proposed
    labels[subject_only & (outside >= 3)] = cv2.GC_BGD
    # Pixels inside both preliminary masks are ambiguous.
    overlap = proposed & object_pixels
    labels[overlap & (inside < 2)] = cv2.GC_PR_BGD
    for x, y in keep_points:
        if nearby[y, x]:
            labels[y, x] = cv2.GC_FGD
    for x, y in object_points:
        labels[y, x] = cv2.GC_BGD
    if not keep_points and np.count_nonzero(clear_foreground) < 12:
        return Image.new("L", contour.size)
    if not np.any(labels == cv2.GC_FGD):
        return Image.new("L", contour.size)
    if not np.any(labels == cv2.GC_BGD):
        return Image.new("L", contour.size)
    models = [np.zeros((1, 65), np.float64) for _ in range(2)]
    try:
        cv2.grabCut(np.asarray(image), labels, None, *models, 2, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return Image.new("L", contour.size)
    fitted = labels % 2 != 0
    # Keep only regions connected to the visible foreground. This prevents
    # similar stone or paint colors in the subject from becoming protection.
    _, components = cv2.connectedComponents(fitted.astype(np.uint8), connectivity=8)
    foreground = np.unique(components[labels == cv2.GC_FGD])
    protected = np.isin(components, foreground)
    return Image.fromarray(protected.astype(np.uint8) * 255)


def discard_islands(matte: Image.Image, *, keep: Image.Image) -> Image.Image:
    bounds = matte.getbbox()
    if bounds is None:
        return matte
    cropped = matte.crop(bounds)
    alpha = np.asarray(cropped)
    count, labels, sizes, _ = cv2.connectedComponentsWithStats((alpha > 80).astype(np.uint8), connectivity=8)
    if count <= 2:
        return matte
    largest = sizes[1:, cv2.CC_STAT_AREA].max()
    kept = sizes[:, cv2.CC_STAT_AREA] >= max(6, min(70, largest * 0.002))
    kept[np.unique(labels[np.asarray(keep.crop(bounds)) != 0])] = True
    kept[0] = False
    support = cv2.dilate(kept[labels].astype(np.uint8), np.ones((3, 3), np.uint8))
    result = Image.new("L", matte.size)
    result.paste(Image.fromarray(alpha * support), bounds[:2])
    return result


def fill_silhouette_gaps(matte: Image.Image, outline: Image.Image, intentional_gaps: Image.Image) -> Image.Image:
    """Bridge narrow cuts inside the initial object where both sides were confirmed."""
    bounds = matte.getbbox()
    if bounds is None:
        return matte
    alpha = np.array(matte.crop(bounds))
    coarse = np.asarray(outline.crop(bounds)) > 0
    gaps = np.asarray(intentional_gaps.crop(bounds)) > 0
    body = cv2.erode((alpha >= 128).astype(np.uint8), np.ones((5, 5), np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx((alpha >= 128).astype(np.uint8), cv2.MORPH_CLOSE, kernel) != 0
    coarse_interior = _distance(coarse) >= 4
    supported = (_distance(body == 0) <= 6) & coarse_interior
    alpha[closed & supported & ~gaps] = 255
    holes = np.pad(alpha < 128, 1, constant_values=True)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=4)
    stable = np.zeros(len(stats), dtype=bool)
    areas = stats[:, cv2.CC_STAT_AREA]
    stable[1:] = areas[1:] <= 400
    stable[labels[0, 0]] = False
    labels = labels[1:-1, 1:-1]
    stable[np.unique(labels[gaps])] = False
    # Gaps near the silhouette are intentional unless the original outline
    # also puts them well inside the body.
    stable[np.unique(labels[~coarse_interior])] = False
    alpha[stable[labels]] = 255
    result = matte.copy()
    result.paste(Image.fromarray(alpha), bounds[:2])
    return result
