"""Detect and repair discontinuities in the repetition of one canvas tile."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw

from .codex import Cancelled, CodexClient, CodexError
from .composite import LINEAR_RGB
from .document import box
from .files import reference_rgb, save_png
from .fill import _smoothstep, _to_srgb, generate_pattern_fill, render_fill
from .geometry import FRAME, Rect
from .periodic import periodic_crop, periodic_pieces

_OVERLAP = 128
_WRITABLE = FRAME - 2 * _OVERLAP
_STRIDE = _WRITABLE - _OVERLAP
_ANCHOR_BLEND = 24
_BINARY_COVERAGE = (0,) + (255,) * 255
_PREVIEW_SIZE = 960

_FIELDS = {
    "axis": {"type": "string", "enum": ["vertical", "horizontal"]},
    "start": {"type": "number"},
    "end": {"type": "number"},
    "half_width": {"type": "number"},
}
_SCHEMA = {
    "type": "object", "required": ["segments"], "additionalProperties": False,
    "properties": {"segments": {"type": "array", "items": {
        "type": "object", "properties": _FIELDS, "required": list(_FIELDS), "additionalProperties": False,
    }}},
}
_DETECTION_PROMPT = """Find visual discontinuities due to repeating this image as a tile.
Image 1 is the clean tile after a circular shift, image 2 marks the potential original joins
on the same tile, and image 3 repeats the original tile in a 3 by 3 grid. Inspect both sides
of the marked joins for abrupt changes in motifs, texture, color, or transparency. Vertical
joins compare left to right and horizontal joins compare above to below. Report only the
segments that need repair, ignoring natural edges that continue across a join and guide ink.
Report a marked join across its full span if tone or texture changes abruptly along it, even
if the jump is subtle. Include the whole interrupted part of a motif when a join cuts one.
Use start and end along the join (top to bottom for vertical, left to right for horizontal),
with 0 at the first edge of image 1 and 1000 at the opposite edge. Use half_width for the
distance the edit should extend to either side of the join, in thousandths of the tile size
perpendicular to the join. Use an empty segments array when both joins already look continuous."""


def _check(cancelled: Event):
    if cancelled.is_set():
        raise Cancelled("Pattern repair canceled.")


def seam_mask(size: tuple[int, int], offset: tuple[int, int], data: dict) -> Image.Image:
    mask = Image.new("L", size)
    width, height = size
    dx, dy = offset
    segments = data.get("segments") if isinstance(data, dict) else None
    if not isinstance(segments, list) or len(segments) > 128:
        raise ValueError("Invalid seam segments.")
    for segment in segments:
        if not isinstance(segment, dict) or set(segment) != _FIELDS.keys():
            raise ValueError("Invalid seam segment.")
        axis = segment["axis"]
        start, end, half_width = (segment[key] for key in ("start", "end", "half_width"))
        if (
            axis not in ("vertical", "horizontal")
            or any(type(n) not in (int, float) or not 0 <= n <= 1000 for n in (start, end, half_width))
            or not start < end or not half_width
        ):
            raise ValueError("Invalid seam coordinates.")
        vertical = axis == "vertical"
        along, across, center = (height, width, dx) if vertical else (width, height, dy)
        if not center:  # A one-pixel dimension already joins itself.
            continue
        first = int(start * along / 1000)
        last = max(first + 1, int(np.ceil(end * along / 1000)))
        half = max(1, round(half_width * across / 1000))
        low, high = max(0, center - half), min(across, center + half)
        bounds = (low, first, high, last) if vertical else (first, low, last, high)
        mask.paste(255, bounds)
    return mask


def detect_pattern_seams(
    original: Image.Image, shifted: Image.Image, offset: tuple[int, int], executable: Path, *,
    cancelled: Event, client_factory=CodexClient,
) -> Image.Image:
    _check(cancelled)
    detail = shifted if max(shifted.size) <= FRAME else shifted.resize(
        tuple(max(1, round(n * FRAME / max(shifted.size))) for n in shifted.size), Image.Resampling.LANCZOS,
    )
    clean = reference_rgb(detail)
    marked = clean.copy()
    draw = ImageDraw.Draw(marked)
    dx, dy = offset
    if dx:
        x = min(marked.width - 1, round(dx * marked.width / shifted.width))
        draw.line((x, 0, x, marked.height - 1), fill="#ff45d6", width=2)
    if dy:
        y = min(marked.height - 1, round(dy * marked.height / shifted.height))
        draw.line((0, y, marked.width - 1, y), fill="#ff45d6", width=2)
    scale = min(1, _PREVIEW_SIZE / (3 * max(original.size)))
    size = tuple(max(1, round(n * scale)) for n in original.size)
    tile = reference_rgb(original.resize(size, Image.Resampling.LANCZOS))
    repeat = periodic_crop(tile, Rect(0, 0, tile.width * 3, tile.height * 3))
    with TemporaryDirectory(prefix="regimagen-pattern-") as folder:
        directory = Path(folder)
        references = tuple(directory / name for name in ("tile.png", "joins.png", "repeat.png"))
        for image, path in zip((clean, marked, repeat), references):
            _check(cancelled)
            save_png(image, path)
        with client_factory(executable, directory.parent, cancelled=cancelled, image_generation=False) as client:
            data = client.analyze_images(_DETECTION_PROMPT, references, _SCHEMA, subject="seam detection")
    _check(cancelled)
    try:
        detected = seam_mask(shifted.size, offset, data)
    except ValueError as error:
        raise CodexError("Codex returned invalid seam coordinates. Detect again or adjust the mask.") from error
    if original.mode == "RGBA":
        detected = ImageChops.lighter(detected, _alpha_seam_mask(shifted, offset))
    elif detected.getbbox():
        # Leave room inside the proposed mask to join reconstructed surface details to the source.
        radius = min(16, max(2, round(min(original.size) * 0.025)))
        detected = _expand_periodic_mask(detected, radius, cancelled)
    return detected


def _expand_periodic_mask(mask: Image.Image, radius: int, cancelled: Event) -> Image.Image:
    width, height = mask.size
    expanded = Image.new("L", mask.size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    rows = max(1, 1_000_000 // (width + 2 * radius))
    for top in range(0, height, rows):
        _check(cancelled)
        bottom = min(top + rows, height)
        area = Rect(0, top, width, bottom).expand(radius, radius)
        padded = np.asarray(periodic_crop(mask, area))
        dilated = cv2.dilate(padded, kernel)
        expanded.paste(Image.fromarray(dilated[radius:radius + bottom - top, radius:radius + width]), (0, top))
    return expanded if expanded.getextrema()[0] == 0 else mask


def _alpha_seam_mask(shifted: Image.Image, offset: tuple[int, int]) -> Image.Image:
    """Include interrupted silhouettes that the visual analysis can miss."""
    alpha = shifted.getchannel("A")
    mask = Image.new("L", shifted.size)
    for axis, center in enumerate(offset):
        if not center:
            continue
        extent, length = shifted.size[axis], shifted.size[1 - axis]
        margin = min(48, max(8, round(extent * 0.12)))
        scan = min(2 * margin, center, extent - center)
        side = np.arange(-scan, scan)
        strip = (alpha.crop((center - scan, 0, center + scan, length)) if axis == 0
                 else alpha.crop((0, center - scan, length, center + scan)).transpose(Image.Transpose.TRANSPOSE))
        pixels = np.asarray(strip).astype(np.int16)
        jump = abs(pixels[:, scan] - pixels[:, scan - 1])
        # A cut has a sustained opacity step across the join, beyond nearby normal changes.
        nearby = np.maximum(abs(pixels[:, scan - 1] - pixels[:, max(0, scan - 3)]),
                            abs(pixels[:, scan] - pixels[:, min(2 * scan - 1, scan + 2)]))
        marked = (jump >= 48) & (jump >= nearby + 32)
        origin = int(np.argmin(marked))
        marked = np.roll(marked, -origin)
        edges = np.flatnonzero(np.diff(np.r_[False, marked, False]))
        for first, last in zip(edges[::2], edges[1::2]):
            if last - first < 3:
                continue
            # Include the rest of the nearby silhouette plus space for the contour to close.
            padding = max(4, min(margin, (last - first) // 2))
            first += origin - padding
            last = min(first + length, last + origin + padding)
            rows = np.arange(first, last) % length
            visible = np.any(pixels[rows] >= 32, axis=0)
            columns = side[visible]
            low = center + max(-scan, min(-margin, int(columns[0]) - 4))
            high = center + min(scan, max(margin, int(columns[-1]) + 5))
            area = Rect(low, first, high, last) if axis == 0 else Rect(first, low, last, high)
            for part, _ in periodic_pieces(area, shifted.size):
                mask.paste(255, box(part))
    return mask


def _axis_windows(length: int):
    count = 1 if length <= _WRITABLE else (length + _STRIDE - 1) // _STRIDE
    spacing = (length + count - 1) // count
    width = min(length, spacing + _OVERLAP)
    for i in range(count):
        # Anchor the first window on the seam and distribute the rest around the period.
        center = length // 2 + (i * length) // count
        left = center - width // 2
        yield left, left + width


def _work_windows(size: tuple[int, int]):
    xs, ys = (_axis_windows(n) for n in size)
    xs = tuple(xs)
    for top, bottom in ys:
        for left, right in xs:
            yield Rect(left, top, right, bottom)


def _window_selection(mask: Image.Image, repaired: Image.Image, area: Rect) -> Image.Image:
    approved = periodic_crop(mask, area)
    done = np.asarray(periodic_crop(repaired, area))
    if not np.any(done):
        return approved
    distance = cv2.distanceTransform(done, cv2.DIST_L2, cv2.DIST_MASK_5)
    return Image.fromarray(np.where(distance <= _ANCHOR_BLEND, np.asarray(approved), 0).astype(np.uint8))


def _polish_join(
    result: Image.Image, mask: Image.Image, axis: int, center: int, cancelled: Event,
):
    """Spread a residual one-pixel step into the approved area on both sides."""
    extent, length = result.size[axis], result.size[1 - axis]
    radius = min(5, center, extent - center)
    if radius < 2:
        return
    for start in range(0, length, 4096):
        _check(cancelled)
        end = min(start + 4096, length)
        low, high = center - radius, center + radius
        bounds = (low, start, high, end) if axis == 0 else (start, low, end, high)
        tile = result.crop(bounds)
        approved = np.asarray(mask.crop(bounds)).astype(np.float32) / 255
        pixels = np.array(tile)
        if axis:
            pixels, approved = pixels.transpose(1, 0, 2), approved.T
        selected = approved != 0
        both = selected[:, radius - 1] & selected[:, radius]
        if not np.any(both):
            continue
        alpha = pixels[..., 3] / np.float32(255) if result.mode == "RGBA" else np.ones(pixels.shape[:2], np.float32)
        # Assess the transition in visible color and alpha, ignoring hidden RGB.
        visible = pixels[..., :3].astype(np.float32) * alpha[..., None]
        samples = np.dstack((visible, alpha * 255))
        left, right = radius - 1, radius
        flagged = _suspect_step(samples[:, left - 1], samples[:, left], samples[:, right], samples[:, right + 1])
        if not np.any(both & flagged):
            continue
        support = (both & flagged).astype(np.float32)
        support = np.maximum(support, cv2.GaussianBlur(support[:, None], (1, 5), 0.9).reshape(-1))
        # Only modify the contiguous mask run around the join, including soft selections.
        distance = np.arange(radius, dtype=np.float32)
        premultiplied = LINEAR_RGB[pixels[..., :3]] * alpha[..., None]
        jump = premultiplied[:, right] - premultiplied[:, left]
        alpha_jump = alpha[:, right] - alpha[:, left]
        for side, sign in ((slice(left, None, -1), 1), (slice(right, None), -1)):
            run = np.cumprod(selected[:, side], axis=1)
            width = run.sum(axis=1)
            t = np.clip(1 - distance / np.maximum(width[:, None] - 1, 1), 0, 1)
            weight = (0.47 * sign * support[:, None]) * _smoothstep(t) * run * approved[:, side]
            use = weight != 0
            new_alpha = np.clip(alpha[:, side] + weight * alpha_jump[:, None], 0, 1)
            new_rgb = premultiplied[:, side] + weight[..., None] * jump[:, None, :]
            denominator = new_alpha[..., None]
            straight = np.divide(new_rgb, denominator, out=np.zeros_like(new_rgb), where=denominator > 0)
            pixels[:, side, :3][use] = _to_srgb(straight[use])
            if result.mode == "RGBA":
                pixels[:, side, 3][use] = np.rint(255 * new_alpha[use]).astype(np.uint8)
        result.paste(Image.fromarray(pixels.transpose(1, 0, 2) if axis else pixels), bounds[:2])


def repair_pattern(
    shifted: Image.Image, mask: Image.Image, executable: Path, *, cancelled: Event,
    progress, client_factory=CodexClient,
) -> tuple[Image.Image, tuple[str, ...]]:
    _check(cancelled)
    if mask.mode != "L" or mask.size != shifted.size or not mask.getbbox():
        raise ValueError("Select the pattern seams before generating.")
    if mask.getextrema()[0] != 0:
        raise ValueError("Leave some unselected pixels to guide the pattern repair.")
    size = shifted.size
    windows = []
    for window in _work_windows(size):
        _check(cancelled)
        if any(mask.crop(box(piece)).getbbox() for piece, _ in periodic_pieces(window, size)):
            windows.append(window)
    result = shifted.copy()
    warnings = set()
    repaired = Image.new("L", size)
    for index, window in enumerate(windows, 1):
        _check(cancelled)
        progress(index - 1, len(windows))
        _check(cancelled)
        selected = periodic_crop(mask, window)
        if ImageChops.subtract(selected, periodic_crop(repaired, window)).getbbox():
            extent = max(window.width, window.height)
            margin = min(_OVERLAP, max(8, extent // 4))
            side = extent + 2 * margin
            left = window.left - (side - window.width) // 2
            top = window.top - (side - window.height) // 2
            frame = Rect(left, top, left + side, top + side)
            local = window.relative_to(frame)
            base = periodic_crop(result, frame)
            guide = Image.new("L", base.size)
            continuation = min(_ANCHOR_BLEND, margin // 2)
            guided = window.expand(continuation, continuation)
            guide.paste(_window_selection(mask, repaired, guided), (guided.left - left, guided.top - top))
            candidate = generate_pattern_fill(base, guide, executable, cancelled=cancelled, client_factory=client_factory)
            preview = render_fill(base, guide, candidate, edit_alpha=base.mode == "RGBA", pattern=True)
            if preview.warning:
                warnings.add(preview.warning)
            core = base.crop(box(local))
            core.paste(preview.patch, (preview.origin[0] - local.left, preview.origin[1] - local.top))
            for area, (x, y) in periodic_pieces(window, size):
                part = x, y, x + area.width, y + area.height
                result.paste(core.crop(part), (area.left, area.top))
                repaired.paste(255, box(area), selected.crop(part).point(_BINARY_COVERAGE))
        progress(index, len(windows))
    for axis, extent in enumerate(size):
        _polish_join(result, mask, axis, extent // 2, cancelled)
    _check(cancelled)
    return result, tuple(sorted(warnings))


@dataclass(frozen=True)
class SeamWarning:
    axis: str
    start: int
    end: int


def _strip_samples(image, axis, pos):
    bounds = (pos, 0, pos + 1, image.height) if axis == "vertical" else (0, pos, image.width, pos + 1)
    pixels = np.asarray(image.crop(bounds)).reshape(-1, len(image.getbands())).astype(np.float32)
    if image.mode == "RGBA":
        pixels[:, :3] *= pixels[:, 3:] / 255
    return pixels


def check_pattern_joins(tile: Image.Image, *, cancelled: Event) -> tuple[SeamWarning, ...]:
    """Compare each cyclic step with the natural color and alpha steps on its sides."""
    warnings = []
    for axis, extent in zip(("vertical", "horizontal"), tile.size):
        _check(cancelled)
        if extent < 2:
            continue
        first, last = (_strip_samples(tile, axis, pos) for pos in (0, extent - 1))
        after, before = (_strip_samples(tile, axis, pos) for pos in (1, extent - 2))
        suspicious = _suspect_step(before, last, first, after)
        edges = np.flatnonzero(np.diff(np.r_[False, suspicious, False]))
        warnings.extend(SeamWarning(axis, int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]))
    return tuple(warnings)


def _suspect_step(before, left, right, after):
    jump = right - left
    start, end = left - before, after - right
    normal = np.maximum(abs(start), abs(end))
    expected = (start + end) / 2
    return np.any((abs(jump) > 12 + 1.5 * normal) | (abs(jump - expected) > 28 + normal), axis=1)
