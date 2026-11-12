"""Locate removal targets with Codex and refine their visible pixels with EfficientSAM."""

import math
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw

from .codex import Cancelled, CodexClient, CodexError
from .document import box
from .files import reference_rgb, save_png
from .fill import _paint_selection, _selection_coverage, selection_has_visible_pixels
from .geometry import FRAME, Rect


@dataclass(frozen=True)
class RemovalTarget:
    label: str
    mask: Image.Image
    origin: tuple[int, int]
    selected: bool = True
    protected: bool = False
    parent: int | None = None


@dataclass(frozen=True)
class _Location:
    label: str
    bounds: Rect
    foreground: tuple[tuple[int, int], ...]
    background: tuple[tuple[int, int], ...]
    contours: tuple[tuple[tuple[int, int], ...], ...]
    uncertain: bool


def _object_schema(fields: dict) -> dict:
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


_POINT_SCHEMA = {"type": "array", "items": {"type": "number"}}
_LOCATION_FIELDS = {
    "label": {"type": "string"},
    "view": {"type": "string", "enum": ["detail", "overview"]},
    "bounds": _POINT_SCHEMA,
    "foreground": {"type": "array", "items": _POINT_SCHEMA},
    "background": {"type": "array", "items": _POINT_SCHEMA},
    "contours": {"type": "array", "items": {"type": "array", "items": _POINT_SCHEMA}},
    "uncertain": {"type": "boolean"},
}
_LOCATION_SCHEMA = _object_schema(_LOCATION_FIELDS)
_DETECTION_SCHEMA = _object_schema({
    "targets": {
        "type": "array",
        "items": _object_schema({**_LOCATION_FIELDS, "effects": {"type": "array", "items": _LOCATION_SCHEMA}}),
    },
    "protected": {"type": "array", "items": _LOCATION_SCHEMA},
})

_DETECTION_PROMPT = """Identify the foreground objects the user is most likely to mean to erase.
Image 1 (detail) shows the area around the lasso. Image 2 (overview) shows the larger scene.
Both mark the same lasso with a high-contrast pink or dark outline. Look at the actual lasso
shape, not just its bounding rectangle. Include objects that intersect the marked region,
even if the mark covers only part of one, and describe the whole visible object. If the detail
view clips that object, use overview for its geometry. Use detail otherwise for precision.
Choose objects, not the background surface. Return an empty targets list when the marked area
contains no removable object. Set uncertain=true for an ambiguous or barely touched target.
Only include shadows and reflections that belong to an identified target as separate effects.
In protected, mark nearby objects that overlap the target or cross in front of it, with their
visible outlines, so removing an occluded target does not erase an unrelated foreground object.

For each entry, label it briefly, select its coordinate view, and give a tight bounds array
[left, top, right, bottom]. All coordinates are in a 0 to 1000 space relative to that view,
with [0, 0] at top left and [1000, 1000] at bottom right. Do not return source-image coordinates.
Give two or three foreground [x,y] points safely inside the visible object, in separate visible
pieces where possible. Give up to two background [x,y] points on adjacent areas to preserve.
Give approximate visible outlines as contours, an array of polygons, each polygon an array
of at least three [x,y] vertices. Use multiple polygons for visible pieces separated by a
foreground obstruction. Outline shadows and reflections separately from their object.
Do not change or generate any image."""


def _views(base: Image.Image, selection: Rect) -> dict[str, Rect]:
    canvas = Rect(0, 0, *base.size)
    margin = max(64, max(selection.width, selection.height) // 2)
    detail = selection.expand(margin, margin).intersect(canvas)
    assert detail is not None
    overview = canvas
    if base.mode == "RGBA" and (visible := base.getbbox()) is not None:
        overview = Rect(
            min(visible[0], selection.left), min(visible[1], selection.top),
            max(visible[2], selection.right), max(visible[3], selection.bottom),
        )
    return {"detail": detail, "overview": overview}


def _guide(base: Image.Image, mask: Image.Image, view: Rect) -> Image.Image:
    scale = FRAME / max(view.width, view.height)
    size = max(1, round(view.width * scale)), max(1, round(view.height * scale))
    guide = reference_rgb(base.resize(size, Image.Resampling.LANCZOS, box=box(view))).copy()
    filled = _selection_coverage(mask, view, size)
    inner = cv2.erode(filled, np.ones((5, 5), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    _paint_selection(guide, filled != inner)
    return guide


def _coordinates(value: object, length: int) -> list:
    if (not isinstance(value, list) or len(value) != length or any(
        type(number) not in (int, float) or not 0 <= number <= 1000 for number in value
    )):
        raise ValueError("Invalid coordinates")
    return value


def _location(data: dict, views: dict[str, Rect]) -> _Location:
    view = views[data["view"]]
    left, top, right, bottom = _coordinates(data["bounds"], 4)
    if left >= right or top >= bottom:
        raise ValueError("Empty bounds")
    bounds = Rect(
        view.left + math.floor(left * view.width / 1000), view.top + math.floor(top * view.height / 1000),
        view.left + math.ceil(right * view.width / 1000), view.top + math.ceil(bottom * view.height / 1000),
    )

    def points(values):
        if not isinstance(values, list) or len(values) > 128:
            raise ValueError("Invalid points")
        result = []
        for value in values:
            x, y = _coordinates(value, 2)
            result.append((
                view.left + min(view.width - 1, int(x * view.width / 1000)),
                view.top + min(view.height - 1, int(y * view.height / 1000)),
            ))
        return tuple(result)

    contours = data["contours"]
    if not isinstance(contours, list) or len(contours) > 32:
        raise ValueError("Invalid contours")
    label = data["label"]
    if not isinstance(label, str) or not label.strip() or type(data["uncertain"]) is not bool:
        raise ValueError("Invalid object description")
    return _Location(
        " ".join(label.split())[:120], bounds,
        points(data["foreground"]), points(data["background"]),
        tuple(polygon for contour in contours if len(polygon := points(contour)) >= 3),
        data["uncertain"],
    )


def _inside(
    points: tuple[tuple[int, int], ...], area: Rect, *, bounds: Rect | None = None,
) -> tuple[tuple[int, int], ...]:
    bounds = bounds or area
    return tuple((x - area.left, y - area.top) for x, y in points
                 if bounds.left <= x < bounds.right and bounds.top <= y < bounds.bottom)


class _EfficientSAM:
    def __init__(self):
        from .remove_model import model_path

        self.net = cv2.dnn.readNetFromONNX(str(model_path()))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def segment(
        self, image: np.ndarray, bounds: Rect, foreground: tuple, background: tuple,
    ) -> np.ndarray | None:
        for mask, _ in self.candidates(image, foreground, background, bounds=bounds):
            return mask
        return None

    def candidates(self, image: np.ndarray, foreground: tuple, background: tuple = (), *, bounds: Rect | None = None):
        if not foreground:
            return
        height, width = image.shape[:2]
        scale = FRAME / max(width, height)
        scaled_width, scaled_height = max(1, round(width * scale)), max(1, round(height * scale))
        resized = cv2.resize(image, (scaled_width, scaled_height))
        frame = cv2.copyMakeBorder(resized, 0, FRAME - scaled_height, 0, FRAME - scaled_width, cv2.BORDER_REPLICATE)
        points = np.zeros((1, 1, 6, 2), np.float32)
        labels = np.full((1, 1, 6, 1), -1, np.float32)
        prompts = [(point, 1) for point in foreground[:3]]
        if bounds is not None:
            prompts[:0] = [((bounds.left, bounds.top), 2), ((bounds.right - 1, bounds.bottom - 1), 3)]
        prompts.extend((point, 0) for point in background[:6 - len(prompts)])
        for index, ((x, y), label) in enumerate(prompts):
            points[0, 0, index] = x * scaled_width / width, y * scaled_height / height
            labels[0, 0, index, 0] = label
        self.net.setInput(cv2.dnn.blobFromImage(frame, 1 / 255), "batched_images")
        self.net.setInput(points, "batched_point_coords")
        self.net.setInput(labels, "batched_point_labels")
        logits, scores = self.net.forward(["output_masks", "iou_predictions"])
        for index in np.argsort(scores[0, 0])[::-1]:
            mask = cv2.resize(
                (logits[0, 0, index, :scaled_height, :scaled_width] >= 0).astype(np.uint8),
                (width, height), interpolation=cv2.INTER_NEAREST,
            )
            if all(mask[y, x] for x, y in foreground) and not any(mask[y, x] for x, y in background):
                yield mask, float(scores[0, 0, index])


def _contour_mask(location: _Location, area: Rect) -> np.ndarray:
    image = Image.new("L", (area.width, area.height))
    draw = ImageDraw.Draw(image)
    for polygon in location.contours:
        draw.polygon([(x - area.left, y - area.top) for x, y in polygon], fill=1)
    return np.array(image)


def _grabcut(image: np.ndarray, coarse: np.ndarray, foreground: tuple, background: tuple) -> np.ndarray | None:
    if not coarse.any() or coarse.all() or not foreground:
        return None
    # Thin or translucent structures do not always provide usable GrabCut samples.
    # Keep their reviewed proposal when the local fit cannot represent it.
    margin = max(2, round(min(coarse.shape) * 0.025))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    labels = cv2.dilate(coarse, kernel) * cv2.GC_PR_BGD
    labels[coarse != 0] = cv2.GC_PR_FGD
    for x, y in foreground:
        labels[y, x] = cv2.GC_FGD
    for x, y in background:
        labels[y, x] = cv2.GC_BGD
    model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(image, labels, None, model.copy(), model.copy(), 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    return ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)).astype(np.uint8)


def _refine(
    base: Image.Image, location: _Location, segmenter, *, use_contour: bool = False,
) -> tuple[Image.Image, tuple[int, int], bool] | None:
    canvas = Rect(0, 0, *base.size)
    margin = max(8, math.ceil(max(location.bounds.width, location.bounds.height) * 0.15))
    area = location.bounds.expand(margin, margin).intersect(canvas)
    assert area is not None
    source = base.crop(box(area))
    pixels = np.asarray(reference_rgb(source))
    foreground = _inside(location.foreground, area, bounds=location.bounds)
    background = _inside(location.background, area)
    bounds = location.bounds.relative_to(area)
    coarse = _contour_mask(location, area)
    refined = None
    if segmenter is not None and not (use_contour and coarse.any()):
        try:
            refined = segmenter.segment(pixels, bounds, foreground, background)
        except cv2.error:
            pass
    if refined is None and not use_contour:
        # GrabCut only needs a local crop. Restrict its work on very large sources.
        scale = min(1, FRAME / max(area.width, area.height))
        if scale < 1:
            size = max(1, round(area.width * scale)), max(1, round(area.height * scale))

            def scaled(points):
                return tuple((min(size[0] - 1, int(x * scale)), min(size[1] - 1, int(y * scale))) for x, y in points)

            fit = _grabcut(
                cv2.resize(pixels, size, interpolation=cv2.INTER_AREA),
                cv2.resize(coarse, size, interpolation=cv2.INTER_NEAREST), scaled(foreground), scaled(background),
            )
            if fit is not None:
                refined = cv2.resize(fit, source.size, interpolation=cv2.INTER_NEAREST)
        else:
            refined = _grabcut(pixels, coarse, foreground, background)
    used_coarse = refined is None
    if used_coarse:
        refined = coarse
    if source.mode == "RGBA":
        refined &= np.asarray(source.getchannel("A")) != 0
    mask = Image.fromarray(refined * 255)
    region = mask.getbbox()
    if region is None:
        return None
    return mask.crop(region), (area.left + region[0], area.top + region[1]), used_coarse


def _entries(data: dict, key: str) -> list:
    entries = data.get(key)
    if not isinstance(entries, list) or len(entries) > 32 or not all(isinstance(entry, dict) for entry in entries):
        raise ValueError("Invalid object list")
    return entries


def _parse_detection(data: dict, views: dict[str, Rect]):
    try:
        targets = [
            (_location(entry, views), [_location(effect, views) for effect in _entries(entry, "effects")])
            for entry in _entries(data, "targets")
        ]
        protected = [_location(entry, views) for entry in _entries(data, "protected")]
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise CodexError("Codex returned invalid object locations. Try again or use the selection manually.") from exc
    return targets, protected


def _check_cancelled(cancelled: Event):
    if cancelled.is_set():
        raise Cancelled("Object detection canceled.")


def _touches_crop_edge(mask: Image.Image, area: Rect, canvas: Rect) -> bool:
    left, top, right, bottom = mask.getbbox()
    return ((left <= 1 and area.left > canvas.left)
            or (top <= 1 and area.top > canvas.top)
            or (right >= area.width - 1 and area.right < canvas.right)
            or (bottom >= area.height - 1 and area.bottom < canvas.bottom))


def _matches_lasso(mask: np.ndarray, lasso: np.ndarray) -> bool:
    overlap = np.count_nonzero(mask & lasso)
    return overlap >= np.count_nonzero(mask) * 0.9 and overlap >= np.count_nonzero(lasso) * 0.55


def _unexplained_lasso_edges(pixels: np.ndarray, unexplained: np.ndarray) -> bool:
    scale = min(1, 512 / max(unexplained.shape))
    if scale < 1:
        height, width = unexplained.shape
        size = max(1, round(width * scale)), max(1, round(height * scale))
        unexplained = cv2.resize(unexplained.astype(np.uint8) * 255, size, interpolation=cv2.INTER_AREA) == 255
        pixels = cv2.resize(pixels, size, interpolation=cv2.INTER_AREA)

    # Only consider edges whose neighborhood stays inside the lasso and away
    # from the proposed object. The unclaimed region may contain another target.
    kernel = np.ones((3, 3), np.uint8)
    interior = cv2.erode(unexplained.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    if not interior.any():
        return False
    contrast = cv2.morphologyEx(pixels, cv2.MORPH_GRADIENT, kernel).max(axis=2)
    edges = interior & (contrast >= 32)
    _, _, stats, _ = cv2.connectedComponentsWithStats(edges)
    components = stats[1:]
    span = np.maximum(components[:, cv2.CC_STAT_WIDTH], components[:, cv2.CC_STAT_HEIGHT])
    return bool(np.any((components[:, cv2.CC_STAT_AREA] >= 8) & (span >= 5)))


def _nearby_translucency(mask: np.ndarray, alpha: np.ndarray, bounds: tuple[int, int, int, int]) -> bool:
    left, top, right, bottom = bounds
    margin = max(5, min(24, round(max(right - left, bottom - top) * 0.1)))
    window = (slice(max(0, top - margin), bottom + margin), slice(max(0, left - margin), right + margin))
    opacity = alpha[window]
    outside = (opacity > 0) & (opacity < 192) & ~mask[window]
    if not outside.any():
        return False
    distances = cv2.distanceTransform((~mask[window]).astype(np.uint8), cv2.DIST_L2, 5)
    # Substantial partial alpha away from the antialiased contour may be a shadow or glow.
    return np.count_nonzero(outside & (distances > 3) & (distances <= margin)) >= max(12, mask.sum() * 0.03)


def _local_target(
    base: Image.Image, area: Rect, point: tuple[int, int], cancelled: Event, segmenter_factory,
    selection: Image.Image | None = None,
) -> tuple[RemovalTarget, ...]:
    source = base.crop(box(area))
    canvas = Rect(0, 0, *base.size)
    lasso = np.asarray(selection.crop(box(area))) != 0 if selection is not None else None
    alpha = np.asarray(source.getchannel("A")) if base.mode == "RGBA" else None
    anchor = point[0] - area.left, point[1] - area.top
    best = None
    confidence = 0.0
    agrees = False
    _check_cancelled(cancelled)
    try:
        segmenter = segmenter_factory()
        pixels = np.asarray(reference_rgb(source))
        for predicted, score in segmenter.candidates(pixels, (anchor,)):
            _check_cancelled(cancelled)
            if not math.isfinite(score) or score < max(0.48, confidence - 0.12):
                continue
            mask = predicted != 0
            if alpha is not None:
                mask &= alpha != 0
            if not mask[anchor[1], anchor[0]] or (lasso is not None and not _matches_lasso(mask, lasso)):
                continue
            if best is None:
                best, confidence = mask, score
            else:
                overlap = np.count_nonzero(best & mask) / np.count_nonzero(best | mask)
                agrees |= overlap >= 0.85
                # Comparable, incompatible outlines can mean a part or the whole object.
                if overlap < 0.85 and score >= confidence - 0.06:
                    return ()
    except Cancelled:
        raise
    except (OSError, RuntimeError, ValueError, cv2.error):
        return ()
    _check_cancelled(cancelled)
    if best is None or (confidence < 0.82 and not agrees) or np.count_nonzero(best) > best.size * 0.65:
        return ()

    _, components, stats, _ = cv2.connectedComponentsWithStats(best.astype(np.uint8), connectivity=8)
    component = components[anchor[1], anchor[0]]
    sizes = stats[1:, cv2.CC_STAT_AREA]
    size = sizes[component - 1]
    if sizes.sum() - size >= max(8, size * 0.01):
        return ()
    best = components == component
    mask = Image.fromarray(best.astype(np.uint8) * 255)
    if _touches_crop_edge(mask, area, canvas):
        return ()
    bounds = mask.getbbox()
    assert bounds is not None
    if alpha is not None and _nearby_translucency(best, alpha, bounds):
        return ()
    if selection is not None:
        region = Rect(*selection.getbbox()).relative_to(area).slices
        if _unexplained_lasso_edges(pixels[region], lasso[region] & ~best[region]):
            return ()
    _check_cancelled(cancelled)
    label = "Clicked object" if selection is None else "Selected object"
    return (RemovalTarget(label, mask.crop(bounds), (area.left + bounds[0], area.top + bounds[1])),)


def segment_removal_click(
    base: Image.Image, point: tuple[float, float], *, cancelled: Event, segmenter_factory=_EfficientSAM,
) -> tuple[RemovalTarget, ...]:
    """Propose a single unambiguous object at the clicked source pixel."""
    _check_cancelled(cancelled)
    if not all(math.isfinite(value) for value in point):
        return ()
    x, y = math.floor(point[0]), math.floor(point[1])
    if not (0 <= x < base.width and 0 <= y < base.height):
        return ()
    if base.mode == "RGBA" and base.getpixel((x, y))[3] == 0:
        return ()
    radius = max(320, min(768, max(base.size) // 6))
    area = Rect(x, y, x + 1, y + 1).expand(radius, radius).intersect(Rect(0, 0, *base.size))
    assert area is not None
    return _local_target(base, area, (x, y), cancelled, segmenter_factory)


def segment_removal_lasso(
    base: Image.Image, selection: Image.Image, *, cancelled: Event, segmenter_factory=_EfficientSAM,
) -> tuple[RemovalTarget, ...]:
    """Resolve a tight lasso locally when one mask explains most of the selection."""
    _check_cancelled(cancelled)
    if selection.mode != "L" or selection.size != base.size:
        raise ValueError("The removal selection must match the image.")
    bounds = selection.getbbox()
    if bounds is None:
        return ()
    selected = Rect(*bounds)
    scale = min(1, 512 / max(selected.width, selected.height))
    size = max(1, round(selected.width * scale)), max(1, round(selected.height * scale))
    lasso = _selection_coverage(selection, selected, size)
    if base.mode == "RGBA":
        alpha = base.resize(size, Image.Resampling.NEAREST, box=bounds).getchannel("A")
        lasso &= np.asarray(alpha) != 0
    if cv2.connectedComponents(lasso, connectivity=8)[0] != 2:
        return ()
    # Pad so a tightly drawn rectangle still has a finite interior distance.
    padded = np.pad(lasso, 1)
    distances = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(distances.argmax(), distances.shape)
    point = (selected.left + int((x + 0.5) * selected.width / size[0]),
             selected.top + int((y + 0.5) * selected.height / size[1]))
    margin = max(64, max(selected.width, selected.height) // 2)
    area = selected.expand(margin, margin).intersect(Rect(0, 0, *base.size))
    assert area is not None
    return _local_target(base, area, point, cancelled, segmenter_factory, selection)


def detect_removal(
    base: Image.Image, selection: Image.Image, executable: Path, *, cancelled: Event,
    client_factory=CodexClient, segmenter_factory=_EfficientSAM,
) -> tuple[RemovalTarget, ...]:
    _check_cancelled(cancelled)
    if base.mode not in ("RGB", "RGBA") or selection.mode != "L" or selection.size != base.size:
        raise ValueError("The removal selection must match the image.")
    bounds = selection.getbbox()
    if bounds is None or (base.mode == "RGBA" and not selection_has_visible_pixels(base, selection, bounds)):
        return ()
    selected = Rect(*bounds)
    views = _views(base, selected)
    with TemporaryDirectory(prefix="regimagen-remove-detect-") as folder:
        directory = Path(folder)
        references = tuple(directory / f"{name}.png" for name in views)
        for view, path in zip(views.values(), references):
            save_png(_guide(base, selection, view), path)
        _check_cancelled(cancelled)
        with client_factory(executable, directory.parent, cancelled=cancelled, image_generation=False) as client:
            data = client.analyze_images(_DETECTION_PROMPT, references, _DETECTION_SCHEMA)
    _check_cancelled(cancelled)
    targets, protected = _parse_detection(data, views)
    if not targets:
        return ()
    try:
        segmenter = segmenter_factory()
    except (OSError, RuntimeError, ValueError, cv2.error):
        segmenter = None

    result = []
    for target, effects in targets:
        _check_cancelled(cancelled)
        if target.bounds.intersect(selected) is None:
            continue
        refined = _refine(base, target, segmenter)
        if refined is None:
            continue
        mask, (x, y), used_coarse = refined
        under_lasso = selection.crop((x, y, x + mask.width, y + mask.height))
        if ImageChops.darker(mask, under_lasso).getbbox() is None:
            continue
        parent = len(result)
        confirmed = not target.uncertain and not used_coarse
        result.append(RemovalTarget(target.label, mask, (x, y), confirmed))
        for effect in effects:
            _check_cancelled(cancelled)
            refined_effect = _refine(base, effect, segmenter, use_contour=True)
            if refined_effect is not None:
                effect_mask, origin, _ = refined_effect
                result.append(RemovalTarget(
                    effect.label, effect_mask, origin, confirmed and not effect.uncertain, parent=parent,
                ))
    if result:
        for location in protected:
            _check_cancelled(cancelled)
            refined = _refine(base, location, segmenter, use_contour=True)
            if refined is not None:
                mask, origin, _ = refined
                result.append(RemovalTarget(location.label, mask, origin, selected=False, protected=True))
    _check_cancelled(cancelled)
    return tuple(result)
