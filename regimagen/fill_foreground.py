"""Isolate a requested foreground in a Generative Fill result."""

import math
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageFilter

from .codex import Cancelled, CodexClient, CodexError
from .document import box
from .files import reference_rgb, save_png
from .fill import _paint_selection, _selection_coverage
from .geometry import FRAME, Rect


def _object_schema(fields: dict) -> dict:
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


_POINT_SCHEMA = {"type": "array", "items": {"type": "number"}}

_VIEW_SCHEMA = {"type": "string", "enum": ["detail", "overview"]}

_CUTOUT_SCHEMA = _object_schema({
    "targets": {"type": "array", "items": _object_schema({
        "label": {"type": "string"}, "view": _VIEW_SCHEMA, "bounds": _POINT_SCHEMA,
        "foreground": {"type": "array", "items": _POINT_SCHEMA},
        "background": {"type": "array", "items": _POINT_SCHEMA},
    })},
})


_COORDINATE_GUIDANCE = """For each entry, label it briefly, select its coordinate view, and give a tight bounds array
[left, top, right, bottom]. All coordinates are in a 0 to 1000 space relative to that view,
with [0, 0] at top left and [1000, 1000] at bottom right. Do not return source-image coordinates."""

_CUTOUT_PROMPT = """Locate the requested foreground to keep in a transparent-background image.
Read the two attached images directly and return the JSON in this turn.
The two views show the same generated image. Image 1 (detail) shows the area around the selection.
Image 2 (overview) shows the whole generated square. Both mark the selection with a pink or dark outline.
Use the requested edit below to identify the intended foreground inside the outline. Put only those subjects
in targets. Treat incidental scenery and support surfaces as background. Give two foreground [x,y] points
inside each subject. Place the first well inside the main body and the second in another visible part.
Give up to two background [x,y] points just outside its silhouette or in visible gaps. Use the detail view
when the whole target fits in it, and overview otherwise. The app finds the edges from the bounds and points.

""" + _COORDINATE_GUIDANCE


def _views(base: Image.Image, selection: Rect) -> dict[str, Rect]:
    canvas = Rect(0, 0, *base.size)
    margin = max(64, max(selection.width, selection.height) // 2)
    detail = selection.expand(margin, margin).intersect(canvas)
    assert detail is not None
    return {"detail": detail, "overview": canvas}


def _view_image(base: Image.Image, view: Rect, *, max_side: int = FRAME) -> Image.Image:
    scale = max_side / max(view.width, view.height)
    size = max(1, round(view.width * scale)), max(1, round(view.height * scale))
    return reference_rgb(base.resize(size, Image.Resampling.LANCZOS, box=box(view)))


def _guide(base: Image.Image, mask: Image.Image, view: Rect, *, max_side: int = FRAME) -> Image.Image:
    guide = _view_image(base, view, max_side=max_side)
    filled = _selection_coverage(mask, view, guide.size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    inner = cv2.erode(filled, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    _paint_selection(guide, filled != inner)
    return guide


def _coordinates(value: object, length: int) -> list:
    if (not isinstance(value, list) or len(value) != length or any(
        type(number) not in (int, float) or not 0 <= number <= 1000 for number in value
    )):
        raise ValueError("Invalid coordinates")
    return value


def _entries(data: dict, key: str) -> list:
    entries = data.get(key)
    if not isinstance(entries, list) or len(entries) > 32 or not all(isinstance(entry, dict) for entry in entries):
        raise ValueError("Invalid object list")
    return entries


def _check_cancelled(cancelled: Event):
    if cancelled.is_set():
        raise Cancelled("Object detection canceled.")


def _view_points(values: object, view: Rect) -> tuple[tuple[int, int], ...]:
    if not isinstance(values, list) or len(values) > 256:
        raise ValueError("Invalid points")
    points = []
    for value in values:
        x, y = _coordinates(value, 2)
        points.append((
            view.left + min(view.width - 1, int(x * view.width / 1000)),
            view.top + min(view.height - 1, int(y * view.height / 1000)),
        ))
    return tuple(points)


def _label(data: dict) -> str:
    label = data["label"]
    if not isinstance(label, str) or not label.strip():
        raise ValueError("Invalid object label")
    return " ".join(label.split())[:120]


@dataclass(frozen=True)
class _Location:
    bounds: Rect
    foreground: tuple[tuple[int, int], ...]
    background: tuple[tuple[int, int], ...]


def _location(data: dict, views: dict[str, Rect]) -> _Location:
    view = views[data["view"]]
    left, top, right, bottom = _coordinates(data["bounds"], 4)
    if left >= right or top >= bottom:
        raise ValueError("Empty bounds")
    bounds = Rect(
        view.left + math.floor(left * view.width / 1000), view.top + math.floor(top * view.height / 1000),
        view.left + math.ceil(right * view.width / 1000), view.top + math.ceil(bottom * view.height / 1000),
    )
    _label(data)
    return _Location(bounds, _view_points(data["foreground"], view), _view_points(data["background"], view))


def _inside(
    points: tuple[tuple[int, int], ...], area: Rect, *, bounds: Rect | None = None,
) -> tuple[tuple[int, int], ...]:
    bounds = bounds or area
    return tuple((x - area.left, y - area.top) for x, y in points
                 if bounds.left <= x < bounds.right and bounds.top <= y < bounds.bottom)


class _EfficientSAM:
    def __init__(self):
        from .segmentation_model import model_path

        self.net = cv2.dnn.readNetFromONNX(str(model_path()))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def segment(
        self, image: np.ndarray, bounds: Rect, foreground: tuple, background: tuple,
    ) -> np.ndarray | None:
        if not foreground:
            return None
        for mask, _ in self.candidates(image, foreground, background, bounds=bounds):
            return mask
        return None

    def candidates(self, image: np.ndarray, foreground: tuple, background: tuple = (), *, bounds: Rect | None = None):
        if not foreground and bounds is None:
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


def _grabcut(
    image: np.ndarray, coarse: np.ndarray, foreground: tuple, background: tuple, *, refine_edges: bool = False,
) -> np.ndarray | None:
    if not coarse.any() or coarse.all() or not foreground:
        return None
    # Thin or translucent structures do not always provide usable GrabCut samples.
    # Keep their reviewed proposal when the local fit cannot represent it.
    margin = max(2, round(min(coarse.shape) * (0.01 if refine_edges else 0.025)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    labels = cv2.dilate(coarse, kernel) * cv2.GC_PR_BGD
    labels[coarse != 0] = cv2.GC_PR_FGD
    if refine_edges:
        # Keep the mask interior fixed and fit colors only along the boundary.
        labels[cv2.erode(coarse, kernel) != 0] = cv2.GC_FGD
    for x, y in foreground:
        labels[y, x] = cv2.GC_FGD
    for x, y in background:
        labels[y, x] = cv2.GC_BGD
    model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(image, labels, None, model.copy(), model.copy(), 2 if refine_edges else 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    return ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)).astype(np.uint8)


def _fit_mask(
    image: np.ndarray, mask: np.ndarray, foreground: tuple, background: tuple, *, refine_edges: bool = False,
) -> np.ndarray | None:
    height, width = mask.shape
    scale = min(1, FRAME / max(width, height))
    if scale == 1:
        return _grabcut(image, mask, foreground, background, refine_edges=refine_edges)
    size = max(1, round(width * scale)), max(1, round(height * scale))

    def scaled(points):
        return tuple((min(size[0] - 1, int(x * scale)), min(size[1] - 1, int(y * scale))) for x, y in points)

    fit = _grabcut(
        cv2.resize(image, size, interpolation=cv2.INTER_AREA),
        cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST), scaled(foreground), scaled(background),
        refine_edges=refine_edges,
    )
    return cv2.resize(fit, (width, height), interpolation=cv2.INTER_NEAREST) if fit is not None else None


def _cutout_mask(base: Image.Image, location: _Location, segmenter: _EfficientSAM) -> tuple[Image.Image, Rect] | None:
    margin = max(8, math.ceil(max(location.bounds.width, location.bounds.height) * 0.15))
    area = location.bounds.expand(margin, margin).intersect(Rect(0, 0, *base.size))
    assert area is not None
    pixels = np.asarray(base.crop(box(area)))
    foreground = _inside(location.foreground, area, bounds=location.bounds)[:2]
    background = _inside(location.background, area)[:2]
    bounds = location.bounds.relative_to(area)
    try:
        refined = segmenter.segment(pixels, bounds, foreground, background)
        if refined is None and len(foreground) > 1:
            foreground = foreground[:1]
            refined = segmenter.segment(pixels, bounds, foreground, background)
    except cv2.error:
        return None
    if refined is None:
        return None
    fit = _fit_mask(pixels, refined, foreground, background, refine_edges=True)
    if fit is not None and fit.any() and not fit.all():
        refined = fit
    return Image.fromarray(refined * 255), area


def fill_foreground_alpha(
    image: Image.Image, selection: Image.Image, requested_edit: str, executable: Path, *,
    cancelled: Event, client_factory=CodexClient,
) -> Image.Image:
    _check_cancelled(cancelled)
    if selection.mode != "L" or selection.size != image.size:
        raise ValueError("The foreground selection must match the image.")
    base = image.convert("RGB")
    bounds = selection.getbbox()
    matte = Image.new("L", image.size)
    if bounds is not None:
        selected = Rect(*bounds)
        views = _views(base, selected)
        with TemporaryDirectory(prefix="regimagen-detect-") as folder:
            directory = Path(folder)
            references = tuple(directory / f"{name}.png" for name in views)
            for view, path in zip(views.values(), references):
                save_png(_guide(base, selection, view), path)
            _check_cancelled(cancelled)
            prompt = f"{_CUTOUT_PROMPT}\n\nRequested edit: {requested_edit}"
            with client_factory(executable, directory.parent, cancelled=cancelled, image_generation=False) as client:
                data = client.analyze_images(prompt, references, _CUTOUT_SCHEMA, effort="low")
        _check_cancelled(cancelled)
        try:
            locations = [_location(entry, views) for entry in _entries(data, "targets")]
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise CodexError("Codex returned invalid foreground locations.") from exc
        try:
            segmenter = _EfficientSAM() if locations else None
        except (OSError, RuntimeError, ValueError, cv2.error):
            segmenter = None
        if segmenter is not None:
            for location in locations:
                _check_cancelled(cancelled)
                if location.bounds.intersect(selected) is None:
                    continue
                fitted = _cutout_mask(base, location, segmenter)
                if fitted is None:
                    continue
                mask, area = fitted
                region = box(area)
                if ImageChops.darker(mask, selection.crop(region)).getbbox() is not None:
                    matte.paste(ImageChops.lighter(matte.crop(region), mask), region[:2])
    _check_cancelled(cancelled)
    # Soften the inner edge without introducing pixels from the surrounding scenery.
    matte = ImageChops.darker(matte, matte.filter(ImageFilter.GaussianBlur(0.75)))
    alpha = ImageChops.multiply(matte, image.getchannel("A"))
    if ImageChops.multiply(alpha, selection).getbbox() is None or matte.getextrema() == (255, 255):
        raise ValueError("Could not isolate the requested foreground from the generated background.")
    return alpha
