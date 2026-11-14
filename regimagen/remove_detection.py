"""Build Generative Remove masks from contours authored by Codex."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

import cv2
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


def _object_schema(fields: dict) -> dict:
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


_POINT_SCHEMA = {"type": "array", "items": {"type": "number"}}

_VIEW_SCHEMA = {"type": "string", "enum": ["detail", "overview"]}
_CONTOUR_FIELDS = {
    "label": {"type": "string"},
    "view": _VIEW_SCHEMA,
    "contours": {"type": "array", "items": {"type": "array", "items": _POINT_SCHEMA}},
    "uncertain": {"type": "boolean"},
}
_CONTOUR_SCHEMA = _object_schema(_CONTOUR_FIELDS)
_DETECTION_SCHEMA = _object_schema({
    "targets": {"type": "array", "items": _object_schema({
        **_CONTOUR_FIELDS, "effects": {"type": "array", "items": _CONTOUR_SCHEMA},
    })},
    "protected": {"type": "array", "items": _CONTOUR_SCHEMA},
})
_EFFECTS_SCHEMA = _object_schema({
    "effects": {"type": "array", "items": _object_schema({
        **_CONTOUR_FIELDS, "view": {"type": "string", "enum": ["detail"]},
    })},
})

_CONTOUR_GUIDANCE = """Trace the visible pixels using polygons. Each polygon is a list of at least three [x,y]
vertices, in a 0 to 1000 space relative to the named view. [0,0] is the top left and [1000,1000]
is the bottom right. Trace the silhouette closely along the actual edges. Use separate narrow
polygons for thin wires or cords and separate polygons for visible pieces separated by gaps or
occlusions. Give concise labels and set uncertain=true for guesses. Respond with the structured
selection only."""

_DETECTION_PROMPT = """Draw the final selection mask for an object removal tool.
Image 1 (detail) is a clean close-up. Image 2 has exactly the same crop and shows the user's
mark in pink or dark ink. Image 3 (overview) shows that mark in the larger scene. Use the
marked images to identify the intended object, then trace it on the clean close-up. Use
view=detail unless the close-up clips the object, in which case use view=overview.

For a lasso, select the whole object the outline encloses. If it surrounds a building from
roof to base, trace the facade, roof and its attached details as one target. If it surrounds
only a rooftop fixture or other detail, trace that detail and leave the support surface
outside the target. If it encloses several distinct intended objects, return each separately.
Ignore unrelated things that merely touch the outline. For a click, the center of the
small circle points to the visible object to trace. If there is no intended object, return
an empty targets list. Do not trace the guide ink.

Look for the selected object's cast shadows and reflections on surrounding surfaces. Trace
their full visible extent separately as effects. Keep natural shading, surface texture
and effects from other objects out of those masks. Use protected
polygons for unrelated foreground that overlaps or crosses in front of a selected object.

""" + _CONTOUR_GUIDANCE

_EFFECTS_PROMPT = """Trace effects cast by the selected object onto other surfaces.
Image 1 (detail) is a clean close-up. Image 2 has exactly the same crop with the selected object
outlined. Inspect the clean image beyond the outline and underneath protruding parts for cast
shadows and reflections caused by this object. Include narrow shadows and soft falloff, even
if offset or disconnected from the object. Exclude shading from other objects and the
surface's own material and joints. Return an empty list if there is no associated effect.
Compare the direction of the light and the shapes of the dark areas with each part of the
selected object. Trace clearly attributable parts in their own polygons even if a separate,
fainter extension is uncertain. A soft edge does not make the effect itself uncertain.
Mark uncertain=true only if the effect could plausibly come from something else.
Use view=detail.

""" + _CONTOUR_GUIDANCE

_DIRECT_EFFECTS_GUIDANCE = """This is a background reconstruction mask. Work from the clean close-up.
The lasso identifies the object. Trace its cast effects in full, including any part outside
the lasso. A shadow can be entirely offset from the selected object.
Follow the projection of the whole object across the receiving surface. For a suspended
part, look past its apparent bottom edge for the offset shadow. For an upright part,
look along the direction of the cast shadow from its contact with the surface. A visible
contact shadow may be just one end of a longer shadow. Include the dark body and soft
falloff of each cast shadow in the same contour, extending to the first unaffected
texture on each side so the edit can blend on evenly lit material. Preserve the shape
of the receiving surface and normal lighting changes from its edges. If the object has
several parts, check the projection of each part before returning the contours.
Return the full contours of attributable effects even if some of their edges are faint.
Use uncertain=true for an area you cannot attribute to the outlined object."""

_DIRECT_EFFECTS_RECHECK = """Image 3 marks areas where an earlier pass found a shadow or
reflection. Use the mark to locate the effect, then examine the clean image. Compare the
surrounding material at the same orientation and illumination. Decide whether a localized
darkening follows the outline or projection of a part of the outlined object. Look past the
marked area in both directions along the projection for the rest of the same effect. If the
shadow falls on a tiled surface, follow the darkening across tile joints to its far edge.
Trace the full attributable area, including the soft fringe on the receiving surface.
Check beyond each marked shadow for connected darker bands and other projections of the
same object, especially farther from the point of contact. Include any missed extensions.
If the mark only covers ordinary surface shading or another object's effect, omit it.
Return the revised contours for these areas."""


def _views(base: Image.Image, selection: Rect, *, margin: int | None = None) -> dict[str, Rect]:
    canvas = Rect(0, 0, *base.size)
    if margin is None:
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


def _view_image(base: Image.Image, view: Rect) -> Image.Image:
    scale = FRAME / max(view.width, view.height)
    size = max(1, round(view.width * scale)), max(1, round(view.height * scale))
    return reference_rgb(base.resize(size, Image.Resampling.LANCZOS, box=box(view)))


def _guide(base: Image.Image, mask: Image.Image, view: Rect) -> Image.Image:
    guide = _view_image(base, view)
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


def _outlined_target(
    base: Image.Image, data: dict, views: dict[str, Rect], *,
    protected: bool = False, parent: int | None = None,
) -> RemovalTarget | None:
    view = views[data["view"]]
    label = _label(data)
    contours, uncertain = data["contours"], data["uncertain"]
    if not isinstance(contours, list) or len(contours) > 32 or type(uncertain) is not bool:
        raise ValueError("Invalid object contours")
    polygons = [_view_points(contour, view) for contour in contours]
    if not polygons:
        return None
    if any(len(polygon) < 3 for polygon in polygons):
        raise ValueError("Incomplete object contour")
    xs, ys = zip(*(point for polygon in polygons for point in polygon))
    left, top = min(xs), min(ys)
    region = left, top, max(xs) + 1, max(ys) + 1
    mask = Image.new("L", (region[2] - left, region[3] - top))
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        draw.polygon([(x - left, y - top) for x, y in polygon], fill=255)
    if base.mode == "RGBA":
        visible = base.crop(region).getchannel("A").point([0] + [255] * 255)
        mask = ImageChops.darker(mask, visible)
        clipped = mask.getbbox()
        if clipped is None:
            return None
        mask = mask.crop(clipped)
        left, top = left + clipped[0], top + clipped[1]
    return RemovalTarget(label, mask, (left, top), not (uncertain or protected), protected, parent)


def _overlaps_selection(target: RemovalTarget, selection: Image.Image) -> bool:
    x, y = target.origin
    under_target = selection.crop((x, y, x + target.mask.width, y + target.mask.height))
    return ImageChops.darker(target.mask, under_target).getbbox() is not None


def detect_removal(
    base: Image.Image, selection: Image.Image, executable: Path, *, cancelled: Event,
    point: tuple[int, int] | None = None, direct: bool = False, client_factory=CodexClient,
) -> tuple[RemovalTarget, ...]:
    """Ask Codex for the actual removal contours and map them onto source pixels."""
    _check_cancelled(cancelled)
    if base.mode not in ("RGB", "RGBA") or selection.mode != "L" or selection.size != base.size:
        raise ValueError("The removal selection must match the image.")
    if point is not None:
        x, y = point
        if not (0 <= x < base.width and 0 <= y < base.height):
            return ()
        if base.mode == "RGBA" and base.getpixel(point)[3] == 0:
            return ()
        radius = max(3, min(14, round(min(base.size) * 0.01)))
        selection = Image.new("L", base.size)
        ImageDraw.Draw(selection).ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    bounds = selection.getbbox()
    if bounds is None or (base.mode == "RGBA" and not selection_has_visible_pixels(base, selection, bounds)):
        return ()
    region = Rect(*bounds)
    span = max(region.width, region.height)
    if direct:
        # Only the lasso can be edited, so use more of the detail image for it.
        margin = max(16, min(128, span // 5))
    else:
        extent = min(base.size)
        context = min(384, extent // 12) if point is not None else min(192, extent // 24)
        margin = max(64, context, span // 2)
    views = _views(base, region, margin=margin)
    with TemporaryDirectory(prefix="regimagen-remove-detect-") as folder:
        directory = Path(folder)
        references = directory / "detail.png", directory / "mark.png", directory / "overview.png"
        save_png(_view_image(base, views["detail"]), references[0])
        save_png(_guide(base, selection, views["detail"]), references[1])
        save_png(_guide(base, selection, views["overview"]), references[2])
        _check_cancelled(cancelled)
        mark = "click" if point is not None else "lasso"
        prompt = f"{_DETECTION_PROMPT}\n\nThe user used a {mark}."
        options = {}
        if direct:
            options["effort"] = "medium"
            prompt += (
                " Trace the intended objects inside the lasso and their visible effects, including outside it. "
                "Follow each fine wire or cord continuously from its attachment to its end, including "
                "where it passes in front of another surface. Trace the whole cast shadow, including "
                "the part offset from the object and the soft edge on the background. "
                "Use protected polygons for nearby unrelated objects that touch or overlap the removal area."
            )
        else:
            prompt += " Also look outside the user's mark for associated effects."
        with client_factory(executable, directory.parent, cancelled=cancelled, image_generation=False) as client:
            data = client.analyze_images(prompt, references, _DETECTION_SCHEMA, **options)
    _check_cancelled(cancelled)

    result = []
    try:
        for entry in _entries(data, "targets"):
            _check_cancelled(cancelled)
            target = _outlined_target(base, entry, views)
            if target is None or not _overlaps_selection(target, selection):
                continue
            parent = len(result)
            result.append(target)
            for effect in _entries(entry, "effects"):
                if (outline := _outlined_target(base, effect, views, parent=parent)) is not None:
                    result.append(outline)
        if result:
            for entry in _entries(data, "protected"):
                _check_cancelled(cancelled)
                if (outline := _outlined_target(base, entry, views, protected=True)) is not None:
                    result.append(outline)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise CodexError("Codex returned an invalid selection. Try again or use Lasso directly.") from exc
    _check_cancelled(cancelled)
    return tuple(result)


def detect_removal_effects(
    base: Image.Image, targets: tuple[RemovalTarget, ...], executable: Path, *,
    cancelled: Event, lasso: Image.Image | None = None, client_factory=CodexClient,
) -> tuple[RemovalTarget, ...]:
    """Ask Codex to check for missed effects around a single outlined object."""
    _check_cancelled(cancelled)
    if base.mode != "RGB":
        return targets
    objects = [(i, target) for i, target in enumerate(targets) if not target.protected and target.parent is None]
    if len(objects) != 1:
        return targets
    parent, target = objects[0]
    previous = [effect for effect in targets if effect.parent == parent]
    if lasso is None and any(effect.selected for effect in previous):
        return targets

    x, y = target.origin
    bounds = Rect(x, y, x + target.mask.width, y + target.mask.height)
    margin = max(64, bounds.width, bounds.height)
    if lasso is not None:
        region = lasso.getbbox()
        if region is None:
            return targets
        bounds = Rect(*region)
        margin = max(32, min(192, max(bounds.width, bounds.height) // 3))
    detail = bounds.expand(margin, margin).intersect(Rect(0, 0, *base.size))
    assert detail is not None
    selection = Image.new("L", base.size)
    selection.paste(target.mask, target.origin)
    outside_object = ImageChops.invert(selection)

    def parse_effects(data):
        effects = []
        for entry in _entries(data, "effects"):
            _check_cancelled(cancelled)
            effect = _outlined_target(base, entry, {"detail": detail}, parent=parent)
            if effect is not None and _overlaps_selection(effect, outside_object):
                effects.append(effect)
        return effects

    with TemporaryDirectory(prefix="regimagen-remove-effects-") as folder:
        directory = Path(folder)
        references = directory / "detail.png", directory / "object-outline.png"
        save_png(_view_image(base, detail), references[0])
        save_png(_guide(base, selection, detail), references[1])
        _check_cancelled(cancelled)
        prompt = f"{_EFFECTS_PROMPT}\n\nSelected object label: {target.label}"
        options = {}
        if lasso is not None:
            prompt += "\n\n" + _DIRECT_EFFECTS_GUIDANCE
            options["effort"] = "high"
        try:
            with client_factory(executable, directory.parent, cancelled=cancelled, image_generation=False) as client:
                effects = parse_effects(client.analyze_images(prompt, references, _EFFECTS_SCHEMA, **options))
                if lasso is not None:
                    # Recheck uncertain candidates and the parts that extend past the lasso.
                    possible = Image.new("L", base.size)
                    for effect in (*previous, *effects):
                        mask = effect.mask
                        if effect.selected:
                            x, y = effect.origin
                            mask = ImageChops.subtract(mask, lasso.crop((x, y, x + mask.width, y + mask.height)))
                        possible.paste(255, effect.origin, mask)
                    possible = ImageChops.darker(possible, outside_object)
                    if possible.getbbox() is not None:
                        candidate_guide = directory / "possible-effects.png"
                        save_png(_guide(base, possible, detail), candidate_guide)
                        revised = parse_effects(client.analyze_images(
                            prompt + "\n\n" + _DIRECT_EFFECTS_RECHECK,
                            (*references, candidate_guide), _EFFECTS_SCHEMA, **options,
                        ))
                        effects = [effect for effect in effects if effect.selected]
                        effects.extend(revised)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise CodexError("Codex returned invalid shadow or reflection locations.") from exc
    _check_cancelled(cancelled)
    if lasso is not None:
        retained = tuple(effect for effect in targets if effect.parent != parent or (
            effect.selected and _overlaps_selection(effect, outside_object)
        ))
        return retained + tuple(effects)
    if not effects or (previous and not any(effect.selected for effect in effects)):
        return targets
    return tuple(target for target in targets if target.parent != parent) + tuple(effects)
