"""Prepare and composite a single generative edit in source coordinates."""

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw

from .codex import Cancelled, CodexClient, GeneratedImage
from .composite import LINEAR_RGB
from .files import TRANSPARENT_MATTE, _srgb, reference_rgb, save_png
from .geometry import FRAME, Rect
from .pattern_courses import match_course_widths

SHIFT_LIMIT = 8
PATCH_SIZE = 32
CONTEXT_PANEL = FRAME // 2
SELECTION_MARK = (255, 73, 215)
SELECTION_DARK = (12, 14, 22)
TRANSPARENT_SELECTION = "Select visible pixels or turn on Edit Transparency."
FILL_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"transparency": {"type": "string", "enum": ["preserve", "edit", "background"]}},
    "required": ["transparency"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class FillCandidate:
    image: Image.Image
    metadata: dict

    @property
    def transparent_background(self) -> bool:
        return self.metadata.get("transparency") == "background"


@dataclass(frozen=True)
class FillPreview:
    patch: Image.Image
    origin: tuple[int, int]
    warning: str = ""

    def compose(self, base: Image.Image) -> Image.Image:
        image = base.convert("RGBA") if self.patch.mode == "RGBA" and base.mode == "RGB" else base.copy()
        image.paste(self.patch, self.origin)
        return image


def selection_has_visible_pixels(base: Image.Image, mask: Image.Image, bounds: tuple[int, int, int, int]) -> bool:
    alpha = base.crop(bounds).getchannel("A")
    return ImageChops.darker(mask.crop(bounds), alpha).getbbox() is not None


def _selection(base: Image.Image, mask: Image.Image, edit_alpha: bool) -> Rect:
    if base.mode not in ("RGB", "RGBA") or (edit_alpha and base.mode != "RGBA"):
        raise ValueError("The image format does not support this edit.")
    if mask.mode != "L" or mask.size != base.size:
        raise ValueError("The selection must match the source image.")
    bounds = mask.getbbox()
    if bounds is None:
        raise ValueError("Select an area to fill.")
    return Rect(*bounds)


def _crop_box(selection: Rect, size: tuple[int, int]) -> Rect:
    longest = max(selection.width, selection.height)
    side = longest + 2 * max(16, math.ceil(longest * 0.20))
    if longest <= min(size):
        side = min(side, *size)

    def start(low: int, high: int, extent: int) -> int:
        center = (low + high - side) // 2
        return max(min(0, extent - side), min(max(0, extent - side), center))

    left = start(selection.left, selection.right, size[0])
    top = start(selection.top, selection.bottom, size[1])
    return Rect(left, top, left + side, top + side)


def _wide_fill_transition(mask: Image.Image, selection: Rect) -> int:
    short, long = sorted((selection.width, selection.height))
    if short < 96 or long < 2 * short:
        return 0
    width = min(128, round(short * 0.16))
    scale = min(1, 512 / long)
    size = max(1, round(selection.width * scale)), max(1, round(selection.height * scale))
    shape = _selection_coverage(mask, selection, size)
    if cv2.connectedComponents(shape, connectivity=8)[0] != 2:
        return 0
    depth = cv2.distanceTransform(np.pad(shape, 1), cv2.DIST_L2, cv2.DIST_MASK_5).max()
    return width if depth >= width * scale else 0


def _working_frame(image: Image.Image, crop: Rect, method: Image.Resampling) -> Image.Image:
    box = crop.left, crop.top, crop.right, crop.bottom
    if crop.intersect(Rect(0, 0, *image.size)) == crop:
        return image.resize((FRAME, FRAME), method, box=box)
    scale = crop.width / FRAME
    frame = image.transform(
        (FRAME, FRAME), Image.Transform.AFFINE, (scale, 0, crop.left, 0, scale, crop.top),
        resample=Image.Resampling.BILINEAR if image.mode == "L" else Image.Resampling.BICUBIC,
    )
    if image.mode == "RGB":
        # Replicate edge samples in the virtual part of the square.
        x, y = _real_axes(crop, image.size)
        columns, rows = np.flatnonzero(x), np.flatnonzero(y)
        if columns.size and rows.size:
            columns = np.clip(np.arange(FRAME), columns[0], columns[-1])
            rows = np.clip(np.arange(FRAME), rows[0], rows[-1])
            frame = Image.fromarray(np.asarray(frame)[rows[:, None], columns])
    return frame


def _real_axes(crop: Rect, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    positions = (np.arange(FRAME) + 0.5) * crop.width / FRAME
    x = positions + crop.left
    y = positions + crop.top
    return (x >= 0) & (x < size[0]), (y >= 0) & (y < size[1])


def _paint_selection(image: Image.Image, outline: np.ndarray):
    pixels = np.asarray(image, dtype=np.int16)
    distance = np.max(np.abs(pixels - np.array(SELECTION_MARK, dtype=np.int16)), axis=2)
    dark = outline & (distance <= 96)
    image.paste(SELECTION_MARK, mask=Image.fromarray(outline & ~dark))
    image.paste(SELECTION_DARK, mask=Image.fromarray(dark))


def _mark_selection(image: Image.Image, selection: Rect, view: Rect):
    relative = selection.relative_to(view)
    sx, sy = image.width / view.width, image.height / view.height
    bounds = (
        max(0, math.floor(relative.left * sx)),
        max(0, math.floor(relative.top * sy)),
        min(image.width - 1, math.ceil(relative.right * sx) - 1),
        min(image.height - 1, math.ceil(relative.bottom * sy) - 1),
    )
    outline = Image.new("1", image.size)
    ImageDraw.Draw(outline).rectangle(bounds, outline=1, width=2)
    _paint_selection(image, np.asarray(outline))


def _selection_coverage(
    mask: Image.Image, crop: Rect, size: tuple[int, int] = (FRAME, FRAME),
) -> np.ndarray:
    # A guide pixel represents every source pixel it overlaps, including fractional
    # edges. Resampling mask intensity can drop isolated, lightly selected pixels.
    view = crop.intersect(Rect(0, 0, *mask.size))
    assert view is not None

    def bounds(origin: int, length: int, count: int, low: int, high: int):
        edges = np.arange(count + 1, dtype=np.int64) * length
        starts = np.clip(origin + edges[:-1] // count, low, high)
        ends = np.clip(origin + (edges[1:] + count - 1) // count, low, high)
        return starts, ends

    width, height = size
    left, right = bounds(crop.left, crop.width, width, view.left, view.right)
    tops, bottoms = bounds(crop.top, crop.height, height, view.top, view.bottom)
    left -= view.left
    right -= view.left
    selected = np.zeros((height, width), dtype=np.uint8)
    prefix = np.zeros(view.width + 1, dtype=np.int64)
    for row in np.flatnonzero(tops < bottoms):
        band = np.asarray(mask.crop((view.left, int(tops[row]), view.right, int(bottoms[row]))))
        np.cumsum(band.any(axis=0), out=prefix[1:])
        selected[row] = prefix[right] > prefix[left]
    return selected


def _selection_reference(
    reference: Image.Image, selected: np.ndarray, *, placement: np.ndarray | None = None,
) -> Image.Image:
    marked = reference_rgb(reference).copy()
    inner = cv2.erode(selected, np.ones((5, 5), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    _paint_selection(marked, selected != inner)
    if placement is not None and placement.any():
        inside = cv2.erode(placement, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        marked.paste((255, 174, 47), mask=Image.fromarray((placement != inside) & (inner != 0)))
    return marked


def _context_reference(
    base: Image.Image, selection: Rect, crop: Rect, *, focus: bool = False,
) -> Image.Image:
    canvas = Rect(0, 0, *base.size)
    if focus:
        margin = min(selection.width, selection.height)
        neighborhood = selection.expand(margin, margin).intersect(canvas)
    else:
        half = crop.width // 2
        extra = crop.width - half
        neighborhood = Rect(crop.left - half, crop.top - half, crop.right + extra, crop.bottom + extra).intersect(canvas)
    assert neighborhood is not None  # The crop contains the selection.
    overview = canvas
    if base.mode == "RGBA":
        visible = base.getbbox()
        if visible:
            overview = Rect(
                min(visible[0], selection.left), min(visible[1], selection.top),
                max(visible[2], selection.right), max(visible[3], selection.bottom),
            )

    context = Image.new("RGB", (2 * CONTEXT_PANEL, CONTEXT_PANEL), TRANSPARENT_MATTE)
    for column, view in enumerate((neighborhood, overview)):
        scale = (CONTEXT_PANEL - 16) / max(view.width, view.height)
        size = max(1, round(view.width * scale)), max(1, round(view.height * scale))
        box = view.left, view.top, view.right, view.bottom
        panel = base.resize(size, Image.Resampling.LANCZOS, box=box)
        panel = reference_rgb(panel)
        _mark_selection(panel, selection, view)
        context.paste(panel, (
            column * CONTEXT_PANEL + (CONTEXT_PANEL - panel.width) // 2,
            (CONTEXT_PANEL - panel.height) // 2,
        ))
    return context


def _decode(data: bytes, require_alpha: bool, *, allow_missing_alpha: bool = False) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data), formats=("PNG",)) as image:
            if image.width != image.height:
                raise ValueError("ImageGen returned a non-square image. Generate another variation.")
            if (require_alpha and not allow_missing_alpha
                    and "A" not in image.getbands() and "transparency" not in image.info):
                raise ValueError("ImageGen returned a PNG without transparency information. Generate another variation.")
            normalized = _srgb(image, preserve_alpha=require_alpha)
            if require_alpha and normalized.mode != "RGBA":
                # Normalize fully opaque results to the requested output mode.
                normalized = normalized.convert("RGBA")
            return normalized
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("ImageGen returned an invalid PNG. Generate another variation.") from exc


def _prompt(prompt: str, *, alpha: bool, edit_alpha: bool, transition: bool = False, pattern: bool = False) -> str:
    prompt = prompt.strip()
    result = (
        "Inspect the three references before calling the native image_gen tool exactly once. "
        "Pass all three image files in their attached order. Image 1 is the clean square to edit. "
        "Image 2 shows the same square and coordinates, with the editable area outlined in "
        "magenta or dark ink. Image 3 is context only: the left panel shows the nearby surroundings "
        "and the right panel shows the whole composition. The marked boxes locate the selection. "
        "Identify the subject, the nearest relevant surface, nearby matching parts, and visible "
        "style cues. Describe the rendering medium, forms, softness of edges, amount of detail, "
        "palette and contrast actually visible in Image 1 and the nearby reference. Use those "
        "observations in the image_gen prompt so new elements look painted, modeled or photographed "
        "in the same style. Use existing elements of the same kind as the closest style examples. "
        "Include specific observations from the references in the image_gen prompt "
        "to ground placement, scale, perspective, material, texture and lighting. "
        "Return one square PNG at the exact framing of Image 1, preferably 1024 by 1024 pixels. "
        "Explicit user requests take priority for the visual attributes they specify. "
    )
    if prompt and not pattern:
        result += (
            "Before calling ImageGen, classify the requested transparency in any language. Choose background "
            "for a transparent or absent background around the requested content. For background, create an "
            "isolated foreground asset. Use all three images for placement, proportions and style, and keep the "
            "subject in the same position in the square. Render only the requested foreground inside the "
            "outline. Set alpha to zero throughout the remaining square, including the original scenery inside "
            "and outside the outline. Include antialiased edges and any foreground effects requested by the user. "
            "Choose edit for another change to pixel opacity, such as clearing an area or making it semitransparent. "
            "For background and edit, set transparent_background=true in the tool call and use real PNG alpha. "
            "Choose preserve for other requests. After the tool call, return the chosen transparency value as "
            "JSON using the response schema. For preserve and edit, follow the in-place edit instructions below. "
        )
    result += (
        "Make added objects sit on or attach to the existing geometry where appropriate. Keep additions and "
        "their contact shadows inside the outline and continue any contours that cross it. Treat the "
        "outline as the edit limit. Give new silhouettes their own natural shape and room to end "
        "inside the edit. Preserve unrelated details and omit the guide marks and panels from the output. "
    )
    if transition:
        result += (
            "For fills that keep the background, use the amber line in Image 2 if present. Place the visible "
            "body and high-contrast details of new additions inside it. Between amber and the outer "
            "magenta or dark outline, match "
            "the original background and its gradients, shapes and texture. Continue the background "
            "through this band even when the requested addition is dark or large. Preserve the "
            "scene's overall lighting, colors and style unless the requested edit changes them. "
        )
    if prompt and not pattern:
        result += "Default alpha handling for preserve: "
    if alpha:
        result += (
            "Set transparent_background=true in the tool call. Use real PNG alpha for transparent "
            "areas and clean semitransparent edges. Do not paint a solid background or checkerboard. "
        )
        result += (
            "Change opacity only where the edit calls for it inside the selection. Use alpha zero "
            "only for existing transparent areas and for requested new gaps. Preserve the rest of "
            "the original opaque material; do not introduce extra holes, gaps, or fading. "
            if edit_alpha else "Preserve the original silhouette and transparency. "
        )
    else:
        result += "Return an opaque PNG. "
    return result + (f"\nRequested edit: {prompt}" if prompt else (
        "Fill the selection naturally with a continuation of its surroundings."
    ))


def _fill_transparency(result: GeneratedImage, *, source_has_alpha: bool) -> str:
    try:
        answer = json.loads(result.final_message)
    except ValueError:
        answer = None
    if isinstance(answer, dict) and answer.get("transparency") in ("preserve", "edit", "background"):
        return answer["transparency"]
    # A native transparency request on an RGB source signals an alpha override
    # even if the server omitted the final JSON.
    return "background" if result.transparent_background and not source_has_alpha else "preserve"


def _native_cutout_alpha(image: Image.Image, selection: Image.Image) -> Image.Image | None:
    alpha = image.getchannel("A")
    pixels = np.asarray(alpha)
    selected = np.asarray(selection) != 0
    selected_count = np.count_nonzero(selected)
    outside_count = selected.size - selected_count
    for ceiling in (0, 8):
        clear = pixels <= ceiling
        if not np.any(selected & ~clear):
            return None
        # Accept a clear area within the edit or a clear surround on a tight selection.
        if (np.count_nonzero(clear & selected) >= max(1, math.ceil(selected_count * 0.02))
                or (outside_count and np.count_nonzero(clear & ~selected) >= outside_count * 0.95)):
            return alpha if ceiling == 0 else alpha.point([0] * (ceiling + 1) + list(range(ceiling + 1, 256)))
    return None


def _context_matches(
    reference: np.ndarray, result: np.ndarray, context: np.ndarray, *, use_alpha: bool = False,
) -> list[tuple[int, int, int, int, float]]:
    ref_gray = cv2.cvtColor(reference[..., :3], cv2.COLOR_RGB2GRAY)
    out_gray = cv2.cvtColor(result[..., :3], cv2.COLOR_RGB2GRAY)
    if use_alpha:
        # Silhouettes provide registration anchors even when the RGB inside them is flat.
        ref_gray = ((0.35 + ref_gray / 392.0) * reference[..., 3]).astype(np.uint8)
        out_gray = ((0.35 + out_gray / 392.0) * result[..., 3]).astype(np.uint8)
    patches = []
    margin = SHIFT_LIMIT
    size = PATCH_SIZE + 2 * margin
    for top in range(margin, FRAME - PATCH_SIZE - margin + 1, PATCH_SIZE):
        for left in range(margin, FRAME - PATCH_SIZE - margin + 1, PATCH_SIZE):
            area = context[top - margin:top - margin + size, left - margin:left - margin + size]
            if not area.all():
                continue
            variation = float(ref_gray[top:top + PATCH_SIZE, left:left + PATCH_SIZE].std())
            if variation >= 5:
                patches.append((variation, left, top))

    matches = []
    for _, left, top in sorted(patches, reverse=True)[:32]:
        template = ref_gray[top:top + PATCH_SIZE, left:left + PATCH_SIZE]
        search = out_gray[top - margin:top - margin + size, left - margin:left - margin + size]
        scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(scores)
        if score < 0.85:
            continue
        competing = scores.copy()
        competing[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = -1
        if score - float(competing.max()) < 0.04:
            continue
        if (x, y) != (margin, margin) and score - float(scores[margin, margin]) < 0.05:
            if scores[margin, margin] < 0.85:
                continue
            x = y = margin
            score = float(scores[margin, margin])
        matches.append((left, top, x - margin, y - margin, score))
    return matches


def _register(
    reference: Image.Image, generated: Image.Image, guide: Image.Image, crop: Rect,
    source_size: tuple[int, int], selection: Rect, *, use_alpha: bool = False,
) -> tuple[list[int], list[float]]:
    ref = np.asarray(reference.convert("RGBA"))
    result = np.asarray(generated.resize((FRAME, FRAME), Image.Resampling.LANCZOS).convert("RGBA"))
    x, y = _real_axes(crop, source_size)
    outside = (np.asarray(guide) == 0).astype(np.uint8)
    near = cv2.distanceTransform(outside, cv2.DIST_L2, cv2.DIST_MASK_5) <= 160
    context = (outside != 0) & near & x[None, :] & y[:, None]
    if not use_alpha:
        context &= (ref[..., 3] >= 250) & (result[..., 3] >= 250)
    matches = _context_matches(ref, result, context, use_alpha=use_alpha)
    if len(matches) < 2:
        return [0, 0], [0.0] * 3

    def agreeing(match):
        return [other for other in matches if max(abs(other[2] - match[2]), abs(other[3] - match[3])) <= 1]

    inliers = max(map(agreeing, matches), key=lambda group: (len(group), sum(match[4] for match in group)))
    if len(inliers) < 2 or 2 * len(inliers) <= len(matches):
        return [0, 0], [0.0] * 3
    dx, dy = (round(value) for value in np.median([match[2:4] for match in inliers], axis=0))
    scale = FRAME / crop.width
    if not (
        0 <= (selection.left - crop.left) * scale + dx
        and (selection.right - crop.left) * scale + dx <= FRAME
        and 0 <= (selection.top - crop.top) * scale + dy
        and (selection.bottom - crop.top) * scale + dy <= FRAME
    ):
        return [0, 0], [0.0] * 3
    differences = []
    for left, top, *_ in inliers:
        a = ref[top:top + PATCH_SIZE, left:left + PATCH_SIZE, :3]
        b = result[top + dy:top + dy + PATCH_SIZE, left + dx:left + dx + PATCH_SIZE, :3]
        visible = ((ref[top:top + PATCH_SIZE, left:left + PATCH_SIZE, 3] >= 250)
                   & (result[top + dy:top + dy + PATCH_SIZE, left + dx:left + dx + PATCH_SIZE, 3] >= 250))
        if visible.any():
            differences.append(LINEAR_RGB[a[visible]] - LINEAR_RGB[b[visible]])
    if not differences:
        return [dx, dy], [0.0] * 3
    correction = np.clip(np.median(np.concatenate(differences), axis=0), -0.04, 0.04).tolist()
    return [dx, dy], correction


def generate_fill(
    base: Image.Image, mask: Image.Image, prompt: str, edit_alpha: bool, executable: Path, *,
    cancelled: Event, client_factory=CodexClient,
) -> FillCandidate:
    prompt = prompt.strip()
    selection = _selection(base, mask, edit_alpha)
    if cancelled.is_set():
        raise Cancelled("Generation canceled.")
    bounds = selection.left, selection.top, selection.right, selection.bottom
    if base.mode == "RGBA" and not edit_alpha and not selection_has_visible_pixels(base, mask, bounds):
        raise ValueError(TRANSPARENT_SELECTION)
    transition = _wide_fill_transition(mask, selection) if base.mode == "RGB" else 0
    tool_prompt = _prompt(prompt, alpha=base.mode == "RGBA", edit_alpha=edit_alpha, transition=bool(transition))
    return _generate_edit(base, mask, selection, tool_prompt, executable, cancelled, client_factory,
                          transition=transition, requested_edit=prompt)


def generate_pattern_fill(
    base: Image.Image, mask: Image.Image, executable: Path, *, cancelled: Event, client_factory=CodexClient,
) -> FillCandidate:
    edit_alpha = base.mode == "RGBA"
    selection = _selection(base, mask, edit_alpha)
    if base.width != base.height or base.width > FRAME:
        raise ValueError("The pattern fill needs a square working window.")
    if cancelled.is_set():
        raise Cancelled("Pattern repair canceled.")
    prompt = _prompt(
        "Repair the marked discontinuities in this repeating tile. Continue the existing motifs, shapes, "
        "spacing, texture, lighting, and color across the joins, including where two joins meet. "
        "Maintain the scale and alignment of the existing repeat. Match the surrounding texture on both "
        "sides of the mask. Use the unmarked content as fixed anchors, including lines or contours that "
        "enter the edit from a previously repaired part of the tile. Keep those lines at the same height, "
        "angle, thickness and spacing, continuing them right up to the anchors. Never reproduce the "
        "original straight cut within the edit. For patterned surfaces with rows or joints, trace each "
        "course from the anchors across the marked area. Keep joints continuous and the same width and "
        "color as their neighbors. The two edges of each joint should follow the same path, at the same "
        "distance apart as the neighboring joints. Preserve course spacing and brick height across the "
        "join; do not create a thin extra row or fill a gap with a second stripe of mortar. "
        "Continue the color of each cut surface from the adjoining piece. Preserve the original photograph's "
        "fine pores, chips, rough edges and quiet color variations. Keep the existing joints when they already "
        "line up and repair only their damaged sections. The repaired strip should have the same texture, "
        "contrast and saturation as its neighbors, with no retouched gloss. "
        "On transparent images, finish each interrupted silhouette "
        "with a single clean contour. Match the edge softness of intact motifs and keep crisp artwork "
        "crisp, including antialiasing but without a feathered halo. Use completely clear alpha in the "
        "gaps and match the opacity of adjacent motifs. Keep untouched empty space entirely clear.",
        alpha=edit_alpha, edit_alpha=edit_alpha, pattern=True,
    )
    return _generate_edit(base, mask, selection, prompt, executable, cancelled, client_factory, pattern=True)


def _generate_edit(
    base: Image.Image, mask: Image.Image, selection: Rect, tool_prompt: str, executable: Path,
    cancelled: Event, client_factory, *, transition: int = 0, requested_edit: str = "", pattern: bool = False,
) -> FillCandidate:
    crop = Rect(0, 0, *base.size) if pattern else _crop_box(selection, base.size)
    reference = _working_frame(base, crop, Image.Resampling.LANCZOS)
    coverage = _selection_coverage(mask, crop)
    guide = _working_frame(mask, crop, Image.Resampling.BILINEAR)
    placement = None
    if transition:
        x, y = _real_axes(crop, base.size)
        real = x[None, :] & y[:, None]
        distance = cv2.distanceTransform((coverage | ~real).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5)
        placement = (real & (coverage != 0) & (distance >= transition * FRAME / crop.width)).astype(np.uint8)
    with TemporaryDirectory(prefix="regimagen-fill-") as folder:
        directory = Path(folder)
        references = directory / "crop.png", directory / "selection.png", directory / "context.png"
        save_png(reference, references[0])
        save_png(_selection_reference(reference, coverage, placement=placement), references[1])
        save_png(_context_reference(base, selection, crop, focus=bool(transition)), references[2])
        if cancelled.is_set():
            raise Cancelled("Generation canceled.")
        # App Server can retain its working directory after the client exits on Windows.
        options = {"output_schema": FILL_OUTPUT_SCHEMA} if requested_edit else {}
        with client_factory(executable, directory.parent, cancelled=cancelled) as client:
            result = client.generate(tool_prompt, references, verify_references=True, **options)
    if cancelled.is_set():
        raise Cancelled("Generation canceled.")
    source_has_alpha = base.mode == "RGBA"
    transparency = _fill_transparency(result, source_has_alpha=source_has_alpha) if requested_edit else "preserve"
    image = _decode(result.data, require_alpha=source_has_alpha or transparency != "preserve",
                    allow_missing_alpha=transparency == "background")
    shift, color = _register(reference, image, guide, crop, base.size, selection,
                             use_alpha=pattern and source_has_alpha)
    if transparency == "background":
        selected = Image.new("L", (FRAME, FRAME))
        selected.paste(Image.fromarray(coverage * 255), tuple(shift))
        selected = selected.resize(image.size, Image.Resampling.BOX)
        alpha = _native_cutout_alpha(image, selected)
        if alpha is None:
            from .fill_foreground import fill_foreground_alpha

            alpha = fill_foreground_alpha(image, selected, requested_edit, executable,
                                          cancelled=cancelled, client_factory=client_factory)
        image.putalpha(alpha)
    feather = max(1, min(12, round(min(selection.width, selection.height) * 0.04)))
    if pattern:
        # Thin selections need a full-strength interior.
        feather = 1
    return FillCandidate(image, {
        "crop": [crop.left, crop.top, crop.right, crop.bottom],
        "shift": shift,
        "color": color,
        "feather": feather,
        "references_verified": result.references_verified,
        "transparency": transparency,
    })


def _smoothstep(value: np.ndarray) -> np.ndarray:
    t = np.clip(value, 0, 1)
    return t * t * (3 - 2 * t)


def _pattern_edge_weight(distance: np.ndarray, original: np.ndarray, candidate_alpha: np.ndarray) -> np.ndarray:
    nearby_depth = np.clip(cv2.dilate(distance, np.ones((25, 25), dtype=np.uint8)), 1, 12)
    if original.shape[2] == 3:
        return _smoothstep(distance / nearby_depth)
    # Where silhouettes disagree, use a shorter transition so they do not ghost.
    mismatch = abs(original[..., 3] / np.float32(255) - candidate_alpha[..., 0])
    width = np.minimum(nearby_depth, 12 - 9 * _smoothstep(mismatch / 0.65))
    return _smoothstep(distance / width)


def _to_srgb(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0, 1)
    scaled = np.where(value <= 0.0031308, 12.92 * value, 1.055 * value ** (1 / 2.4) - 0.055)
    return np.rint(scaled * 255).astype(np.uint8)


def _match_context_tones(
    base: Image.Image, mask: Image.Image, image: Image.Image, crop: Rect, shift: tuple[int, int], transition: int,
) -> Image.Image:
    # Use nearby, unedited pixels to estimate local color drift in the generated
    # crop. The correction fades as the reconstruction gets farther from context.
    reference = np.asarray(_working_frame(base, crop, Image.Resampling.LANCZOS))
    generated = np.asarray(image.resize((FRAME, FRAME), Image.Resampling.LANCZOS))
    dx, dy = shift
    offset = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned = cv2.warpAffine(generated, offset, (FRAME, FRAME), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REPLICATE)
    x, y = _real_axes(crop, base.size)
    real = x[None, :] & y[:, None]
    selected = _selection_coverage(mask, crop) != 0
    outside = cv2.distanceTransform((~selected).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5)
    width = max(2.0, transition * FRAME / crop.width)
    source = LINEAR_RGB[reference]
    rendered = LINEAR_RGB[aligned]
    delta = source - rendered
    context = real & ~selected & (outside <= width) & (np.abs(delta).max(axis=2) <= 0.2)
    kernel = np.ones((3, 3), np.uint8)
    source_range = cv2.morphologyEx(reference, cv2.MORPH_GRADIENT, kernel).max(axis=2)
    aligned_range = cv2.morphologyEx(aligned, cv2.MORPH_GRADIENT, kernel).max(axis=2)
    context &= (source_range <= 12) & (aligned_range <= 12)
    if not context.any():
        return image

    sigma = width * 0.75
    support = cv2.GaussianBlur(context.astype(np.float32), (0, 0), sigma, borderType=cv2.BORDER_CONSTANT)
    residual = cv2.GaussianBlur(np.clip(delta, -0.12, 0.12) * context[..., None], (0, 0), sigma,
                                borderType=cv2.BORDER_CONSTANT)
    field = residual / np.maximum(support[..., None], 0.0001)
    inside = cv2.distanceTransform((selected | ~real).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5)
    strength = _smoothstep(support / 0.035) * (1 - _smoothstep(inside / (2 * width)))
    field *= strength[..., None]
    field = cv2.warpAffine(field, offset, (FRAME, FRAME), borderMode=cv2.BORDER_REPLICATE)
    field = cv2.resize(field, image.size, interpolation=cv2.INTER_LINEAR)
    pixels = LINEAR_RGB[np.asarray(image)]
    return Image.fromarray(_to_srgb(pixels + field))


def _match_pattern_context(
    base: Image.Image, mask: Image.Image, image: Image.Image, shift: tuple[int, int],
) -> Image.Image:
    """Fit an opaque repair to the geometry and color of its unselected neighbors."""
    size = base.size
    reference = np.asarray(base)
    generated = np.asarray(image.resize(size, Image.Resampling.LANCZOS))
    dx, dy = (delta * size[0] / FRAME for delta in shift)
    x, y = np.meshgrid(np.arange(size[0], dtype=np.float32), np.arange(size[1], dtype=np.float32))
    aligned = cv2.remap(generated, x + dx, y + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    selected = np.asarray(mask) != 0
    outside = (~selected).astype(np.uint8)
    distance_out = cv2.distanceTransform(outside, cv2.DIST_L2, cv2.DIST_MASK_5)
    context = ~selected & (distance_out <= 32)
    if not context.any():
        return image

    dx_field, dy_field = _local_displacement(reference, aligned, selected, context, x, y)
    aligned = cv2.remap(generated, x + dx + dx_field, y + dy + dy_field,
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    aligned = match_course_widths(reference, aligned, selected)
    source = cv2.cvtColor(reference.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    patch = cv2.cvtColor(aligned.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    bias = np.clip(source - patch, (-12, -16, -16), (12, 16, 16)).astype(np.float32)
    # Estimate a separate local correction for red or otherwise chromatic material and for
    # neutral material such as mortar. Ignore edges where the two materials meet.
    material = _smoothstep((np.hypot(patch[..., 1], patch[..., 2]) - 12) / 22)
    source_material = _smoothstep((np.hypot(source[..., 1], source[..., 2]) - 12) / 22)
    similar = context & (abs(source[..., 0] - patch[..., 0]) < 22)
    correction = np.zeros_like(patch)
    for weight, reference_weight in ((material, source_material), (1 - material, 1 - source_material)):
        field, support = _pattern_context_field(bias, similar * weight * reference_weight, sigma=14)
        correction += field * (_smoothstep(support / 0.12) * weight)[..., None]
    fitted = patch + correction
    # Match source color throughout the repair while keeping the generated small-scale detail.
    low_texture = cv2.bilateralFilter(fitted, 7, 9.0, 2.0)
    targets = _pattern_material_targets(source, low_texture, selected)
    if targets is not None:
        surface_delta = np.clip(targets - low_texture[selected], (-18, -28, -24), (18, 28, 24))
        fitted[selected] += surface_delta * (0.55, 0.95, 0.95)
    # Retain the source-resolution texture instead of round-tripping the RGB through Lab.
    corrected = cv2.cvtColor(fitted, cv2.COLOR_LAB2RGB)
    neutral = cv2.cvtColor(patch, cv2.COLOR_LAB2RGB)
    rgb = aligned.astype(np.float32) + (corrected - neutral) * 255
    return Image.fromarray(np.rint(np.clip(rgb, 0, 255)).astype(np.uint8))


def _local_displacement(reference, aligned, selected, context, x, y):
    gray, edited = (cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY) for pixels in (reference, aligned))
    if min(gray.shape) < 32 or not context.any() or gray[context].std() < 5:
        return 0, 0
    solver = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    solver.setFinestScale(0)
    flow = solver.calc(gray, edited, None)
    warped = cv2.remap(edited, x + flow[..., 0], y + flow[..., 1], cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    gray = gray.astype(np.float32)
    before, after = (cv2.GaussianBlur(abs(gray - pixels), (0, 0), 2) for pixels in (edited, warped))
    valid = context & (np.linalg.norm(flow, axis=-1) <= 16) & (after + 2 < before)
    field, support = _pattern_context_field(flow, valid, sigma=8)
    depth = cv2.distanceTransform(selected.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5)
    strength = _smoothstep(support / 0.12) * (1 - _smoothstep(depth / 30))
    return field[..., 0] * strength, field[..., 1] * strength


def _pattern_context_field(values, weight, *, sigma):
    weight = weight.astype(np.float32)
    support = cv2.GaussianBlur(weight, (0, 0), sigma, borderType=cv2.BORDER_CONSTANT)
    field = cv2.GaussianBlur(values * weight[..., None], (0, 0), sigma, borderType=cv2.BORDER_CONSTANT)
    return field / np.maximum(support[..., None], 0.0001), support


def _pattern_material_targets(reference, candidate, selected):
    """Find the color of nearby source material with the same surface color."""
    outside = cv2.distanceTransform((~selected).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5)
    nearby = ~selected & (outside <= 32)
    grid = np.s_[::2, ::2]
    colors = reference[grid][nearby[grid]]
    positions = np.argwhere(nearby[grid]).astype(np.float32) * 2
    if not len(colors):
        return None
    stride = max(1, (len(colors) + 20000 - 1) // 20000)
    colors, positions = colors[::stride], positions[::stride]

    def features(points, lab):
        # Luminance and hue identify the material. Saturation is left free to match the source.
        hue = np.arctan2(lab[:, 2], lab[:, 1])
        chroma = np.hypot(lab[:, 1], lab[:, 2])
        hue_weight = np.minimum(chroma, 22)
        return np.column_stack((points / 2, lab[:, 0] * 1.35, hue_weight * np.cos(hue) * 2,
                                hue_weight * np.sin(hue) * 2)).astype(np.float32)

    samples = features(positions, colors)
    index = cv2.flann_Index(samples, {"algorithm": 1, "trees": 2})
    points = np.argwhere(selected)
    rgb = candidate[selected]
    matches = np.zeros_like(rgb)
    for left in range(0, len(points), 32768):
        region = slice(left, left + 32768)
        ids, distances = index.knnSearch(features(points[region], rgb[region]), min(5, len(colors)), params={"checks": 24})
        weights = 1 / (25 + distances)
        matches[region] = (colors[ids] * weights[..., None]).sum(axis=1) / weights.sum(axis=1)[:, None]
    return matches


def _candidate_box(
    side: int, region: Rect, crop: Rect, shift: tuple[int, int],
) -> tuple[float, float, float, float]:
    scale = side / crop.width
    dx, dy = (value * side / FRAME for value in shift)
    return (
        (region.left - crop.left) * scale + dx, (region.top - crop.top) * scale + dy,
        (region.right - crop.left) * scale + dx, (region.bottom - crop.top) * scale + dy,
    )


def _candidate_channels(
    image: Image.Image, box: tuple[float, float, float, float],
) -> tuple[tuple[Image.Image, ...], tuple[int, int]]:
    pixels = np.asarray(image.convert("RGBA"))
    alpha = pixels[..., 3].astype(np.float32) / 255
    left, top, right, bottom = box
    pad = (max(0, math.ceil(-top)), max(0, math.ceil(bottom - image.height))), (
        max(0, math.ceil(-left)), max(0, math.ceil(right - image.width)),
    )

    def channel_image(channel: np.ndarray) -> Image.Image:
        if any(pad[0]) or any(pad[1]):
            channel = np.pad(channel, pad, mode="edge")
        return Image.fromarray(channel)

    channels = tuple(channel_image(LINEAR_RGB[pixels[..., index]] * alpha) for index in range(3))
    return (*channels, channel_image(alpha)), (pad[1][0], pad[0][0])


def _sample_candidate(
    channels: tuple[Image.Image, ...], region: Rect,
    box: tuple[float, float, float, float], offset: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    # Use the same antialiasing filter at crop edges and across every stripe.
    x, y = offset
    left, top, right, bottom = box
    box = left + x, top + y, right + x, bottom + y
    sampled = [np.asarray(channel.resize((region.width, region.height), Image.Resampling.LANCZOS, box=box))
               for channel in channels]
    alpha = np.clip(sampled[3], 0, 1)[..., None]
    return np.clip(np.stack(sampled[:3], axis=-1), 0, alpha), alpha


def _blend(
    original: np.ndarray, mask: np.ndarray, distance: np.ndarray | None, candidate: np.ndarray,
    candidate_alpha: np.ndarray, color: np.ndarray, feather: int, edit_alpha: bool,
    *, transition: int = 0, soft_feather: int = 0,
) -> np.ndarray:
    source = LINEAR_RGB[original[..., :3]]
    source_alpha = original[..., 3:4].astype(np.float32) / 255 if original.shape[2] == 4 else 1.0

    coverage = mask[..., None].astype(np.float32) / 255
    weight = coverage
    if distance is not None:
        if transition:
            # A strong edge in the generated image must not tighten the blend and
            # reveal the contour of a wide lasso against smooth backgrounds.
            width = transition
        else:
            # Narrow the blend when the two images disagree along strong structure.
            matte = LINEAR_RGB[list(TRANSPARENT_MATTE)]
            a = cv2.cvtColor(source * source_alpha + matte * (1 - source_alpha), cv2.COLOR_RGB2GRAY)
            b = cv2.cvtColor(candidate + matte * (1 - candidate_alpha), cv2.COLOR_RGB2GRAY)
            difference = np.abs(cv2.Laplacian(a - b, cv2.CV_32F))
            width = np.maximum(1, feather * (1 - 0.65 * _smoothstep(difference / 0.35)))
        if transition:
            distance = np.maximum(0, distance - 0.5)
        color_correction = color * (1 - _smoothstep(distance / (2 * feather)))[..., None]
        if soft_feather > feather and np.any(mask == 0):
            # Broaden the transition where both images show the same smooth surface.
            kernel = np.ones((3, 3), np.uint8)
            detail = np.maximum(cv2.morphologyEx(a, cv2.MORPH_GRADIENT, kernel),
                                cv2.morphologyEx(b, cv2.MORPH_GRADIENT, kernel))
            delta = source - candidate
            smooth = (1 - _smoothstep(detail / 0.12)) * (1 - _smoothstep((np.abs(delta).max(axis=2) - 0.04) / 0.12))
            smooth = cv2.GaussianBlur(smooth, (5, 5), 0)
            width += (soft_feather - feather) * smooth
            context = smooth * (mask == 0)
            window = (2 * soft_feather + 1,) * 2
            support = cv2.boxFilter(context, -1, window, borderType=cv2.BORDER_CONSTANT)
            residual = cv2.boxFilter(delta * context[..., None], -1, window, borderType=cv2.BORDER_CONSTANT)
            local_color = residual / np.maximum(support[..., None], 0.001)
            strength = smooth * _smoothstep(support / 0.15) * (1 - _smoothstep(distance / soft_feather))
            color_correction += (local_color - color_correction) * strength[..., None]
        weight *= _smoothstep(distance / width)[..., None]
        candidate = np.clip(candidate + candidate_alpha * color_correction, 0, candidate_alpha)
    result = original.copy()
    if edit_alpha:
        alpha = source_alpha + (candidate_alpha - source_alpha) * weight
        premult = source * source_alpha * (1 - weight) + candidate * weight
        color_result = np.divide(premult, alpha, out=np.zeros_like(premult), where=alpha > 0)
        result[..., 3] = np.rint(alpha[..., 0] * 255).astype(np.uint8)
        use = mask != 0
    else:
        straight = np.divide(candidate, candidate_alpha, out=source.copy(), where=candidate_alpha > 0)
        if original.shape[2] == 4:
            weight *= np.minimum(1, candidate_alpha / np.maximum(source_alpha, 1 / 255))
        color_result = source + (straight - source) * weight
        use = (mask != 0) & (candidate_alpha[..., 0] > 0)
        if original.shape[2] == 4:
            use &= original[..., 3] != 0
    result[..., :3][use] = _to_srgb(color_result[use])
    return result


def _gray_for_edges(image: Image.Image, background: tuple[int, int, int]) -> np.ndarray:
    if image.mode == "RGBA":
        matte = Image.new("RGBA", image.size, (*background, 255))
        image = Image.alpha_composite(matte, image).convert("RGB")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)


def _alpha_seam_warning(
    reference: Image.Image, composed: Image.Image, rim: np.ndarray, dx: np.ndarray, dy: np.ndarray,
    crop: Rect,
) -> bool:
    original = np.asarray(reference.getchannel("A"), dtype=np.int16)
    edited = np.asarray(composed.getchannel("A"), dtype=np.int16)
    if np.array_equal(original, edited):
        return False
    # Sample across roughly one source pixel even when the guide is enlarged.
    step = max(1, math.ceil(FRAME / crop.width))
    for axis, normal, tangent in ((0, dy, dx), (1, dx, dy)):
        before = np.moveaxis(original, axis, 0)
        after = np.moveaxis(edited, axis, 0)
        boundary = np.moveaxis(rim & (np.abs(normal) >= np.abs(tangent)) & (normal != 0), axis, 0)
        original_jump = np.abs(before[step:] - before[:-step])
        edited_jump = np.abs(after[step:] - after[:-step])
        new_jumps = (boundary[step:] | boundary[:-step]) & (edited_jump - original_jump >= 32)
        if new_jumps.any():
            _, _, stats, _ = cv2.connectedComponentsWithStats(new_jumps.astype(np.uint8), connectivity=8)
            if np.any(stats[1:, cv2.CC_STAT_AREA] >= max(8, 4 * step)):
                return True
    return False


def _seam_warning(
    base: Image.Image, preview: FillPreview, mask: Image.Image, crop: Rect, feather: int, *, wide: bool = False,
) -> bool:
    guide = _working_frame(mask, crop, Image.Resampling.BILINEAR)
    x, y = _real_axes(crop, mask.size)
    real = x[None, :] & y[:, None]
    selected = np.asarray(guide) >= 128
    inside = cv2.distanceTransform((selected | ~real).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5)
    scale = FRAME / crop.width
    rim = selected & real & (inside <= max(6, (feather + 3) * scale))
    if not rim.any():
        return False
    reference = _working_frame(base, crop, Image.Resampling.BILINEAR)
    composed = _working_frame(preview.compose(base), crop, Image.Resampling.BILINEAR)
    # The distance gradient gives the boundary normal through the whole blend.
    distance = np.minimum(inside, 2 * FRAME)
    dx = cv2.Sobel(distance, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(distance, cv2.CV_32F, 0, 1, ksize=3)
    if base.mode == "RGBA" and _alpha_seam_warning(reference, composed, rim, dx, dy, crop):
        return True
    length = np.hypot(dx, dy)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    backgrounds = (TRANSPARENT_MATTE, (225, 225, 225)) if base.mode == "RGBA" else (TRANSPARENT_MATTE,)
    for background in backgrounds:
        original = _gray_for_edges(reference, background)
        edited = _gray_for_edges(composed, background)
        original_edges = cv2.Canny(original, 30, 70)
        edited_edges = cv2.Canny(edited, 30, 70)
        gx = cv2.Sobel(original, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(original, cv2.CV_32F, 0, 1, ksize=3)
        crossing = np.abs(gx * dx + gy * dy) <= 0.8 * length * np.hypot(gx, gy)
        check = rim & (length > 2) & crossing & (original_edges != 0)
        total = np.count_nonzero(check)
        if total >= 12:
            missing = cv2.dilate(edited_edges, kernel) == 0
            if wide:
                # A smooth tonal change can break Canny hysteresis even when
                # the local edge still has the same direction and strength.
                ex = cv2.Sobel(edited, cv2.CV_32F, 1, 0, ksize=3)
                ey = cv2.Sobel(edited, cv2.CV_32F, 0, 1, ksize=3)
                projection = gx * ex + gy * ey
                supported = (projection >= 0.65 * (gx * gx + gy * gy)) & (np.abs(gx * ey - gy * ex) <= 0.75 * projection)
                missing &= ~supported
            if np.count_nonzero(check & missing) / total > 0.35:
                return True
        step = np.abs(cv2.Laplacian(edited.astype(np.float32) - original, cv2.CV_32F))
        if np.count_nonzero(rim & (step > 34)) > max(16, np.count_nonzero(rim) // 20):
            return True
    return False


def render_fill(
    base: Image.Image, mask: Image.Image, candidate: FillCandidate, *, edit_alpha: bool,
    isolated_layer: bool = False, pattern: bool = False,
) -> FillPreview:
    if candidate.metadata.get("transparency") in ("edit", "background"):
        edit_alpha = True
        if base.mode == "RGB":
            base = base.convert("RGBA")
    selection = _selection(base, mask, edit_alpha)
    if candidate.image.width != candidate.image.height or candidate.image.mode != base.mode:
        raise ValueError("The saved fill does not match the source image format.")
    try:
        crop = Rect(*candidate.metadata["crop"])
        shift = tuple(candidate.metadata["shift"])
        color = np.array(candidate.metadata["color"], dtype=np.float32)
        feather = candidate.metadata["feather"]
        if (
            any(type(value) is not int for value in (crop.left, crop.top, crop.right, crop.bottom))
            or crop.width != crop.height or selection.intersect(crop) != selection
            or len(shift) != 2 or any(type(value) is not int or abs(value) > SHIFT_LIMIT for value in shift)
            or color.shape != (3,) or not np.isfinite(color).all() or np.abs(color).max() > 0.041
            or type(feather) is not int or not 1 <= feather <= 12
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("The saved fill geometry is invalid for this selection.") from exc

    patch = (Image.new("RGBA", (selection.width, selection.height)) if isolated_layer
             else base.crop((selection.left, selection.top, selection.right, selection.bottom)))
    canvas = Rect(0, 0, *base.size)
    opaque_fill = base.mode == "RGB" and not pattern
    transition = _wide_fill_transition(mask, selection) if opaque_fill else 0
    short_side = min(selection.width, selection.height)
    soft_feather = min(24, round(short_side * 0.16)) if opaque_fill and not transition and short_side >= 96 else 0
    generated = (_match_context_tones(base, mask, candidate.image, crop, shift, transition)
                 if transition else candidate.image)
    if pattern and base.mode == "RGB":
        generated = _match_pattern_context(base, mask, generated, shift)
        if generated is not candidate.image:
            shift = (0, 0)
    margin = max(24 if pattern else 2 * feather, transition, soft_feather + 2) + 2
    work = selection.expand(margin, margin).intersect(canvas)
    assert work is not None
    channels, offset = _candidate_channels(generated, _candidate_box(generated.width, work, crop, shift))
    rows = max(1, min(128, 500_000 // selection.width))
    for top in range(selection.top, selection.bottom, rows):
        region = Rect(selection.left, top, selection.right, min(top + rows, selection.bottom))
        # Include enough neighbors for both the feather and the edge filter.
        expanded = region.expand(margin, margin).intersect(canvas)
        assert expanded is not None
        part = region.relative_to(expanded).slices
        expanded_box = expanded.left, expanded.top, expanded.right, expanded.bottom
        selected = np.asarray(mask.crop(expanded_box))
        if not selected[part].any():
            continue
        # Use the selection coverage directly for cutouts to preserve generated alpha.
        distance = (None if candidate.transparent_background
                    else cv2.distanceTransform((selected != 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5))
        original = (np.zeros((*selected.shape, 4), dtype=np.uint8) if isolated_layer
                    else np.asarray(base.crop(expanded_box)))
        box = _candidate_box(generated.width, expanded, crop, shift)
        pixels, alpha = _sample_candidate(channels, expanded, box, offset)
        if pattern:
            # Blend back to the original texture near the mask boundary.
            # A narrow region still reaches full weight at its center.
            selected = np.rint(selected * _pattern_edge_weight(distance, original, alpha)).astype(np.uint8)
            distance = None
        blended = _blend(original, selected, distance, pixels, alpha, color, feather, edit_alpha,
                         transition=transition, soft_feather=soft_feather)
        patch.paste(Image.fromarray(blended[part]), (0, top - selection.top))

    preview = FillPreview(patch, (selection.left, selection.top))
    warnings = []
    if candidate.metadata.get("references_verified", True) is None:
        warnings.append("Could not verify that ImageGen used the context references.")
    if candidate.image.width < crop.width:
        warnings.append("ImageGen returned a smaller image. The edit was enlarged to fit the source.")
    if not candidate.transparent_background and _seam_warning(
        base, preview, mask, crop, max(12 if pattern else feather, transition), wide=bool(transition),
    ):
        warnings.append("Check the selection edge for a visible seam.")
    return FillPreview(patch, preview.origin, " ".join(warnings))
