"""Prepare and composite a single generative edit in source coordinates."""

import io
import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw

from .codex import Cancelled, CodexClient
from .composite import LINEAR_RGB
from .files import TRANSPARENT_MATTE, _srgb, reference_rgb, save_png
from .geometry import FRAME, Rect

SHIFT_LIMIT = 8
PATCH_SIZE = 32
CONTEXT_PANEL = FRAME // 2
SELECTION_MARK = (255, 73, 215)
SELECTION_DARK = (12, 14, 22)
TRANSPARENT_SELECTION = "Select visible pixels or turn on Edit Transparency."
ALPHA_CROSSING_LIMIT = 128
ALPHA_CONTACT_MARGIN = 18
ALPHA_CONTEXT_PIXELS = 4_000_000


@dataclass(frozen=True)
class FillCandidate:
    image: Image.Image
    metadata: dict


@dataclass(frozen=True)
class FillPreview:
    patch: Image.Image
    origin: tuple[int, int]
    warning: str = ""

    def compose(self, base: Image.Image) -> Image.Image:
        image = base.copy()
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

    def start(low: int, high: int, extent: int) -> int:
        center = (low + high - side) // 2
        return max(min(0, extent - side), min(max(0, extent - side), center))

    left = start(selection.left, selection.right, size[0])
    top = start(selection.top, selection.bottom, size[1])
    return Rect(left, top, left + side, top + side)


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
    reference: Image.Image, selected: np.ndarray, *, show_extent: bool = False,
) -> Image.Image:
    marked = reference_rgb(reference).copy()
    if show_extent:
        # The original object may be completely hidden in this reference.
        marked.paste((255, 170, 255), mask=Image.fromarray(selected * 44))
    inner = cv2.erode(selected, np.ones((5, 5), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    _paint_selection(marked, selected != inner)
    return marked


def _context_reference(
    base: Image.Image, selection: Rect, crop: Rect, *, hide: Image.Image | None = None,
) -> Image.Image:
    canvas = Rect(0, 0, *base.size)
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
        if hide is not None:
            panel = _provisional_background(panel, _selection_coverage(hide, view, size))
        panel = reference_rgb(panel)
        _mark_selection(panel, selection, view)
        context.paste(panel, (
            column * CONTEXT_PANEL + (CONTEXT_PANEL - panel.width) // 2,
            (CONTEXT_PANEL - panel.height) // 2,
        ))
    return context


def _provisional_background(image: Image.Image, hidden: np.ndarray) -> Image.Image:
    """Hide the old subject with a local approximation used only as a reference."""
    if not hidden.any():
        return image
    if hidden.all():
        matte = (0, 0, 0, 0) if image.mode == "RGBA" else TRANSPARENT_MATTE
        return Image.new(image.mode, image.size, matte)
    pixels = np.asarray(image)
    if image.mode == "RGB":
        return Image.fromarray(cv2.inpaint(pixels, hidden, 3, cv2.INPAINT_TELEA))

    # Inpaint color already weighted by opacity so hidden RGB cannot contaminate edges.
    premultiplied = np.asarray(image.convert("RGBa"))
    color = cv2.inpaint(np.ascontiguousarray(premultiplied[..., :3]), hidden, 3, cv2.INPAINT_TELEA)
    alpha = cv2.inpaint(premultiplied[..., 3].copy(), hidden, 3, cv2.INPAINT_TELEA)
    pixels = pixels.copy()
    edited = hidden != 0
    pixels[edited] = np.asarray(Image.frombytes(
        "RGBa", image.size, np.dstack((np.minimum(color, alpha[..., None]), alpha)).tobytes(),
    ).convert("RGBA"))[edited]
    return Image.fromarray(pixels)


def _prepared_crop(
    reference: Image.Image, hidden: np.ndarray, crop: Rect, source_size: tuple[int, int],
) -> Image.Image:
    if reference.mode == "RGBA" or crop.intersect(Rect(0, 0, *source_size)) == crop:
        return _provisional_background(reference, hidden)
    x, y = _real_axes(crop, source_size)
    columns, rows = np.flatnonzero(x), np.flatnonzero(y)
    if not columns.size or not rows.size:
        return _provisional_background(reference, hidden)
    # Regenerate padding from the prepared frame, so an edge object cannot
    # reappear as copies of its old colors in virtual context.
    left, right = int(columns[0]), int(columns[-1]) + 1
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    prepared = _provisional_background(reference.crop((left, top, right, bottom)), hidden[top:bottom, left:right])
    columns = np.clip(np.arange(FRAME) - left, 0, prepared.width - 1)
    rows = np.clip(np.arange(FRAME) - top, 0, prepared.height - 1)
    return Image.fromarray(np.asarray(prepared)[rows[:, None], columns])


def _decode(data: bytes, require_alpha: bool) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data), formats=("PNG",)) as image:
            if image.width != image.height:
                raise ValueError("ImageGen returned a non-square image. Generate another variation.")
            if require_alpha and "A" not in image.getbands() and "transparency" not in image.info:
                raise ValueError("ImageGen returned a PNG without transparency information. Generate another variation.")
            normalized = _srgb(image, preserve_alpha=require_alpha)
            if require_alpha and normalized.mode != "RGBA":
                # An encoded, fully opaque alpha channel is valid too.
                normalized = normalized.convert("RGBA")
            return normalized
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("ImageGen returned an invalid PNG. Generate another variation.") from exc


def _prompt(prompt: str, *, alpha: bool, edit_alpha: bool) -> str:
    result = (
        "Inspect the three references before calling the native image_gen tool exactly once. "
        "Pass all three image files in their attached order. Image 1 is the clean square to edit. "
        "Image 2 shows the same square and coordinates, with the editable area outlined in "
        "magenta or dark ink. Image 3 is context only: the left panel shows the nearby surroundings "
        "and the right panel shows the whole composition. The marked boxes locate the selection. "
        "Identify the subject, the nearest relevant surface, nearby matching parts, and visible "
        "style cues. Include specific observations from the references in the image_gen prompt "
        "to ground placement, scale, perspective, material, texture, lighting and contact shadows. "
        "Make added objects sit on or attach to the existing geometry where appropriate. Explicit "
        "user requests take priority for any visual attribute they specify. Keep additions and "
        "their shadows inside the outline and continue any contours that cross it. Preserve "
        "unrelated details and omit the guide marks and panels from the output. Return one square "
        "PNG of image 1, preferably 1024 by 1024 pixels, with its exact framing and subject positions. "
    )
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
    return result + (f"\nRequested edit: {prompt.strip()}" if prompt.strip() else (
        "Fill the selection naturally with a continuation of its surroundings."
    ))


def _remove_prompt(labels: Iterable[str], *, alpha: bool, prepared: bool, clear_restored: bool) -> str:
    targets = json.dumps(list(labels), ensure_ascii=False)
    crop_description = "square working crop" if prepared or clear_restored else "original square crop"
    source = f"Image 1 is the {crop_description}. " + (
        "The target and its fringe have been hidden with a rough local background approximation. "
        "Replace that approximation with a coherent reconstruction. "
        if prepared else "Remove the marked objects and their marked effects. "
    )
    prompt = (
        "Call the native image_gen tool exactly once. Pass all three attached images in their given order. "
        + source
        + "Image 2 marks the approved edit area on the same square and coordinates. "
        "Image 3 shows local and whole-image context. Use those views to infer what was behind "
        "the removed objects. Continue crossing lines, texture, surfaces, perspective and illumination. "
        "Preserve neighboring and overlapping objects marked outside the edit, including small details. "
        "Do not recreate a removed silhouette, shadow, reflection or object. Omit guide ink, boxes and panels. "
        "Return one square PNG at exactly the field of view and alignment of image 1, preferably 1024 by 1024. "
        f"Removed object labels, for reference only: {targets}. "
    )
    if clear_restored:
        prompt += (
            "The reliably empty part of the old target is already transparent in the references. "
            "Leave that part empty and reconstruct only the remaining marked area, where the target "
            "may have covered a visible surface. "
        )
    if alpha:
        return prompt + (
            "Set transparent_background=true. Restore true PNG transparency where the background is empty. "
            "Continue visible surfaces where the objects occluded them. Keep existing transparency and "
            "semitransparent contours elsewhere and do not paint a checkerboard or a solid backdrop."
        )
    return prompt + "Return an opaque PNG."


def _context_matches(
    reference: np.ndarray, result: np.ndarray, context: np.ndarray,
) -> list[tuple[int, int, int, int, float]]:
    ref_gray = cv2.cvtColor(reference[..., :3], cv2.COLOR_RGB2GRAY)
    out_gray = cv2.cvtColor(result[..., :3], cv2.COLOR_RGB2GRAY)
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
    source_size: tuple[int, int], selection: Rect,
) -> tuple[list[int], list[float]]:
    ref = np.asarray(reference.convert("RGBA"))
    result = np.asarray(generated.resize((FRAME, FRAME), Image.Resampling.LANCZOS).convert("RGBA"))
    x, y = _real_axes(crop, source_size)
    outside = (np.asarray(guide) == 0).astype(np.uint8)
    near = cv2.distanceTransform(outside, cv2.DIST_L2, cv2.DIST_MASK_5) <= 160
    context = (outside != 0) & near & x[None, :] & y[:, None] & (ref[..., 3] >= 250) & (result[..., 3] >= 250)
    matches = _context_matches(ref, result, context)
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
        differences.append((LINEAR_RGB[a] - LINEAR_RGB[b]).reshape(-1, 3))
    correction = np.clip(np.median(np.concatenate(differences), axis=0), -0.04, 0.04).tolist()
    return [dx, dy], correction


def generate_fill(
    base: Image.Image, mask: Image.Image, prompt: str, edit_alpha: bool, executable: Path, *,
    cancelled: Event, client_factory=CodexClient,
) -> FillCandidate:
    selection = _selection(base, mask, edit_alpha)
    if cancelled.is_set():
        raise Cancelled("Generation canceled.")
    bounds = selection.left, selection.top, selection.right, selection.bottom
    if base.mode == "RGBA" and not edit_alpha and not selection_has_visible_pixels(base, mask, bounds):
        raise ValueError(TRANSPARENT_SELECTION)
    tool_prompt = _prompt(prompt, alpha=base.mode == "RGBA", edit_alpha=edit_alpha)
    return _generate_edit(base, mask, selection, tool_prompt, executable, cancelled, client_factory)


def _removal_selection(base: Image.Image, core: Image.Image, footprint: Image.Image) -> Rect:
    selection = _selection(base, footprint, base.mode == "RGBA")
    if core.mode != "L" or core.size != base.size:
        raise ValueError("The removal masks must match the source image.")
    bounds = core.getbbox()
    if bounds is None:
        raise ValueError("Select at least one object or area to remove.")
    left, top, right, bottom = bounds
    for y in range(top, bottom, 128):
        box = (left, y, right, min(y + 128, bottom))
        unapproved = footprint.crop(box).point([255] + [0] * 255)
        if ImageChops.multiply(core.crop(box), unapproved).getbbox() is not None:
            raise ValueError("The approved area must contain the entire removal mask.")
    return selection


def generate_remove(
    base: Image.Image, core: Image.Image, footprint: Image.Image, labels: Iterable[str],
    executable: Path | Callable[[], Path], *, cancelled: Event, client_factory=CodexClient, prepare: bool = True,
    prepare_crop: Callable[[Image.Image, np.ndarray, Rect, tuple[int, int]], Image.Image] | None = None,
) -> FillCandidate:
    if cancelled.is_set():
        raise Cancelled("Removal canceled.")
    selection = _removal_selection(base, core, footprint)
    clear = _removal_alpha_mask(base, footprint, selection)
    if cancelled.is_set():
        raise Cancelled("Removal canceled.")
    if clear is not None:
        footprint = ImageChops.subtract(footprint, clear)
        if footprint.getbbox() is None:
            crop = _crop_box(selection, base.size)
            return FillCandidate(Image.new("RGBA", (1, 1)), {
                "crop": [crop.left, crop.top, crop.right, crop.bottom],
                "shift": [0, 0], "color": [0, 0, 0], "feather": 1, "generation_backend": "alpha",
            })
        base = _clear_background(base, clear)
    tool_prompt = _remove_prompt(labels, alpha=base.mode == "RGBA", prepared=prepare, clear_restored=clear is not None)
    return _generate_edit(
        base, footprint, selection, tool_prompt, executable() if callable(executable) else executable,
        cancelled, client_factory, removal=True, prepare=prepare, prepare_crop=prepare_crop,
    )


def _generate_edit(
    base: Image.Image, mask: Image.Image, selection: Rect, tool_prompt: str, executable: Path,
    cancelled: Event, client_factory, *, removal: bool = False, prepare: bool = False,
    prepare_crop: Callable[[Image.Image, np.ndarray, Rect, tuple[int, int]], Image.Image] | None = None,
) -> FillCandidate:
    crop = _crop_box(selection, base.size)
    reference = _working_frame(base, crop, Image.Resampling.LANCZOS)
    coverage = _selection_coverage(mask, crop)
    guide = Image.fromarray(coverage * 255) if removal else _working_frame(mask, crop, Image.Resampling.BILINEAR)
    target = (prepare_crop or _prepared_crop)(reference, coverage, crop, base.size) if prepare else reference
    context_selection = Rect(*mask.getbbox()) if removal else selection
    with TemporaryDirectory(prefix="regimagen-fill-") as folder:
        directory = Path(folder)
        references = directory / "crop.png", directory / "selection.png", directory / "context.png"
        save_png(target, references[0])
        save_png(_selection_reference(target, coverage, show_extent=removal), references[1])
        save_png(_context_reference(base, context_selection, crop, hide=mask if prepare else None), references[2])
        if cancelled.is_set():
            raise Cancelled("Generation canceled.")
        # App Server can retain its working directory after the client exits on Windows.
        with client_factory(executable, directory.parent, cancelled=cancelled) as client:
            result = client.generate(tool_prompt, references, verify_references=True)
    if cancelled.is_set():
        raise Cancelled("Generation canceled.")
    image = _decode(result.data, require_alpha=base.mode == "RGBA")
    shift, color = _register(reference, image, guide, crop, base.size, selection)
    return FillCandidate(image, {
        "crop": [crop.left, crop.top, crop.right, crop.bottom],
        "shift": shift,
        "color": color,
        "feather": max(1, min(12, round(min(selection.width, selection.height) * 0.04))),
        "references_verified": result.references_verified,
    })


def _smoothstep(value: np.ndarray) -> np.ndarray:
    t = np.clip(value, 0, 1)
    return t * t * (3 - 2 * t)


def _to_srgb(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0, 1)
    scaled = np.where(value <= 0.0031308, 12.92 * value, 1.055 * value ** (1 / 2.4) - 0.055)
    return np.rint(scaled * 255).astype(np.uint8)


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
    original: np.ndarray, mask: np.ndarray, distance: np.ndarray, candidate: np.ndarray,
    candidate_alpha: np.ndarray, color: np.ndarray, feather: int, edit_alpha: bool,
    replace_core: np.ndarray | None = None,
) -> np.ndarray:
    source = LINEAR_RGB[original[..., :3]]
    source_alpha = original[..., 3:4].astype(np.float32) / 255 if original.shape[2] == 4 else 1.0

    # Narrow the blend when the two images disagree along strong structure.
    matte = LINEAR_RGB[list(TRANSPARENT_MATTE)]
    a = cv2.cvtColor(source * source_alpha + matte * (1 - source_alpha), cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(candidate + matte * (1 - candidate_alpha), cv2.COLOR_RGB2GRAY)
    difference = np.abs(cv2.Laplacian(a - b, cv2.CV_32F))
    width = np.maximum(1, feather * (1 - 0.65 * _smoothstep(difference / 0.35)))
    if replace_core is not None:
        # The first pixel of a narrow approved band still needs a soft edge.
        distance = np.maximum(0, distance - 0.5)
    weight = (_smoothstep(distance / width) * (mask.astype(np.float32) / 255))[..., None]
    color_strength = (1 - _smoothstep(distance / (2 * feather)))[..., None]
    if replace_core is not None:
        weight[replace_core] = 1
        color_strength[replace_core] = 0
    candidate = np.clip(candidate + candidate_alpha * color * color_strength, 0, candidate_alpha)
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


def _seam_warning(base: Image.Image, preview: FillPreview, mask: Image.Image, crop: Rect, feather: int) -> bool:
    guide = _working_frame(mask, crop, Image.Resampling.BILINEAR)
    x, y = _real_axes(crop, mask.size)
    real = x[None, :] & y[:, None]
    selected = np.asarray(guide) >= 128
    inside = cv2.distanceTransform((selected | ~real).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5)
    rim = selected & real & (inside <= max(6, (feather + 3) * FRAME / crop.width))
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
        if total >= 12 and np.count_nonzero(check & (cv2.dilate(edited_edges, kernel) == 0)) / total > 0.35:
            return True
        step = np.abs(cv2.Laplacian(edited.astype(np.float32) - original, cv2.CV_32F))
        if np.count_nonzero(rim & (step > 34)) > max(16, np.count_nonzero(rim) // 20):
            return True
    return False


def render_fill(
    base: Image.Image, mask: Image.Image, candidate: FillCandidate, *, edit_alpha: bool,
    replace_core: Image.Image | None = None,
) -> FillPreview:
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

    patch = base.crop((selection.left, selection.top, selection.right, selection.bottom))
    canvas = Rect(0, 0, *base.size)
    margin = 2 * feather + 2
    work = selection.expand(margin, margin).intersect(canvas)
    assert work is not None
    channels, offset = _candidate_channels(candidate.image, _candidate_box(candidate.image.width, work, crop, shift))
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
        distance = cv2.distanceTransform((selected != 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5)
        original = np.asarray(base.crop(expanded_box))
        box = _candidate_box(candidate.image.width, expanded, crop, shift)
        pixels, alpha = _sample_candidate(channels, expanded, box, offset)
        core = np.asarray(replace_core.crop(expanded_box)) != 0 if replace_core is not None else None
        blended = _blend(original, selected, distance, pixels, alpha, color, feather, edit_alpha, core)
        patch.paste(Image.fromarray(blended[part]), (0, top - selection.top))

    preview = FillPreview(patch, (selection.left, selection.top))
    warnings = []
    if candidate.metadata.get("references_verified", True) is None:
        warnings.append("Could not verify that ImageGen used the context references.")
    if candidate.image.width < crop.width:
        warnings.append("ImageGen returned a smaller image. The edit was enlarged to fit the source.")
    if _seam_warning(base, preview, mask, crop, feather):
        warnings.append("Check the selection edge for a visible seam.")
    return FillPreview(patch, preview.origin, " ".join(warnings))


def render_remove(
    base: Image.Image, core: Image.Image, footprint: Image.Image, candidate: FillCandidate,
) -> FillPreview:
    selection = _removal_selection(base, core, footprint)
    clear = _removal_alpha_mask(base, footprint, selection)
    remaining = ImageChops.subtract(footprint, clear) if clear is not None else footprint
    needs_reconstruction = clear is None or remaining.getbbox() is not None
    if needs_reconstruction and candidate.metadata.get("generation_backend") == "alpha":
        raise ValueError("The saved transparency removal does not match this selection.")
    if clear is None:
        return render_fill(base, footprint, candidate, edit_alpha=base.mode == "RGBA", replace_core=core)

    bounds = selection.left, selection.top, selection.right, selection.bottom
    origin = selection.left, selection.top
    if not needs_reconstruction:
        return FillPreview(_clear_background(base.crop(bounds), clear.crop(bounds)), origin)

    prepared = _clear_background(base, clear)
    remaining_core = ImageChops.subtract(core, clear)
    preview = render_fill(prepared, remaining, candidate, edit_alpha=True, replace_core=remaining_core)
    patch = prepared.crop(bounds)
    patch.paste(preview.patch, (preview.origin[0] - selection.left, preview.origin[1] - selection.top))
    return FillPreview(patch, origin, preview.warning)


def _clear_background(base: Image.Image, clear: Image.Image) -> Image.Image:
    prepared = base.copy()
    prepared.paste((0, 0, 0, 0), mask=clear)
    return prepared


def removal_alpha_mask(base: Image.Image, footprint: Image.Image) -> Image.Image | None:
    """Find approved source pixels that can be replaced with clear background."""
    selection = _selection(base, footprint, base.mode == "RGBA")
    return _removal_alpha_mask(base, footprint, selection)


def _transparency_components(selected: np.ndarray, transparent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count, groups, bounds, _ = cv2.connectedComponentsWithStats(selected.astype(np.uint8))
    exterior = ~selected
    visible = exterior & ~transparent
    kernel = np.ones((3, 3), np.uint8)
    touches_exterior = cv2.dilate(exterior.astype(np.uint8), kernel) != 0
    touching = selected & (cv2.dilate(visible.astype(np.uint8), kernel) != 0)
    y, x = np.nonzero(touching)
    connected = groups[y, x]
    clear = np.zeros(count, dtype=bool)
    clear[groups[selected & touches_exterior]] = True
    clear[connected] = False

    # Two distinct visible structures can be the exposed ends of one object.
    # Keep the entire connection ambiguous, even when it crosses diagonally.
    exterior_count, structures = cv2.connectedComponents(visible.astype(np.uint8))
    structures = np.pad(structures, 1)
    first = np.full(count, exterior_count, dtype=np.int32)
    last = np.zeros(count, dtype=np.int32)
    for dy in range(3):
        for dx in range(3):
            contacts = structures[y + dy, x + dx]
            present = contacts != 0
            np.minimum.at(first, connected[present], contacts[present])
            np.maximum.at(last, connected[present], contacts[present])
    return clear[groups], _protrusion_pixels(groups, bounds, touching, first == last)


def _protrusion_pixels(
    groups: np.ndarray, bounds: np.ndarray, touching: np.ndarray, single_structure: np.ndarray,
) -> np.ndarray:
    # A continuous structure can meet a selection in separate places, as when a
    # mask crosses a closed frame. Count separate runs along the mask boundary.
    contact_count, contacts = cv2.connectedComponents(touching.astype(np.uint8))
    y, x = np.nonzero(touching)
    selected_groups, contact_labels = groups[y, x], contacts[y, x]
    first = np.full(len(bounds), contact_count, dtype=np.int32)
    last = np.zeros(len(bounds), dtype=np.int32)
    np.minimum.at(first, selected_groups, contact_labels)
    np.maximum.at(last, selected_groups, contact_labels)
    eligible = (single_structure & (first == last))[groups]

    for group in np.flatnonzero(single_structure & (first != last)):
        left, top, width, height, _ = bounds[group]
        area = np.s_[top:top + height, left:left + width]
        region = groups[area] == group
        local_contacts = touching[area] & region
        output = eligible[area]
        if width > height:
            local_contacts, region, output = local_contacts.T, region.T, output.T
        positions = np.flatnonzero(local_contacts.any(axis=1))
        start, end = positions[0], positions[-1]
        length = len(output)
        # Contacts clustered at one end still allow the tip of a protrusion to
        # clear. Keep the space between the contacts available for reconstruction.
        if 3 * end < length:
            output[end + 1:] = region[end + 1:]
        elif 3 * start >= 2 * length:
            output[:start] = region[:start]
    return eligible


def _clear_sides(selected: np.ndarray, exterior_clear: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Test the first real unselected pixel on each side of an approved run."""
    width = selected.shape[1]
    columns = np.arange(width, dtype=np.int32)
    left = np.maximum.accumulate(np.where(selected, -1, columns), axis=1)
    right = np.minimum.accumulate(np.where(selected, width, columns)[:, ::-1], axis=1)[:, ::-1]
    before = (left >= 0) & np.take_along_axis(exterior_clear, np.maximum(left, 0), axis=1)
    after = (right < width) & np.take_along_axis(exterior_clear, np.minimum(right, width - 1), axis=1)
    # Use opposite sides only across narrow gaps so remote transparency cannot
    # imply that the middle of a broad surface should disappear.
    return before, after, before & after & ((right - left) <= ALPHA_CROSSING_LIMIT)


def _removal_alpha_mask(base: Image.Image, footprint: Image.Image, selection: Rect) -> Image.Image | None:
    if base.mode != "RGBA":
        return None
    if _surrounded_by_transparency(base, footprint, selection):
        return footprint.point([0] + [255] * 255)

    tile_size, halo = 512, ALPHA_CROSSING_LIMIT
    area = selection.expand(halo, halo).intersect(Rect(0, 0, *base.size))
    assert area is not None
    if area.width * area.height > ALPHA_CONTEXT_PIXELS:
        return None
    box = area.left, area.top, area.right, area.bottom
    selected = np.asarray(footprint.crop(box)) != 0
    transparent = np.asarray(base.crop(box).getchannel("A")) == 0
    if not (transparent & ~selected).any():
        return None
    islands, attached = _transparency_components(selected, transparent)
    if not (islands.any() or attached.any()):
        return None
    clear = Image.new("L", base.size)
    clear.paste(Image.fromarray(islands.astype(np.uint8) * 255), (area.left, area.top))

    # At an attached protrusion, require three clear sides and a gap from
    # known visible material. A bridge between opaque edges stays ambiguous.
    for top in range(selection.top, selection.bottom, tile_size):
        for left in range(selection.left, selection.right, tile_size):
            tile = Rect(left, top, min(left + tile_size, selection.right), min(top + tile_size, selection.bottom))
            expanded = tile.expand(halo, halo).intersect(area)
            assert expanded is not None
            part = tile.relative_to(expanded).slices
            source = expanded.relative_to(area).slices
            tile_selected = selected[source]
            eligible = attached[source] & tile_selected
            if not eligible[part].any():
                continue
            exterior_clear = ~tile_selected & transparent[source]
            left_clear, right_clear, across = _clear_sides(tile_selected, exterior_clear)
            top_clear, bottom_clear, down = _clear_sides(tile_selected.T, exterior_clear.T)
            supported = eligible & ((across & (top_clear | bottom_clear).T)
                                    | (down.T & (left_clear | right_clear)))
            if not supported[part].any():
                continue
            visible = ~tile_selected & ~transparent[source]
            distance = cv2.distanceTransform((~visible).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5)
            supported &= distance >= ALPHA_CONTACT_MARGIN
            patch = Image.fromarray(supported[part].astype(np.uint8) * 255)
            clear.paste(255, (tile.left, tile.top, tile.right, tile.bottom), patch)
    return clear if clear.getbbox() is not None else None


def _surrounded_by_transparency(base: Image.Image, mask: Image.Image, selection: Rect) -> bool:
    # Use actual source pixels, including a one-pixel halo for each stripe. A
    # downsample could miss a thin object touching the approved removal area.
    canvas = Rect(0, 0, *base.size)
    area = selection.expand(1, 1).intersect(canvas)
    assert area is not None
    kernel = np.ones((3, 3), np.uint8)
    found_background = False
    for top in range(area.top, area.bottom, 128):
        region = Rect(area.left, top, area.right, min(top + 128, area.bottom))
        halo = region.expand(0, 1).intersect(area)
        assert halo is not None
        box = halo.left, halo.top, halo.right, halo.bottom
        selected = (np.asarray(mask.crop(box)) != 0).astype(np.uint8)
        ring = (cv2.dilate(selected, kernel) != selected)[region.relative_to(halo).slices]
        if ring.any():
            alpha = np.asarray(base.crop((region.left, region.top, region.right, region.bottom)).getchannel("A"))
            if alpha[ring].any():
                return False
            found_background = True
    return found_background
