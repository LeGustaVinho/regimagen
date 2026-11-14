"""Adapt the generative fill engine to a layered document."""

from dataclasses import dataclass, replace
from threading import Event

from PIL import Image, ImageChops

from .codex import Cancelled
from .document import Document, Layer, box, paste_pixels, render
from .fill import FillCandidate, FillPreview, render_fill
from .geometry import Rect
from .remove_detection import RemovalTarget


def fill_reference(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA" and image.getchannel("A").getextrema() == (255, 255):
        return image.convert("RGB")
    return image


@dataclass(frozen=True)
class ActiveFillTarget:
    layer: Layer
    area: Rect
    image: Image.Image


def active_fill_target(document: Document, layer: Layer | None) -> ActiveFillTarget | None:
    if layer is None or not layer.visible or layer.locked or layer.coverage is not None:
        return None
    area = layer.bounds.intersect(document.bounds)
    if area is None:
        return None
    return ActiveFillTarget(layer, area, fill_reference(layer.image.crop(box(area.relative_to(layer.bounds)))))


@dataclass(frozen=True)
class LayeredVariant:
    document: Document
    layer_id: str
    preview: FillPreview
    candidate: FillCandidate | None = None


@dataclass(frozen=True, kw_only=True)
class RemovalVariant(LayeredVariant):
    background: FillPreview


def removed_object_layers(
    source: Image.Image, core: Image.Image, objects: tuple[RemovalTarget, ...], *, cancelled: Event,
) -> tuple[Layer, ...]:
    """Extract each visible object from the original composition in review order."""
    area = core.getbbox()
    if area is None:
        return ()
    origin = Rect(*area)
    available = core.crop(area)
    layers = []
    for target in objects:
        if cancelled.is_set():
            raise Cancelled("Removal canceled.")
        x, y = target.origin
        bounds = Rect(x, y, x + target.mask.width, y + target.mask.height)
        overlap = bounds.intersect(origin)
        if overlap is None:
            continue
        region = box(overlap.relative_to(origin))
        subject = target.mask.crop(box(overlap.relative_to(bounds))).point([0] + [255] * 255)
        mask = ImageChops.darker(available.crop(region), subject)
        selected = mask.getbbox()
        if selected is None:
            continue
        x, y = overlap.left, overlap.top
        left, top, right, bottom = selected
        patch = source.crop((x + left, y + top, x + right, y + bottom)).convert("RGBA")
        alpha = ImageChops.multiply(patch.getchannel("A"), mask.crop(selected))
        visible = alpha.getbbox()
        if visible is None:
            continue
        patch.putalpha(alpha)
        patch = patch.crop(visible)
        clear = patch.getchannel("A").point([255] + [0] * 255)
        patch.paste((0, 0, 0, 0), (0, 0, *patch.size), clear)
        layers.append(Layer(target.label, patch, x + left + visible[0], y + top + visible[1]))
        available.paste(0, region, mask)
    if cancelled.is_set():
        raise Cancelled("Removal canceled.")
    return tuple(layers)


def removal_variant(
    document: Document, footprint: Image.Image, preview: FillPreview, objects: tuple[Layer, ...], *,
    cancelled: Event,
) -> RemovalVariant:
    background = layered_variant(document, footprint, preview, None, name="Generative Remove")
    document = replace(background.document, layers=background.document.layers + objects)
    x, y = background.preview.origin
    width, height = background.preview.patch.size
    after = render(document, Rect(x, y, x + width, y + height), cancelled=cancelled.is_set)
    return RemovalVariant(
        document, objects[-1].id, FillPreview(after, (x, y), preview.warning), background=background.preview,
    )


def render_layer_fill(
    document: Document, image: Image.Image, selection: Image.Image, candidate: FillCandidate,
    target: ActiveFillTarget | None, *, edit_alpha: bool,
) -> LayeredVariant:
    preview = render_fill(image, selection, candidate, edit_alpha=edit_alpha)
    return replace(layered_variant(document, selection, preview, target), candidate=candidate)


def layered_variant(
    document: Document, selection: Image.Image, preview: FillPreview, target: ActiveFillTarget | None,
    *, name: str = "Generative Fill",
) -> LayeredVariant:
    left, top = preview.origin
    patch = preview.patch.convert("RGBA")
    area = Rect(left, top, left + patch.width, top + patch.height)
    if target is None:
        # The engine already feathered the pixels. Coverage identifies exactly
        # where to replace the composite, including pixels with alpha zero.
        coverage = selection.crop(box(area)).point([0] + [255] * 255)
        layer = Layer(name, patch, left, top, coverage=coverage)
        document = document.insert(layer)
    else:
        layer = target.layer
        left, top = left + target.area.left, top + target.area.top
        area = Rect(left, top, left + patch.width, top + patch.height)
        edited = paste_pixels(layer.image, patch, (left - layer.x, top - layer.y))
        layer = replace(layer, image=edited)
        document = document.update(layer)
    composed = render(document, area)
    return LayeredVariant(document, layer.id, FillPreview(composed, (left, top), preview.warning))
