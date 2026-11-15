"""Adapt the generative fill engine to a layered document."""

from dataclasses import dataclass, replace

from PIL import Image

from .document import Document, Layer, box, paste_pixels, render
from .fill import FillCandidate, FillPreview, render_fill
from .geometry import Rect


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


def replacement_layer(
    selection: Image.Image, patch: Image.Image, origin: tuple[int, int], name: str, *, isolated: bool = False,
) -> Layer:
    left, top = origin
    area = left, top, left + patch.width, top + patch.height
    # The patch already includes the blend. Coverage also replaces fully transparent pixels.
    coverage = None if isolated else selection.crop(area).point([0] + [255] * 255)
    return Layer(name, patch if patch.mode == "RGBA" else patch.convert("RGBA"), left, top, coverage=coverage)


def layered_variant(
    document: Document, selection: Image.Image, preview: FillPreview, target: ActiveFillTarget | None,
    *, name: str = "Generative Fill", isolated_layer: bool = False,
) -> LayeredVariant:
    left, top = preview.origin
    patch = preview.patch.convert("RGBA")
    area = Rect(left, top, left + patch.width, top + patch.height)
    if target is None:
        layer = replacement_layer(selection, patch, (left, top), name, isolated=isolated_layer)
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


def render_layer_fill(
    document: Document, image: Image.Image, selection: Image.Image, candidate: FillCandidate,
    target: ActiveFillTarget | None, *, edit_alpha: bool,
) -> LayeredVariant:
    isolated = target is None and candidate.transparent_background
    preview = render_fill(image, selection, candidate, edit_alpha=edit_alpha, isolated_layer=isolated)
    return replace(layered_variant(document, selection, preview, target, isolated_layer=isolated), candidate=candidate)
