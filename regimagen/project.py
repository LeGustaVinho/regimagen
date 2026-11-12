"""Read and atomically write portable Regimagen documents."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO
from zipfile import BadZipFile, ZipFile

import png
from PIL import Image

from .codex import Cancelled
from .document import Document, Layer, PixelAsset, Raster, validate_size
from .files import SRGB_BYTES, SrgbPngWriter, atomic_output

PROJECT_SUFFIX = ".regimagen"
VERSION = 1


def _write_asset(image: PixelAsset, stream: BinaryIO, check_cancelled: Callable[[], None]):
    rgba = image.mode == "RGBA"
    if not isinstance(image, Raster):
        profile = {"icc_profile": SRGB_BYTES} if rgba else {}
        image.save(stream, format="PNG", **profile)
        return

    width, height = image.size
    row_bytes = width * (4 if rgba else 1)
    band_height = max(1, min(128, (1024 * 1024) // row_bytes))

    def rows():
        for top in range(0, height, band_height):
            check_cancelled()
            bottom = min(top + band_height, height)
            band = memoryview(image.crop((0, top, width, bottom)).tobytes())
            for start in range(0, len(band), row_bytes):
                yield band[start:start + row_bytes]

    if rgba:
        writer = SrgbPngWriter(width, height, greyscale=False, alpha=True, embed_profile=True)
    else:
        writer = png.Writer(width, height, greyscale=True)
    writer.write(stream, rows())


def save_project(
    document: Document, destination: Path, *, cancelled: Callable[[], bool] = lambda: False,
    before_publish: Callable[[Path], None] | None = None,
):
    def check_cancelled():
        if cancelled():
            raise Cancelled("Save canceled.")

    def publish(path):
        check_cancelled()
        if before_publish is not None:
            before_publish(path)

    with atomic_output(destination, before_publish=publish) as stream, ZipFile(stream, "w") as archive:
        assets: dict[int, str] = {}

        def asset(image: PixelAsset | None) -> str | None:
            if image is None:
                return None
            key = id(image)
            if key not in assets:
                check_cancelled()
                path = f"assets/{len(assets)}.png"
                with archive.open(path, "w") as output:
                    _write_asset(image, output, check_cancelled)
                assets[key] = path
            return assets[key]

        layers = [{
            "id": layer.id, "name": layer.name, "x": layer.x, "y": layer.y,
            "opacity": layer.opacity, "visible": layer.visible, "locked": layer.locked,
            "image": asset(layer.image), "mask": asset(layer.mask), "coverage": asset(layer.coverage),
            "adjustments": layer.adjustments,
        } for layer in document.layers]
        archive.writestr("manifest.json", json.dumps({
            "version": VERSION, "id": document.id, "name": document.name,
            "width": document.width, "height": document.height, "layers": layers,
        }, ensure_ascii=False))


def _read_manifest(archive: ZipFile, cancelled: Callable[[], bool]) -> Document:
    entries = archive.namelist()
    if len(entries) != len(set(entries)):
        raise ValueError("The project contains duplicate entries.")
    data = json.loads(archive.read("manifest.json"))
    if not isinstance(data, dict) or type(data.get("version")) is not int:
        raise ValueError("The project manifest is invalid.")
    if data["version"] != VERSION:
        raise ValueError("This project version is unsupported. Open it with a compatible Regimagen version.")

    def text(row, key):
        value = row[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"The project has an invalid {key}.")
        return value

    assets: dict[str, Image.Image] = {}

    def asset(path, mode):
        if path is None and mode == "L":
            return None
        if not isinstance(path, str) or not path.startswith("assets/") or not path.endswith(".png"):
            raise ValueError("The project has an invalid image reference.")
        index = path[len("assets/"):-len(".png")]
        if not index.isascii() or not index.isdecimal():
            raise ValueError("The project has an invalid image reference.")
        if path not in assets:
            with archive.open(path) as stream, Image.open(stream, formats=("PNG",)) as image:
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Project images must be still PNGs.")
                validate_size(*image.size)
                assets[path] = image.copy()
        if assets[path].mode != mode:
            raise ValueError("A project image has an invalid pixel format.")
        return assets[path]

    width, height = data["width"], data["height"]
    validate_size(width, height)
    if not isinstance(data["layers"], list):
        raise ValueError("The project layer list is invalid.")
    layers = []
    seen = set()
    for row in data["layers"]:
        if cancelled():
            raise Cancelled("Open canceled.")
        layer_id = text(row, "id")
        if layer_id in seen:
            raise ValueError("The project contains duplicate layers.")
        seen.add(layer_id)
        x, y, opacity, visible, locked = (row[key] for key in ("x", "y", "opacity", "visible", "locked"))
        values = row.get("adjustments", [0, 0, 0])
        if (
            any(type(value) is not int for value in (x, y, opacity)) or not 0 <= opacity <= 100
            or type(visible) is not bool or type(locked) is not bool
            or not isinstance(values, list) or len(values) != 3
            or any(type(value) is not int or not -100 <= value <= 100 for value in values)
        ):
            raise ValueError("A project layer has invalid settings.")
        image = asset(row["image"], "RGBA")
        mask, coverage = (asset(row.get(key), "L") for key in ("mask", "coverage"))
        if any(item is not None and item.size != image.size for item in (mask, coverage)):
            raise ValueError("A project mask does not match its layer.")
        layers.append(Layer(
            text(row, "name"), image, x, y, opacity, visible, locked, mask, coverage, tuple(values), layer_id,
        ))
    return Document(text(data, "name"), width, height, tuple(layers), text(data, "id"))


def load_project(path: Path, *, cancelled: Callable[[], bool] = lambda: False) -> Document:
    try:
        with ZipFile(path) as archive:
            return _read_manifest(archive, cancelled)
    except Cancelled:
        raise
    except (BadZipFile, Image.DecompressionBombError, KeyError, OSError, RuntimeError, TypeError) as exc:
        raise ValueError(f"Could not open {path.name} as a Regimagen project.") from exc
