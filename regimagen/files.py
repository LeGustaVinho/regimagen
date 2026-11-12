"""Image loading and atomic file writes."""

import io
import os
import tempfile
import zlib
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

import png
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError

from .geometry import FRAME

SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
SRGB_BYTES = SRGB.tobytes()
FORMATS = ("PNG", "JPEG", "WEBP")
TRANSPARENT_MATTE = (30, 33, 39)


class InvalidStageInput(ValueError):
    code = "INVALID_STAGE_INPUT"


class SrgbPngWriter(png.Writer):
    def __init__(self, *args, embed_profile: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed_profile = embed_profile

    def write_preamble(self, stream):
        super().write_preamble(stream)
        if self._embed_profile:
            png.write_chunk(stream, b"iCCP", b"ICC Profile\x00\x00" + zlib.compress(SRGB_BYTES))
        else:
            png.write_chunk(stream, b"sRGB", b"\x00")


@contextmanager
def atomic_output(destination: Path, *, overwrite: bool = True, before_publish: Callable[[Path], None] | None = None):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.stem}-", suffix=".part", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        if before_publish is not None:
            before_publish(temporary)
        if overwrite:
            os.replace(temporary, destination)
        elif os.name == "nt":
            os.rename(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def file_identity(path: Path) -> list[int] | None:
    try:
        info = path.stat()
    except OSError:
        return None
    # Renaming a file on the same filesystem preserves its file ID and mtime.
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns] if info.st_ino else None


def _srgb(image: Image.Image, *, preserve_alpha: bool) -> Image.Image:
    if getattr(image, "n_frames", 1) != 1:
        raise ValueError("Choose a still image.")
    oriented = ImageOps.exif_transpose(image)
    alpha = None
    if "A" in oriented.getbands() or "transparency" in oriented.info:
        coverage = oriented.convert("RGBA").getchannel("A")
        if coverage.getextrema() != (255, 255):
            if not preserve_alpha:
                raise ValueError("The image returned by ImageGen must be opaque.")
            alpha = coverage
    profile = oriented.info.get("icc_profile")
    result = None
    if profile:
        try:
            source_profile = ImageCms.ImageCmsProfile(io.BytesIO(profile))
            result = ImageCms.profileToProfile(oriented, source_profile, SRGB, outputMode="RGB")
        except (ImageCms.PyCMSError, OSError, ValueError):
            # Some image generators write incomplete ICC metadata.
            pass
    if result is None:
        result = oriented.convert("RGB")
    if alpha is not None:
        result.putalpha(alpha)
    return result


def load_source(path: Path | BinaryIO, *, formats: tuple[str, ...] = FORMATS) -> Image.Image:
    try:
        with Image.open(path, formats=formats) as image:
            return _srgb(image, preserve_alpha=True)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ValueError("Could not open the image. Use PNG, JPEG, or WebP.") from exc


def load_generated(data: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data), formats=FORMATS) as image:
            normalized = _srgb(image, preserve_alpha=False)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ValueError("ImageGen returned an invalid image file.") from exc
    if normalized.width != normalized.height or normalized.width < FRAME:
        raise ValueError(
            f"ImageGen returned {normalized.width}×{normalized.height} px. "
            f"The tile must be square and at least {FRAME} px on each side."
        )
    return normalized.resize((FRAME, FRAME), Image.Resampling.LANCZOS)


def alpha_channel(image: Image.Image) -> Image.Image | None:
    return image.getchannel("A") if image.mode == "RGBA" else None


def reference_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    matte = Image.new("RGBA", image.size, (*TRANSPARENT_MATTE, 255))
    return Image.alpha_composite(matte, image).convert("RGB")


def save_png(
    image: Image.Image, destination: Path, *, overwrite: bool = True,
    before_publish: Callable[[Path], None] | None = None,
):
    with atomic_output(destination, overwrite=overwrite, before_publish=before_publish) as stream:
        image.save(stream, format="PNG", icc_profile=SRGB_BYTES)
