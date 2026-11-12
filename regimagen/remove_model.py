"""Locate or prepare the pinned EfficientSAM model for Generative Remove."""

import sys
from hashlib import file_digest, sha256
from http.client import HTTPException
from pathlib import Path
from urllib.request import urlopen

from .files import atomic_output

MODEL_NAME = "image_segmentation_efficientsam_ti_2025april_int8.onnx"
MODEL_REVISION = "1e791988b34b2098736003057da0f3e6bc826259"
MODEL_SHA256 = "5ecc8d59a2802c32246e68553e1cf8ce74cf74ba707b84f206eb9181ff774b4e"
MODEL_SIZE = 20_479_928
MODEL_URL = (
    "https://huggingface.co/opencv/image_segmentation_efficientsam/resolve/"
    f"{MODEL_REVISION}/{MODEL_NAME}?download=true"
)


def _verified(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size != MODEL_SIZE:
        return False
    with path.open("rb") as model:
        return file_digest(model, "sha256").hexdigest() == MODEL_SHA256


def model_path() -> Path:
    package = Path(__file__).resolve().parent
    bundled = package / "assets" / MODEL_NAME
    if _verified(bundled):
        return bundled
    if getattr(sys, "frozen", False):
        raise RuntimeError("The bundled object selection model is missing or damaged. Extract the portable package again.")

    cached = package.parent / "build" / "models" / MODEL_NAME
    if _verified(cached):
        return cached
    try:
        with urlopen(MODEL_URL, timeout=30) as response:
            data = response.read(MODEL_SIZE + 1)
    except (OSError, HTTPException) as exc:
        raise RuntimeError("Could not download the object selection model. Try again or use a manual selection.") from exc
    if len(data) != MODEL_SIZE or sha256(data).hexdigest() != MODEL_SHA256:
        raise RuntimeError("The downloaded object selection model did not match its expected checksum.")
    cached.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output(cached) as stream:
        stream.write(data)
    return cached


if __name__ == "__main__":
    try:
        print(model_path())
    except RuntimeError as exc:
        sys.exit(str(exc))
