"""Pixel coordinates on a repeating tile."""

from PIL import Image, ImageChops

from .geometry import Rect


def wrap_offset(image: Image.Image, dx: int, dy: int) -> Image.Image:
    dx, dy = dx % image.width, dy % image.height
    return ImageChops.offset(image, dx, dy) if dx or dy else image


def periodic_pieces(area: Rect, size: tuple[int, int]):
    """Yield tile-local rectangles and their offsets within the requested area."""
    width, height = size
    for y in range(area.top // height * height, area.bottom, height):
        for x in range(area.left // width * width, area.right, width):
            tile = Rect(x, y, x + width, y + height)
            part = area.intersect(tile)
            yield part.relative_to(tile), (part.left - area.left, part.top - area.top)


def periodic_crop(image: Image.Image, area: Rect) -> Image.Image:
    width, height = min(image.width, area.width), min(image.height, area.height)
    period = Rect(area.left, area.top, area.left + width, area.top + height)
    result = Image.new(image.mode, (area.width, area.height))
    for part, destination in periodic_pieces(period, image.size):
        result.paste(image.crop((part.left, part.top, part.right, part.bottom)), destination)
    while width < area.width:
        cols = min(width, area.width - width)
        result.paste(result.crop((0, 0, cols, height)), (width, 0))
        width += cols
    while height < area.height:
        rows = min(height, area.height - height)
        result.paste(result.crop((0, 0, width, rows)), (0, height))
        height += rows
    return result
