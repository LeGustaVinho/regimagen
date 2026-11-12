"""Render the desktop's inspection views to local PNG files."""

import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw

from .automation import CommandError
from .composite import Compositor, STRIPE_HEIGHT
from .files import atomic_output, load_source, reference_rgb, save_png
from .geometry import Cell, Rect
from .imaging import prepare_frame
from .jobs import prepare_tile_request
from .seams import seam_region
from .storage import Job, Snapshot


def cell_at(job: Job, position) -> Cell:
    cell = Cell(position[0] - 1, position[1] - 1)
    if cell not in job.layout.cells:
        raise ValueError("The tile coordinates are outside the grid.")
    return cell


def cell_json(cell: Cell) -> dict:
    return {"column": cell.column + 1, "row": cell.row + 1}


def rectangle(area: Rect) -> dict:
    return {"x": area.left, "y": area.top, "width": area.width, "height": area.height}


def canvas_crop(image: Image.Image, canvas: Rect, area: Rect, size: tuple[int, int]) -> Image.Image:
    sx, sy = image.width / canvas.width, image.height / canvas.height
    source_box = (
        (area.left - canvas.left) * sx, (area.top - canvas.top) * sy,
        min(image.width, (area.right - canvas.left) * sx),
        min(image.height, (area.bottom - canvas.top) * sy),
    )
    return image.convert("RGBA").resize(size, Image.Resampling.BICUBIC, box=source_box)


def render_reconstruction(job: Job, snapshot: Snapshot, sprite, area: Rect, size: tuple[int, int], *, fit: bool):
    if not snapshot.artifacts:
        return Image.new("RGBA", size)
    compositor = Compositor(job.layout, snapshot.artifacts, alpha=sprite)
    if area.width * area.height <= 2_500_000:
        return Image.fromarray(compositor.render(area)).convert("RGBA").resize(size, Image.Resampling.LANCZOS)

    if fit and size[0] < area.width and size[1] < area.height:
        output = Image.new("RGBA", size)
        for cell, tile in snapshot.tiles.items():
            core = job.layout.core(cell)
            overlap = core.intersect(area)
            if tile.artifact is None or overlap is None:
                continue
            x0 = round((overlap.left - area.left) * size[0] / area.width)
            x1 = round((overlap.right - area.left) * size[0] / area.width)
            y0 = round((overlap.top - area.top) * size[1] / area.height)
            y1 = round((overlap.bottom - area.top) * size[1] / area.height)
            if x1 <= x0 or y1 <= y0:
                continue
            if tile.thumbnail is not None:
                with Image.open(tile.thumbnail) as thumbnail:
                    image = canvas_crop(thumbnail, core, overlap, (x1 - x0, y1 - y0))
            else:
                image = Image.fromarray(compositor.render(overlap)).convert("RGBA").resize(
                    (x1 - x0, y1 - y0), Image.Resampling.LANCZOS,
                )
            output.paste(image, (x0, y0))
        return output

    resize = size != (area.width, area.height)
    mode = "RGBa" if resize else "RGBA"
    panel = Image.new(mode, (size[0], area.height))
    for top in range(area.top, area.bottom, STRIPE_HEIGHT):
        bottom = min(top + STRIPE_HEIGHT, area.bottom)
        band = Image.fromarray(compositor.render(Rect(area.left, top, area.right, bottom))).convert(mode)
        panel.paste(band.resize((size[0], band.height), Image.Resampling.LANCZOS), (0, top - area.top))
    if not resize:
        return panel
    # Filter vertically in canvas coordinates so Lanczos spans stripe boundaries.
    scaled = panel.resize(size, Image.Resampling.LANCZOS)
    del panel
    return scaled.convert("RGBA")


def inspect_image(job: Job, snapshot: Snapshot, args, output: Path) -> dict:
    bounds = job.layout.bounds
    area = bounds
    selection = cell_at(job, args.tile) if args.tile else None
    if selection:
        area = job.layout.core(selection)
    elif args.region:
        x, y, width, height = args.region
        area = Rect(x, y, x + width, y + height)
        if width < 1 or height < 1 or area.intersect(bounds) != area:
            raise ValueError("The inspection region must be inside the output image.")
    elif args.seam:
        first, second = cell_at(job, args.seam[:2]), cell_at(job, args.seam[2:])
        if second not in job.layout.neighbors(first):
            raise ValueError("Select two neighboring tiles for seam inspection.")
        first, second = sorted((first, second), key=lambda cell: (cell.row, cell.column))
        area = seam_region(job.layout, first, second).intersect(bounds)
        if area is None:
            raise ValueError("These tiles have no visible seam region.")
    if area.width < 1 or area.height < 1:
        raise ValueError("This selection has no pixels in the output canvas.")

    width, height = args.viewport
    if min(width, height) < 1:
        raise ValueError("The viewport dimensions must be positive.")
    fit = args.zoom is None
    if fit:
        if args.center:
            raise ValueError("--center requires a numeric --zoom.")
        factor = min(width / area.width, height / area.height)
    else:
        if not 0.1 <= args.zoom <= 800:
            raise ValueError("Zoom must be between 0.1 and 800 percent.")
        factor = args.zoom / 100
        center_x, center_y = args.center or ((area.left + area.right) / 2, (area.top + area.bottom) / 2)
        if not all(map(math.isfinite, (center_x, center_y))):
            raise ValueError("The viewport center must contain finite coordinates.")
        limits = area if args.region else bounds
        w = limits.width if width >= limits.width * factor else max(1, int(width / factor))
        h = limits.height if height >= limits.height * factor else max(1, int(height / factor))
        x = round(max(limits.left, min(limits.right - w, center_x - w / 2)))
        y = round(max(limits.top, min(limits.bottom - h, center_y - h / 2)))
        area = Rect(x, y, x + w, y + h)
    size = max(1, min(width, round(area.width * factor))), max(1, min(height, round(area.height * factor)))
    panels = []
    if args.view in ("original", "input", "compare"):
        path = job.source.path if args.view == "original" else job.input_path
        panels.append(canvas_crop(load_source(path), bounds, area, size))
    if args.view in ("reconstruction", "compare"):
        source = load_source(job.input_path)
        sprite = source if source.mode == "RGBA" else None
        panels.append(render_reconstruction(job, snapshot, sprite, area, size, fit=fit))

    visible = [cell for cell in job.layout.cells if job.layout.core(cell).intersect(area)]
    for panel in panels:
        draw = ImageDraw.Draw(panel)
        for cell in visible:
            if not args.grid and not (args.highlight and cell == selection):
                continue
            core = job.layout.core(cell)
            x0 = round((core.left - area.left) * size[0] / area.width)
            y0 = round((core.top - area.top) * size[1] / area.height)
            x1 = round((core.right - area.left) * size[0] / area.width) - 1
            y1 = round((core.bottom - area.top) * size[1] / area.height) - 1
            if x1 >= x0 and y1 >= y0:
                draw.rectangle((x0, y0, x1, y1), outline="#75d8be" if cell == selection else "#888888")
    combined = Image.new("RGBA", (size[0] * len(panels), size[1]))
    for index, panel in enumerate(panels):
        combined.paste(panel, (size[0] * index, 0))
    save_png(combined, output, overwrite=args.overwrite)
    return {
        "path": str(output), "width": combined.width, "height": combined.height,
        "view": args.view, "region": rectangle(area), "panel_size": {"width": size[0], "height": size[1]},
        "zoom": factor * 100,
        "missing_tiles": [cell_json(cell) for cell in visible if snapshot.tiles[cell].artifact is None],
        "stale_tiles": [cell_json(cell) for cell in visible if snapshot.tiles[cell].stale],
    }


def save_references(job: Job, snapshot: Snapshot, cell: Cell, directory: Path, *, overwrite: bool) -> dict:
    source = load_source(job.input_path)
    original = reference_rgb(source)
    frame = prepare_frame(np.asarray(original), job.layout, cell)
    with TemporaryDirectory() as scratch:
        prompt, images = prepare_tile_request(
            Path(scratch), original, frame, job.layout, cell, snapshot, sprite=source.mode == "RGBA",
        )
        prompt_file = Path(scratch) / "prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        if not overwrite and any((directory / path.name).exists() for path in (*images, prompt_file)):
            raise CommandError("OUTPUT_EXISTS", "A reference output file already exists.", status=4)
        directory.mkdir(parents=True, exist_ok=True)
        for path in (*images, prompt_file):
            with atomic_output(directory / path.name, overwrite=overwrite) as stream:
                stream.write(path.read_bytes())
    return {
        "images": [str(directory / path.name) for path in images], "prompt": prompt,
        "prompt_path": str(directory / prompt_file.name),
    }
