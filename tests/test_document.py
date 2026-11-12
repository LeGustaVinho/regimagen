from dataclasses import replace
from zipfile import ZipFile

import pytest
from PIL import Image

from regimagen.codex import Cancelled
from regimagen.document import RASTER_TILE_SIZE, Document, Layer, Raster, export_document, paint, render
from regimagen.files import SRGB_BYTES
from regimagen.geometry import Rect
from regimagen.project import load_project, save_project


def test_project_round_trip_keeps_assets_and_composites_like_export(tmp_path):
    backdrop = Layer("Fundo", Image.new("RGBA", (12, 10), (200, 20, 30, 255)), -2, -1)
    pixels = Image.new("RGBA", (8, 8), (40, 100, 210, 200))
    mask = Image.new("L", pixels.size, 128)
    accent = Layer("Detalhe", pixels, 2, 2, opacity=60, locked=True, mask=mask, adjustments=(4, 12, -20))
    copy = Layer("Cópia", pixels, 8, 6, visible=False, mask=mask)
    cutout = Layer("Fill", Image.new("RGBA", (2, 2)), 1, 1, coverage=Image.new("L", (2, 2), 255))
    document = Document("Composição", 18, 12, (backdrop, accent, copy, cutout))

    project = tmp_path / "original.regimagen"
    save_project(document, project)
    moved_folder = tmp_path / "moved"
    moved_folder.mkdir()
    moved = project.rename(moved_folder / project.name)
    loaded = load_project(moved)

    assert (loaded.id, loaded.size, loaded.name) == (document.id, document.size, document.name)
    assert [layer.name for layer in loaded.layers] == ["Fundo", "Detalhe", "Cópia", "Fill"]
    _, active, duplicate, fill = loaded.layers
    assert (active.x, active.y, active.opacity, active.locked, active.adjustments) == (2, 2, 60, True, (4, 12, -20))
    assert active.image is duplicate.image and active.mask is duplicate.mask
    assert not duplicate.visible and fill.coverage.getextrema() == (255, 255)

    expected = render(document)
    assert expected.tobytes() == render(loaded).tobytes()
    assert expected.getpixel((1, 1)) == (0, 0, 0, 0)
    assert expected.getpixel((0, 0)) == (200, 20, 30, 255)
    assert render(loaded, Rect(6, 4, 15, 9)).tobytes() == expected.crop((6, 4, 15, 9)).tobytes()
    for suffix in ("png", "webp"):
        destination = tmp_path / f"export.{suffix}"
        export_document(loaded, destination)
        with Image.open(destination) as image:
            assert image.mode == "RGBA" and image.tobytes() == expected.tobytes()

    cropped = document.crop(Rect(3, 1, 12, 10))
    assert cropped.layers[0].image is backdrop.image
    assert render(cropped).tobytes() == expected.crop((3, 1, 12, 10)).tobytes()


def test_fill_opacity_interpolates_alpha_and_jpeg_uses_white_matte(tmp_path):
    base = Layer("Opaque", Image.new("RGBA", (24, 24), (255, 0, 0, 255)))
    fill = Layer("Clear", Image.new("RGBA", (16, 16)), opacity=50, coverage=Image.new("L", (16, 16), 255))
    doc = Document("Cutout", 32, 24, (base, fill))
    result = render(doc)
    assert result.getpixel((4, 4)) == (255, 0, 0, 127)
    assert result.getpixel((18, 8)) == (255, 0, 0, 255)
    assert result.getpixel((28, 8)) == (0, 0, 0, 0)
    assert render(doc.update(replace(fill, visible=False))).getpixel((4, 4)) == (255, 0, 0, 255)
    output = tmp_path / "matte.jpg"
    export_document(doc, output)
    with Image.open(output) as image:
        assert image.mode == "RGB"
        assert all(channel >= 250 for channel in image.getpixel((29, 8)))


def test_brush_history_shares_pixels_and_saves_edited_tiles(tmp_path, monkeypatch):
    edge = RASTER_TILE_SIZE
    size = (edge * 3, edge * 2)
    pixels = Image.new("RGBA", size, "red")
    mask = Image.new("L", size, 255)
    layer = Layer("Layer", pixels, mask=mask)
    document = Document("Brush", *size, (layer,))

    def stroke(current, point, color, *, edit_mask=False):
        return paint(document, current, [point], size=1, strength=100, color=color, edit_mask=edit_mask, selection=None)

    blue, green = (0, 0, 255), (0, 255, 0)
    first = stroke(layer, (edge - 1, 10), blue)
    second = stroke(first, (edge, 10), green)
    assert isinstance(first.image, Raster) and isinstance(second.image, Raster)
    assert first.image._base is second.image._base is pixels
    assert set(first.image._tiles) == {(0, 0)}
    assert set(second.image._tiles) == {(0, 0), (edge, 0)}
    assert second.image._tiles[(0, 0)] is first.image._tiles[(0, 0)]
    assert stroke(second, (edge, 10), green) is second
    assert layer.image.getpixel((edge - 1, 10)) == (255, 0, 0, 255)
    assert first.image.getpixel((edge, 10)) == (255, 0, 0, 255)
    crossing = second.image.crop((edge - 1, 10, edge + 1, 11))
    assert crossing.getpixel((0, 0)) == (*blue, 255)
    assert crossing.getpixel((1, 0)) == (*green, 255)

    masked = stroke(second, (edge, 10), None, edit_mask=True)
    assert masked.image is second.image and masked.mask._base is mask
    assert set(masked.mask._tiles) == {(edge, 0)}
    assert masked.mask.getpixel((edge, 10)) == 0 and mask.getpixel((edge, 10)) == 255
    edited = document.update(masked)
    project = tmp_path / "brush.regimagen"
    crops = []
    crop = Raster.crop

    def record_band(image, region):
        left, top, right, bottom = region
        assert (left, right) == (0, size[0])
        assert bottom - top < size[1]
        crops.append((image.mode, top, bottom))
        return crop(image, region)

    with monkeypatch.context() as patch:
        patch.setattr(Raster, "crop", record_band)
        save_project(edited, project)
    for mode in ("RGBA", "L"):
        bands = [(top, bottom) for band_mode, top, bottom in crops if band_mode == mode]
        assert len(bands) > 1
        assert bands[0][0] == 0 and bands[-1][1] == size[1]
        assert all(first[1] == second[0] for first, second in zip(bands, bands[1:]))
    loaded = load_project(project)
    assert loaded.layers[0].image.getpixel((edge, 10)) == (*green, 255)
    assert loaded.layers[0].image.info["icc_profile"] == SRGB_BYTES
    assert loaded.layers[0].mask.getpixel((edge, 10)) == 0
    assert render(loaded).tobytes() == render(edited).tobytes()


def test_canceling_between_png_bands_keeps_the_previous_project(tmp_path, monkeypatch):
    size = (RASTER_TILE_SIZE * 2,) * 2
    document = Document.from_image(Image.new("RGB", size, "red"), "Brush")
    edited = document.update(paint(
        document, document.layers[0], [(8, 8)], size=1, strength=100, color=(0, 0, 255),
        edit_mask=False, selection=None,
    ))
    path = tmp_path / "brush.regimagen"
    save_project(document, path)
    original = path.read_bytes()
    bands_read = 0
    crop = Raster.crop

    def read_band(image, region):
        nonlocal bands_read
        bands_read += 1
        return crop(image, region)

    monkeypatch.setattr(Raster, "crop", read_band)
    with pytest.raises(Cancelled, match="Save canceled"):
        save_project(edited, path, cancelled=lambda: bands_read >= 2)
    assert bands_read == 2
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_resize_fill_ignores_excluded_pixels_and_preserves_cutouts():
    pixels = Image.new("RGBA", (9, 1), (0, 255, 0, 0))
    pixels.putpixel((0, 0), (255, 0, 0, 255))
    pixels.putpixel((1, 0), (0, 255, 0, 255))
    pixels.putpixel((2, 0), (255, 0, 0, 128))
    coverage = Image.new("L", pixels.size, 255)
    coverage.putpixel((1, 0), 0)
    mask = Image.new("L", pixels.size, 255)
    resized = Layer("Fill", pixels, mask=mask, coverage=coverage).resize((72, 8))
    assert resized.image.size == resized.mask.size == resized.coverage.size == (72, 8)
    assert resized.coverage.getpixel((70, 4)) == 255
    assert resized.image.getchannel("A").getextrema() == (0, 255)

    lower = Layer("Lower", Image.new("RGBA", (72, 8), "blue"))
    result = render(Document("Fill", 72, 8, (lower, resized)))
    assert result.getchannel("G").getextrema() == (0, 0)
    assert result.getpixel((70, 4)) == (0, 0, 0, 0)
    assert result.getchannel("R").getextrema()[1] > 0


def test_failed_project_save_keeps_previous_file(tmp_path):
    doc = Document.from_image(Image.new("RGB", (8, 12), "blue"), "Blue")
    project = tmp_path / "blue.regimagen"
    save_project(doc, project)
    original = project.read_bytes()

    def fail(_path):
        raise OSError("Cannot publish project")

    with pytest.raises(OSError, match="Cannot publish"):
        save_project(replace(doc, name="new"), project, before_publish=fail)
    with pytest.raises(Cancelled):
        save_project(replace(doc, name="new"), project, cancelled=lambda: True)
    assert project.read_bytes() == original and load_project(project).name == "Blue"
    assert list(tmp_path.iterdir()) == [project]


def test_incompatible_and_incomplete_projects_report_errors(tmp_path):
    project = tmp_path / "unknown.regimagen"
    with ZipFile(project, "w") as archive:
        archive.writestr("manifest.json", '{"version":100}')
    with pytest.raises(ValueError, match="unsupported"):
        load_project(project)
    project.write_bytes(b"interrupted write")
    with pytest.raises(ValueError, match="Could not open"):
        load_project(project)
