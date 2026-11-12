import hashlib
import io
import json
import os
import sqlite3
from pathlib import Path
from threading import Event
from zipfile import ZipFile

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from regimagen.codex import Cancelled, CodexClient, CodexError, GeneratedImage, QuotaPaused, find_codex
from regimagen.composite import Compositor, export_parts, export_png
from regimagen.files import TRANSPARENT_MATTE, load_generated, load_source
from regimagen.geometry import FRAME, Cell, Grid, Layout, Rect, parse_scales, plan_scales, scaled_grid
from regimagen.imaging import align_tile, seam_warning
from regimagen.jobs import pending_cells, run_scale_batch, run_tiles
from regimagen.storage import Store


def png_bytes(size=(FRAME, FRAME), color=(130, 150, 170)):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    return stream.getvalue()


class FakeImageGen:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def __call__(self, *_args, **_kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def generate(self, prompt, references, *, on_image=None):
        with Image.open(references[0]) as primary:
            assert primary.size == (FRAME, FRAME)
        self.calls.append((prompt, len(references)))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        result = GeneratedImage(reply, f"image-{len(self.calls)}")
        if on_image is not None:
            on_image(result)
        return result


def make_job(directory, grid, source_size=(320, 320)):
    store = Store(directory)
    source = store.import_source(Image.new("RGB", source_size, (90, 110, 130)), "source.png")
    return store, store.ensure_job(source, grid)


def run(store, job, selection, generator, **kwargs):
    run_tiles(
        store, job, selection, Path("codex.exe"), pause=Event(), cancelled=Event(),
        client_factory=generator, **kwargs,
    )


def test_preview_quota_and_resume_reuse_results(tmp_path):
    store, job = make_job(tmp_path, Grid(3, 3))
    preview = job.layout.preview()
    frame = png_bytes()
    first = FakeImageGen([frame, frame, QuotaPaused(1_798_000_000)])
    run(store, job, preview, first)
    snapshot = store.snapshot(job)
    assert set(snapshot.artifacts) == set(preview[:2])
    assert snapshot.quota_reset == 1_798_000_000

    reopened = Store(tmp_path)
    selection = pending_cells(reopened.snapshot(job), preview)
    assert selection == list(preview[2:])
    resume = FakeImageGen([frame] * 7)
    run(reopened, job, selection, resume)
    assert set(reopened.snapshot(job).artifacts) == set(preview)
    remaining = pending_cells(reopened.snapshot(job), job.layout.cells)
    run(reopened, job, remaining, resume)
    assert len(resume.calls) == 7
    assert len(reopened.snapshot(job).artifacts) == 9

    other = reopened.ensure_job(job.source, Grid(2, 2))
    assert other.id != job.id
    assert reopened.snapshot(other).artifacts == {}
    assert reopened.ensure_job(job.source, job.grid).id == job.id


def test_invalid_output_and_explicit_tile_repair(tmp_path):
    store, job = make_job(tmp_path, Grid(2, 2))
    frame = png_bytes(color=(100, 120, 140))
    first = FakeImageGen([png_bytes((1024, 1536))])
    run(store, job, job.layout.cells, first)
    assert len(first.calls) == 1
    assert "must be square" in store.snapshot(job).notice
    assert not store.snapshot(job).artifacts

    normal = FakeImageGen([frame] * 4)
    run(store, job, job.layout.cells, normal)
    before = store.snapshot(job).artifacts
    assert len(before) == 4
    repair = FakeImageGen([frame])
    run(store, job, [Cell(0, 0)], repair)
    after = store.snapshot(job).artifacts
    assert repair.calls[0][1] == 3
    assert after[Cell(0, 0)] != before[Cell(0, 0)]
    assert all(after[cell] == before[cell] for cell in job.layout.cells[1:])


def test_recover_artifact_already_received_without_new_generation(tmp_path):
    store, job = make_job(tmp_path, Grid(2, 2))
    cell = Cell(0, 0)
    store.begin_tile(job, cell, "attempt")
    (job.folder / "attempt.image").write_bytes(png_bytes())
    provider = FakeImageGen([])
    run(store, job, [cell], provider)
    assert provider.calls == []
    assert set(store.snapshot(job).artifacts) == {cell}
    assert not (job.folder / "attempt.image").exists()


@pytest.mark.parametrize("turn_status", ["local_cancel", "interrupted", "failed"])
def test_single_tile_checkpoints_image_event_before_turn_completion(tmp_path, turn_status):
    store, job = make_job(tmp_path, Grid(2, 2))
    cell = job.layout.cells[0]
    cancelled = Event()
    data = png_bytes()
    interrupted = turn_status != "failed"

    class DeliveredImage(CodexClient):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def _run_turn(self, _inputs, wait, **_options):
            image = {"method": "item/completed", "params": {
                "threadId": "thread", "turnId": "turn", "item": {"type": "imageGeneration", "status": "completed"},
            }}
            return wait("thread", "turn", [image])

        def _image_from_event(self, *_):
            return GeneratedImage(data, "delivered-image")

        def _receive(self, _deadline):
            if turn_status == "local_cancel":
                cancelled.set()
                raise Cancelled("Generation canceled.")
            return {"method": "turn/completed", "params": {
                "threadId": "thread", "turn": {"id": "turn", "status": turn_status, "error": {"message": "Rejected."}},
            }}

    outcome = run_tiles(store, job, [cell], Path("codex.exe"), pause=Event(), cancelled=cancelled,
                        client_factory=DeliveredImage)
    pending = store.snapshot(job).tiles[cell]
    assert pending.artifact is None
    assert (job.folder / f"{pending.attempt}.image").is_file() == interrupted
    assert outcome.state == ("canceled" if interrupted else "failed")
    resume = FakeImageGen([] if interrupted else [data])
    run(store, job, [cell], resume)
    tile = store.snapshot(job).tiles[cell]
    assert tile.artifact is not None
    assert tile.generation_id == ("delivered-image" if interrupted else "image-1")
    assert len(resume.calls) == (0 if interrupted else 1)


def test_prompt_changes_regenerate_only_affected_tiles(tmp_path):
    store, job = make_job(tmp_path, Grid(2, 2))
    cell, neighbor, *_ = job.layout.cells
    frame = png_bytes()
    run(store, job, [cell, neighbor], FakeImageGen([frame, frame]))
    store.begin_tile(job, cell, "old")
    (job.folder / "old.image").write_bytes(frame)
    store.set_prompt(job, "Brushed copper", cell)
    store = Store(tmp_path)
    snapshot = store.snapshot(job)
    assert snapshot.tiles[cell].prompt == "Brushed copper"
    assert snapshot.tiles[cell].stale
    assert snapshot.tiles[cell].attempt is None
    assert set(snapshot.current_artifacts) == {neighbor}
    repair = FakeImageGen([frame])
    run(store, job, [cell], repair)
    assert "Brushed copper" in repair.calls[0][0]
    assert set(store.snapshot(job).current_artifacts) == {cell, neighbor}

    store.set_prompt(job, "Warm lighting")
    assert not store.snapshot(job).current_artifacts
    regenerated = FakeImageGen([frame, frame])
    run(store, job, [cell, neighbor], regenerated)
    first, second = [prompt for prompt, _ in regenerated.calls]
    assert first.index("Warm lighting") < first.index("Brushed copper")
    assert "Warm lighting" in second and "Brushed copper" not in second


def test_partial_delegated_group_can_resume(tmp_path):
    store, job = make_job(tmp_path, Grid(2, 2))
    cells = job.layout.cells
    frame = png_bytes()

    class Delegated(FakeImageGen):
        def generate_delegated(self, requests, on_image):
            assert len(requests) == 4
            on_image(2, GeneratedImage(frame, "agent-tile-3"))
            raise QuotaPaused(1_798_000_000)

    run(store, job, cells, Delegated([]), subagents=True)
    snapshot = store.snapshot(job)
    assert snapshot.quota_reset == 1_798_000_000
    assert set(snapshot.current_artifacts) == {cells[2]}
    remaining = pending_cells(snapshot, cells)
    assert remaining == [cells[0], cells[1], cells[3]]
    run(store, job, remaining, FakeImageGen([frame] * 3))
    assert not pending_cells(store.snapshot(job), cells)


def test_scaled_jobs_and_transparent_parts(tmp_path):
    store = Store(tmp_path)
    image = Image.new("RGBA", (33, 21))
    ImageDraw.Draw(image).rectangle((8, 6, 26, 16), fill=(130, 150, 170, 240))
    source = store.import_source(image, "sprite.png")
    assert parse_scales("2x, 3 ×; 4 2") == (2, 3, 4)
    assert scaled_grid(1000, 500, 2) == Grid(3, 2)
    with pytest.raises(ValueError):
        scaled_grid(1000, 500, 11)
    jobs = [store.ensure_job(source, scaled_grid(*image.size, scale), scale) for scale in (2, 3)]
    assert jobs[0].grid == jobs[1].grid
    assert jobs[0].id != jobs[1].id
    job = Store(tmp_path).find_job(source, jobs[0].grid, 2)
    assert job.layout.size == (66, 42)
    cell = Cell(0, 0)
    run(store, job, [cell], FakeImageGen([png_bytes()]))
    destination = tmp_path / "parts.zip"
    artifacts = store.snapshot(job).current_artifacts
    export_parts(job.layout, artifacts, destination, alpha=image)
    with ZipFile(destination) as archive:
        manifest = json.loads(archive.read("positions.json"))
        assert (manifest["width"], manifest["height"]) == (66, 42)
        part, = manifest["tiles"]
        assert part["core"] == {"x": 0, "y": 0, "width": 33, "height": 21}
        assert (part["x"], part["y"]) == (0, 0)
        assert part["file"] == "row-01_column-01.png"
        with Image.open(io.BytesIO(archive.read(part["file"]))) as saved:
            assert saved.size == (part["width"], part["height"])
            assert saved.mode == "RGBA"
            expected = Compositor(job.layout, artifacts, alpha=image).render(Rect(0, 0, *saved.size))
            np.testing.assert_array_equal(np.asarray(saved), expected)
            assert saved.getpixel((20, 16))[3] == 240


def test_legacy_store_preserves_results(tmp_path, capsys):
    from regimagen.cli import main

    db_path = tmp_path / "projetos.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.executescript("""
            CREATE TABLE sources (id TEXT PRIMARY KEY, name TEXT, width INT, height INT);
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY, source_id TEXT REFERENCES sources(id), columns INT, rows INT,
                notice TEXT, quota_reset INT, UNIQUE (source_id, columns, rows)
            );
            CREATE TABLE tiles (
                job_id TEXT REFERENCES jobs(id), col INT, row INT, attempt TEXT, result TEXT,
                generation_id TEXT, warning TEXT, error TEXT, PRIMARY KEY (job_id, col, row)
            );
            INSERT INTO sources VALUES ('source', 'old.png', 128, 128);
            INSERT INTO jobs VALUES (
                'job', 'source', 2, 2, 'Coluna 1, linha 1. O Codex atingiu o limite de uso.', 123
            );
            INSERT INTO tiles VALUES (
                'job', 0, 0, NULL, 'result', 'image-1', 'Confira o alinhamento com o original.', NULL
            );
        """)
    folder = tmp_path / "lotes" / "job"
    folder.mkdir(parents=True)
    (folder / "result.png").write_bytes(png_bytes())
    assert main(["--data-dir", str(tmp_path), "job", "show", "job"]) == 0
    result = json.loads(capsys.readouterr().out)["result"]
    assert result["ready"] == 1 and result["pending"] == 3 and result["quota_reset"] == 123
    assert result["tiles"][0]["generation_id"] == "image-1"
    store = Store(tmp_path)
    job, = store.recent_jobs()
    assert job.id == "job" and job.scale == 0
    snapshot = store.snapshot(job)
    assert snapshot.quota_reset == 123 and snapshot.prompt == ""
    assert snapshot.notice == "Column 1, row 1. Codex has reached its usage limit."
    assert snapshot.tiles[Cell(0, 0)].warning == "Check alignment with the original."
    assert set(snapshot.current_artifacts) == {Cell(0, 0)}
    cell = Cell(1, 0)
    store.set_prompt(job, "Preserve the original tones.", cell)
    store.begin_tile(job, cell, "new-attempt")
    assert store.snapshot(job).tiles[cell].attempt == "new-attempt"
    assert store.snapshot(job).tiles[cell].prompt == "Preserve the original tones."
    assert store.ensure_job(job.source, job.grid, 2).id != job.id
    with sqlite3.connect(db_path) as db:
        assert not db.execute("PRAGMA foreign_key_check").fetchall()


def test_scale_batch_uses_previous_export_and_resumes_by_input(tmp_path, monkeypatch):
    store = Store(tmp_path)
    image = Image.new("RGBA", (32, 20), (60, 80, 100, 150))
    source = store.import_source(image, "sprite.png")
    layouts = plan_scales(*image.size, "2×, 3×", chained=True)
    with pytest.raises(ValueError, match="ascending order"):
        plan_scales(*image.size, "3×, 2×", chained=True)

    frame = png_bytes(color=(120, 140, 160))
    rendered = []
    pause = Event()

    def generate(store, job, cells, executable, **kwargs):
        rendered.append(job)
        if job.input_scale > 1:
            assert kwargs["input_image"].size == (64, 40)
            with Image.open(job.input_path) as cached:
                np.testing.assert_array_equal(kwargs["input_image"], cached)
            assert kwargs["input_image"].getpixel((10, 10)) != image.getpixel((10, 10))
        run_tiles(store, job, cells, executable, client_factory=FakeImageGen([frame] * len(cells)), **kwargs)

    monkeypatch.setattr("regimagen.jobs.run_tiles", generate)

    def batch(layouts, on_export=lambda _: None):
        return run_scale_batch(
            Store(tmp_path), source, layouts, Path("codex.exe"), tmp_path,
            pause=pause, cancelled=Event(), on_export=on_export,
        )

    _, count, _ = batch(layouts, lambda _: pause.set())
    assert count == 1
    first, = rendered
    pause.clear()
    folder, count, _ = batch(layouts)
    assert count == 2
    first, second = rendered
    assert second.source == source and second.input_scale == 2
    assert (folder / "sprite-2x.png").read_bytes() == second.input_path.read_bytes()
    with Image.open(folder / "sprite-3x.png") as result:
        assert result.size == (96, 60) and result.mode == "RGBA"

    batch(layouts)
    assert rendered == [first, second]
    independent = plan_scales(*image.size, "3×", chained=False)
    batch(independent)
    direct = rendered[-1]
    assert direct.input_path == source.path and direct.id != second.id

    run(store, first, [Cell(0, 0)], FakeImageGen([frame]))
    batch(layouts)
    replacement = rendered[-1]
    assert replacement.id != second.id and replacement.input_path != second.input_path
    assert second.input_path.is_file()


@pytest.mark.parametrize(("grid", "source_size"), [
    (Grid(2, 2), (320, 320)),
    (Grid(16, 6), (800, 300)),
])
def test_full_simulated_export_uses_generated_pixels(tmp_path, grid, source_size):
    store, job = make_job(tmp_path, grid, source_size)
    generator = FakeImageGen([png_bytes()] * grid.count)
    run(store, job, job.layout.cells, generator)
    snapshot = store.snapshot(job)
    assert len(generator.calls) == len(snapshot.artifacts) == grid.count
    assert not pending_cells(snapshot, job.layout.cells)
    export = tmp_path / "final.png"
    export_png(job.layout, snapshot.artifacts, export)
    with Image.open(export) as decoded:
        assert decoded.size == job.layout.size
        assert decoded.mode == "RGB"
        assert decoded.info["srgb"] == 0
        assert decoded.getextrema() == ((122, 122), (142, 142), (162, 162))


def test_stripes_match_the_zoom_compositor_at_both_seams(tmp_path):
    layout = Layout(320, 320, Grid(2, 2))
    artifacts = {}
    for i, cell in enumerate(layout.cells):
        path = tmp_path / f"{i}.png"
        image = Image.new("RGB", (FRAME, FRAME), (30 + i * 22, 84 + i * 19, 150))
        draw = ImageDraw.Draw(image)
        draw.line((80, 75, 950, 945), fill=(144, 150, 156), width=17)
        image.save(path)
        artifacts[cell] = path
    compositor = Compositor(layout, artifacts)
    region = Rect(820, 840, 975, 941)
    expected = compositor.render(region)
    stitched = np.concatenate([
        compositor.render(Rect(820, 840, 975, 895)),
        compositor.render(Rect(820, 895, 975, 941)),
    ])
    assert np.array_equal(expected, stitched)

    destination = tmp_path / "final.png"
    destination.write_bytes(b"existing")
    with pytest.raises(Cancelled):
        export_png(layout, artifacts, destination, cancelled=lambda: True)
    assert destination.read_bytes() == b"existing"


@pytest.mark.parametrize("vertical", [True, False])
def test_join_registers_a_shifted_line_without_softening_a_lone_tile(tmp_path, vertical):
    layout = Layout(256, 256, Grid(2, 2), scale=2)
    cells = Cell(0, 0), Cell(1, 0) if vertical else Cell(0, 1)
    artifacts = {}
    for index, cell in enumerate(cells):
        frame = layout.frame(cell)
        generated = Image.new("RGB", layout.size, (180, 180, 180))
        y = 125 + 6 * index
        ImageDraw.Draw(generated).rectangle((160, y, 350, y + 4), fill=(25, 25, 25))
        if not vertical:
            generated = generated.transpose(Image.Transpose.TRANSPOSE)
        path = tmp_path / f"{index}.png"
        generated.crop((frame.left, frame.top, frame.right, frame.bottom)).save(path)
        artifacts[cell] = path

    compositor = Compositor(layout, artifacts)
    seam = compositor._seam(*cells)
    assert seam.shift[128] > 5
    cut = round(seam.position[128])
    area = Rect(cut - 4, 121, cut + 4, 141) if vertical else Rect(121, cut - 4, 141, cut + 4)
    result = compositor.render(area)[:, :, 0]
    if not vertical:
        result = result.T
    dark = np.argwhere(result < 60)
    assert set(dark[:, 1]) == set(range(8))
    assert dark[:, 0].min() >= 6 and dark[:, 0].max() <= 12

    single = Compositor(layout, {cells[0]: artifacts[cells[0]]})
    margin = Rect(268, 121, 282, 141) if vertical else Rect(121, 268, 141, 282)
    frame = layout.frame(cells[0])
    expected = np.asarray(Image.open(artifacts[cells[0]]))[margin.relative_to(frame).slices]
    np.testing.assert_array_equal(single.render(margin), expected)


def test_join_routes_around_incompatible_shapes_and_shares_a_four_tile_junction(tmp_path):
    layout = Layout(256, 256, Grid(2, 2), scale=3)
    artifacts = {}
    for cell in layout.cells:
        frame = layout.frame(cell)
        generated = Image.new("RGB", layout.size, (75, 75, 75))
        draw = ImageDraw.Draw(generated)
        draw_shape = draw.rectangle if cell.column == 0 else draw.ellipse
        draw_shape((353, 163, 415, 221), outline=(205, 205, 205), width=7)
        draw.rectangle((354 + 4 * cell.column, 350 + 4 * cell.row, 406, 416), fill=(175, 175, 175))
        path = tmp_path / f"{cell.column}-{cell.row}.png"
        generated.crop((frame.left, frame.top, frame.right, frame.bottom)).save(path)
        artifacts[cell] = path

    compositor = Compositor(layout, artifacts)
    lens = Rect(348, 157, 421, 227)
    result = compositor.render(lens)
    assert not np.any(compositor._seam(Cell(0, 0), Cell(1, 0)).shift[lens.top:lens.bottom])
    assert any(np.abs(result.astype(int) - np.asarray(Image.open(artifacts[cell]))[
        lens.relative_to(layout.frame(cell)).slices
    ]).mean() < 0.1 for cell in (Cell(0, 0), Cell(1, 0)))
    intersection = Rect(320, 320, 448, 448)
    pixels, missing = compositor._render_rgb(intersection)
    assert not np.any(missing)
    alternate = Compositor(layout, dict(reversed(tuple(artifacts.items()))))
    stripes = [alternate.render(Rect(320, y, 448, y + 16)) for y in range(320, 448, 16)]
    np.testing.assert_array_equal(pixels, np.concatenate(stripes))


def test_alignment_and_displaced_contour():
    layout = Layout(320, 320, Grid(2, 2))
    target = Cell(0, 0)
    frame = Image.new("RGB", (FRAME, FRAME), (115, 125, 130))
    draw = ImageDraw.Draw(frame)
    draw.rectangle((251, 309, 584, 467), fill=(205, 205, 205))
    draw.ellipse((600, 324, 722, 496), fill=(25, 25, 25))
    pixels = np.asarray(frame)
    shifted = cv2.warpAffine(
        pixels, np.float32([[1, 0, 13], [0, 1, -11]]), frame.size, borderMode=cv2.BORDER_REPLICATE,
    )
    aligned, warning = align_tile(Image.fromarray(shifted), pixels, layout, target)
    assert np.array_equal(aligned[150:850, 150:850], pixels[150:850, 150:850])
    assert warning is None
    neighbor = np.full_like(pixels, 230)
    assert seam_warning(layout, target, Cell(1, 0), pixels, neighbor)


def test_boundary_detector_checks_crossing_contours_at_source_scale():
    layout = Layout(320, 320, Grid(2, 2))
    left, right = Cell(0, 0), Cell(1, 0)
    first = np.full((FRAME, FRAME, 3), 150, dtype=np.uint8)
    second = first.copy()
    first[447:457] = 30
    second[478:488] = 30
    assert "contours" in seam_warning(layout, left, right, first, second)
    assert seam_warning(layout, left, right, first, first) is None

    # Stripes parallel to the seam do not need to cross it.
    first[:] = 150
    second[:] = 150
    first[:, 944:948] = 30
    second[:, 61:65] = 30
    assert seam_warning(layout, left, right, first, second) is None


def test_image_decode_and_event_provenance(tmp_path):
    target = tmp_path / "input.png"
    target.write_bytes(png_bytes())
    generated_folder = tmp_path / "generated_images"
    generated_folder.mkdir()
    generated = generated_folder / "result.png"
    generated.write_bytes(png_bytes((1254, 1254)))
    client = CodexClient(Path("codex.exe"), tmp_path)
    client.generated_dir = generated_folder
    result = client._image_from_event({
        "savedPath": str(generated),
        "inputImages": [{"sha256": hashlib.sha256(target.read_bytes()).hexdigest()}],
    }, (target,))
    assert load_generated(result.data).size == (1024, 1024)
    with pytest.raises(CodexError, match="different tile"):
        client._image_from_event({
            "savedPath": str(generated), "inputImages": [{"sha256": "different"}],
        }, (target,))
    with pytest.raises(CodexError, match="accessible image file"):
        client._image_from_event({"savedPath": str(target)}, (target,))

    alpha = tmp_path / "alpha.png"
    Image.new("RGBA", (16, 16), (230, 30, 20, 0)).save(alpha)
    assert load_source(alpha).mode == "RGBA"
    with pytest.raises(ValueError, match="opaque"):
        load_generated(alpha.read_bytes())


def test_transparent_input_keeps_coverage_in_preview_and_export(tmp_path):
    source_path = tmp_path / "sprite.png"
    sprite = Image.new("RGBA", (256, 256))
    ImageDraw.Draw(sprite).polygon(
        [(128, 27), (171, 163), (128, 223), (86, 162)], fill=(90, 110, 130, 224),
    )
    sprite.save(source_path)
    store = Store(tmp_path / "local")
    source = store.import_source(load_source(source_path), source_path.name)
    job = store.ensure_job(source, Grid(2, 2))
    mock = FakeImageGen([png_bytes()] * 4)
    run(store, job, job.layout.cells, mock)
    snapshot = store.snapshot(job)
    assert len(mock.calls) == len(snapshot.artifacts) == 4
    assert all("flat, dark charcoal matte" in prompt for prompt, _ in mock.calls)
    with Image.open(snapshot.tiles[Cell(0, 0)].thumbnail) as preview:
        assert preview.mode == "RGBA"
        assert preview.getpixel((0, 0)) == (0, 0, 0, 0)

    destination = tmp_path / "sprite-final.png"
    export_png(job.layout, snapshot.artifacts, destination, alpha=load_source(source.path))
    with Image.open(destination) as result:
        assert result.mode == "RGBA"
        assert result.size == (1792, 1792)
        assert result.info["srgb"] == 0
        assert result.getpixel((0, 0)) == (0, 0, 0, 0)
        assert result.getpixel((900, 900))[3] == 224
        assert result.getpixel((900, 900))[:3] != sprite.getpixel((128, 128))[:3]
        actual = result.getchannel("A")
        assert actual.getextrema() == (0, 224)
        original = sprite.getchannel("A")
        expected = np.asarray(original, dtype=np.int16)
        bilinear = original.resize(result.size, Image.Resampling.BILINEAR)
        restored = np.asarray(actual.resize(original.size, Image.Resampling.BOX), dtype=np.int16)
        bilinear_restored = np.asarray(bilinear.resize(original.size, Image.Resampling.BOX), dtype=np.int16)
        assert np.abs(restored - expected).mean() < np.abs(bilinear_restored - expected).mean()


def test_sprite_edge_correction_uses_generated_color_across_stripes(tmp_path):
    layout = Layout(256, 256, Grid(2, 2))
    alpha = Image.new("L", (256, 256), 0)
    ImageDraw.Draw(alpha).rectangle((64, 64, 191, 191), fill=255)
    core_color = (168, 87, 52)
    artifacts = {}
    for cell in layout.cells:
        image = Image.new("RGB", (FRAME, FRAME), TRANSPARENT_MATTE)
        inner = Rect(453, 453, 1338, 1338).intersect(layout.frame(cell))
        if inner:
            box = inner.relative_to(layout.frame(cell))
            ImageDraw.Draw(image).rectangle(
                (box.left, box.top, box.right - 1, box.bottom - 1), fill=core_color,
            )
        path = tmp_path / f"{cell.column}-{cell.row}.png"
        image.save(path)
        artifacts[cell] = path
    compositor = Compositor(layout, artifacts, alpha=alpha)
    rendered = compositor.render(Rect(650, 430, 714, 464))
    assert tuple(rendered[20, 32, :3]) == core_color
    assert rendered[20, 32, 3] > 0
    assert tuple(rendered[0, 32]) == (0, 0, 0, 0)
    stripes = np.concatenate([
        compositor.render(Rect(650, 430, 714, 449)),
        compositor.render(Rect(650, 449, 714, 464)),
    ])
    assert np.max(np.abs(rendered.astype(np.int16) - stripes)) <= 1


def test_sprite_matte_removes_false_coverage_and_preserves_thin_translucent_detail(tmp_path):
    layout = Layout(64, 64, Grid(2, 2), scale=8)
    source = Image.new("RGBA", (64, 64), (255, 0, 255, 0))
    draw = ImageDraw.Draw(source)
    body = (173, 102, 66)
    dark = (34, 36, 42)
    antenna = (160, 190, 210)
    draw.rectangle((12, 16, 27, 42), fill=(*body, 255))
    draw.rectangle((10, 9, 10, 17), fill=(*antenna, 112))
    draw.rectangle((41, 16, 50, 39), fill=(*dark, 255))

    generated = Image.new("RGB", layout.size, TRANSPARENT_MATTE)
    draw = ImageDraw.Draw(generated)
    draw.rectangle((102, 128, 223, 343), fill=body)
    draw.rectangle((328, 128, 407, 319), fill=dark)
    artifacts = {}
    for cell in layout.cells:
        box = layout.frame(cell)
        path = tmp_path / f"{cell.column}-{cell.row}.png"
        generated.crop((box.left, box.top, box.right, box.bottom)).save(path)
        artifacts[cell] = path

    compositor = Compositor(layout, artifacts, alpha=source)
    full = compositor.render(layout.bounds)
    assert full[200, 95, 3] < 20  # The generated tile filled this source strip with charcoal.
    assert tuple(full[200, 150]) == (*body, 255)
    assert tuple(full[200, 330, :3]) == dark
    assert full[200, 330, 3] >= 180
    assert full[117, 83, 3] >= 106  # A haste omitida pelo recorte tem alfa 112 na fonte.
    assert np.max(np.abs(full[117, 83, :3].astype(int) - antenna)) <= 1
    assert tuple(full[10, 10]) == (0, 0, 0, 0)
    for area in (Rect(32, 71, 241, 132), Rect(89, 164, 350, 221)):
        np.testing.assert_array_equal(compositor.render(area), full[area.slices])


@pytest.mark.skipif(os.name != "nt", reason="The npm installation uses a .cmd shim on Windows.")
def test_npm_install_discovers_native_binary(tmp_path, monkeypatch):
    shim = tmp_path / "codex.cmd"
    executable = tmp_path / "node_modules/@openai/codex-win32-x64/vendor/windows/codex/codex.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(
        "regimagen.codex.shutil.which",
        lambda name: str(shim) if name in {"codex", "codex.cmd"} else None,
    )
    assert find_codex() == executable
