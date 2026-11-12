"""Queue ImageGen edits and recover previously received results."""

from collections.abc import Callable, Iterable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import file_digest, sha256
from itertools import batched
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory, mkdtemp
from threading import Event
from uuid import uuid4

import numpy as np
from PIL import Image

from .codex import AGENT_GROUP_SIZE, AuthRequired, Cancelled, CodexClient, CodexError, GeneratedImage, QuotaPaused
from .composite import Compositor, export_png
from .files import atomic_output, load_generated, load_source, reference_rgb, save_png
from .geometry import Cell, Layout
from .imaging import (
    EDIT_PROMPT, REPAIR_PROMPT, align_tile, prepare_frame, read_tile, seam_warning, write_references,
)
from .seams import SEAM_MARGIN
from .storage import Job, Snapshot, Source, Store, thumbnail_path

SPRITE_PROMPT = (
    " The source is a sprite shown on a flat, dark charcoal matte. Preserve the exact outer "
    "silhouette, orientation, cut-off edges and proportions of the sprite. Keep the matte flat. "
    "Keep the same flat matte in the empty spaces and internal holes. Preserve thin parts and "
    "clean, antialiased contours. Avoid outlines, shadows or glow on the matte. "
    "Reconstruct the existing surface details in place."
)


def pending_cells(snapshot: Snapshot, selection: Iterable[Cell]) -> list[Cell]:
    return [
        cell for cell in selection
        if snapshot.tiles[cell].artifact is None or snapshot.tiles[cell].stale
        or snapshot.tiles[cell].attempt is not None
    ]


@dataclass(frozen=True)
class RunOutcome:
    state: str = "completed"
    code: str | None = None
    message: str | None = None
    quota_reset: int | None = None


def prepare_tile_request(
    directory: Path, original: Image.Image, frame: np.ndarray, layout: Layout, cell: Cell,
    snapshot: Snapshot, *, sprite: bool,
) -> tuple[str, tuple[Path, ...]]:
    repair = {
        neighbor: tile.artifact
        for neighbor in layout.neighbors(cell)
        if (tile := snapshot.tiles[neighbor]).artifact is not None
        and not (tile.alignment_warning or tile.stale)
    }
    inputs = write_references(directory, original, frame, layout, cell, repair)
    prompt = EDIT_PROMPT + (SPRITE_PROMPT if sprite else "")
    if repair:
        prompt += REPAIR_PROMPT
    if snapshot.prompt:
        prompt += f"\nAdditional art direction for the entire image:\n{snapshot.prompt}"
    tile_prompt = snapshot.tiles[cell].prompt
    if tile_prompt:
        prompt += f"\nAdditional art direction for this tile (takes precedence over shared art direction):\n{tile_prompt}"
    return prompt, inputs


def run_tiles(
    store: Store,
    job: Job,
    cells: Iterable[Cell],
    executable: Path | Callable[[], Path],
    *,
    pause: Event,
    cancelled: Event,
    on_change: Callable[[Cell | None], None] = lambda _: None,
    client_factory: Callable = CodexClient,
    subagents: bool = False,
    input_image: Image.Image | None = None,
) -> RunOutcome:
    layout = job.layout
    if not layout.enlarges:
        raise ValueError("The selected grid must enlarge the image in both dimensions.")
    job.folder.mkdir(exist_ok=True)
    imported = input_image if input_image is not None else load_source(job.input_path)
    alpha = imported if imported.mode == "RGBA" else None
    original = reference_rgb(imported)
    source = np.asarray(original)
    store.set_notice(job, None)

    with ExitStack() as stack:
        client = None
        for group in batched(cells, AGENT_GROUP_SIZE if subagents else 1):
            if pause.is_set() or cancelled.is_set():
                store.set_notice(job, "Job paused. Completed tiles have been saved.")
                return RunOutcome("canceled" if cancelled.is_set() else "paused")
            current = group[0]
            try:
                snapshot = store.snapshot(job)
                attempts = {}
                queued = []
                generated = {}
                generation_ids = {}
                with ExitStack() as references:
                    for current in group:
                        previous = snapshot.tiles[current]
                        attempt = previous.attempt or uuid4().hex
                        store.begin_tile(job, current, attempt)
                        on_change(current)
                        staged = job.folder / f"{attempt}.image"
                        frame = prepare_frame(source, layout, current)
                        attempts[current] = (attempt, staged, frame)
                        if staged.is_file():
                            try:
                                generated[current] = load_generated(staged.read_bytes())
                                continue
                            except ValueError:
                                staged.unlink()
                        directory = references.enter_context(
                            TemporaryDirectory(dir=job.folder, prefix="reference-")
                        )
                        prompt, inputs = prepare_tile_request(
                            Path(directory), original, frame, layout, current, snapshot, sprite=alpha is not None,
                        )
                        queued.append((current, prompt, inputs))

                    def stage(index: int, result: GeneratedImage):
                        cell = queued[index][0]
                        image = load_generated(result.data)
                        with atomic_output(attempts[cell][1]) as stream:
                            stream.write(result.data)
                        generated[cell] = image
                        generation_ids[cell] = result.generation_id

                    failure = None
                    if queued:
                        if client is None:
                            options = {"subagents": True} if subagents else {}
                            path = executable() if callable(executable) else executable
                            client = stack.enter_context(client_factory(path, job.folder, cancelled=cancelled, **options))
                        try:
                            if subagents:
                                client.generate_delegated([(prompt, refs) for _, prompt, refs in queued], stage)
                            else:
                                _, prompt, refs = queued[0]
                                stage(0, client.generate(prompt, refs))
                        except (CodexError, OSError, ValueError) as exc:
                            failure = exc

                if cancelled.is_set():
                    raise Cancelled("Generation canceled.")
                for current, image in generated.items():
                    on_change(current)
                    attempt, staged, frame = attempts[current]
                    _commit_tile(store, job, current, attempt, staged, frame, image, alpha, generation_ids.get(current))
                if failure is not None and len(generated) < len(attempts):
                    current = next(cell for cell in attempts if cell not in generated)
                    raise failure
            except QuotaPaused as exc:
                store.set_notice(job, str(exc), exc.resets_at)
                return RunOutcome("quota_limited", "QUOTA_EXCEEDED", str(exc), exc.resets_at)
            except Cancelled:
                store.set_notice(job, "Job paused. Completed tiles have been saved.")
                return RunOutcome("canceled")
            except (CodexError, OSError, ValueError) as exc:
                store.fail_tile(job, current, str(exc))
                store.set_notice(job, f"{current.label}. {exc}")
                code = "AUTH_REQUIRED" if isinstance(exc, AuthRequired) else "GENERATION_FAILED"
                return RunOutcome("failed", code, f"{current.label}. {exc}")
            finally:
                on_change(None)
    return RunOutcome()


def _file_sha256(path: Path) -> str | None:
    try:
        with path.open("rb") as stream:
            return file_digest(stream, "sha256").hexdigest()
    except FileNotFoundError:
        return None


def _scale_cache_digest(layout, artifacts, cache, alpha, cancelled) -> str:
    checksum = cache.with_suffix(".sha256")
    try:
        recorded_hash = checksum.read_text(encoding="ascii").strip()
    except (FileNotFoundError, UnicodeError):
        recorded_hash = None
    digest = recorded_hash
    if not digest or _file_sha256(cache) != digest:
        export_png(layout, artifacts, cache, alpha=alpha, cancelled=cancelled.is_set)
        digest = _file_sha256(cache)
        if digest is None:
            raise FileNotFoundError(f"The scale cache is missing: {cache}")
    if recorded_hash != digest:
        with atomic_output(checksum) as stream:
            stream.write(digest.encode("ascii") + b"\n")
    return digest


def run_scale_batch(
    store: Store,
    source: Source,
    layouts: Sequence[Layout],
    executable: Path | Callable[[], Path],
    parent: Path,
    *,
    pause: Event,
    cancelled: Event,
    prompt: str | None = "",
    subagents: bool = False,
    on_stage: Callable[[Job, Image.Image], None] = lambda *_: None,
    on_change: Callable[[Cell | None], None] = lambda _: None,
    on_export: Callable[[int], None] = lambda _: None,
    folder: Path | None = None,
    published: dict[int, tuple[str, str]] | None = None,
    on_published: Callable[[Job, Path, str, str], None] | None = None,
    on_outcome: Callable[[RunOutcome], None] = lambda _: None,
) -> tuple[Path, int, int]:
    stem = Path(source.name).stem
    if folder is None:
        folder = Path(mkdtemp(prefix=f"{stem}-scales-", dir=parent))
    else:
        folder.mkdir(exist_ok=True)
    completed = 0
    revision = ""
    input_path = None
    publications = published or {}
    for index, layout in enumerate(layouts):
        if pause.is_set() or cancelled.is_set():
            on_outcome(RunOutcome("canceled" if cancelled.is_set() else "paused"))
            break
        job = store.ensure_job(
            source, layout.grid, layout.scale, prompt=prompt or "",
            input_revision=revision, input_scale=layout.input_scale,
        )
        if prompt is not None:
            store.set_prompt(job, prompt)
        if job.input_path != input_path:
            imported = load_source(job.input_path)
            input_path = job.input_path
        on_stage(job, imported)
        cells = pending_cells(store.snapshot(job), job.layout.cells)
        if cells:
            outcome = run_tiles(
                store, job, cells, executable, pause=pause, cancelled=cancelled,
                on_change=on_change, subagents=subagents, input_image=imported,
            )
            if outcome is not None and outcome.state != "completed":
                on_outcome(outcome)
                break
        snapshot = store.snapshot(job)
        if pending_cells(snapshot, job.layout.cells) or cancelled.is_set():
            on_outcome(RunOutcome("canceled" if cancelled.is_set() else "paused"))
            break
        on_export(job.scale)
        destination = folder / f"{stem}-{job.scale}x.png"
        alpha = imported if imported.mode == "RGBA" else None
        revision = ""
        artifacts = snapshot.current_artifacts
        identity = [job.id, *(artifacts[cell].name for cell in job.layout.cells)]
        output_revision = sha256("\n".join(identity).encode("ascii")).hexdigest()
        previous_revision, previous_hash = publications.get(job.scale, (None, None))
        expected_hash = previous_hash if previous_revision == output_revision else None
        if index + 1 < len(layouts) and layouts[index + 1].input_scale == job.scale:
            # Published attempts have immutable names. Regenerating a tile changes
            # the input revision and creates a separate job for the next stage.
            revision = output_revision
            cache = source.scale_result_path(revision)
            digest = _scale_cache_digest(job.layout, artifacts, cache, alpha, cancelled)
            if _file_sha256(destination) != digest:
                with cache.open("rb") as stream, atomic_output(destination) as output:
                    copyfileobj(stream, output)
                    if cancelled.is_set():
                        raise Cancelled("Export canceled.")
        elif expected_hash and _file_sha256(destination) == expected_hash:
            digest = expected_hash
        else:
            export_png(job.layout, artifacts, destination, alpha=alpha, cancelled=cancelled.is_set)
            digest = _file_sha256(destination) if on_published is not None else None
        completed += 1
        if on_published is not None:
            if digest is None:
                raise FileNotFoundError(f"The published PNG is missing: {destination}")
            on_published(job, destination, output_revision, digest)
    return folder, completed, len(layouts)


def _commit_tile(store, job, cell, attempt, staged, frame, generated, alpha, generation_id):
    layout = job.layout
    snapshot = store.snapshot(job)
    previous = snapshot.tiles[cell]
    corrected, warning = align_tile(generated, frame, layout, cell)
    notes = {
        neighbor: seam_warning(layout, cell, neighbor, corrected, read_tile(path))
        for neighbor in layout.neighbors(cell)
        if (path := snapshot.tiles[neighbor].artifact) is not None
    }
    path = job.folder / f"{attempt}.png"
    save_png(Image.fromarray(corrected), path)
    store.finish_tile(job, cell, attempt, generation_id, warning, notes)
    refresh_thumbnails(store, job, changed=None if snapshot.previews_dirty else cell, alpha=alpha)
    old_thumbnail = thumbnail_path(previous.artifact) if previous.artifact else None
    for old in (staged, previous.artifact, old_thumbnail):
        if old is not None:
            try:
                old.unlink(missing_ok=True)
            except OSError:
                # The previous preview may still be reading this file.
                pass


def refresh_thumbnails(
    store: Store, job: Job, *, changed: Cell | None = None, force: bool = False,
    alpha: Image.Image | None = None, cancelled: Callable[[], bool] = lambda: False,
):
    layout = job.layout
    snapshot = store.snapshot(job)
    artifacts = snapshot.artifacts
    compositor = Compositor(layout, artifacts, cache_size=12, alpha=alpha)
    changed_support = layout.core(changed).expand(*layout.band(SEAM_MARGIN)) if changed is not None else None
    for cell, path in artifacts.items():
        if cancelled():
            raise Cancelled("Preview canceled.")
        core = layout.core(cell)
        destination = thumbnail_path(path)
        if core.intersect(layout.bounds) is None:
            continue
        if changed_support is None and not (force or snapshot.previews_dirty) and destination.is_file():
            continue
        if changed_support is not None and core.intersect(changed_support) is None:
            continue
        preview = Image.fromarray(compositor.render(core))
        preview.thumbnail((256, 256), Image.Resampling.LANCZOS)
        save_png(preview, destination)
    if snapshot.previews_dirty:
        store.mark_previews_ready(job)
