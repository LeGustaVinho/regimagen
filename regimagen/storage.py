"""Local project and result storage with per-tile checkpoints."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from hashlib import file_digest, sha256
from pathlib import Path
from struct import pack

from PIL import Image

from .codex import Cancelled
from .files import atomic_output, save_png
from .geometry import Cell, Grid, Layout
from .legacy_messages import translate_saved_message

JOBS_SCHEMA = """
CREATE TABLE IF NOT EXISTS {table} (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(id),
    columns INTEGER NOT NULL,
    rows INTEGER NOT NULL,
    scale INTEGER NOT NULL DEFAULT 0,
    input_revision TEXT NOT NULL DEFAULT '',
    input_scale INTEGER NOT NULL DEFAULT 1,
    prompt TEXT NOT NULL DEFAULT '',
    notice TEXT,
    quota_reset INTEGER,
    previews_dirty INTEGER NOT NULL DEFAULT 0,
    UNIQUE (source_id, columns, rows, scale, input_revision)
);
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS source_fills (
    source_id TEXT PRIMARY KEY REFERENCES sources(id),
    base_id TEXT NOT NULL REFERENCES sources(id),
    root_id TEXT NOT NULL REFERENCES sources(id),
    base_sha256 TEXT NOT NULL,
    prompt TEXT NOT NULL,
    edit_alpha INTEGER NOT NULL,
    metadata TEXT NOT NULL,
    derived_from TEXT REFERENCES source_fills(source_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS source_fills_root ON source_fills(root_id);
CREATE TABLE IF NOT EXISTS document_sources (
    document_id TEXT NOT NULL,
    pixel_hash TEXT NOT NULL,
    source_id TEXT NOT NULL REFERENCES sources(id),
    PRIMARY KEY (document_id, pixel_hash)
);
""" + JOBS_SCHEMA.format(table="jobs") + """
CREATE TABLE IF NOT EXISTS tiles (
    job_id TEXT NOT NULL REFERENCES jobs(id),
    col INTEGER NOT NULL,
    row INTEGER NOT NULL,
    attempt TEXT,
    result TEXT,
    generation_id TEXT,
    warning TEXT,
    error TEXT,
    prompt TEXT NOT NULL DEFAULT '',
    stale INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (job_id, col, row)
);
CREATE TABLE IF NOT EXISTS seams (
    job_id TEXT NOT NULL REFERENCES jobs(id),
    col INTEGER NOT NULL,
    row INTEGER NOT NULL,
    axis TEXT NOT NULL,
    warning TEXT NOT NULL,
    PRIMARY KEY (job_id, col, row, axis)
);
"""


def thumbnail_path(artifact: Path) -> Path:
    return artifact.with_name(f"{artifact.stem}-preview-v3.png")


@dataclass(frozen=True)
class Source:
    id: str
    name: str
    width: int
    height: int
    path: Path

    def scale_result_path(self, revision: str) -> Path:
        path = self.path.with_name(f"stage-{revision}.png")
        legacy_path = self.path.with_name(f"etapa-{revision}.png")
        return legacy_path if legacy_path.is_file() and not path.exists() else path


@dataclass(frozen=True)
class FillEdit:
    source: Source
    base: Source
    base_sha256: str
    prompt: str
    edit_alpha: bool
    mask_path: Path
    candidate_path: Path
    metadata: dict
    derived_from: str | None


@dataclass(frozen=True)
class Job:
    id: str
    source: Source
    grid: Grid
    folder: Path
    scale: int = 0
    input_revision: str = ""
    input_scale: int = 1

    @property
    def input_path(self) -> Path:
        return self.source.scale_result_path(self.input_revision) if self.input_revision else self.source.path

    @property
    def layout(self) -> Layout:
        return Layout(
            self.source.width * self.input_scale, self.source.height * self.input_scale,
            self.grid, self.scale, self.input_scale,
        )


@dataclass(frozen=True)
class Tile:
    attempt: str | None = None
    artifact: Path | None = None
    thumbnail: Path | None = None
    warning: str | None = None
    error: str | None = None
    alignment_warning: str | None = None
    prompt: str = ""
    stale: bool = False
    generation_id: str | None = None


@dataclass(frozen=True)
class Snapshot:
    tiles: dict[Cell, Tile]
    notice: str | None = None
    quota_reset: int | None = None
    prompt: str = ""
    seams: tuple[tuple[Cell, Cell, str], ...] = ()
    previews_dirty: bool = False

    @property
    def artifacts(self) -> dict[Cell, Path]:
        return {cell: tile.artifact for cell, tile in self.tiles.items() if tile.artifact is not None}

    @property
    def current_artifacts(self) -> dict[Cell, Path]:
        return {cell: path for cell, path in self.artifacts.items() if not self.tiles[cell].stale}


class Store:
    def __init__(self, root: Path, *, readonly: bool = False):
        self.root = root
        self.readonly = readonly
        self.db_path = root / "projects.sqlite3"
        self.sources = root / "originals"
        self.lots = root / "jobs"
        # Keep existing databases and their artifact paths together.
        legacy_database = root / "projetos.sqlite3"
        if legacy_database.is_file() and not self.db_path.exists():
            self.db_path = legacy_database
            self.sources = root / "originais"
            self.lots = root / "lotes"
        if readonly:
            return
        self.sources.mkdir(parents=True, exist_ok=True)
        self.lots.mkdir(exist_ok=True)
        with self._connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
        self._migrate()

    def check_export_destination(self, path: Path):
        path = path.resolve()
        managed_dirs = (self.sources.resolve(), self.lots.resolve())
        databases = (self.db_path, self.root / "automation.sqlite3")
        managed_files = {(self.root / "regimagen.lock").resolve()}
        managed_files.update(
            database.with_name(database.name + suffix).resolve()
            for database in databases for suffix in ("", "-wal", "-shm", "-journal")
        )
        if any(path.is_relative_to(folder) for folder in managed_dirs) or path in managed_files:
            raise ValueError("Choose an output path outside Regimagen's managed files.")

    def _migrate(self):
        with self._connection() as db:
            db.execute("PRAGMA foreign_keys=OFF")
            db.execute("BEGIN IMMEDIATE")
            columns = {column["name"] for column in db.execute("PRAGMA table_info(jobs)")}
            if "input_revision" not in columns:
                db.execute(JOBS_SCHEMA.format(table="jobs_new"))
                fields = "id, source_id, columns, rows, notice, quota_reset"
                if "scale" in columns:
                    fields += ", scale, prompt"
                db.execute(
                    f"INSERT INTO jobs_new ({fields}) SELECT {fields} FROM jobs ORDER BY rowid"
                )
                db.execute("DROP TABLE jobs")
                db.execute("ALTER TABLE jobs_new RENAME TO jobs")
            elif "previews_dirty" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN previews_dirty INTEGER NOT NULL DEFAULT 0")
            if "prompt" not in {column["name"] for column in db.execute("PRAGMA table_info(tiles)")}:
                db.execute("ALTER TABLE tiles ADD COLUMN prompt TEXT NOT NULL DEFAULT ''")
                db.execute("ALTER TABLE tiles ADD COLUMN stale INTEGER NOT NULL DEFAULT 0")

    @contextmanager
    def _connection(self):
        if self.readonly:
            db = sqlite3.connect(self.db_path.as_uri() + "?mode=ro", uri=True, timeout=10)
        else:
            db = sqlite3.connect(self.db_path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def import_source(self, image: Image.Image, name: str) -> Source:
        source_id = uuid.uuid4().hex
        path = self.sources / f"{source_id}.png"
        save_png(image, path)
        with self._connection() as db:
            db.execute("INSERT INTO sources VALUES (?, ?, ?, ?)", (source_id, name, *image.size))
        return Source(source_id, name, *image.size, path)

    def document_source(
        self, document_id: str, image: Image.Image, name: str, *,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> Source:
        """Associate an immutable reconstruction input with a rendered document."""
        def check_cancelled(_path=None):
            if cancelled():
                raise Cancelled("Reconstruction canceled.")

        digest = sha256(pack(">II", *image.size))
        digest.update(image.mode.encode("ascii"))
        for top in range(0, image.height, 64):
            check_cancelled()
            digest.update(image.crop((0, top, image.width, min(top + 64, image.height))).tobytes())
        pixel_hash = digest.hexdigest()
        written = None
        try:
            with self._connection() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute(
                    "SELECT sources.* FROM document_sources JOIN sources ON sources.id = source_id "
                    "WHERE document_id=? AND pixel_hash=?", (document_id, pixel_hash),
                ).fetchone()
                if row is not None:
                    path = self.sources / f"{row['id']}.png"
                    if not path.is_file():
                        save_png(image, path, overwrite=False, before_publish=check_cancelled)
                    return Source(row["id"], row["name"], row["width"], row["height"], path)
                source_id = uuid.uuid4().hex
                path = self.sources / f"{source_id}.png"
                save_png(image, path, overwrite=False, before_publish=check_cancelled)
                written = path
                db.execute("INSERT INTO sources VALUES (?, ?, ?, ?)", (source_id, name, *image.size))
                db.execute("INSERT INTO document_sources VALUES (?, ?, ?)", (document_id, pixel_hash, source_id))
                check_cancelled()
        except Exception:
            if written is not None:
                with suppress(OSError):
                    written.unlink(missing_ok=True)
            raise
        return Source(source_id, name, *image.size, path)

    def _fill_paths(self, source_id: str) -> tuple[Path, Path]:
        return (
            self.sources / f"{source_id}-fill-mask.png",
            self.sources / f"{source_id}-fill-candidate.png",
        )

    def _has_fill_schema(self, db: sqlite3.Connection) -> bool:
        # Read-only callers can open databases that have never run the additive migration.
        return not self.readonly or db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='source_fills'"
        ).fetchone() is not None

    def fill_sources(self) -> list[Source]:
        with self._connection() as db:
            if not self._has_fill_schema(db):
                return []
            rows = db.execute("""
                SELECT * FROM sources
                WHERE id IN (SELECT source_id FROM source_fills UNION SELECT root_id FROM source_fills)
                ORDER BY rowid DESC
            """).fetchall()
        return [
            Source(row["id"], row["name"], row["width"], row["height"], self.sources / f"{row['id']}.png")
            for row in rows
        ]

    def get_fill(self, source: Source) -> FillEdit | None:
        with self._connection() as db:
            if not self._has_fill_schema(db):
                return None
            row = db.execute("""
                SELECT fills.*, base.name, base.width, base.height
                FROM source_fills AS fills JOIN sources AS base ON base.id = fills.base_id
                WHERE fills.source_id = ?
            """, (source.id,)).fetchone()
        if row is None:
            return None
        base_id = row["base_id"]
        base = Source(base_id, row["name"], row["width"], row["height"], self.sources / f"{base_id}.png")
        mask_path, candidate_path = self._fill_paths(source.id)
        return FillEdit(
            source, base, row["base_sha256"], row["prompt"], bool(row["edit_alpha"]),
            mask_path, candidate_path, json.loads(row["metadata"]), row["derived_from"],
        )

    def save_fill(
        self, base: Source, image: Image.Image, mask: Image.Image, candidate: Image.Image,
        *, prompt: str, edit_alpha: bool, metadata: dict, base_sha256: str,
        derived_from: str | None = None, cancelled: Callable[[], bool] = lambda: False,
    ) -> Source:
        """Publish an accepted fill after all of its PNGs have been written."""
        if self.readonly:
            raise ValueError("This project is read-only.")
        size = base.width, base.height
        if image.size != size or mask.size != size or mask.mode != "L" or not mask.getbbox():
            raise ValueError("The result and selection must match the base image.")
        if image.mode not in ("RGB", "RGBA") or candidate.mode not in ("RGB", "RGBA"):
            raise ValueError("The result and generated image must be RGB or RGBA.")
        if not isinstance(metadata, dict):
            raise ValueError("The fill parameters must be an object.")
        parameters = json.dumps(metadata, allow_nan=False)

        def check_cancelled(_path: Path | None = None):
            if cancelled():
                raise Cancelled("Saving the fill was canceled.")

        check_cancelled()
        source_id = uuid.uuid4().hex
        path = self.sources / f"{source_id}.png"
        mask_path, candidate_path = self._fill_paths(source_id)
        written = []
        try:
            save_png(image, path, overwrite=False, before_publish=check_cancelled)
            written.append(path)
            with atomic_output(mask_path, overwrite=False, before_publish=check_cancelled) as stream:
                mask.save(stream, format="PNG")
            written.append(mask_path)
            save_png(candidate, candidate_path, overwrite=False, before_publish=check_cancelled)
            written.append(candidate_path)

            with self._connection() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("""
                    SELECT base.width, base.height, root.id AS root_id, root.name AS root_name
                    FROM sources AS base LEFT JOIN source_fills AS parent ON parent.source_id = base.id
                    JOIN sources AS root ON root.id = COALESCE(parent.root_id, base.id)
                    WHERE base.id = ?
                """, (base.id,)).fetchone()
                base_path = self.sources / f"{base.id}.png"
                if row is None or (row["width"], row["height"]) != size or base.path.resolve() != base_path.resolve():
                    raise ValueError("The fill base is unavailable in this project.")
                with base_path.open("rb") as stream:
                    if file_digest(stream, "sha256").hexdigest() != base_sha256:
                        raise ValueError("The fill base has changed. Open the image again to continue.")
                if derived_from is not None and db.execute(
                    "SELECT 1 FROM source_fills WHERE source_id = ? AND base_id = ?", (derived_from, base.id),
                ).fetchone() is None:
                    raise ValueError("The earlier fill belongs to a different base image.")

                root_id = row["root_id"]
                sequence = db.execute(
                    "SELECT COUNT(*) FROM source_fills WHERE root_id = ?", (root_id,),
                ).fetchone()[0] + 1
                name = f"{Path(row['root_name']).stem} — Fill {sequence}.png"
                db.execute("INSERT INTO sources VALUES (?, ?, ?, ?)", (source_id, name, *size))
                db.execute("""
                    INSERT INTO source_fills (
                        source_id, base_id, root_id, base_sha256, prompt, edit_alpha, metadata, derived_from
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (source_id, base.id, root_id, base_sha256, prompt, int(edit_alpha), parameters, derived_from))
                check_cancelled()
        except Exception:
            for artifact in written:
                with suppress(OSError):
                    artifact.unlink(missing_ok=True)
            raise
        return Source(source_id, name, *size, path)

    def _job_from_row(self, row: sqlite3.Row) -> Job:
        source_id = row["source_id"]
        source = Source(
            source_id, row["name"], row["width"], row["height"], self.sources / f"{source_id}.png",
        )
        return self._read_job(row, source)

    def list_sources(self, source_id: str | None = None) -> list[Source]:
        with self._connection() as db:
            if source_id is None:
                rows = db.execute("SELECT * FROM sources ORDER BY rowid DESC").fetchall()
            else:
                rows = db.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchall()
        return [
            Source(row["id"], row["name"], row["width"], row["height"], self.sources / f"{row['id']}.png")
            for row in rows
        ]

    def _read_job(self, row: sqlite3.Row, source: Source) -> Job:
        return Job(
            row["id"], source, Grid(row["columns"], row["rows"]), self.lots / row["id"],
            row["scale"], row["input_revision"], row["input_scale"],
        )

    def recent_jobs(self, *, job_id: str | None = None, source_id: str | None = None) -> list[Job]:
        with self._connection() as db:
            rows = db.execute(
                "SELECT jobs.*, sources.name, sources.width, sources.height "
                "FROM jobs JOIN sources ON sources.id=jobs.source_id "
                "WHERE (? IS NULL OR jobs.id=?) AND (? IS NULL OR jobs.source_id=?) ORDER BY jobs.rowid DESC",
                (job_id, job_id, source_id, source_id),
            ).fetchall()
        return [self._job_from_row(row) for row in rows]

    def find_job(self, source: Source, grid: Grid, scale: int = 0) -> Job | None:
        with self._connection() as db:
            row = db.execute(
                "SELECT * FROM jobs WHERE source_id=? AND columns=? AND rows=? AND scale=? AND input_revision=''",
                (source.id, grid.columns, grid.rows, scale),
            ).fetchone()
        if row is None:
            return None
        return self._read_job(row, source)

    def ensure_job(
        self, source: Source, grid: Grid, scale: int = 0, *, prompt: str = "",
        input_revision: str = "", input_scale: int = 1,
    ) -> Job:
        job_id = uuid.uuid4().hex
        job = Job(job_id, source, grid, self.lots / job_id, scale, input_revision, input_scale)
        if job.layout.grid_error:
            raise ValueError(job.layout.grid_error)
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM jobs WHERE source_id=? AND columns=? AND rows=? AND scale=? AND input_revision=?",
                (source.id, grid.columns, grid.rows, scale, input_revision),
            ).fetchone()
            if row is not None:
                return self._read_job(row, source)
            job.folder.mkdir()
            db.execute(
                "INSERT INTO jobs (id, source_id, columns, rows, scale, prompt, input_revision, input_scale) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (job_id, source.id, grid.columns, grid.rows, scale, prompt, input_revision, input_scale),
            )
            db.executemany(
                "INSERT INTO tiles (job_id, col, row) VALUES (?, ?, ?)",
                ((job_id, cell.column, cell.row) for cell in job.layout.cells),
            )
        return job

    def snapshot(self, job: Job) -> Snapshot:
        with self._connection() as db:
            db.execute("BEGIN")
            status = db.execute("SELECT notice, quota_reset, prompt, previews_dirty FROM jobs WHERE id=?", (job.id,)).fetchone()
            rows = db.execute(
                "SELECT * FROM tiles WHERE job_id=? ORDER BY row, col", (job.id,),
            ).fetchall()
            seams = db.execute("SELECT * FROM seams WHERE job_id=?", (job.id,)).fetchall()
        borders: dict[Cell, list[str]] = {}
        seam_warnings = []
        for seam in seams:
            first = Cell(seam["col"], seam["row"])
            second = Cell(first.column + (seam["axis"] == "x"), first.row + (seam["axis"] == "y"))
            reason = translate_saved_message(seam["warning"])
            seam_warnings.append((first, second, reason))
            for cell in (first, second):
                borders.setdefault(cell, []).append(reason)
        tiles: dict[Cell, Tile] = {}
        for row in rows:
            cell = Cell(row["col"], row["row"])
            alignment_warning = translate_saved_message(row["warning"])
            notes = [alignment_warning, *borders.get(cell, [])]
            warning = " ".join(dict.fromkeys(note for note in notes if note)) or None
            artifact, thumbnail = None, None
            error = translate_saved_message(row["error"])
            if row["result"]:
                result = job.folder / f"{row['result']}.png"
                if result.is_file():
                    artifact = result
                    if not status["previews_dirty"]:
                        small = thumbnail_path(result)
                        thumbnail = small if small.is_file() else None
                else:
                    error = "This tile file is missing. Regenerate the tile."
            tiles[cell] = Tile(
                row["attempt"], artifact, thumbnail, warning, error, alignment_warning, row["prompt"], bool(row["stale"]),
                row["generation_id"],
            )
        return Snapshot(
            {cell: tiles.get(cell, Tile()) for cell in job.layout.cells},
            translate_saved_message(status["notice"]), status["quota_reset"], status["prompt"],
            tuple(seam_warnings), bool(status["previews_dirty"]),
        )

    def set_prompt(self, job: Job, text: str, cell: Cell | None = None):
        with self._connection() as db:
            if cell is None:
                changed = db.execute(
                    "UPDATE jobs SET prompt=?, notice=NULL, quota_reset=NULL WHERE id=? AND prompt<>?",
                    (text, job.id, text),
                )
                if changed.rowcount:
                    db.execute(
                        "UPDATE tiles SET stale=(result IS NOT NULL), attempt=NULL, error=NULL WHERE job_id=?",
                        (job.id,),
                    )
            else:
                db.execute(
                    "INSERT INTO tiles (job_id, col, row, prompt) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT (job_id, col, row) DO UPDATE SET prompt=excluded.prompt, "
                    "stale=(result IS NOT NULL), attempt=NULL, error=NULL WHERE prompt<>excluded.prompt",
                    (job.id, cell.column, cell.row, text),
                )

    def set_notice(self, job: Job, message: str | None, quota_reset: int | None = None):
        with self._connection() as db:
            db.execute("UPDATE jobs SET notice=?, quota_reset=? WHERE id=?", (message, quota_reset, job.id))

    def begin_tile(self, job: Job, cell: Cell, attempt: str):
        with self._connection() as db:
            db.execute(
                "INSERT INTO tiles (job_id, col, row, attempt) VALUES (?, ?, ?, ?) "
                "ON CONFLICT (job_id, col, row) DO UPDATE SET attempt=excluded.attempt, error=NULL",
                (job.id, cell.column, cell.row, attempt),
            )

    def fail_tile(self, job: Job, cell: Cell, message: str):
        with self._connection() as db:
            db.execute(
                "UPDATE tiles SET error=? WHERE job_id=? AND col=? AND row=?",
                (message, job.id, cell.column, cell.row),
            )

    def finish_tile(
        self, job: Job, cell: Cell, attempt: str, generation_id: str | None,
        warning: str | None, seams: dict[Cell, str | None],
    ):
        with self._connection() as db:
            result = db.execute(
                "UPDATE tiles SET result=?, generation_id=?, warning=?, error=NULL, attempt=NULL, stale=0 "
                "WHERE job_id=? AND col=? AND row=? AND attempt=?",
                (attempt, generation_id, warning, job.id, cell.column, cell.row, attempt),
            )
            if result.rowcount != 1:
                raise ValueError("This tile has already been replaced by another edit.")
            db.execute("UPDATE jobs SET previews_dirty=1 WHERE id=?", (job.id,))
            for neighbor, reason in seams.items():
                key = (
                    job.id, min(cell.column, neighbor.column), min(cell.row, neighbor.row),
                    "x" if cell.row == neighbor.row else "y",
                )
                db.execute("DELETE FROM seams WHERE job_id=? AND col=? AND row=? AND axis=?", key)
                if reason:
                    db.execute("INSERT INTO seams VALUES (?, ?, ?, ?, ?)", (*key, reason))

    def mark_previews_ready(self, job: Job):
        with self._connection() as db:
            db.execute("UPDATE jobs SET previews_dirty=0 WHERE id=?", (job.id,))
