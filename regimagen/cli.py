"""Regimagen's local, JSON-first command line interface."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import time

from PySide6.QtCore import QCoreApplication

from regimagen import __version__
from regimagen.automation import ACTIVE, DEFAULT_CONFIG, CommandError, Operations
from regimagen.cli_images import cell_at, cell_json, inspect_image, rectangle, save_references
from regimagen.codex import AuthRequired, CodexClient, CodexError, find_codex
from regimagen.composite import export_parts, export_png, render_part
from regimagen.files import load_source, save_png
from regimagen.geometry import Grid, Layout, plan_scales, parse_scales, scale_limit, suggested_grid
from regimagen.jobs import RunOutcome, pending_cells, refresh_thumbnails, run_scale_batch, run_tiles
from regimagen.seams import seam_region
from regimagen.storage import Store
from regimagen.workspace import WorkspaceBusy, exclusive_workspace


SCHEMA_VERSION = 1


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise CommandError("INVALID_ARGUMENT", message)


def output_json(result: dict, *, ok: bool):
    response = {"schema_version": SCHEMA_VERSION, "ok": ok, "result" if ok else "error": result}
    print(json.dumps(response, ensure_ascii=False, allow_nan=False), flush=True)


def default_root() -> Path:
    return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "Regimagen" / "cli"


def operation_json(item: dict) -> dict:
    request = item["request"]
    progress = {
        key: value for key, value in item["progress"].items() if key not in ("baseline", "revisions", "publications")
    }
    return {
        "operation_id": item["id"], "request_id": item["request_id"], "kind": request["kind"],
        "job_id": request.get("job_id"), "source_id": request.get("source_id"), "state": item["state"],
        "requested_action": item["action"], "worker_pid": item["pid"],
        "created_at": item["created_at"], "updated_at": item["updated_at"],
        "quota_reset": item["quota_reset"], "error": item["error"], "progress": progress,
    }


def source_info(source) -> dict:
    grid = suggested_grid(source.width, source.height)
    return {
        "source_id": source.id, "name": source.name, "path": str(source.path), "available": source.path.is_file(),
        "width": source.width, "height": source.height, "suggested_grid": grid_info(grid),
        "max_scale": scale_limit(source.width, source.height),
    }


def grid_info(grid: Grid) -> dict:
    return {"columns": grid.columns, "rows": grid.rows}


def layout_info(layout: Layout) -> dict:
    width, height = layout.size
    return {
        "width": width, "height": height, "grid": grid_info(layout.grid), "edits": layout.grid.count,
        "input_width": layout.source_width, "input_height": layout.source_height,
        "scale": layout.scale or None, "input_scale": layout.input_scale, "enlarges": layout.enlarges,
    }


def source_by_id(store: Store, source_id: str):
    if store.db_path.is_file() and (items := store.list_sources(source_id)):
        return items[0]
    raise CommandError("NOT_FOUND", "Source not found.", status=3)


def job_by_id(store: Store, job_id: str):
    if store.db_path.is_file() and (items := store.recent_jobs(job_id=job_id)):
        return items[0]
    raise CommandError("NOT_FOUND", "Job not found.", status=3)


def readable_store(root: Path) -> Store:
    store = Store(root, readonly=True)
    if not store.db_path.is_file():
        return store
    # Upgrade workspaces written by earlier desktop or CLI versions.
    with store._connection() as db:
        jobs = {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}
        tiles = {row["name"] for row in db.execute("PRAGMA table_info(tiles)")}
    if not {"input_revision", "previews_dirty"} <= jobs or "prompt" not in tiles:
        with exclusive_workspace(root):
            Store(root)
    return store


def tile_info(job, cell, tile) -> dict:
    return {
        **cell_json(cell), "core": rectangle(job.layout.core(cell)), "frame": rectangle(job.layout.frame(cell)),
        "neighbors": [cell_json(other) for other in job.layout.neighbors(cell)],
        "artifact_path": str(tile.artifact) if tile.artifact else None,
        "thumbnail_path": str(tile.thumbnail) if tile.thumbnail else None,
        "generation_id": tile.generation_id, "exportable": tile.artifact is not None and not tile.stale,
        "pending": tile.artifact is None or tile.stale or tile.attempt is not None,
        "stale": tile.stale, "attempt_pending": tile.attempt is not None,
        "warning": tile.warning, "alignment_warning": tile.alignment_warning,
        "error": tile.error, "prompt": tile.prompt,
    }


def job_info(job, snapshot, *, tiles: bool = False) -> dict:
    ready = len(snapshot.current_artifacts)
    data = {
        "job_id": job.id, "source": source_info(job.source), **layout_info(job.layout),
        "input_path": str(job.input_path), "input_revision": job.input_revision or None,
        "ready": ready, "pending": len(pending_cells(snapshot, job.layout.cells)),
        "review": sum(bool(tile.warning) and not tile.stale for tile in snapshot.tiles.values()),
        "can_export_png": ready == job.grid.count, "can_export_tiles": ready > 0,
        "notice": snapshot.notice, "quota_reset": snapshot.quota_reset, "prompt": snapshot.prompt,
    }
    if tiles:
        data["tiles"] = [tile_info(job, cell, tile) for cell, tile in snapshot.tiles.items()]
        data["seams"] = [
            {"first": cell_json(first), "second": cell_json(second), "warning": warning,
             "region": rectangle(seam_region(job.layout, first, second))}
            for first, second, warning in snapshot.seams
        ]
    return data


def executable_for(value: str):
    path = find_codex() if value == "auto" else Path(value)
    if path is None or not path.is_file():
        raise CommandError("CODEX_NOT_FOUND", "Install Codex CLI or specify its executable with --codex.", status=1)
    return path.resolve()


def check_destination(store: Store, path: Path, *, overwrite: bool, directory: bool = False):
    managed = (store.sources.resolve(), store.lots.resolve())
    if any(path.is_relative_to(folder) for folder in managed) or path in {
        store.db_path.resolve(), (store.root / "regimagen.lock").resolve(),
        (store.root / "automation.sqlite3").resolve(),
        *(store.db_path.with_name(store.db_path.name + suffix).resolve() for suffix in ("-wal", "-shm", "-journal")),
        *(store.root / f"automation.sqlite3{suffix}" for suffix in ("-wal", "-shm", "-journal")),
    }:
        raise ValueError("Choose an output path outside Regimagen's managed files.")
    if directory and path.exists() and not path.is_dir():
        raise ValueError("The output directory is a file.")
    if not directory and path.exists() and (not overwrite or path.is_dir()):
        raise CommandError("OUTPUT_EXISTS", "The output already exists.", status=4, details={"path": str(path)})


def checked_snapshot(job, store, render):
    try:
        return render(store.snapshot(job))
    except FileNotFoundError:
        # Another process can replace a tile between a snapshot and a file read.
        return render(store.snapshot(job))


def scale_options(width: int, height: int, supplied, config: dict):
    if supplied:
        return parse_scales(" ".join(supplied))
    if config["scales"] is not None:
        return tuple(config["scales"])
    limit = scale_limit(width, height)
    if limit < 2:
        raise ValueError("This image is too large for a 2× scale.")
    return tuple(range(2, min(5, limit) + 1))


def plan_for(args, store: Store, config: dict) -> dict:
    if bool(args.file) == bool(args.source_id):
        raise ValueError("Specify a source ID or --file.")
    if args.file:
        width, height = load_source(Path(args.file)).size
    else:
        source = source_by_id(store, args.source_id)
        width, height = source.width, source.height
    if args.scales or args.input_mode:
        if args.grid:
            raise ValueError("Specify a grid or a scale batch.")
        mode = args.input_mode or config["input_mode"]
        scales = scale_options(width, height, args.scales, config)
        stages = plan_scales(width, height, " ".join(map(str, scales)), chained=mode == "previous")
        return {"input_mode": mode, "stages": [layout_info(layout) for layout in stages],
                "max_edits": sum(layout.grid.count for layout in stages)}
    suggested = suggested_grid(width, height)
    layout = Layout(width, height, Grid(*args.grid) if args.grid else suggested)
    return {"layout": layout_info(layout), "suggested": layout_info(Layout(width, height, suggested)),
            "max_scale": scale_limit(width, height)}


def generation_plan(args, job, snapshot) -> tuple[list[list[int]], dict]:
    if not job.layout.enlarges:
        raise ValueError("The selected grid must enlarge the image in both dimensions. Run plan to see the suggested grid.")
    if args.anchor and not args.preview:
        raise ValueError("--anchor requires --preview.")
    if args.regenerate and not args.tile:
        raise ValueError("--regenerate requires one or more --tile options.")
    if args.preview:
        selection = list(job.layout.preview(cell_at(job, args.anchor) if args.anchor else None))
    elif args.tile:
        selection = list(dict.fromkeys(cell_at(job, position) for position in args.tile))
    else:
        selection = list(job.layout.cells)
    pending = selection if args.regenerate else pending_cells(snapshot, selection)
    positions = [[cell.column + 1, cell.row + 1] for cell in selection]
    return positions, {
        "job_id": job.id, "selected": [cell_json(cell) for cell in selection],
        "pending": [cell_json(cell) for cell in pending], "max_edits": len(pending),
        "reused": [cell_json(cell) for cell in selection if cell not in pending],
    }


def export_selection(job, snapshot, kind: str, positions):
    current = snapshot.current_artifacts
    if kind == "png":
        if positions:
            raise ValueError("A composed PNG uses the full grid.")
        if len(current) != job.grid.count:
            raise CommandError("NOT_READY", "Complete the grid before exporting a composed PNG.", status=4)
    if kind == "tile-png" and len(positions or ()) != 1:
        raise ValueError("A tile PNG requires exactly one --tile.")
    if positions:
        cells = list(dict.fromkeys(cell_at(job, position) for position in positions))
        if any(cell not in current for cell in cells):
            raise CommandError("NOT_READY", "One or more selected tiles do not have current results.", status=4)
        current = {cell: current[cell] for cell in cells}
    if not current:
        raise CommandError("NOT_READY", "Generate at least one tile before exporting.", status=4)
    return current


def run_operation(root: Path, item, pause, cancel, report) -> RunOutcome:
    request = item["request"]
    store = Store(root)
    kind = request["kind"]
    if cancel.is_set():
        return RunOutcome("canceled")
    if kind == "batch":
        return run_batch_operation(store, item, pause, cancel, report)
    job = job_by_id(store, request["job_id"])
    snapshot = store.snapshot(job)
    if kind == "generate":
        selection = [cell_at(job, value) for value in request["tiles"]]
        baseline = item["progress"].get("baseline")
        if request["regenerate"] and baseline is None:
            baseline = [str(snapshot.tiles[cell].artifact) for cell in selection]
            report(baseline=baseline)

        def remaining():
            current = store.snapshot(job)
            cells = pending_cells(current, selection)
            if baseline is not None:
                cells = [cell for cell, previous in zip(selection, baseline, strict=True)
                         if cell in cells or str(current.tiles[cell].artifact) == previous]
            return cells

        def change(active):
            report(job_id=job.id, completed=len(selection) - len(remaining()), total=len(selection),
                   active_tile=cell_json(active) if active else None)

        cells = remaining()
        change(None)
        if not cells:
            return RunOutcome()
        return run_tiles(
            store, job, cells, lambda: executable_for(request["codex"]), pause=pause, cancelled=cancel,
            on_change=change, subagents=request["subagents"],
        )

    source = load_source(job.input_path)
    sprite = source if source.mode == "RGBA" else None
    if kind == "refresh_previews":
        refresh_thumbnails(store, job, force=True, alpha=sprite, cancelled=cancel.is_set)
        refreshed = store.snapshot(job)
        report(job_id=job.id, outputs=[{"path": str(tile.thumbnail)} for tile in refreshed.tiles.values() if tile.thumbnail])
        return RunOutcome()

    output = Path(request["output"])
    overwrite = request["overwrite"]
    check_destination(store, output, overwrite=overwrite)
    format_name = request["format"]
    artifacts = export_selection(job, snapshot, format_name, request["tiles"])
    output.parent.mkdir(parents=True, exist_ok=True)

    def progress(done, total):
        report(job_id=job.id, completed=done, total=total, unit="rows" if format_name == "png" else "tiles")

    artifact = {"path": str(output), "format": format_name}
    if format_name == "tile-png":
        cell, path = next(iter(artifacts.items()))
        image, position = render_part(job.layout, cell, path, alpha=sprite)
        if cancel.is_set():
            return RunOutcome("canceled")
        save_png(image, output, overwrite=overwrite)
        artifact.update(position, **cell_json(cell))
        progress(1, 1)
    else:
        exporter = export_png if format_name == "png" else export_parts
        exporter(job.layout, artifacts, output, progress=progress, cancelled=cancel.is_set, alpha=sprite, overwrite=overwrite)
        artifact.update(width=job.layout.size[0], height=job.layout.size[1])
        if format_name == "tiles-zip":
            artifact["manifest"] = "positions.json"
    report(outputs=[artifact])
    return RunOutcome()


def run_batch_operation(store, item, pause, cancel, report):
    request = item["request"]
    source = source_by_id(store, request["source_id"])
    layouts = plan_scales(
        source.width, source.height, " ".join(map(str, request["scales"])),
        chained=request["input_mode"] == "previous",
    )
    parent = Path(request["output_dir"])
    folder = parent / f"{Path(source.name).stem}-scales-{item['id']}"
    check_destination(store, folder, overwrite=False, directory=True)
    parent.mkdir(parents=True, exist_ok=True)
    publications = {int(key): tuple(value) for key, value in item["progress"].get("publications", {}).items()}
    # Rebuild the public output list in stage order. An earlier stage can change
    # the input of every stage that follows it in a chained batch.
    outputs = []
    report(outputs=outputs, completed_stages=0, active_tile=None)
    stage_job = None
    outcome = RunOutcome()

    def stage(job, _image):
        nonlocal stage_job
        stage_job = job
        report(job_id=job.id, scale=job.scale, input_path=str(job.input_path),
               input_revision=job.input_revision or None, phase="generate", active_tile=None,
               completed=0, total=job.grid.count, stages=len(layouts), output_dir=str(folder))
        change(None)

    def change(active):
        snapshot = store.snapshot(stage_job)
        report(completed=stage_job.grid.count - len(pending_cells(snapshot, stage_job.layout.cells)),
               active_tile=cell_json(active) if active else None)

    def published(job, destination, revision, digest):
        outputs.append({"scale": job.scale, "job_id": job.id, "path": str(destination),
                        "width": job.layout.size[0], "height": job.layout.size[1],
                        "input_path": str(job.input_path), "input_revision": job.input_revision or None})
        publications[job.scale] = (revision, digest)
        report(outputs=outputs, publications=publications, completed_stages=len(outputs))

    def stopped(result):
        nonlocal outcome
        outcome = result

    run_scale_batch(
        store, source, layouts, lambda: executable_for(request["codex"]), parent, pause=pause, cancelled=cancel,
        prompt=request["prompt"], subagents=request["subagents"], on_stage=stage, on_change=change,
        on_export=lambda _: report(phase="export", active_tile=None), folder=folder, published=publications,
        on_published=published, on_outcome=stopped,
    )
    report(active_tile=None)
    return outcome


def command_parser():
    parser = Parser(prog="regimagen-cli", description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-dir", metavar="DIR", help="Workspace directory. Defaults to the local Regimagen CLI folder.")
    parser.add_argument("--version", action="version", version=__version__)
    groups = parser.add_subparsers(dest="command", required=True)

    def leaf(group, name, description):
        command = group.add_parser(name, help=description, description=description, allow_abbrev=False)
        return command

    def group(name, description):
        return leaf(groups, name, description).add_subparsers(dest="action", required=True)

    def job_id(command):
        command.add_argument("job_id")
        return command

    def tile(command, *, repeated=False):
        command.add_argument("--tile", nargs=2, type=int, action="append" if repeated else "store",
                             metavar=("COLUMN", "ROW"), help="One-based tile coordinates.")

    def background(command):
        command.add_argument("--request-id", help="Reuse the operation ID for an identical submission.")

    def codex_options(command, *, subagents=False):
        command.add_argument("--codex", metavar="PATH_OR_AUTO", help="Codex executable override.")
        if subagents:
            command.add_argument("--subagents", action=argparse.BooleanOptionalAction, default=None,
                                 help="Override the workspace subagent preference.")

    leaf(groups, "capabilities", "Describe commands and arguments as JSON.")
    doctor = leaf(groups, "doctor", "Locate Codex and optionally verify the ChatGPT login.")
    codex_options(doctor)
    doctor.add_argument("--check-login", action="store_true")

    config = group("config", "Manage workspace preferences.")
    leaf(config, "show", "Show workspace preferences.")
    keys = tuple(DEFAULT_CONFIG)
    set_config = leaf(config, "set", "Set codex, subagents, scales, or input_mode.")
    set_config.add_argument("key", choices=keys)
    set_config.add_argument("value")
    leaf(config, "reset", "Restore the default for one setting or all settings.").add_argument(
        "key", choices=keys, nargs="?",
    )

    sources = group("source", "Import and inspect local source images.")
    leaf(sources, "import", "Import a still PNG, JPEG, or WebP image.").add_argument("file")
    leaf(sources, "list", "List saved source images.")
    leaf(sources, "show", "Show the source image and suggested grid.").add_argument("source_id")

    plan = leaf(groups, "plan", "Calculate dimensions and edit counts without generating images.")
    plan.add_argument("source_id", nargs="?")
    plan.add_argument("--file")
    plan.add_argument("--grid", nargs=2, type=int, metavar=("COLUMNS", "ROWS"))
    plan.add_argument("--scales", nargs="+", metavar="SCALE")
    plan.add_argument("--input-mode", choices=("original", "previous"))

    jobs = group("job", "Open and manage reconstruction jobs.")
    create = leaf(jobs, "create", "Create or find a job for a source and grid.")
    create.add_argument("source_id")
    create.add_argument("--grid", nargs=2, type=int, metavar=("COLUMNS", "ROWS"))
    leaf(jobs, "list", "List saved jobs.").add_argument("--source", dest="source_id")
    job_id(leaf(jobs, "show", "Show geometry, progress, tiles, and seam warnings."))
    background(job_id(leaf(jobs, "refresh-previews", "Rebuild thumbnails from committed tiles.")))

    prompts = group("prompt", "Read and edit global or per-tile art direction.")
    for action, description in (
        ("get", "Read the saved prompt and its precedence."),
        ("set", "Set the prompt from text, a UTF-8 file, or standard input."),
        ("clear", "Clear a saved prompt."),
    ):
        command = job_id(leaf(prompts, action, description))
        tile(command)
        if action == "set":
            inputs = command.add_mutually_exclusive_group(required=True)
            inputs.add_argument("--text")
            inputs.add_argument("--file")
            inputs.add_argument("--stdin", action="store_true")

    generate = job_id(leaf(groups, "generate", "Generate pending tiles, a 2×2 preview, or selected tiles."))
    selection = generate.add_mutually_exclusive_group()
    selection.add_argument("--preview", action="store_true")
    tile(selection, repeated=True)
    generate.add_argument("--anchor", nargs=2, type=int, metavar=("COLUMN", "ROW"))
    generate.add_argument("--regenerate", action="store_true")
    generate.add_argument("--dry-run", action="store_true")
    codex_options(generate, subagents=True)
    background(generate)

    batches = group("batch", "Generate multiple scales.")
    start = leaf(batches, "start", "Generate and export one PNG per scale.")
    start.add_argument("source_id")
    start.add_argument("--scales", nargs="+", metavar="SCALE")
    start.add_argument("--input-mode", choices=("original", "previous"))
    start.add_argument("--output-dir", required=True)
    batch_prompt = start.add_mutually_exclusive_group()
    batch_prompt.add_argument("--prompt-file")
    batch_prompt.add_argument("--prompt-from-job")
    codex_options(start, subagents=True)
    background(start)

    tiles = group("tile", "Inspect one tile or save its ImageGen references.")
    for action, description in (
        ("show", "Show tile geometry, files, prompts, and diagnostics."),
        ("references", "Save the current prompt and ImageGen input images."),
    ):
        command = job_id(leaf(tiles, action, description))
        command.add_argument("column", type=int)
        command.add_argument("row", type=int)
        if action == "references":
            command.add_argument("--output-dir", required=True)
            command.add_argument("--overwrite", action="store_true")

    inspect = job_id(leaf(groups, "inspect", "Render an image, crop, seam, or before/after comparison."))
    inspect.add_argument("--view", choices=("original", "input", "reconstruction", "compare"), default="compare")
    inspect.add_argument("--output", required=True)
    inspect.add_argument("--overwrite", action="store_true")
    focus = inspect.add_mutually_exclusive_group()
    tile(focus)
    focus.add_argument("--region", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"),
                       help="Limit the inspection to this output-space rectangle.")
    focus.add_argument("--seam", nargs=4, type=int, metavar=("COLUMN1", "ROW1", "COLUMN2", "ROW2"))
    inspect.add_argument("--viewport", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"), default=(1024, 1024))
    inspect.add_argument("--zoom", type=float, metavar="PERCENT",
                         help="Use 0.1 to 800 percent, or omit to fit. Use 100 for actual pixels.")
    inspect.add_argument("--center", nargs=2, type=float, metavar=("X", "Y"))
    inspect.add_argument("--grid", action="store_true")
    inspect.add_argument("--highlight", action="store_true")

    export = job_id(leaf(groups, "export", "Export a full PNG, a tile ZIP, or one tile PNG."))
    export.add_argument("--format", choices=("png", "tiles-zip", "tile-png"), required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--overwrite", action="store_true")
    tile(export, repeated=True)
    background(export)

    operations = group("operation", "Monitor and control background work.")
    leaf(operations, "list", "List background operations.")
    for action, description in (
        ("status", "Show current progress and files."),
        ("wait", "Wait for completion or the timeout."),
        ("pause", "Pause generation after the active group."),
        ("cancel", "Cancel an operation and retain published files."),
        ("resume", "Resume a generation or scale batch."),
    ):
        command = leaf(operations, action, description)
        command.add_argument("operation_id")
        if action == "wait":
            command.add_argument("--timeout", type=float, default=20)
    return parser


def describe_commands(parser, path=()):
    arguments = []
    subcommands = {}
    for arg in parser._actions:
        if isinstance(arg, argparse._SubParsersAction):
            subcommands.update(arg.choices)
        elif arg.help != argparse.SUPPRESS:
            value_type = "boolean" if arg.nargs == 0 else (
                {int: "integer", float: "number"}.get(arg.type, "string")
            )
            arguments.append({
                "name": arg.dest, "flags": arg.option_strings, "required": arg.required,
                "type": value_type, "nargs": arg.nargs, "choices": list(arg.choices) if arg.choices else None,
                "default": arg.default, "help": arg.help,
            })
    result = [{
        "command": " ".join(path), "description": parser.description, "arguments": arguments,
        "subcommands": list(subcommands),
        "mutually_exclusive": [
            {"required": group.required, "arguments": [arg.dest for arg in group._group_actions]}
            for group in parser._mutually_exclusive_groups
        ],
    }]
    for name, child in subcommands.items():
        result.extend(describe_commands(child, (*path, name)))
    return result


def read_prompt(args) -> str:
    if args.file is not None:
        return Path(args.file).read_text(encoding="utf-8-sig").strip()
    if args.stdin:
        return sys.stdin.read().strip()
    return args.text.strip()


def manage_operations(args, operations: Operations) -> dict:
    operations.recover()
    if args.action == "list":
        return {"operations": [operation_json(item) for item in operations.rows()]}
    if args.action in ("pause", "cancel", "resume"):
        return operation_json(operations.control(args.operation_id, args.action))
    item = operations.get(args.operation_id)
    if args.action == "status":
        return operation_json(item)
    if not math.isfinite(args.timeout) or args.timeout < 0:
        raise ValueError("The wait timeout must be a finite, nonnegative number.")
    deadline = time.monotonic() + args.timeout
    while item["state"] in ACTIVE:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.2, remaining))
        operations.recover()
        item = operations.get(args.operation_id)
    return operation_json(item) | {"timed_out": item["state"] in ACTIVE}


def preferences(args, operations: Operations) -> dict:
    if args.action == "reset":
        values = DEFAULT_CONFIG if args.key is None else {args.key: DEFAULT_CONFIG[args.key]}
        operations.set_config(values)
    elif args.action == "set":
        value = args.value
        if args.key == "codex" and value != "auto":
            value = str(executable_for(value))
        elif args.key == "subagents":
            if value not in ("true", "false"):
                raise ValueError("Use true or false for subagents.")
            value = value == "true"
        elif args.key == "scales":
            value = list(parse_scales(value))
            if min(value) < 2:
                raise ValueError("Scales must be at least 2.")
        elif args.key == "input_mode" and value not in ("original", "previous"):
            raise ValueError("Use original or previous for input_mode.")
        operations.set_config({args.key: value})
    return operations.config()


def backend_options(args, config: dict) -> dict:
    codex = args.codex or config["codex"]
    if codex != "auto":
        codex = str(Path(codex).resolve())
    return {"codex": codex, "subagents": config["subagents"] if args.subagents is None else args.subagents}


def dispatch(args, parser, root: Path, operations: Operations) -> dict:
    command = args.command
    if command == "capabilities":
        return {"version": __version__, "commands": describe_commands(parser), "default_data_dir": str(default_root())}
    if command == "operation":
        return manage_operations(args, operations)
    if command == "config":
        return preferences(args, operations)
    config = operations.config()
    if command == "doctor":
        executable = executable_for(args.codex or config["codex"])
        result = {"codex": str(executable), "authenticated": None, "data_dir": str(root)}
        if args.check_login:
            with TemporaryDirectory() as directory, CodexClient(executable, Path(directory)):
                result["authenticated"] = True
        return result
    store = readable_store(root)
    if command == "source":
        if args.action == "list":
            return {"sources": [source_info(source) for source in store.list_sources()] if store.db_path.is_file() else []}
        if args.action == "show":
            return source_info(source_by_id(store, args.source_id))
        path = Path(args.file)
        image = load_source(path)
        with exclusive_workspace(root):
            operations.require_idle()
            return source_info(Store(root).import_source(image, path.name))
    if command == "plan":
        return plan_for(args, store, config)
    if command == "job" and args.action == "list":
        jobs = store.recent_jobs(source_id=args.source_id) if store.db_path.is_file() else []
        return {"jobs": [job_info(job, store.snapshot(job)) for job in jobs]}
    if command == "job" and args.action == "create":
        source = source_by_id(store, args.source_id)
        grid = Grid(*args.grid) if args.grid else suggested_grid(source.width, source.height)
        with exclusive_workspace(root):
            operations.require_idle()
            writable = Store(root)
            job = writable.ensure_job(source, grid)
            return job_info(job, writable.snapshot(job))
    if command == "batch":
        source = source_by_id(store, args.source_id)
        scales = scale_options(source.width, source.height, args.scales, config)
        mode = args.input_mode or config["input_mode"]
        plan_scales(source.width, source.height, " ".join(map(str, scales)), chained=mode == "previous")
        prompt = None
        if args.prompt_file:
            prompt = Path(args.prompt_file).read_text(encoding="utf-8-sig").strip()
        elif args.prompt_from_job:
            origin = job_by_id(store, args.prompt_from_job)
            if origin.source.id != source.id:
                raise ValueError("The prompt job must use the same source.")
            prompt = store.snapshot(origin).prompt
        destination = Path(args.output_dir).resolve()
        request = {
            "kind": "batch", "source_id": source.id, "scales": list(scales), "input_mode": mode,
            "output_dir": str(destination), "prompt": prompt, **backend_options(args, config),
        }
        return operation_json(operations.submit(
            request, args.request_id, validate=lambda: check_destination(store, destination, overwrite=False, directory=True),
        ))

    job = job_by_id(store, args.job_id)
    snapshot = store.snapshot(job)
    if command == "job":
        if args.action == "show":
            return job_info(job, snapshot, tiles=True)
        request = {"kind": "refresh_previews", "job_id": job.id}
        return operation_json(operations.submit(request, args.request_id))
    if command == "prompt":
        cell = cell_at(job, args.tile) if args.tile else None
        if args.action != "get":
            value = read_prompt(args) if args.action == "set" else ""
            with exclusive_workspace(root):
                operations.require_idle()
                writable = Store(root)
                writable.set_prompt(job, value, cell)
                snapshot = writable.snapshot(job)
        return {"job_id": job.id, "global_prompt": snapshot.prompt,
                "tile": cell_json(cell) if cell else None, "tile_prompt": snapshot.tiles[cell].prompt if cell else None,
                "precedence": ["tile", "global"]}
    if command == "generate":
        positions, preview = generation_plan(args, job, snapshot)
        if args.dry_run:
            return preview
        request = {"kind": "generate", "job_id": job.id, "tiles": positions,
                   "regenerate": args.regenerate, **backend_options(args, config)}
        return operation_json(operations.submit(request, args.request_id))
    if command == "tile":
        cell = cell_at(job, (args.column, args.row))
        if args.action == "show":
            return tile_info(job, cell, snapshot.tiles[cell])
        destination = Path(args.output_dir).resolve()
        check_destination(store, destination, overwrite=args.overwrite, directory=True)
        return checked_snapshot(
            job, store, lambda fresh: save_references(job, fresh, cell, destination, overwrite=args.overwrite),
        )
    destination = Path(args.output).resolve()
    extension = ".zip" if command == "export" and args.format == "tiles-zip" else ".png"
    if destination.suffix.lower() != extension:
        raise ValueError(f"The output filename must end in {extension}.")
    if command == "inspect":
        check_destination(store, destination, overwrite=args.overwrite)
        destination.parent.mkdir(parents=True, exist_ok=True)
        return checked_snapshot(job, store, lambda fresh: inspect_image(job, fresh, args, destination))
    request = {
        "kind": "export", "job_id": job.id, "output": str(destination),
        "format": args.format, "tiles": args.tile, "overwrite": args.overwrite,
    }

    def validate_export():
        check_destination(store, destination, overwrite=args.overwrite)
        export_selection(job, store.snapshot(job), args.format, args.tile)

    return operation_json(operations.submit(request, args.request_id, validate=validate_export))


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    for stream in (sys.stdin, sys.stdout):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    application = QCoreApplication.instance() or QCoreApplication([])
    try:
        if len(argv) == 5 and argv[0] == "--data-dir" and argv[2] == "_worker":
            root = Path(argv[1]).resolve()
            Operations(root).worker(argv[3], argv[4], lambda *args: run_operation(root, *args))
            return 0
        parser = command_parser()
        args = parser.parse_args(argv)
        root = Path(args.data_dir or default_root()).resolve()
        output_json(dispatch(args, parser, root, Operations(root)), ok=True)
        return 0
    except CommandError as exc:
        error = exc
    except WorkspaceBusy as exc:
        error = CommandError("WORKSPACE_BUSY", str(exc), status=4)
    except AuthRequired as exc:
        error = CommandError("AUTH_REQUIRED", str(exc), status=1)
    except CodexError as exc:
        error = CommandError("CODEX_UNAVAILABLE", str(exc), status=1)
    except FileExistsError as exc:
        error = CommandError("OUTPUT_EXISTS", str(exc), status=4)
    except ValueError as exc:
        error = CommandError("INVALID_ARGUMENT", str(exc))
    except (OSError, sqlite3.Error) as exc:
        error = CommandError("IO_ERROR", str(exc), status=1)
    except KeyboardInterrupt:
        error = CommandError("INTERRUPTED", "The CLI command was interrupted.", status=130)
    output_json(error.as_dict(), ok=False)
    return error.status


if __name__ == "__main__":
    raise SystemExit(main())
