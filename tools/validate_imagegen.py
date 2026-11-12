"""Run the application's pipeline with a live ImageGen connection."""

import argparse
import sys
from pathlib import Path
from threading import Event

from regimagen.automation import Operations
from regimagen.codex import find_codex
from regimagen.composite import export_png
from regimagen.files import load_source
from regimagen.geometry import Grid, suggested_grid
from regimagen.jobs import pending_cells, run_tiles
from regimagen.storage import Store
from regimagen.workspace import WorkspaceBusy, exclusive_workspace


_LEGACY_FLAGS = {
    "--dados": "--data-dir",
    "--origem": "--source",
    "--grade": "--grid",
    "--previa": "--preview",
    "--saida": "--output",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Isolated directory for the test job.")
    parser.add_argument("--source", type=Path, help="Input image for a new job.")
    parser.add_argument("--grid", nargs=2, type=int, metavar=("COLUMNS", "ROWS"))
    parser.add_argument("--preview", action="store_true", help="Generate only the four center tiles.")
    parser.add_argument("--output", type=Path, help="PNG path for the completed grid.")
    arguments = []
    for argument in sys.argv[1:]:
        flag, equals, value = argument.partition("=")
        arguments.append(_LEGACY_FLAGS.get(flag, flag) + equals + value)
    args = parser.parse_args(arguments)
    executable = find_codex()
    if executable is None:
        parser.error("Install the Codex CLI and sign in with your ChatGPT account.")

    try:
        with exclusive_workspace(args.data_dir.resolve()):
            Operations(args.data_dir.resolve()).require_idle()
            return validate(args, executable, parser)
    except WorkspaceBusy as exc:
        parser.error(str(exc))


def validate(args, executable, parser) -> int:
    store = Store(args.data_dir.resolve())
    previous = store.recent_jobs()
    if args.source is not None:
        image = load_source(args.source)
        grid = Grid(*args.grid) if args.grid else suggested_grid(*image.size)
        source = store.import_source(image, args.source.name)
        job = store.ensure_job(source, grid)
    elif previous:
        job = previous[0]
    else:
        parser.error("Provide --source to create the first job in this directory.")

    layout = job.layout
    if not layout.enlarges:
        parser.error("The selected grid is too small to enlarge this image.")
    selection = layout.preview() if args.preview else layout.cells

    def report(active):
        if active is not None:
            print(f"Editing column {active.column + 1}, row {active.row + 1}…", flush=True)
        else:
            snapshot = store.snapshot(job)
            print(f"{len(snapshot.artifacts)}/{job.grid.count} tiles ready", flush=True)
            if snapshot.notice:
                print(snapshot.notice, flush=True)

    width, height = layout.size
    print(f"Job {job.grid.columns}×{job.grid.rows} · {width}×{height} px", flush=True)
    run_tiles(
        store, job, pending_cells(store.snapshot(job), selection), executable,
        pause=Event(), cancelled=Event(), on_change=report,
    )
    snapshot = store.snapshot(job)
    if pending_cells(snapshot, selection):
        return 4
    if args.output and len(snapshot.current_artifacts) == job.grid.count:
        source = load_source(job.input_path)
        export_png(layout, snapshot.current_artifacts, args.output, alpha=source if source.mode == "RGBA" else None)
        print(f"Exported PNG to {args.output}", flush=True)
    warnings = sum(bool(tile.warning) for tile in snapshot.tiles.values())
    print(f"{warnings} tiles flagged for review.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
