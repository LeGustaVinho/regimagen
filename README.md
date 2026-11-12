# Regimagen

Regimagen is a desktop app for Windows 10 and 11 x64. It divides an image into overlapping tiles, edits each tile with Codex CLI ImageGen, and assembles an sRGB PNG.

## Use the app

Extract `Regimagen-Windows-x64.zip` and open `Regimagen.exe` in the extracted folder. Keep the `_internal` folder alongside the executable. The package includes Python and the app's libraries. ImageGen uses the Codex CLI installed on your computer and your existing ChatGPT sign-in. Run `codex login` in a terminal to sign in. If necessary, select the executable in the **Codex CLI** menu.

1. Open or drag in a PNG, JPEG, or WebP image. The exported PNG preserves transparency from the source image.
2. Review the suggested grid. Adjust **Columns** and **Rows** to see the output dimensions and number of edits.
3. Use **Global prompt** to guide all tiles in the job. Select a tile and open **Tile prompt** to enter instructions that take priority over the global prompt. Changing a prompt queues the affected tiles for regeneration.
4. Use **Generate 2×2 preview** to check four neighboring tiles, or **Generate image** to start the full job. The full job reuses the preview tiles for the same grid. Turn on **Use Codex subagents** to delegate up to four tiles in parallel. Turn it off to generate one tile at a time.
5. Select a tile in the grid to inspect or regenerate it. Use the percentage button to view both images at 100% with synchronized zoom.
6. Use **Export PNG** to assemble the image once all tiles are ready. Use **Export tiles (ZIP)** to save completed tiles as individual PNGs, even when the job is still incomplete. The PNGs include the overlap margins. The `positions.json` file lists the canvas dimensions, each PNG's position, and each tile's core rectangle. Coordinates are measured in pixels from the top-left corner.

In **Generate scale batch**, enter multipliers such as `2×, 3×, 4×, 5×`. Each multiplier sets the output dimensions relative to the original image. Under **Input image**, choose **Original image** to generate each scale directly from the source or **Previous stage result** to chain the stages. When chaining, enter the scales in ascending order. For example, `2×, 4×` first generates the 2× PNG and uses it as input for the 4× stage.

Each scale produces a separate PNG in the batch folder. The open job's global prompt applies to every scale. Tile prompts are saved with their respective jobs. To resume, open the image in **Saved projects** and run the same scales with the same input option. Regimagen reuses completed tiles when the prompts and input image match.

Regimagen stores the results and queue in its local data folder. Open **Saved projects** to continue a job after restarting the app. Changing the grid selects a different job for the same image. The app displays a message when Codex reaches its usage limit.

## Command-line automation

Run `regimagen-cli.exe` from the same portable package. The CLI runs without opening a desktop window. Run `regimagen-cli.exe capabilities` for the command and argument schema, and add `--help` to any command for terminal help. From source, use `uv run regimagen-cli`.

Each command returns one UTF-8 JSON object on stdout. Successful commands use `{"schema_version":1,"ok":true,"result":{...}}`. Command errors use `{"schema_version":1,"ok":false,"error":{"code":"...","message":"...","details":{...}}}`. Help and version flags print text. Exit codes are `0` for a successful command, `1` for runtime or backend errors, `2` for invalid arguments, `3` for a missing ID, `4` for an operation or workspace conflict, and `130` for an interrupted command. For accepted background work, read the operation's `state` and `error` fields to determine its outcome.

### Generate, inspect, and export an image

```powershell
$cli = '.\regimagen-cli.exe'
& $cli doctor --check-login
$source = ((& $cli source import '.\ship.png') | ConvertFrom-Json).result.source_id
& $cli plan $source --grid 4 4
$job = ((& $cli job create $source --grid 4 4) | ConvertFrom-Json).result.job_id
& $cli prompt set $job --text 'Preserve the rivets and brushed copper.'
& $cli prompt set $job --tile 2 2 --file '.\tile-prompt.txt'
& $cli generate $job --preview --anchor 2 2 --dry-run
$preview = ((& $cli generate $job --preview --anchor 2 2 --request-id ship-preview) | ConvertFrom-Json).result.operation_id
do {
    $wait = ((& $cli operation wait $preview --timeout 30) | ConvertFrom-Json)
    if (-not $wait.ok) { throw $wait.error.message }
} while ($wait.result.timed_out)
if ($wait.result.state -ne 'completed') {
    $wait.result | ConvertTo-Json -Depth 8
    throw "Preview stopped in state '$($wait.result.state)'."
}
& $cli inspect $job --view compare --tile 2 2 --zoom 100 --viewport 600 600 --output '.\compare.png'
$full = ((& $cli generate $job --subagents --request-id ship-full) | ConvertFrom-Json).result.operation_id
& $cli operation status $full
```

Once the full operation reports `completed`, run `export $job --format png --output .\ship-rebuilt.png` and wait for its operation to complete. For manual assembly, use `--format tiles-zip` with a `.zip` path, or `--format tile-png --tile 2 2` with a `.png` path. The ZIP can contain any current tiles, including a partial grid. Both formats use the same per-tile pixels and report the placement and core rectangle in output coordinates. CLI exports and inspection files require `--overwrite` to replace an existing destination.

### Plan and run scale batches

```powershell
& $cli plan $source --scales 2 4 --input-mode previous
$batch = ((& $cli batch start $source --scales 2 4 --input-mode previous --output-dir '.\exports' --prompt-from-job $job) | ConvertFrom-Json).result.operation_id
& $cli operation wait $batch --timeout 30
```

Scales are relative to the original size. `--input-mode original` uses the original for every scale. `--input-mode previous` requires ascending scales and uses each completed stage's PNG for the next stage. A batch can inherit a global prompt from a job for the same source or read one from `--prompt-file`. With no prompt override, each saved stage keeps its global and per-tile prompts. `operation status` reports each verified PNG, its job ID, and the input used for that stage. On resume, Regimagen checks saved exports in stage order and rebuilds missing or changed PNGs from the committed tiles. It also verifies shared scale caches by checksum before reusing them in another batch. An outdated downstream PNG can remain on disk until its stage is reached.

When opening a saved chained stage, Regimagen verifies its input against the cached checksum. A scale batch can rebuild missing or changed caches for its current stages. To recover a historical stage that predates checksums, check the PNG at the `input_path` from `job show`, then run `job adopt-input JOB_ID` to record its checksum and reopen that exact revision. The command validates the PNG and its dimensions. An existing checksum remains authoritative.

### Desktop workflow coverage

| Desktop capability | CLI |
| --- | --- |
| Open or drop PNG, JPEG, and WebP, including alpha | `source import FILE` |
| Browse original images and saved projects | `source list/show`, `job list [--source ID]`, `job show ID`, `job adopt-input ID` for a verified historical PNG |
| Suggested grid, independent column and row counts, output size, and edit count | `plan ID` or `plan --file FILE`, `--grid COLUMNS ROWS`, `job create` |
| Global and selected-tile art direction | `prompt get/set/clear` with optional `--tile COLUMN ROW`. Set from `--text`, a UTF-8 `--file`, or `--stdin`. |
| Centered or selected 2×2 preview, full generation, and selected-tile repair | `generate --preview [--anchor COLUMN ROW]`, `generate`, `generate --tile COLUMN ROW [--tile ...] --regenerate`. Add `--dry-run` to count pending and reusable results. |
| Codex location, automatic detection, and optional subagent delegation | `config show/set/reset`, `doctor --check-login`, `--codex PATH_OR_AUTO`, `--subagents` or `--no-subagents` on each run |
| Tile selection, progress, seam diagnostics, and repair references | `job show`, `tile show`, `tile references --output-dir DIR`, `job refresh-previews` to rebuild thumbnails from committed tiles |
| Before/after views, synchronized fit and actual-size zoom, pan, and grid overlay | `inspect --view original\|input\|reconstruction\|compare`, `--viewport WIDTH HEIGHT`, `--zoom 100`, `--center X Y`, `--grid`, `--highlight` |
| Inspect a tile, crop, seam, or partial reconstruction | `inspect --tile COLUMN ROW`, `--region X Y WIDTH HEIGHT`, or `--seam COLUMN1 ROW1 COLUMN2 ROW2`. Results report missing and stale tiles. |
| Full PNG, tile ZIP, and individual tile PNG | `export --format png\|tiles-zip\|tile-png` |
| Saved scale preferences and both batch input modes | `config set scales "2x, 4x"`, `config set input_mode previous`, `plan --scales`, `batch start` |
| Pause, cancel, monitor, and resume | `operation list/status/wait/pause/cancel/resume` |

Tile columns and rows start at 1. Image coordinates start at 0 in the output canvas. Inspection uses a shared coordinate system for both panels. `--viewport` sets the maximum size per panel. Omit `--zoom` to fit the selection, or use a zoom from 0.1 to 800 percent. `--zoom 100` displays output pixels at actual size. With numeric zoom, `--center` pans to a point in the output canvas. `--region` bounds the crop at any zoom and keeps panning inside that rectangle.

### Workspaces and operation state

The CLI defaults to `%LOCALAPPDATA%\Regimagen\cli`. Put `--data-dir DIR` before the command to select a workspace. To share saved sources, prompts, and jobs with the desktop, point it at the desktop's `%LOCALAPPDATA%\Regimagen` folder. Close the desktop before starting work in the shared folder. Read commands can inspect saved state while the desktop or worker has the workspace open. CLI preferences are saved per workspace and can be overridden on each run.

Generation, scale batches, thumbnail refreshes, and exports return an `operation_id` and run in a detached local process. Use `operation wait ID --timeout SECONDS` for a bounded wait, or `--timeout 0` for an immediate status snapshot. `timed_out: true` means the latest observed state was active when the wait expired. `OPERATION_STATE_BUSY` means the database was locked throughout the wait and no snapshot was available. `operation pause` lets the active tile or delegated group finish. `operation cancel` requests cancellation and keeps committed tiles and published files. Resume a paused, canceled, failed, quota-limited, or interrupted generation or scale batch using `operation resume ID`. If Codex reports a quota reset, `quota_reset` is a Unix timestamp. A new generation operation can also continue a saved job by selecting its pending tiles.

Use `--request-id` on a background command to retrieve the existing operation after a client retry. Reusing a key with different parameters returns `REQUEST_ID_CONFLICT`. The worker writes state to the selected workspace, and stopping the terminal or a wait command leaves it running. Workspace mutations and background work are serialized. Run operations in different workspaces if you need independent project state.

If a worker exits during an export, use `operation resume ID` to recover the same operation. Regimagen reports completion if it recognizes the published file. Otherwise it retries with the original destination and overwrite preference. Single-tile generation checkpoints a delivered ImageGen result while the turn is active, so a cancellation after delivery can reuse the image and its generation ID.

## Run from source

```powershell
uv sync --locked --group dev
uv run python -m regimagen
```

Run `uv run pytest -q` to check desktop and CLI project compatibility, command workflows, background resumption, compositing, and simulated export of 96 tiles. To exercise the packaged CLI with a restricted `PATH`, set `REGIMAGEN_CLI_TEST_EXE` to the absolute path of `regimagen-cli.exe` and run `uv run pytest -q tests/test_cli.py -k detached`.

Use a separate test directory to run a live ImageGen test. The first command edits the four center tiles with `--preview`. The second reuses the saved results and completes the grid.

```powershell
uv run python tools/validate_imagegen.py --data-dir .\test-run --source .\image.png --grid 16 6 --preview
uv run python tools/validate_imagegen.py --data-dir .\test-run --output .\result.png
```

## Build the portable package

Run `./tools/build-portable.ps1` on Windows x64. The script writes `dist/Regimagen-Windows-x64.zip` with `Regimagen.exe`, `regimagen-cli.exe`, and their shared `_internal` runtime. Keep both executables next to that folder when extracting the package.
