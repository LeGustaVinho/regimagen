# Regimagen CLI reference

Use this reference to drive Regimagen from a script or an AI agent. The CLI imports local images, plans a reconstruction, runs Codex ImageGen, and exports local PNG or ZIP files. Each workspace stores its own sources, jobs, preferences, and operation history.

## Start here

- In the Windows archive, invoke `regimagen-cli.exe` from the extracted folder. Keep `_internal` next to it. From the repository, use `uv run regimagen-cli` after `uv sync --locked --group dev`.
- Run `capabilities` to discover the command tree, flags, argument types, defaults, and mutually exclusive groups for the installed version. Use `COMMAND --help` for human-readable help.
- Run `doctor --check-login` before generation. It checks the locally installed Codex CLI and ChatGPT sign-in. Import, planning, inspection, and export of existing results run without ImageGen.
- Put the global `--data-dir DIR` flag **before** the command. Use the same directory throughout a workflow. The default is `%LOCALAPPDATA%\Regimagen\cli`. To share desktop jobs, point the CLI at `%LOCALAPPDATA%\Regimagen` and close the desktop before starting a mutation or background task.
- Pass paths and prompts as individual arguments. Prefer absolute paths when running from changing working directories. Use a UTF-8 prompt file or stdin for multiline prompts.

## Protocol

Every normal command writes one UTF-8 JSON object to stdout. The exceptions are `--help` and `--version`, which write text.

```json
{"schema_version":1,"ok":true,"result":{"operation_id":"example","state":"starting"}}
```

```json
{"schema_version":1,"ok":false,"error":{"code":"WORKSPACE_BUSY","message":"Another Regimagen process is using this workspace.","details":{}}}
```

Read `schema_version` and `ok` before reading the payload. Use `error.code` for branching and `error.message` for display. The success example is an excerpt of an operation result.

| Exit code | Meaning |
| --- | --- |
| `0` | The command succeeded or accepted background work. |
| `1` | A runtime, I/O, or backend error occurred. |
| `2` | The arguments or request were invalid. |
| `3` | The requested ID was unavailable. |
| `4` | The operation, workspace, input, or output is in conflict. |
| `130` | The CLI command was interrupted. |

A `source_id` identifies an imported image. A `job_id` identifies its reconstruction grid or scale stage. An `operation_id` identifies one background run, including its retries. Use the IDs returned by the CLI. Tile coordinates are **one-based** in column, row order. Pixel regions are **zero-based** in the output canvas, measured from the top left.

## Recommended agent workflow

1. Import the input once and retain its `source_id`.
2. Run `plan` and check the output dimensions, grid, and edit count. Use `generate --dry-run` on an existing job to see the pending and reusable tiles.
3. Set art direction before submitting generation. A prompt change marks affected tiles stale and can require new edits.
4. Submit a 2×2 preview when the user wants a visual check. Wait for it and save a comparison with `inspect`. For a full image, submit generation on the same job to reuse the preview tiles.
5. Wait for `completed` before submitting a dependent operation in the same workspace. For a composed PNG, check `can_export_png` in `job show`. For partial work, export the current tiles in a ZIP.
6. Return the published path from `progress.outputs` after export completes. Retain the operation ID when work pauses, encounters a quota limit, or needs a retry.

### PowerShell example

Run from the extracted folder with a still image at `.\image.png`. This example generates a full 2×2 job and consumes ImageGen quota.

```powershell
$cli = (Resolve-Path -LiteralPath '.\regimagen-cli.exe').Path
$workspace = Join-Path $env:LOCALAPPDATA 'Regimagen\automation-example'

function Invoke-Regimagen {
    $reply = (& $cli --data-dir $workspace @args) | ConvertFrom-Json
    if (-not $reply.ok) {
        throw "Regimagen $($reply.error.code) $($reply.error.message)"
    }
    $reply.result
}

function Wait-Regimagen([string] $OperationId) {
    do {
        $state = Invoke-Regimagen operation wait $OperationId --timeout 30
    } while ($state.timed_out)
    if ($state.state -ne 'completed') {
        throw "Operation $OperationId stopped in state $($state.state). Inspect it with operation status."
    }
    $state
}

$null = Invoke-Regimagen doctor --check-login
$source = (Invoke-Regimagen source import '.\image.png').source_id
$plan = Invoke-Regimagen plan $source --grid 2 2
if (-not $plan.layout.enlarges) { throw 'Choose a grid that enlarges this image.' }
$job = (Invoke-Regimagen job create $source --grid 2 2).job_id
Invoke-Regimagen generate $job --dry-run
$run = Invoke-Regimagen generate $job --request-id "${job}-generate"
$null = Wait-Regimagen $run.operation_id
$export = Invoke-Regimagen export $job --format png --output '.\rebuilt.png' --request-id "${job}-export"
(Wait-Regimagen $export.operation_id).progress.outputs
```

## Commands

### Setup, import, and planning

| Command | Behavior |
| --- | --- |
| `capabilities` | List the installed commands and argument schema as JSON. |
| `doctor [--codex PATH_OR_AUTO] [--check-login]` | Locate Codex and optionally check authentication. |
| `config show`, `config set KEY VALUE`, `config reset [KEY]` | Read, set, or reset workspace defaults. |
| `source import FILE` | Import a still PNG, JPEG, or WebP and return its source ID. |
| `source list`, `source show SOURCE_ID` | Discover sources and their dimensions, paths, and suggested grids. |
| `plan SOURCE_ID` or `plan --file FILE` | Calculate geometry. Add `--grid COLUMNS ROWS` for a grid or `--scales SCALE ... [--input-mode original\|previous]` for a batch. |
| `job create SOURCE_ID [--grid COLUMNS ROWS]` | Create or find the job for the selected grid. Omit the grid to use the suggestion. |
| `job list [--source SOURCE_ID]`, `job show JOB_ID` | Find saved jobs or inspect their tile counts, input revisions, seams, and export readiness. |
| `job refresh-previews JOB_ID [--request-id KEY]` | Rebuild thumbnails in the background. |
| `job adopt-input JOB_ID` | Record a checksum for the checked PNG of a historical chained job. Verify the `input_path` from `job show` first. The command validates the file and its expected dimensions and preserves an existing checksum. |

Choose at least 2 columns and 2 rows. Both axes can grow as long as every tile covers at least one output pixel and the result fits the PNG format. `plan` suggests the smallest grid that can fit at least 2× the source dimensions. Check that `layout.enlarges` is true and `layout.grid_error` is null before submitting generation. The `--file` form can plan without importing.

Workspace keys are `codex` (an executable path or `auto`), `subagents` (`true` or `false`), `scales` (a list such as `"2x, 4x"`), and `input_mode` (`original` or `previous`). Generation and batch commands can override the Codex path and subagent setting for the run.

### Prompts and generation

- `prompt get JOB_ID [--tile COLUMN ROW]`, `prompt clear JOB_ID [--tile COLUMN ROW]`, and `prompt set JOB_ID [--tile COLUMN ROW] (--text TEXT | --file FILE | --stdin)`. Omit `--tile` for the global prompt. Tile prompts take precedence.
- `generate JOB_ID` submits all pending tiles. Use `--preview [--anchor COLUMN ROW]` for a 2×2 region or repeat `--tile COLUMN ROW` to select specific tiles. Combine `--regenerate` with an explicit tile selection to replace current results.
- Generation accepts `--dry-run`, `--codex PATH_OR_AUTO`, `--subagents` or `--no-subagents`, and `--request-id KEY`. A dry run returns `selected`, `pending`, `reused`, `transparent`, and `max_edits` without starting a worker. The `transparent` list contains pending tiles with zero source alpha across their contribution to the output, including overlap. Regimagen completes those tiles locally, includes them in exports, and excludes them from `max_edits`.
- `batch start SOURCE_ID --output-dir DIR [--scales SCALE ...] [--input-mode original|previous]` runs and exports scale stages. It accepts the same Codex, subagent, and request-ID flags. Choose at most one of `--prompt-file FILE` and `--prompt-from-job JOB_ID`. The prompt job must use the same source.

Scales are integer multiples of the original image dimensions, starting at 2×. The reported `max_scale` is the PNG format limit for those dimensions. For `2 4`, `previous` mode uses the completed 2× PNG as the input to the 4× stage and requires ascending scales. `original` uses the original image at every stage. If a batch has no prompt override, saved stage prompts are reused. The operation reports verified output paths and the input revision for each stage. If the current chain's cache needs repair, rerun the same batch. To recover an intact older PNG for a specific historical stage, use `job adopt-input` after checking that file.

### Inspect and export

- `tile show JOB_ID COLUMN ROW` returns status, geometry, prompts, generation ID, and artifact paths for a cell.
- `tile references JOB_ID COLUMN ROW --output-dir DIR [--overwrite]` writes the prompt and ImageGen reference images.
- `inspect JOB_ID --output FILE.png` saves a PNG of the comparison. Choose `--view original|input|reconstruction|compare`. `input` follows the stage's input revision. Use one of `--tile COLUMN ROW`, `--region X Y WIDTH HEIGHT`, or `--seam COLUMN1 ROW1 COLUMN2 ROW2` to focus the view.
- Inspection accepts `--viewport WIDTH HEIGHT` (default `1024 1024` per panel), `--zoom PERCENT` (0.1–800, with 100 for output pixels), `--center X Y` with numeric zoom, `--grid`, `--highlight`, and `--overwrite`. Omit zoom to fit. The response includes the rendered region, panel size, and lists of missing and stale tiles.
- `export JOB_ID --format png|tiles-zip|tile-png --output FILE [--overwrite] [--request-id KEY]` submits an export. Full PNG requires every tile to be current. ZIP supports a partial grid and optional repeated `--tile COLUMN ROW` filters. A tile PNG requires exactly one `--tile COLUMN ROW`. Use the matching `.png` or `.zip` extension.

The tile ZIP includes `positions.json`, with canvas dimensions and each tile's placement and core rectangle. Tile files include overlap margins. Inspection and export protect existing destinations unless the request includes `--overwrite`.

## Background operations and retries

`generate`, `batch start`, `job refresh-previews`, and `export` submit detached work. The submitting CLI process can exit while the worker runs. A workspace allows one background operation at a time. Inspect or list saved state while a worker is active. Use a separate workspace for independent concurrent projects.

| Command | Use |
| --- | --- |
| `operation list`, `operation status OPERATION_ID` | Discover operations or read the latest state, progress, and output paths. |
| `operation wait OPERATION_ID [--timeout SECONDS]` | Wait for a terminal state or return the most recent state at the deadline. The default is 20 seconds. Use 0 for an immediate snapshot. |
| `operation pause OPERATION_ID` | Request a pause after the active tile or delegated group of a generation or batch. |
| `operation cancel OPERATION_ID` | Request cancellation while retaining committed tiles and published files. |
| `operation resume OPERATION_ID` | Continue a stopped generation or batch, or reconcile or retry an interrupted export. |

`starting` and `running` are active. `completed` is successful. Other terminal states are `paused`, `canceled`, `quota_limited`, `interrupted`, and `failed`. Read `error` for a failure reason and `quota_reset` for the reset time if Codex provides it, as a Unix timestamp. A wait with `timed_out: true` can be followed by another wait. `OPERATION_STATE_BUSY` means the database was busy and the wait obtained no snapshot.

Use a stable, unique `--request-id` for each intended submission. On an ambiguous client-side failure, repeat the same command with the same key. It returns the existing operation, while a key already associated with different parameters returns `REQUEST_ID_CONFLICT`. Use the operation ID to resume stopped work. A repeated export can recognize an unchanged file already published before a worker interruption. For `WORKSPACE_BUSY`, inspect the active operation or close the desktop if it owns the workspace.
