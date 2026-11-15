# Regimagen

[![Latest GitHub release](https://img.shields.io/github/v/release/LeGustaVinho/regimagen?display_name=tag&sort=semver&label=release&color=2f6feb)](https://github.com/LeGustaVinho/regimagen/releases/latest)
[![Windows 10 and 11 x64](https://img.shields.io/badge/Windows-10%20%2F%2011%20x64-0078D4)](#quick-start)
[![CLI JSON schema v1](https://img.shields.io/badge/CLI-JSON%20v1-2EA043)](docs/CLI.md)
[![Python 3.14](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](#develop-from-source)

**Layered image editing and generative reconstruction with Codex ImageGen.** Compose and retouch images in the Windows app, use Generative Fill on a selection, or reconstruct the visible composition at a larger size. The JSON CLI automates reconstruction and tile export.

[Download Windows ZIP](https://github.com/LeGustaVinho/regimagen/releases/latest/download/Regimagen-Windows-x64.zip) · [CLI reference](docs/CLI.md) · [Report an issue](https://github.com/LeGustaVinho/regimagen/issues/new)

![Regimagen desktop with the original image, a four-tile reconstruction preview, and the job controls](docs/assets/desktop.png)

*Reconstruction mode showing a 2×2 preview within a 4×4 job, using simulated tiles.*

## Features

- **Canvas and layers.** Move, reorder, duplicate, hide, and lock layers. Use rectangular and lasso selections to crop the canvas or work on part of a layer. Save editable layers, masks, and pixels together in a portable `.regimagen` project.
- **Manual retouching.** Paint pixels or masks with the brush and eraser. Rotate, flip, and resize layers, or preview brightness, contrast, and saturation adjustments. Undo and redo edits in the current session. Export the composition as PNG or WebP with transparency, or JPEG with a white background.
- **Repeating patterns.** Use **View → Pattern Preview** to see the canvas repeat as you edit the center tile. **Filter → Other → Offset** shifts the active layer with wraparound. **Filter → Fix Pattern Seams** uses Codex to propose a seam mask and Generative Fill to repair it. [Pattern workflow](#repeating-patterns)
- **Generative Fill.** Choose **New layer** to use the visible composition, or **Active layer** to edit a visible, unlocked raster layer. Select an area, describe the edit, and compare variations before applying. Use **Subtract** to trim the selected variation locally. A prompt for a transparent background creates a cutout on **New layer**. You can also enable **Edit Transparency** to allow coverage changes. To reduce the selection of a previously saved image version, open it from **Recent** and choose **Image version → Reopen fill**.
- **Configurable reconstruction.** Choose at least 2 columns and 2 rows. Regimagen previews the output size and tile count before generation and suggests the smallest grid that can fit at least 2× the source dimensions.
- **Reusable 2×2 previews.** Check a four-tile region before running the full grid. The full job reuses completed tiles from the same grid.
- **Art direction and review.** Set a global prompt and per-tile overrides, inspect seams with synchronized zoom, and regenerate individual tiles.
- **Scale batches.** Generate exact multiples of the original size. Each stage can use the original input or the preceding stage's result.
- **Flexible exports.** Export a lossless PNG with source transparency, or a ZIP of completed tiles with placement metadata. Fully transparent tiles are completed locally and included in both exports. The CLI can also export an individual tile.
- **Local checkpoints.** Completed tiles and prompts are saved in the workspace. Pause or resume long jobs and enable Codex subagents when you want parallel generation. **Resume batch** continues an interrupted scale batch in the same folder. If a destination PNG conflicts with the batch, move or rename that file before resuming, or start a new batch in another folder.

Use the preview and seam review to check generated tiles before committing to a large grid. Regimagen sends image references to Codex under your account when it requests generation or visual analysis.

## Quick start

1. [Download the portable Windows ZIP](https://github.com/LeGustaVinho/regimagen/releases/latest/download/Regimagen-Windows-x64.zip) and extract the whole folder. Keep `_internal` beside the executables. The ZIP bundles the Python runtime and application dependencies. [Build from source](#develop-from-source) to try changes on a development branch.
2. Open `Regimagen.exe` on Windows 10 or 11 x64. Drop in a still PNG, JPEG, WebP, or `.regimagen` project. To start from transparency, choose **New canvas**. Use **Cancel** in the task bar to stop opening a file and keep the current document.
3. Use **Add layer** to import another image or create a paintable layer. Choose **File → Save project** to keep your layers, or **File → Export image** to publish the composition.
4. Install Codex CLI and run `codex login` for **Generative Fill** and **Reconstruct**. If automatic detection misses it, choose the executable from the **Codex CLI** menu. [Codex CLI documentation](https://developers.openai.com/codex/cli/)
5. In **Reconstruct**, run **Generate 2×2 preview**, inspect the result, then choose **Generate image**. If a run is interrupted, use the retry action in the task bar to resume the same region. Use **Open result in editor** to work on the completed image at its generated size. You can also export the PNG or a ZIP of tiles.

## Repeating patterns

1. Open an image in the layered editor and turn on **View → Pattern Preview**. Zoom and pan to inspect the joins. Changes to the center tile update the surrounding copies.
2. To inspect the wrap of one layer, choose **Filter → Other → Offset** and preview the horizontal and vertical shifts. **Apply** commits the shift to the active layer.
3. Choose **Filter → Fix Pattern Seams…** to repair the visible composition. Regimagen opens an offset view so you can see the joins in the middle of the tile. Choose **Detect Seams** for a proposed mask, or draw one with **Lasso** or **Rectangle**. Use **Add** and **Subtract** to refine the mask, then choose **Generate**.
4. Turn on **Pattern Preview** in the repair view and compare **Before** and **After** across the tile boundaries. Amber marks joins that may need another pass. Use **Generate Another** to compare a different repair.
5. Click **Apply** to add a **Pattern Seam Repair** layer at the original tile position. **File → Export image** saves one tile at the canvas dimensions. Use PNG or WebP to retain transparency.

## CLI and AI automation

The portable archive also includes `regimagen-cli.exe`. It runs without a desktop window and returns UTF-8 JSON. Use the runtime command schema for discovery and the reference for workflows, state transitions, and retry behavior. [CLI reference](docs/CLI.md)

```powershell
.\regimagen-cli.exe capabilities
.\regimagen-cli.exe doctor --check-login
.\regimagen-cli.exe plan --file .\image.png --grid 4 4
```

## Develop from source

On Windows x64, use [uv](https://docs.astral.sh/uv/) with Python 3.14.

```powershell
uv sync --locked --group dev
uv run python -m regimagen
uv run regimagen-cli capabilities
uv run pytest -q
```

The test suite covers the editor, generative selection and compositing, portable projects, saved-job compatibility, and CLI recovery using simulated ImageGen results. The optional `tools/validate_imagegen.py` script runs a live check and uses your Codex quota. See `--help` for its options.

Build the portable archive with `.\tools\build-portable.ps1`. The build prepares and verifies the pinned EfficientSAM model for Generative Fill cutouts, then writes `dist\Regimagen-Windows-x64.zip` with the desktop app, CLI, model, and documentation. Pass `-DistSubdirectory NAME` to write the archive under `dist\NAME`. To prepare the model ahead of a source run, use `uv run python -m regimagen.segmentation_model`.

## Third-party components

The Windows package includes Qt for Python, CPython, and other redistributable components. See the bundled notices for license texts and corresponding source links. [Third-party notices](THIRD_PARTY_NOTICES.md)
