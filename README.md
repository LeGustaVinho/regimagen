# Regimagen

Regimagen reconstructs images at larger resolutions using Codex ImageGen. It divides the input into overlapping tiles, regenerates each tile with new visual detail, and blends the results into a single sRGB PNG. Use the Windows desktop app for visual review or the JSON CLI for automation.

[Download for Windows x64](https://github.com/LeGustaVinho/regimagen/releases/latest) · [CLI reference](docs/CLI.md) · [Report an issue](https://github.com/LeGustaVinho/regimagen/issues/new)

## Features

- **Configurable reconstruction.** Choose 2–16 columns and 2–6 rows. Regimagen previews the output size and tile count before generation.
- **Reusable 2×2 previews.** Check a four-tile region before running the full grid. The full job reuses completed tiles from the same grid.
- **Art direction and review.** Set a global prompt and per-tile overrides, inspect seams with synchronized zoom, and regenerate individual tiles.
- **Scale batches.** Generate exact multiples of the original size. Each stage can use the original input or the preceding stage's result.
- **Flexible exports.** Export a lossless PNG with source transparency, or a ZIP of completed tiles with placement metadata. The CLI can also export an individual tile.
- **Local checkpoints.** Completed tiles and prompts are saved in the workspace. Pause or resume long jobs and enable Codex subagents when you want parallel generation.

Generated tiles may differ from the source or neighboring tiles. Use the preview and seam review to check the result before committing to a large grid. Regimagen sends tile and reference images to Codex for generation under your account and its usage limits.

## Quick start

1. Download `Regimagen-Windows-x64.zip` from the latest release and extract the whole folder. Keep `_internal` beside the executables. The ZIP bundles the Python runtime and application dependencies.
2. Install Codex CLI and run `codex login` with your ChatGPT account. If automatic detection misses it, select the executable from the desktop app's **Codex CLI** menu. [Codex CLI documentation](https://developers.openai.com/codex/cli/)
3. Open `Regimagen.exe` on Windows 10 or 11 x64. Drop in a still PNG, JPEG, or WebP and review the proposed grid.
4. Run **Generate 2×2 preview**, inspect the result, then choose **Generate image**. Use **Export PNG** for the complete image or **Export tiles (ZIP)** for manual assembly.

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

The test suite covers geometry, compositing, saved-job compatibility, and CLI recovery using simulated ImageGen results. The optional `tools/validate_imagegen.py` script runs a live check and uses your Codex quota. See `--help` for its options.

Build the portable archive with `.\tools\build-portable.ps1`. The build writes `dist\Regimagen-Windows-x64.zip` with the desktop app, CLI, and bundled documentation.

## Third-party components

The Windows package includes Qt for Python, CPython, and other redistributable components. See the bundled notices for license texts and corresponding source links. [Third-party notices](THIRD_PARTY_NOTICES.md)
