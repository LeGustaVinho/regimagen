param(
    [ValidatePattern('^[a-zA-Z0-9_-]+$')]
    [string] $DistSubdirectory
)

$ErrorActionPreference = 'Stop'
$regimagenRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Push-Location -LiteralPath $regimagenRoot
try {
    uv sync --locked --group dev
    if ($LASTEXITCODE -ne 0) { throw 'Could not install the build dependencies.' }

    uv run python -c "import platform, struct; assert platform.machine().upper() == 'AMD64' and struct.calcsize('P') == 8"
    if ($LASTEXITCODE -ne 0) { throw 'Use x64 Python to build the portable package.' }

    $regimagenDist = Join-Path $regimagenRoot 'dist'
    if ($DistSubdirectory) {
        $regimagenDist = Join-Path $regimagenDist $DistSubdirectory
    }
    $regimagenBuild = Join-Path $regimagenRoot 'build'
    New-Item -ItemType Directory -Path $regimagenBuild -Force | Out-Null
    uv run python -m regimagen.remove_model
    if ($LASTEXITCODE -ne 0) { throw 'Could not prepare the object selection model.' }

    $regimagenIcon = Join-Path $regimagenBuild 'Regimagen.ico'
    uv run python -c "import sys; from PySide6.QtWidgets import QApplication; from regimagen.app import logo; app=QApplication([]); assert logo(64).save(sys.argv[1], 'ICO')" $regimagenIcon
    if ($LASTEXITCODE -ne 0) { throw 'Could not generate the application icon.' }
    $regimagenBuildArgs = @(
        '--noconfirm', '--distpath', $regimagenDist,
        '--workpath', (Join-Path $regimagenBuild 'pyinstaller'), 'tools/regimagen.spec'
    )
    uv run pyinstaller @regimagenBuildArgs
    if ($LASTEXITCODE -ne 0) { throw 'Could not build the executable.' }

    $regimagenPortable = Join-Path $regimagenDist 'Regimagen'
    # Qt uses the Windows ICU symbols without version suffixes. PyInstaller may
    # collect an incompatible version from the build machine's PATH.
    $regimagenLibraries = Join-Path $regimagenPortable '_internal'
    foreach ($regimagenDll in (Get-ChildItem -LiteralPath $regimagenLibraries -File)) {
        if ($regimagenDll.Name -match '^icu(?:uc|in|dt)?\d*\.dll$') {
            Remove-Item -LiteralPath $regimagenDll.FullName
        }
    }
    Copy-Item -LiteralPath (Join-Path $regimagenRoot 'README.md') -Destination $regimagenPortable -Force
    Copy-Item -LiteralPath (Join-Path $regimagenRoot 'THIRD_PARTY_NOTICES.md') -Destination $regimagenPortable -Force
    Copy-Item -LiteralPath (Join-Path $regimagenRoot 'docs') -Destination $regimagenPortable -Recurse -Force
    Copy-Item -LiteralPath (Join-Path $regimagenRoot 'licenses') -Destination $regimagenPortable -Recurse -Force
    $regimagenLicenses = Join-Path $regimagenPortable 'licenses'
    uv run python -c "import sys; from importlib.metadata import distribution; from pathlib import Path; from shutil import copyfile; dest = Path(sys.argv[1]); bootloader = distribution('pyinstaller'); license_path = next(path for path in bootloader.files if path.name == 'COPYING.txt'); copyfile(Path(sys.base_prefix) / 'LICENSE.txt', dest / 'CPython.txt'); copyfile(bootloader.locate_file(license_path), dest / 'PyInstaller.txt')" $regimagenLicenses
    if ($LASTEXITCODE -ne 0) { throw 'Could not bundle the Python and PyInstaller license texts.' }
    $regimagenArchive = Join-Path $regimagenDist 'Regimagen-Windows-x64.zip'
    Compress-Archive -LiteralPath $regimagenPortable -DestinationPath $regimagenArchive -Force
    Write-Output $regimagenArchive
} finally {
    Pop-Location
}
