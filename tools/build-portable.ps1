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
    $regimagenArchive = Join-Path $regimagenDist 'Regimagen-Windows-x64.zip'
    Compress-Archive -LiteralPath $regimagenPortable -DestinationPath $regimagenArchive -Force
    Write-Output $regimagenArchive
} finally {
    Pop-Location
}
