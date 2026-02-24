param(
    [switch]$Deep,
    [switch]$PipCache
)

$dirsToRemove = @(
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache"
)

Get-ChildItem -Path . -Recurse -Force -Directory -ErrorAction SilentlyContinue |
    Where-Object { $dirsToRemove -contains $_.Name } |
    ForEach-Object {
        Remove-Item -Path $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }

Get-ChildItem -Path . -Recurse -Force -Include *.pyc,*.pyo -File -ErrorAction SilentlyContinue |
    Remove-Item -Force -ErrorAction SilentlyContinue

if (Test-Path "models\runs") {
    Remove-Item "models\runs" -Recurse -Force -ErrorAction SilentlyContinue
}

if ($Deep) {
    if (Test-Path "models\latest") {
        Remove-Item "models\latest" -Recurse -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path "logs") {
        Remove-Item "logs" -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if ($PipCache) {
    python -m pip cache purge | Out-Null
}

[System.GC]::Collect()
Write-Host "Clean completed."
