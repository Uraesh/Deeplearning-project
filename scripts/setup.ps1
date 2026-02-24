param(
    [string]$PythonExe = "python",
    [switch]$ForceInstall
)

if (-not (Test-Path .venv)) {
    & $PythonExe -m venv .venv
}

. .\.venv\Scripts\Activate.ps1

$hashFile = ".venv\.requirements.sha256"
$currentHash = (Get-FileHash -Path requirements.txt -Algorithm SHA256).Hash
$storedHash = ""

if (Test-Path $hashFile) {
    $storedHash = (Get-Content $hashFile -Raw).Trim()
}

if ($ForceInstall -or ($storedHash -ne $currentHash)) {
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    Set-Content -Path $hashFile -Value $currentHash -Encoding ascii
    Write-Host "Dependencies installed/updated."
} else {
    Write-Host "Dependencies unchanged, install skipped (cache hit)."
}

Write-Host "Setup completed."
Write-Host "Use: .\scripts\run_train.ps1"
