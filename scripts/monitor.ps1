param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [string]$ProjectPath = ".",
    [int]$RoundDigits = 3
)

Set-Location $ProjectPath

function Convert-ToFeatureMap {
    param([Parameter(Mandatory = $true)]$CsvRow)

    $features = @{}
    foreach ($prop in $CsvRow.PSObject.Properties) {
        if ([string]::IsNullOrWhiteSpace($prop.Name)) { continue }
        if ($prop.Name -in @("id", "diagnosis")) { continue }
        if ($prop.Name -like "Unnamed*") { continue }
        if ([string]::IsNullOrWhiteSpace([string]$prop.Value)) { continue }
        $key = $prop.Name.Trim().Replace(" ", "_")
        $features[$key] = [double]$prop.Value
    }
    return $features
}

function Get-ApiProcess {
    $process = Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -match "^python(\\.exe)?$" -and
            $_.CommandLine -match "breast_cancer_ai\\.api|uvicorn"
        } |
        Select-Object -First 1

    if ($null -eq $process) {
        return $null
    }
    return Get-Process -Id $process.ProcessId -ErrorAction SilentlyContinue
}

Write-Host "`n=== API Health ==="
try {
    $health = Invoke-RestMethod "$BaseUrl/health" -Method Get -ErrorAction Stop
    $health | ConvertTo-Json
}
catch {
    Write-Host "Health check failed: $($_.Exception.Message)"
}

Write-Host "`n=== Dashboard Metrics (/performance) ==="
try {
    $performance = Invoke-RestMethod "$BaseUrl/performance" -Method Get -ErrorAction Stop
    $report = $performance.report
    [PSCustomObject]@{
        model_version = $report.model_version
        roc_auc_test = $report.test_metrics.roc_auc
        sensitivity_test = $report.test_metrics.sensitivity
        specificity_test = $report.test_metrics.specificity
        threshold = $report.threshold
    } | Format-List
}
catch {
    Write-Host "Performance endpoint failed: $($_.Exception.Message)"
}

Write-Host "`n=== API Predict Smoke Test ==="
try {
    $row = Import-Csv "data.csv" | Select-Object -First 1
    $features = Convert-ToFeatureMap -CsvRow $row
    $payload = @{ features = $features } | ConvertTo-Json -Depth 6
    $prediction = Invoke-RestMethod "$BaseUrl/predict" -Method Post -ContentType "application/json" -Body $payload
    $prediction | ConvertTo-Json -Depth 6
}
catch {
    Write-Host "Predict test failed: $($_.Exception.Message)"
}

Write-Host "`n=== API Process Memory ==="
$apiProcess = Get-ApiProcess
if ($null -eq $apiProcess) {
    Write-Host "No local python API process detected (possible Docker execution)."
}
else {
    $apiProcess |
        Select-Object Id, ProcessName,
        @{N = "WorkingSet_MB"; E = { [math]::Round($_.WorkingSet64 / 1MB, $RoundDigits) }},
        @{N = "Private_MB"; E = { [math]::Round($_.PrivateMemorySize64 / 1MB, $RoundDigits) }},
        CPU |
        Format-Table -AutoSize
}

Write-Host "`n=== Filesystem Space ==="
Get-PSDrive -PSProvider FileSystem |
    Select-Object Name,
    @{N = "UsedGB"; E = { [math]::Round($_.Used / 1GB, 2) }},
    @{N = "FreeGB"; E = { [math]::Round($_.Free / 1GB, 2) }} |
    Format-Table -AutoSize

Write-Host "`n=== Project Size ==="
$projectBytes = (Get-ChildItem . -Recurse -File | Measure-Object Length -Sum).Sum
"{0:N2} MB ({1:N0} bytes)" -f ($projectBytes / 1MB), $projectBytes

Write-Host "`n=== Models Size ==="
if (Test-Path "models") {
    $modelsBytes = (Get-ChildItem models -Recurse -File | Measure-Object Length -Sum).Sum
    "Total models: {0:N3} MB ({1:N0} bytes)" -f ($modelsBytes / 1MB), $modelsBytes
    Get-ChildItem models -Recurse -File |
        Select-Object FullName,
        @{N = "SizeKB"; E = { [math]::Round($_.Length / 1KB, $RoundDigits) }},
        @{N = "Bytes"; E = { $_.Length }} |
        Format-Table -AutoSize
}
else {
    Write-Host "models/ folder not found."
}
