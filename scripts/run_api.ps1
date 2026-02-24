param(
    [string]$ModelPath = "models/latest/model.pt",
    [string]$HostAddress = "0.0.0.0",
    [int]$Port = 8000
)

$env:PYTHONPATH = "src"
$env:MODEL_PATH = $ModelPath
python -m breast_cancer_ai.api --host $HostAddress --port $Port
