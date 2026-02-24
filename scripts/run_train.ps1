param(
    [string]$ConfigPath = "configs/train_config.yaml",
    [switch]$ForceRetrain
)

$env:PYTHONPATH = "src"

$pythonArgs = @("-m", "breast_cancer_ai.train", "--config", $ConfigPath)
if ($ForceRetrain) {
    $pythonArgs += "--force-retrain"
}

python @pythonArgs
