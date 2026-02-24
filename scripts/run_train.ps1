param(
    [string]$ConfigPath = "configs/train_config.yaml",
    [switch]$ForceRetrain,
    [switch]$ShowEpochs
)

$env:PYTHONPATH = "src"

$pythonArgs = @("-m", "breast_cancer_ai.train", "--config", $ConfigPath)
if ($ForceRetrain) {
    $pythonArgs += "--force-retrain"
}
if ($ShowEpochs) {
    $pythonArgs += "--show-epochs"
}

python @pythonArgs
