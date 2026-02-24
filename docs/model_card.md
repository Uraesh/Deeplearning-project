# Model Card - Wisconsin PyTorch V1

## Model details

- Name: `wisconsin_torch_v1`
- Framework: PyTorch
- Type: Tabular MLP + clinical thresholding
- Runtime: CPU-first

## Intended use

- Triage support for malignant vs benign prediction.
- Positive class: MALIGNANT.
- Not a standalone diagnostic decision tool.

## Data

- Source: Wisconsin Diagnostic Breast Cancer dataset.
- Features: 30 numeric morphology features.
- Excluded: direct identifiers (`id`) and non-feature fields.

## Training and validation

- Stratified train/val/test split.
- Early stopping on validation loss.
- Threshold chosen to satisfy sensitivity target first.

## Key metrics

- Sensitivity / Specificity
- PPV / NPV
- ROC-AUC / PR-AUC
- Brier score

## Limitations

- Single-center academic dataset.
- Limited sample size for hospital-wide claims.
- No image branch (MIAS) yet in production path.
