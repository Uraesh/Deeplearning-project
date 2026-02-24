# Training Report

- Model version: `20260224T004529Z`
- Created at UTC: `2026-02-24T00:45:29.807637+00:00`
- Threshold selected (OOF): `0.508603`
- Threshold reason: `target_sensitivity_met`
- CV folds: `5`
- Final training epochs: `155`
- Training signature: `fb06a5b88336acdb83e9b85691a59d4838717536b1c7164cadb8b7cc0d610512`

## CV OOF
- Sensitivity: `0.9706`
- Specificity: `0.9860`
- ROC-AUC: `0.9931`
- PR-AUC: `0.9896`

## Train+Val (Final model)
- Sensitivity: `1.0000`
- Specificity: `0.9965`
- ROC-AUC: `0.9982`
- PR-AUC: `0.9940`

## Test
- Sensitivity: `0.9286`
- Specificity: `1.0000`
- ROC-AUC: `0.9954`
- PR-AUC: `0.9936`

## Overfitting Gap (trainval - test)
- Sensitivity gap: `0.0714`
- Specificity gap: `-0.0035`
- ROC-AUC gap: `0.0028`
