# Training Report

- Model version: `20260224T135220Z`
- Created at UTC: `2026-02-24T13:52:20.055837+00:00`
- Threshold selected (OOF): `0.290783`
- Threshold reason: `target_sensitivity_met`
- CV folds: `5`
- Final training epochs: `185`
- Training signature: `552e292affd32e21cb11ea8f72ff1f14eb7549f5b9134d6bc7839cc20514e4ee`

## CV OOF
- Sensitivity: `0.9706`
- Specificity: `0.7692`
- ROC-AUC: `0.9801`
- PR-AUC: `0.9779`

## Train+Val (Final model)
- Sensitivity: `1.0000`
- Specificity: `1.0000`
- ROC-AUC: `1.0000`
- PR-AUC: `1.0000`

## Test
- Sensitivity: `0.9286`
- Specificity: `0.9155`
- ROC-AUC: `0.9913`
- PR-AUC: `0.9873`

## Overfitting Gap (trainval - test)
- Sensitivity gap: `0.0714`
- Specificity gap: `0.0845`
- ROC-AUC gap: `0.0087`
