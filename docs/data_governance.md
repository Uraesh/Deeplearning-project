# Data Governance Policy (V1)

## 1) Identity protection

- No direct patient identifiers in model inputs.
- Remove or pseudonymize IDs before ingestion.
- API requests must only contain model features.

## 2) Access policy

- Limit raw data access to authorized team members.
- Use role-based permissions for data, model, and reports.
- Keep artifact folders read-only for reviewers.

## 3) Auditability

- Store model version, threshold, and metrics per run.
- Log request IDs only, avoid full patient vectors in logs.
- Keep training config and report for reproducibility.

## 4) Data lifecycle

- Raw data: protected and not shared publicly.
- Processed data: reproducible transformations only.
- Model artifacts: versioned in `models/runs` and `models/latest`.

## 5) Clinical safety gate

- Sensitivity target must be explicitly validated.
- Threshold selection method documented.
- Final release requires internal review by at least 2 members.
