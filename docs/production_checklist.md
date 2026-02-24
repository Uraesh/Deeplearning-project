# Production Checklist (V1)

1. Verify `metrics.json` from latest run.
2. Confirm sensitivity target is met on validation split.
3. Keep rollback artifact in `models/runs`.
4. Validate API schema on `/docs`.
5. Run tests before deployment.
6. Execute `scripts/clean.ps1` to remove stale caches.
7. Deploy with Docker Compose and health check.
