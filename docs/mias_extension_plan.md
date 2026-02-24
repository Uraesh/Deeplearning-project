# MIAS Extension Plan

The MIAS dataset will be integrated later as a separate image pipeline:

1. Build image preprocessing (resize, normalization, split by patient).
2. Train lightweight PyTorch CNN (CPU-aware).
3. Add independent validation and threshold policy.
4. Add `/predict_image` endpoint.
5. Optionally fuse tabular and image probabilities.
