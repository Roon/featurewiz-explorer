# Precompute Reference runs offline and ship them as a Kaggle dataset

The Lessons are Kaggle notebooks. Kaggle's image follows the Colab runtime (numpy 2, recent scikit-learn), but featurewiz 0.6.1 pins `numpy<2.0`, `scikit-learn<=1.5.2`, `xgboost<=1.7.6` and pulls in `tensorflow`. Installing it in a notebook also needs Internet switched on, which requires a phone-verified account. So we run the real featurewiz offline in a pinned environment, save its outputs, and publish them as a Kaggle dataset that the Lessons attach. Lessons then compare their Re-implementation against those saved outputs with Internet off.

## Considered options

- **`pip install featurewiz` in the notebook**: rejected because it downgrades core packages, needs a restart, and fails for learners without Internet access.
- **Isolated venv plus subprocess in the notebook**: kept only as an optional, clearly flagged cell for learners who want to rerun featurewiz live.

## Consequences

- Saved outputs can go stale if a Synthetic dataset's generator or seed changes. They must be regenerated together.
- The pinned featurewiz version is part of the saved outputs' identity.
- The Reference run uses xgboost 1.7.6, but the Lessons run on Kaggle's current xgboost, whose default tree method is `hist` instead of `exact`. CI therefore runs the Re-implementation in both environments. The pinned run must match the Reference run exactly (with `tree_method="exact"` forced). Differences in the current environment are reported and explained, not counted as failures.
