# Re-implement featurewiz's stages instead of calling the library

The project teaches featurewiz by showing every intermediate step of SULOV and Recursive XGBoost: correlated pairs, mutual-information scores, each XGBoost round. featurewiz 0.6.1 doesn't expose these. `FE_remove_variables_using_SULOV_method` and `FE_perform_recursive_xgboost` return only the final kept-feature list and print or plot the rest. So we write our own small, readable Re-implementation of each stage and check it against a Reference run of the real library on the same data.

## Considered options

- **Call featurewiz and visualise only inputs and outputs**: rejected because it can't show the steps, and the steps are the point.
- **Monkey-patch or copy featurewiz internals to capture intermediates**: rejected as brittle against upstream changes and unreadable as teaching material.

## Consequences

- The Re-implementation is faithful, not idealised. It reproduces featurewiz 0.6.1's quirks (column-order dependence in Recursive XGBoost, the 25%/20% top-N rule, the SULOV-internal 0.70 correlation limit vs the 0.8 top-level default) so that a Reference run comparison is meaningful. Lessons point out the quirks instead of fixing them.

- The Re-implementation can drift from featurewiz's real behaviour. The Reference run comparison is the guard against that drift.
- featurewiz pins old dependencies (`numpy<2.0`, `xgboost<=1.7.6`, `scikit-learn<=1.5.2`, `tensorflow`). The Re-implementation doesn't need them; only the Reference run does.
