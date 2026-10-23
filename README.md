# featurewiz-explorer

**featurewiz, explained**: a four-part Kaggle series that teaches how [featurewiz](https://github.com/AutoViML/featurewiz)
selects features. The series rebuilds its two stages, **SULOV** then **Recursive XGBoost**, step by step.
The Re-implementation is faithful to featurewiz 0.6.1, quirks included. It is checked against Reference runs
of the real library. Vocabulary is defined in [CONTEXT.md](CONTEXT.md).

## The Lessons on Kaggle

| # | Lesson | What it covers |
|---|---|---|
| 1 | [Intuition](https://www.kaggle.com/code/amayzes/featurewiz-explained-1-4-intuition) | The Canonical dataset, and why redundant and noise features hurt |
| 2 | [SULOV, step by step](https://www.kaggle.com/code/amayzes/featurewiz-explained-2-4-sulov-step-by-step) | Correlated pairs, mutual information, the knockout walk; Breast Cancer Wisconsin |
| 3 | [Recursive XGBoost, step by step](https://www.kaggle.com/code/amayzes/featurewiz-explained-3-4-recursive-xgboost) | Rounds, windows and nominations; why column order changes the answer |
| 4 | [Full pipeline](https://www.kaggle.com/code/amayzes/featurewiz-explained-4-4-full-pipeline) | Comparison with real featurewiz, Signal-group scoring, the Madelon-style capstone |

All four run with Internet off. They attach the dataset
[featurewiz, explained: reference runs](https://www.kaggle.com/datasets/amayzes/featurewiz-explained-reference-runs),
which holds the shared code, the Breast Cancer data, and the saved Reference runs.

## Running locally

```bash
uv venv .venv && uv pip install -p .venv -e ".[dev]"
.venv/bin/nbstripout --install                      # strips notebook outputs on commit
.venv/bin/pytest                                    # unit + Reference-run report
.venv/bin/pytest --nbmake --nbmake-timeout=900 notebooks/
```

Exact matching against the Reference runs only happens in the pinned environment (xgboost 1.7.6, scikit-learn 1.5.2).
See `requirements-pinned.txt`, the `pinned` CI job and [ADR 0002](docs/adr/0002-precomputed-reference-runs.md).
To regenerate the Reference runs with the real featurewiz, see the docstring of `scripts/make_reference_runs.py`.

## Publishing to Kaggle

```bash
.venv/bin/python scripts/build_kaggle.py --public    # without --public, a push makes the Lessons private
.venv/bin/kaggle datasets version -p build/kaggle/dataset -m "<notes>"
for d in build/kaggle/lesson-*; do .venv/bin/kaggle kernels push -p "$d"; done
```

## Licence

Code is Apache-2.0 ([LICENSE](LICENSE)). Data in `reference_runs/` is CC BY 4.0 ([LICENSE-CC-BY-4.0](LICENSE-CC-BY-4.0)).
Attributions to featurewiz and UCI, and the changes made to the UCI data, are in [NOTICE](NOTICE).
