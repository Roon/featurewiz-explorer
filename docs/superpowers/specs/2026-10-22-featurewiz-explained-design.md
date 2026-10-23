# featurewiz, explained: design

Agreed in a grilling session on 2026-10-22. The vocabulary is defined in `CONTEXT.md`, and the hard-to-reverse decisions are recorded in `docs/adr/`.

## Purpose and audience

A free, public, four-part series of **Kaggle notebooks** (Lessons) that teaches how featurewiz's feature *selection* works by rebuilding it step by step. The audience is Kaggle users who want to understand featurewiz, not just call it.

## Scope

- In scope: **SULOV** and **Recursive XGBoost**, on **classification** targets.
- Out of scope: featurewiz's feature-engineering helpers (`FE_*` groupby, interactions, encoders), and regression. Regression gets a short sidebar in Lesson 4 and no code.

## Re-implementation (ADR 0001)

- Our own readable code for each stage, **faithful to featurewiz 0.6.1 including its quirks**: column-order dependence in Recursive XGBoost, the 25%/20% top-N rule plus the "≥ 50% of max gain" rule, SULOV's `>=` comparison, and pairs sorted by name.
- Its correctness is checked against **Reference runs** of the real library on the same data.
- Prototype finding (2026-10-22): with xgboost 1.7.6 and scikit-learn 1.5.2, the prototype reproduced featurewiz 0.6.1 bit-for-bit (identical SULOV output, round windows, total-gain values and final selection) on the Canonical dataset, the Madelon-style dataset, Breast Cancer, and a shuffled column order.

## Reference runs (ADR 0002)

- Precomputed offline in a pinned environment: Python 3.11, featurewiz 0.6.1, xgboost 1.7.6, scikit-learn 1.5.2, numpy 1.26.4, pandas 2.2.3, pyarrow 16.1.0.
- Saved as JSON and shipped in the Kaggle dataset `amayzes/featurewiz-explained-reference-runs`. Lessons run with **Internet off**.
- Matching rule: in the pinned environment the Re-implementation must match exactly (with `tree_method="exact"`). In the current environment (Kaggle-like) differences are **reported, not failures**.
- An optional "rerun featurewiz live" recipe is shown as a non-executed code block, because it needs Internet.

## Lessons

1. **Intuition**: introduces the Canonical dataset and its planted roles, shows that redundant copies split importance and that noise and redundancy hurt a model, then gives an overview of SULOV → Recursive XGBoost. No featurewiz code.
2. **SULOV, step by step**: on the Canonical dataset, then Breast Cancer Wisconsin.
3. **Recursive XGBoost, step by step**: including the column-shuffle **Quirk** demonstration.
4. **Full pipeline**: compares our version with the Reference run, scores against Planted ground truth by **Signal group**, runs Breast Cancer, then the **Madelon-style** capstone.

Each Lesson builds its stage inline so learners can read it. The finished code lives in the module `featurewiz_explained.py` (plus `featurewiz_explained_plots.py`), which later Lessons import.

## Data

- **Canonical dataset**: generated in the notebook from a fixed seed. 2,000 rows and 12 features: `x1..x4` informative, `x1_copy..x3_copy` near-linear copies, `x4_squared` a nonlinear copy that SULOV can't see, and `noise_1..noise_4`. Binary `target`.
- **Madelon-style dataset**: `make_classification` with `shuffle=False`. 2,000 rows and 300 features: 5 informative, 15 redundant, 280 noise. Generated in the notebook. Real Madelon is not used because its feature roles were never published.
- **Breast Cancer Wisconsin**: prepared from UCI (CC BY 4.0) into `breast_cancer.csv`. Columns are renamed without spaces (`concave_points_mean`), because featurewiz silently strips spaces from column names. The target is `diagnosis` (M → 1, B → 0).

## Scoring

- Canonical dataset, by **Signal group**: groups covered, groups missed, redundant extras kept, noise kept. Keeping a copy instead of the original is not a miss.
- Madelon-style dataset: relevant (informative + redundant) features kept out of 20, and noise kept out of 280.

## Delivery

- GitHub (`Roon/featurewiz-explorer`) is the source of truth. Lessons are stored as `.ipynb` with outputs stripped (nbstripout).
- **Charts: Plotly only**, no ipywidgets, so they work in Kaggle's read-only view.
- **Shared code**: the module ships *inside* the reference-runs Kaggle dataset, and each Lesson adds that folder to `sys.path`. This replaces Kaggle Utility Scripts, because the Kaggle API's support for pushing utility scripts couldn't be confirmed.
- **CI** (GitHub Actions) has two jobs:
  - *pinned*: exact match against the Reference runs.
  - *current*: unit tests, all Lessons executed with `pytest --nbmake`, and a reference-difference report.
- **Kaggle publishing**: under `amayzes`, with notebooks titled "featurewiz, explained N/4: …". Pushed **private first**; making them public needs the owner's explicit go-ahead.
- **License**: Apache-2.0, with a NOTICE crediting AutoViML/featurewiz and the UCI datasets.
