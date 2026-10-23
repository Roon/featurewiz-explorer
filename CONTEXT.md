# featurewiz-explorer

A public learning resource that teaches how the featurewiz feature-selection algorithm works, by rebuilding it step by step on toy data. Its public name is **featurewiz, explained**.

## Language

**featurewiz**:
The third-party Python library (AutoViML) whose feature-selection algorithm this project teaches.
_Avoid_: "the library" when it could be confused with our own code

**Reference run**:
Running the real **featurewiz** library on a dataset, used as the answer key that a **Re-implementation** must match.
_Avoid_: baseline, ground truth (ground truth means something else here, see **Planted ground truth**)

**Re-implementation**:
This project's own from-scratch, step-by-step version of a featurewiz stage, written to expose every intermediate result.
_Avoid_: port, clone, wrapper

**SULOV**:
"Searching for Uncorrelated List Of Variables": the first selection stage. Among every pair of features correlated above the **correlation limit**, the one with lower mutual information with the target is dropped.

**Correlation limit**:
The absolute-correlation threshold above which two features count as a redundant pair in **SULOV** (featurewiz default 0.8).

**Recursive XGBoost**:
The second selection stage. Over several **rounds**, an XGBoost model is trained on a shrinking window of the features **SULOV** kept. Each round drops the next block of features from the front of the column order. The top features by gain from every round are pooled into the final selection.

**Round**:
One XGBoost fit inside **Recursive XGBoost**, with the top-N features it nominates.

**Quirk**:
A real featurewiz behaviour that is surprising or arguably wrong, such as **Recursive XGBoost** depending on column order. The **Re-implementation** reproduces quirks on purpose, and **Lessons** point them out.
_Avoid_: bug (we don't judge upstream; we show it)

**Signal group**:
An informative feature in a **Synthetic dataset** together with its redundant copies. A selection is perfect when it keeps exactly one feature per signal group and no noise. Which member it keeps doesn't matter.

**Lesson**:
One notebook in the series, teaching exactly one idea: intuition, **SULOV**, **Recursive XGBoost**, or the full pipeline.
_Avoid_: chapter, part, kernel (Kaggle's old word for a notebook)

**Synthetic dataset**:
Generated data whose features are deliberately built as informative, redundant (derived from informative ones), or pure noise.

**Canonical dataset**:
The one small **Synthetic dataset** used in every **Lesson**, so learners get to know its features by name. It deliberately includes a nonlinear copy that **SULOV** can't detect.

**Planted ground truth**:
The known role (informative, redundant, or noise) of each feature in a **Synthetic dataset**, against which a selection can be scored.
_Avoid_: "probe" (Madelon's word for a noise feature). Say noise.

**Madelon-style dataset**:
A larger **Synthetic dataset** built with the same recipe as the NIPS 2003 Madelon benchmark (scikit-learn's `make_classification`). It has a few informative features, many redundant linear mixtures of them, and hundreds of noise features. Real Madelon is *not* used, because its feature roles were never published.

## Relationships

- A selection is **SULOV** followed by **Recursive XGBoost**. Only features that survive **SULOV** enter **Recursive XGBoost**
- A **Re-implementation** is checked against a **Reference run** on the same dataset
- **Planted ground truth** exists only for **Synthetic datasets** (the **Canonical dataset** and the **Madelon-style dataset**). Real datasets (Breast Cancer Wisconsin) have a **Reference run** but no ground truth
- A **Recursive XGBoost** selection is made of many **Rounds**
- A selection on the **Canonical dataset** is scored by **Signal group**: groups covered, redundant extras kept, noise kept. On the **Madelon-style dataset**, where redundant features mix several informative ones, it is scored as relevant features kept vs noise kept

## Example dialogue

> **Learner:** "featurewiz dropped `x3_copy`. Was that right?"
> **Author:** "In this **Synthetic dataset** the **Planted ground truth** says `x3_copy` is redundant with `x3`. **SULOV** paired them because they're correlated above the **correlation limit**, and dropped `x3_copy` because its mutual information was lower. Our **Re-implementation** shows that pair, and the **Reference run** made the same call."

## Flagged ambiguities

- "Ground truth" vs "the featurewiz answer": resolved. The real library's output is the **Reference run**, and only synthetic feature roles are **Planted ground truth**. featurewiz can be wrong with respect to the ground truth.
