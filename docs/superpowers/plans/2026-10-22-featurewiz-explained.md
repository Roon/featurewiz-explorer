# featurewiz, explained: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a four-notebook Kaggle series that teaches featurewiz 0.6.1's feature selection by re-implementing SULOV and Recursive XGBoost faithfully, checked against Reference runs of the real library.

**Architecture:** One importable module, `src/featurewiz_explained.py`, holds the datasets, the faithful Re-implementation, scoring, and Reference-run loading and comparison. A second module, `src/featurewiz_explained_plots.py`, holds the Plotly charts. A script run once in a pinned Python 3.11 environment executes the real featurewiz and records Reference runs as JSON in `reference_runs/`. Four Lessons (`notebooks/*.ipynb`) import the modules. For Kaggle, the modules and `reference_runs/` are flattened into one Kaggle dataset, which each Lesson finds under `/kaggle/input` and adds to `sys.path`.

**Tech Stack:** Python, numpy, pandas, scikit-learn (`mutual_info_classif`, `make_classification`), xgboost, Plotly, pytest, nbmake, nbstripout, jupytext (authoring only), GitHub Actions, Kaggle CLI.

**Spec:** `docs/superpowers/specs/2026-10-22-featurewiz-explained-design.md`. Also read `CONTEXT.md` (vocabulary) and `docs/adr/0001-*.md`, `docs/adr/0002-*.md`.

## Global Constraints

- The reference library is **featurewiz 0.6.1**. The Re-implementation reproduces its behaviour *including quirks*. Never "fix" a quirk in code; Lessons explain it instead.
- Pinned (reference) environment: Python 3.11, `featurewiz==0.6.1`, `xgboost==1.7.6`, `scikit-learn==1.5.2`, `numpy==1.26.4`, `pandas==2.2.3`, `pyarrow==16.1.0`, `scipy==1.17.1`.
- Current environment: latest packages, Python 3.12. It stands in for Kaggle's notebook image.
- The Re-implementation forces `tree_method="exact"` by default. The pinned environment must match the Reference runs **exactly**. Differences in the current environment are **reported, never failures**.
- Classification only (binary and multiclass). The target must already be encoded as integers `0..k-1`.
- Charts: Plotly only, no ipywidgets. Colours come from the reference palette: informative `#2a78d6`, redundant `#eb6834`, noise `#1baf7a`, dropped/neutral `#a3a29c`, sequential scale `Blues`. Identity is never shown by colour alone; there's always a label or legend.
- Lessons run with **Internet off**: no `pip install`, no downloads, no URLs fetched in executed cells.
- Notebooks are committed with **outputs stripped**.
- Any copy of SULOV must include featurewiz's SULOV attribution text. The library's docstring says "YOU MUST INCLUDE THE ABOVE MESSAGE IF YOU COPY SULOV method".
- Kaggle owner `amayzes`. Dataset slug `featurewiz-explained-reference-runs`. Notebook titles `featurewiz, explained N/4: <Topic>`. Push **private**; only the owner makes them public.
- License: Apache-2.0, with a NOTICE crediting AutoViML/featurewiz (Apache-2.0) and UCI Breast Cancer Wisconsin (CC BY 4.0).
- Glossary words from `CONTEXT.md` are used verbatim in code comments and Lesson text: Reference run, Re-implementation, SULOV, Recursive XGBoost, Round, Quirk, Lesson, Canonical dataset, Madelon-style dataset, Planted ground truth, Signal group.

## Review Focus

1. **Column names containing spaces.** featurewiz silently strips spaces, so a learner's own data would diverge from real featurewiz without warning. `select_features` must raise a `ValueError` that explains this. Test in Task 5.
2. **Targets not encoded as `0..k-1`** (strings, `{1, 2}`, `{-1, 1}`). featurewiz label-encodes these first. `recursive_xgboost` must raise a `ValueError` naming the found labels, not train a wrong model. Test in Task 4.
3. **Missing values or non-numeric columns.** featurewiz adds missing-value flag columns and encodes categoricals, which is out of scope. `select_features` must refuse with a clear message. Test in Task 5.
4. **A Lesson that quietly needs the Internet** (a `pip install`, a URL) would break on Kaggle with Internet off. A test scans every code cell. Test in Task 12.
5. **Committed notebook outputs or slow Lessons.** Outputs bloat diffs and go stale; the Madelon-style run takes minutes on Kaggle's 4 CPUs. CI runs `nbstripout --verify` and `nbmake` with an explicit timeout. Task 12.

---

## File map

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata; exposes the two modules from `src/`; pytest config |
| `LICENSE`, `NOTICE` | Apache-2.0 text; attribution |
| `src/featurewiz_explained.py` | Datasets, SULOV, Recursive XGBoost, pipeline, scoring, Reference-run loading and comparison |
| `src/featurewiz_explained_plots.py` | Plotly figures used by the Lessons |
| `scripts/prepare_breast_cancer.py` | One-off: UCI zip → `reference_runs/breast_cancer.csv` |
| `scripts/make_reference_runs.py` | Pinned environment only: run real featurewiz and write `reference_runs/reference_*.json` and `manifest.json` |
| `scripts/build_kaggle.py` | Assemble `build/kaggle/` (flat dataset folder and one folder per Lesson with `kernel-metadata.json`) |
| `requirements-reference.txt` | Pinned environment *with* featurewiz (for making Reference runs) |
| `requirements-pinned.txt` | Pinned environment *without* featurewiz (CI exact-match job) |
| `reference_runs/` | `breast_cancer.csv`, `reference_<name>.json`, `manifest.json` |
| `notebooks/01-intuition.ipynb` … `04-full-pipeline.ipynb` | The four Lessons |
| `tests/test_*.py` | Unit tests per area, the reference-match test, and notebook hygiene tests |
| `.github/workflows/ci.yml` | `pinned` and `current` jobs |

---

### Task 1: Project scaffold and the Canonical dataset

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `LICENSE`, `NOTICE`, `src/featurewiz_explained.py`, `tests/test_datasets.py`

**Interfaces:**
- Produces:
  - `SyntheticDataset` (frozen dataclass) with fields `frame: pd.DataFrame`, `target: str`, `roles: dict[str, str]` (values `"informative" | "redundant" | "noise"`), `groups: dict[str, str]` (feature → Signal group name; empty when groups aren't defined), and property `features -> list[str]` (frame columns minus target, in column order).
  - `make_canonical_dataset(seed: int = 7, n_rows: int = 2000) -> SyntheticDataset`.

- [ ] **Step 1: Write the scaffold files**

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "featurewiz-explained"
version = "0.1.0"
description = "A faithful, readable re-implementation of featurewiz's feature selection, for learning."
license = { text = "Apache-2.0" }
requires-python = ">=3.10"
dependencies = ["numpy", "pandas", "scikit-learn", "xgboost", "plotly"]

[project.optional-dependencies]
dev = ["pytest", "nbmake", "nbstripout", "jupytext", "ipykernel"]

[tool.setuptools]
package-dir = { "" = "src" }
py-modules = ["featurewiz_explained", "featurewiz_explained_plots"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.gitignore`:

```
__pycache__/
*.egg-info/
.venv*/
build/
.ipynb_checkpoints/
.pytest_cache/
```

`LICENSE`: run `curl -sL https://www.apache.org/licenses/LICENSE-2.0.txt -o LICENSE`.

`NOTICE`:

```
featurewiz, explained
Copyright 2026 Aaron Mayzes

This project re-implements, for teaching purposes, the SULOV and Recursive
XGBoost feature-selection methods of featurewiz (https://github.com/AutoViML/featurewiz),
Copyright Ram Seshadri / AutoViML, licensed under the Apache License 2.0.
SULOV ("Searching for Uncorrelated List Of Variables") was created by Ram Seshadri in 2018.

Breast Cancer Wisconsin (Diagnostic) data: Wolberg, W., Mangasarian, O., Street, N.,
& Street, W. (1993). UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B
Licensed under CC BY 4.0.
```

- [ ] **Step 2: Set up the dev environment**

Run: `uv venv -p 3.12 .venv && uv pip install -p .venv -e ".[dev]"`
Expected: installs without error. Use `.venv/bin/pytest` and `.venv/bin/python` from now on.

- [ ] **Step 3: Write the failing test**

`tests/test_datasets.py`:

```python
import numpy as np

import featurewiz_explained as fwx


def test_canonical_dataset_shape_and_columns():
    data = fwx.make_canonical_dataset()
    assert data.target == "target"
    assert data.features == [
        "x1", "x2", "x3", "x4",
        "x1_copy", "x2_copy", "x3_copy", "x4_squared",
        "noise_1", "noise_2", "noise_3", "noise_4",
    ]
    assert data.frame.shape == (2000, 13)
    assert set(data.frame["target"].unique()) == {0, 1}


def test_canonical_dataset_roles_and_signal_groups():
    data = fwx.make_canonical_dataset()
    assert [f for f in data.features if data.roles[f] == "informative"] == ["x1", "x2", "x3", "x4"]
    assert [f for f in data.features if data.roles[f] == "redundant"] == ["x1_copy", "x2_copy", "x3_copy", "x4_squared"]
    assert [f for f in data.features if data.roles[f] == "noise"] == ["noise_1", "noise_2", "noise_3", "noise_4"]
    assert data.groups == {
        "x1": "x1", "x2": "x2", "x3": "x3", "x4": "x4",
        "x1_copy": "x1", "x2_copy": "x2", "x3_copy": "x3", "x4_squared": "x4",
    }


def test_canonical_dataset_plants_linear_copies_and_a_nonlinear_copy():
    frame = fwx.make_canonical_dataset().frame
    for i in (1, 2, 3):
        assert frame[f"x{i}"].corr(frame[f"x{i}_copy"]) > 0.9  # SULOV (limit 0.8) will see these
    assert abs(frame["x4"].corr(frame["x4_squared"])) < 0.2  # ...but not this one


def test_canonical_dataset_is_deterministic_per_seed():
    a, b = fwx.make_canonical_dataset(seed=7), fwx.make_canonical_dataset(seed=7)
    assert a.frame.equals(b.frame)
    assert not np.allclose(a.frame["x1"], fwx.make_canonical_dataset(seed=8).frame["x1"])
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_datasets.py -v`
Expected: FAIL. The module is empty, so `AttributeError: module 'featurewiz_explained' has no attribute 'make_canonical_dataset'` (or ImportError until Step 5 creates the file).

- [ ] **Step 5: Write the implementation**

`src/featurewiz_explained.py`:

```python
"""featurewiz, explained: a faithful, readable Re-implementation of featurewiz 0.6.1's
feature selection (SULOV, then Recursive XGBoost), for learning.

Vocabulary is defined in CONTEXT.md. Every Quirk reproduced here is deliberate
(docs/adr/0001-reimplement-featurewiz-stages.md): the Re-implementation must match a
Reference run of the real library exactly, so don't "fix" anything here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATUREWIZ_VERSION = "0.6.1"


# --------------------------------------------------------------------------- datasets

@dataclass(frozen=True)
class SyntheticDataset:
    """A Synthetic dataset plus its Planted ground truth."""

    frame: pd.DataFrame
    target: str
    roles: dict[str, str]   # feature -> "informative" | "redundant" | "noise"
    groups: dict[str, str]  # feature -> Signal group (empty if the dataset has none)

    @property
    def features(self) -> list[str]:
        return [c for c in self.frame.columns if c != self.target]


def make_canonical_dataset(seed: int = 7, n_rows: int = 2000) -> SyntheticDataset:
    """The Canonical dataset: 4 informative features, 3 near-linear copies,
    1 nonlinear copy (which SULOV can't detect), and 4 noise features."""
    rng = np.random.default_rng(seed)
    x = {f"x{i}": rng.normal(size=n_rows) for i in range(1, 5)}
    columns = dict(x)
    for i in (1, 2, 3):
        columns[f"x{i}_copy"] = x[f"x{i}"] + rng.normal(scale=0.3, size=n_rows)
    columns["x4_squared"] = x["x4"] ** 2 + rng.normal(scale=0.1, size=n_rows)
    for i in range(1, 5):
        columns[f"noise_{i}"] = rng.normal(size=n_rows)
    logit = 1.2 * x["x1"] - 1.0 * x["x2"] + 0.8 * x["x3"] + 0.6 * x["x4"] + rng.logistic(size=n_rows)
    columns["target"] = (logit > 0).astype(int)

    roles = {f"x{i}": "informative" for i in range(1, 5)}
    roles |= {f"x{i}_copy": "redundant" for i in (1, 2, 3)} | {"x4_squared": "redundant"}
    roles |= {f"noise_{i}": "noise" for i in range(1, 5)}
    groups = {f"x{i}": f"x{i}" for i in range(1, 5)}
    groups |= {f"x{i}_copy": f"x{i}" for i in (1, 2, 3)} | {"x4_squared": "x4"}
    return SyntheticDataset(pd.DataFrame(columns), "target", roles, groups)
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_datasets.py -v`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore LICENSE NOTICE src/featurewiz_explained.py tests/test_datasets.py
git commit -m "feat: scaffold package and Canonical dataset"
```

---

### Task 2: Madelon-style dataset and Breast Cancer data

**Files:**
- Modify: `src/featurewiz_explained.py` (append to the datasets section; add imports)
- Create: `scripts/prepare_breast_cancer.py`, `reference_runs/breast_cancer.csv` (generated)
- Test: `tests/test_datasets.py` (append)

**Interfaces:**
- Consumes: `SyntheticDataset` (Task 1).
- Produces:
  - `make_madelon_style_dataset(seed: int = 7, n_rows: int = 2000) -> SyntheticDataset`, with features `f000..f299`, roles `f000–f004` informative, `f005–f019` redundant, the rest noise, and `groups == {}`.
  - `BREAST_CANCER_FEATURES: list[str]` (30 names, e.g. `"concave_points_mean"`).
  - `reference_dir() -> pathlib.Path`.
  - `load_breast_cancer() -> pd.DataFrame` (30 features plus `diagnosis` in 0/1, 569 rows).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_datasets.py`)

```python
def test_madelon_style_dataset_roles_follow_make_classification_layout():
    data = fwx.make_madelon_style_dataset()
    assert data.frame.shape == (2000, 301)
    assert data.features[0] == "f000" and data.features[-1] == "f299"
    assert [f for f in data.features if data.roles[f] == "informative"] == [f"f{i:03d}" for i in range(5)]
    assert [f for f in data.features if data.roles[f] == "redundant"] == [f"f{i:03d}" for i in range(5, 20)]
    assert sum(r == "noise" for r in data.roles.values()) == 280
    assert data.groups == {}


def test_breast_cancer_is_prepared_without_spaces_in_names():
    frame = fwx.load_breast_cancer()
    assert frame.shape == (569, 31)
    assert list(frame.columns) == [*fwx.BREAST_CANCER_FEATURES, "diagnosis"]
    assert not any(" " in c for c in frame.columns)  # featurewiz strips spaces (a Quirk)
    assert frame["diagnosis"].sum() == 212  # malignant cases


def test_reference_dir_can_be_overridden(monkeypatch, tmp_path):
    monkeypatch.setenv("FWX_REFERENCE_DIR", str(tmp_path))
    assert fwx.reference_dir() == tmp_path
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_datasets.py -v`
Expected: the 3 new tests FAIL with `AttributeError` (no `make_madelon_style_dataset` / `load_breast_cancer` / `reference_dir`).

- [ ] **Step 3: Write the implementation**

Add these imports at the top of `src/featurewiz_explained.py` (keep them sorted with the existing ones):

```python
import os
from pathlib import Path

from sklearn.datasets import make_classification
```

Append to the datasets section:

```python
def make_madelon_style_dataset(seed: int = 7, n_rows: int = 2000) -> SyntheticDataset:
    """A Madelon-style dataset: the NIPS 2003 Madelon recipe (scikit-learn's
    make_classification is Guyon's generator). With shuffle=False the columns come
    out as 5 informative, then 15 redundant (linear mixtures of the informative
    ones), then 280 noise. Real Madelon is not used because its feature roles
    were never published."""
    X, y = make_classification(
        n_samples=n_rows, n_features=300, n_informative=5, n_redundant=15, n_repeated=0,
        n_classes=2, n_clusters_per_class=16, shuffle=False, random_state=seed,
    )
    names = [f"f{i:03d}" for i in range(300)]
    roles = {name: "informative" if i < 5 else "redundant" if i < 20 else "noise" for i, name in enumerate(names)}
    return SyntheticDataset(pd.DataFrame(X, columns=names).assign(target=y), "target", roles, {})


BREAST_CANCER_FEATURES = [
    f"{measure}_{stat}"
    for stat in ("mean", "se", "worst")
    for measure in (
        "radius", "texture", "perimeter", "area", "smoothness", "compactness",
        "concavity", "concave_points", "symmetry", "fractal_dimension",
    )
]


def reference_dir() -> Path:
    """Where Reference runs and prepared data live.

    Order: $FWX_REFERENCE_DIR; next to this module (the flat Kaggle dataset);
    the repo's reference_runs/ folder."""
    if "FWX_REFERENCE_DIR" in os.environ:
        return Path(os.environ["FWX_REFERENCE_DIR"])
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent / "reference_runs"):
        if (candidate / "breast_cancer.csv").is_file():
            return candidate
    raise FileNotFoundError(
        "Reference runs not found. On Kaggle, attach the dataset "
        "amayzes/featurewiz-explained-reference-runs; locally, run from the repo."
    )


def load_breast_cancer() -> pd.DataFrame:
    """Breast Cancer Wisconsin (Diagnostic), UCI, CC BY 4.0. diagnosis: 1 = malignant."""
    return pd.read_csv(reference_dir() / "breast_cancer.csv")
```

`scripts/prepare_breast_cancer.py`:

```python
"""One-off: download UCI Breast Cancer Wisconsin (Diagnostic), CC BY 4.0, into
reference_runs/breast_cancer.csv.

Column names use underscores instead of spaces ("concave_points_mean") because
featurewiz strips spaces from column names. Spaces would make the Re-implementation
and the Reference run see different names.

Usage: .venv/bin/python scripts/prepare_breast_cancer.py
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from featurewiz_explained import BREAST_CANCER_FEATURES

URL = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
OUT = Path(__file__).resolve().parents[1] / "reference_runs" / "breast_cancer.csv"


def main() -> None:
    raw = urllib.request.urlopen(URL, timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive, archive.open("wdbc.data") as f:
        frame = pd.read_csv(f, header=None, names=["id", "diagnosis", *BREAST_CANCER_FEATURES])
    frame["diagnosis"] = (frame["diagnosis"] == "M").astype(int)
    OUT.parent.mkdir(exist_ok=True)
    frame[[*BREAST_CANCER_FEATURES, "diagnosis"]].to_csv(OUT, index=False)
    print(f"wrote {OUT} {frame.shape}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Generate the data file**

Run: `.venv/bin/python scripts/prepare_breast_cancer.py`
Expected: `wrote .../reference_runs/breast_cancer.csv (569, 32)`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_datasets.py -v`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add src/featurewiz_explained.py scripts/prepare_breast_cancer.py reference_runs/breast_cancer.csv tests/test_datasets.py
git commit -m "feat: Madelon-style dataset and prepared Breast Cancer data"
```

---

### Task 3: SULOV Re-implementation

**Files:**
- Modify: `src/featurewiz_explained.py` (new section after datasets)
- Test: `tests/test_sulov.py`

**Interfaces:**
- Produces:
  - `SulovResult` (frozen dataclass): `correlated_pairs: list[tuple[str, str]]`, `mutual_info: dict[str, float]` (only for features in a pair), `kept: list[str]`, `dropped: list[str]`.
  - `sulov(frame: pd.DataFrame, features: list[str], target: str, corr_limit: float = 0.8) -> SulovResult`.

These are the featurewiz 0.6.1 behaviours to reproduce (from `featurewiz/sulov_method.py`, `FE_remove_variables_using_SULOV_method`, pairwise mode). Each one is pinned by a test below:
- Missing values are filled with 0. Pearson `|corr|` is used. A pair is `(features[i], features[j])` for `i < j` with `|corr| >= corr_limit` (greater than *or equal*).
- Pairs are **sorted** as tuples, by name.
- Mutual information (`mutual_info_classif(..., n_neighbors=5, discrete_features=False, random_state=42)`) is computed on the paired features only, in order of first appearance in the sorted pairs.
- Walk features by mutual information, highest first. Keep the current feature, and for each not-yet-analysed pair that contains it, remove its partner from the walk.
- `kept` = features in no pair (input order), then the kept paired features (walk order).

- [ ] **Step 1: Write the failing tests**

`tests/test_sulov.py`:

```python
import numpy as np
import pandas as pd

import featurewiz_explained as fwx


def _chain_frame():
    """a predicts y; b = a + noise; c = b + noise.
    |corr|: a-b 0.895, b-c 0.912, a-c 0.815."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=3000)
    b = a + rng.normal(scale=0.5, size=3000)
    c = b + rng.normal(scale=0.5, size=3000)
    frame = pd.DataFrame({"a": a, "b": b, "c": c, "z": rng.normal(size=3000)})
    frame["y"] = (a > 0).astype(int)
    return frame


def test_no_correlated_pairs_keeps_everything_in_input_order():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"p": rng.normal(size=500), "q": rng.normal(size=500), "y": rng.integers(0, 2, 500)})
    result = fwx.sulov(frame, ["q", "p"], "y")
    assert result == fwx.SulovResult([], {}, ["q", "p"], [])


def test_pairs_use_greater_or_equal_and_are_sorted_by_name():
    frame = _chain_frame()
    result = fwx.sulov(frame, ["c", "b", "a", "z"], "y", corr_limit=0.85)
    # input order is c, b, a, but pairs are sorted tuples, so ('b', 'a') sorts before ('c', 'b')
    assert result.correlated_pairs == [("b", "a"), ("c", "b")]
    # same code path as sulov() (DataFrame.corr), so the float is bit-identical
    exact = frame[["a", "b"]].corr(method="pearson").abs().iloc[0, 1]
    assert fwx.sulov(frame, ["a", "b"], "y", corr_limit=exact).correlated_pairs == [("a", "b")]


def test_chain_keeps_both_ends_when_the_middle_is_knocked_out():
    result = fwx.sulov(_chain_frame(), ["a", "b", "c", "z"], "y", corr_limit=0.85)
    assert result.correlated_pairs == [("a", "b"), ("b", "c")]
    assert list(result.mutual_info) == ["a", "b", "c"]  # first appearance in sorted pairs
    assert result.mutual_info["a"] > result.mutual_info["b"] > result.mutual_info["c"]
    # a wins its pair and knocks out b; c's only partner (b) is gone, so c survives
    assert result.kept == ["z", "a", "c"]
    assert result.dropped == ["b"]


def test_uncorrelated_features_come_first_then_mutual_information_order():
    result = fwx.sulov(_chain_frame(), ["z", "a", "b"], "y")
    assert result.kept == ["z", "a"]
    assert result.dropped == ["b"]


def test_canonical_dataset_drops_linear_copies_but_not_the_squared_one():
    data = fwx.make_canonical_dataset()
    result = fwx.sulov(data.frame, data.features, data.target)
    assert sorted(result.dropped) == ["x1_copy", "x2_copy", "x3"]  # x3_copy happened to win its pair
    assert "x4_squared" in result.kept
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_sulov.py -v`
Expected: FAIL with `AttributeError: module 'featurewiz_explained' has no attribute 'sulov'`.

- [ ] **Step 3: Write the implementation**

Add the import `from sklearn.feature_selection import mutual_info_classif` at the top, then append:

```python
# --------------------------------------------------------------------------- SULOV

@dataclass(frozen=True)
class SulovResult:
    correlated_pairs: list[tuple[str, str]]
    mutual_info: dict[str, float]
    kept: list[str]
    dropped: list[str]


def sulov(frame: pd.DataFrame, features: list[str], target: str, corr_limit: float = 0.8) -> SulovResult:
    """SULOV, re-implemented faithfully from featurewiz 0.6.1 (pairwise mode).

    Attribution, as featurewiz requires of anyone copying SULOV:
    SULOV stands for Searching Uncorrelated List Of Variables. SULOV method was
    created by Ram Seshadri in 2018. This highly efficient method removes variables
    that are highly correlated using a series of pair-wise correlation knockout
    rounds. To decide order of variables to keep, we use mutual information score
    to select. MIS returns a ranked list of these correlated variables: when we
    select one, we knock out others that are highly correlated to it. Then we
    select next variable to inspect. This continues until we knock out all highly
    correlated variables in each set of variables. Finally we are left with only
    uncorrelated variables that are also highly important in mutual score.

    Note: featurewiz()'s default corr_limit is 0.8, but the SULOV function's own
    default is 0.70. We follow featurewiz() (a Quirk the Lessons point out).
    """
    features = list(features)
    X = frame[features].fillna(0)
    corr = X.corr(method="pearson").abs()

    pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if corr.iloc[i, j] >= corr_limit:
                pairs.append((features[i], features[j]))
    pairs.sort()
    if not pairs:
        return SulovResult([], {}, features, [])

    paired = list(dict.fromkeys(f for pair in pairs for f in pair))
    scores = mutual_info_classif(
        X[paired], frame[target], n_neighbors=5, discrete_features=False, random_state=42
    )
    mutual_info = {f: float(s) for f, s in zip(paired, scores)}

    # Walk features from highest mutual information down. `walk` shrinks while we
    # iterate over it, exactly as in featurewiz.
    walk = sorted(mutual_info, key=mutual_info.get, reverse=True)
    kept_paired, analysed = [], set()
    for feature in walk:
        kept_paired.append(feature)
        for a, b in pairs:
            pair = tuple(sorted((a, b)))
            if feature == a and pair not in analysed:
                analysed.add(pair)
                if b in walk:
                    walk.remove(b)
            elif feature == b and pair not in analysed:
                analysed.add(pair)
                if a in walk:
                    walk.remove(a)

    kept = [f for f in features if f not in paired] + kept_paired
    return SulovResult(pairs, mutual_info, kept, [f for f in features if f not in kept])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_sulov.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/featurewiz_explained.py tests/test_sulov.py
git commit -m "feat: faithful SULOV re-implementation"
```

---

### Task 4: Recursive XGBoost Re-implementation

**Files:**
- Modify: `src/featurewiz_explained.py` (new section after SULOV)
- Test: `tests/test_recursive_xgboost.py`

**Interfaces:**
- Produces:
  - `round_plan(n_features: int) -> tuple[int, int]`, returning `(step, top_num)`.
  - `nominate(gains: dict[str, float], top_num: int) -> list[str]`.
  - `XgbRound` (frozen dataclass): `window: list[str]`, `gains: dict[str, float]`, `nominated: list[str]`.
  - `RecursiveXgbResult` (frozen dataclass): `step: int`, `top_num: int`, `rounds: list[XgbRound]`, `selected: list[str]`.
  - `recursive_xgboost(frame: pd.DataFrame, features: list[str], target: str, tree_method: str | None = "exact") -> RecursiveXgbResult`. `tree_method=None` means "use the installed xgboost's default", which is what Lesson 3 uses to show the version gap.

These are the featurewiz 0.6.1 behaviours to reproduce (`featurewiz/featurewiz.py`, `FE_perform_recursive_xgboost`, single-label, non-dask):
- `step = 2` if there are ≤ 10 features, else `int(n/5 + 0.5)`. `top_num = int(max(2, n*0.25))` if n ≤ 50, else `int(n*0.20)`.
- Rounds start at `i = 0, step, 2*step, …`. The window is `features[i:]` (a **suffix**, so later columns appear in more rounds). A window with fewer than 2 features is skipped.
- `xgb.train(params, DMatrix(window, enable_categorical=True, feature_names=window), num_boost_round=100)` (20 if rows ≥ 100,000). Binary: `{"objective": "binary:logistic", "num_class": 1, "min_child_weight": 0.5}`. Multiclass: `{"objective": "multi:softmax", "num_class": k, "min_child_weight": 0.5}`. featurewiz sets no `tree_method`; xgboost 1.7.6's `auto` resolves to `exact` on small data.
- Gains are `get_score(importance_type="total_gain")`.
- Nomination: the features with gain ≥ 50% of the round's max (kept in *gains-dict order*). If there are more than 1, use them, but if that's fewer than `top_num`, take the top `top_num` by gain instead. If there's at most 1, take the top `top_num` by gain.
- `selected` = de-duplicated union of nominations, in order of first nomination.

- [ ] **Step 1: Write the failing tests**

`tests/test_recursive_xgboost.py`:

```python
import numpy as np
import pandas as pd
import pytest

import featurewiz_explained as fwx


@pytest.mark.parametrize(
    "n, expected",
    [(5, (2, 2)), (9, (2, 2)), (10, (2, 2)), (11, (2, 2)), (20, (4, 5)), (50, (10, 12)), (51, (10, 10)), (295, (59, 59))],
)
def test_round_plan(n, expected):
    assert fwx.round_plan(n) == expected


def test_nominate_takes_strong_features_in_gains_order():
    # b and a are both >= 50% of the max; gains-dict order is kept (b before a)
    assert fwx.nominate({"b": 6.0, "a": 10.0, "c": 1.0}, top_num=2) == ["b", "a"]


def test_nominate_tops_up_to_top_num_when_too_few_strong_features():
    assert fwx.nominate({"a": 10.0, "b": 6.0, "c": 1.0, "d": 0.5}, top_num=3) == ["a", "b", "c"]


def test_nominate_falls_back_to_top_num_when_one_feature_dominates():
    assert fwx.nominate({"a": 10.0, "b": 1.0, "c": 0.5}, top_num=2) == ["a", "b"]


def _small_frame(labels=(0, 1)):
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({f"f{i}": rng.normal(size=400) for i in range(5)})
    frame["y"] = np.where(frame["f0"] + frame["f3"] > 0, labels[1], labels[0])
    return frame


def test_windows_are_suffixes_and_short_windows_are_skipped():
    features = ["f0", "f1", "f2", "f3", "f4"]
    result = fwx.recursive_xgboost(_small_frame(), features, "y")
    assert (result.step, result.top_num) == (2, 2)
    # i = 0, 2, 4; the window at i = 4 holds one feature, so it is skipped
    assert [r.window for r in result.rounds] == [features, ["f2", "f3", "f4"]]
    assert "f0" in result.rounds[0].gains


def test_selected_is_union_of_nominations_in_first_seen_order():
    result = fwx.recursive_xgboost(_small_frame(), ["f0", "f1", "f2", "f3", "f4"], "y")
    union = list(dict.fromkeys(f for r in result.rounds for f in r.nominated))
    assert result.selected == union


def test_multiclass_targets_are_supported():
    frame = _small_frame()
    frame["y"] = pd.cut(frame["f0"], 3, labels=False)
    result = fwx.recursive_xgboost(frame, ["f0", "f1", "f2", "f3", "f4"], "y")
    assert "f0" in result.selected


@pytest.mark.parametrize("labels", [(-1, 1), (1, 2)])
def test_targets_must_be_encoded_as_0_to_k_minus_1(labels):
    with pytest.raises(ValueError, match="0..k-1"):
        fwx.recursive_xgboost(_small_frame(labels), ["f0", "f1", "f2"], "y")


def test_string_targets_are_rejected():
    frame = _small_frame()
    frame["y"] = frame["y"].map({0: "B", 1: "M"})
    with pytest.raises(ValueError, match="0..k-1"):
        fwx.recursive_xgboost(frame, ["f0", "f1", "f2"], "y")


def test_canonical_dataset_after_sulov_misses_x4():
    data = fwx.make_canonical_dataset()
    kept = fwx.sulov(data.frame, data.features, data.target).kept
    result = fwx.recursive_xgboost(data.frame, kept, data.target)
    assert result.selected == ["x1", "x2", "x3_copy"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_recursive_xgboost.py -v`
Expected: FAIL with `AttributeError: ... no attribute 'round_plan'`.

- [ ] **Step 3: Write the implementation**

Add `import xgboost as xgb` at the top, then append:

```python
# --------------------------------------------------------------------------- Recursive XGBoost

@dataclass(frozen=True)
class XgbRound:
    window: list[str]
    gains: dict[str, float]
    nominated: list[str]


@dataclass(frozen=True)
class RecursiveXgbResult:
    step: int
    top_num: int
    rounds: list[XgbRound]
    selected: list[str]


def round_plan(n_features: int) -> tuple[int, int]:
    """(step, top_num): how many features each Round slides past, and how many it nominates."""
    step = 2 if n_features <= 10 else int(n_features / 5 + 0.5)
    top_num = int(max(2, n_features * 0.25)) if n_features <= 50 else int(n_features * 0.20)
    return step, top_num


def nominate(gains: dict[str, float], top_num: int) -> list[str]:
    """Which features a Round nominates, with featurewiz's exact pandas expressions
    (including the boolean-mask alignment that keeps strong features in gains order)."""
    s = pd.Series(gains)
    strong = s[s.sort_values(ascending=False) / s.values.max() >= 0.5]
    if len(strong) > 1:
        nominated = strong.index.tolist()
        if len(nominated) < top_num:
            nominated = s.sort_values(ascending=False)[:top_num].index.tolist()
        return nominated
    return s.sort_values(ascending=False)[:top_num].index.tolist()


def _xgb_params(y: pd.Series) -> dict:
    classes = np.sort(pd.unique(y))
    if len(classes) < 2 or not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError(
            "target must be encoded as integers 0..k-1 with at least two classes "
            f"(featurewiz label-encodes it before this stage); found {classes.tolist()}"
        )
    if len(classes) == 2:
        return {"objective": "binary:logistic", "num_class": 1, "min_child_weight": 0.5}
    return {"objective": "multi:softmax", "num_class": len(classes), "min_child_weight": 0.5}


def recursive_xgboost(
    frame: pd.DataFrame, features: list[str], target: str, tree_method: str | None = "exact"
) -> RecursiveXgbResult:
    """Recursive XGBoost, re-implemented faithfully from featurewiz 0.6.1.

    Column order matters (a Quirk): each Round trains on the suffix features[i:],
    so the first columns appear in fewer Rounds than the last ones.
    tree_method="exact" reproduces xgboost 1.7.6's choice; None uses the installed default.
    """
    features = list(features)
    y = frame[target]
    params = _xgb_params(y)
    if tree_method is not None:
        params["tree_method"] = tree_method
    step, top_num = round_plan(len(features))
    num_boost_round = 20 if len(frame) >= 100_000 else 100

    rounds, selected = [], []
    for i in range(0, len(features), step):
        if len(features) - i < 2:
            continue
        window = features[i:]
        dtrain = xgb.DMatrix(frame[window], y, enable_categorical=True, feature_names=window)
        booster = xgb.train(params, dtrain, num_boost_round=num_boost_round)
        gains = booster.get_score(fmap="", importance_type="total_gain")
        nominated = nominate(gains, top_num)
        selected = list(dict.fromkeys(selected + nominated))
        rounds.append(XgbRound(window, gains, nominated))
    return RecursiveXgbResult(step, top_num, rounds, selected)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_recursive_xgboost.py -v`
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
git add src/featurewiz_explained.py tests/test_recursive_xgboost.py
git commit -m "feat: faithful Recursive XGBoost re-implementation"
```

---

### Task 5: Pipeline, input checks, column shuffling and scoring

**Files:**
- Modify: `src/featurewiz_explained.py` (new sections after Recursive XGBoost)
- Test: `tests/test_pipeline_and_scoring.py`

**Interfaces:**
- Consumes: `sulov`, `SulovResult`, `recursive_xgboost`, `RecursiveXgbResult`, `SyntheticDataset`.
- Produces:
  - `Selection` (frozen dataclass): `sulov: SulovResult`, `xgb: RecursiveXgbResult`, property `selected -> list[str]`.
  - `select_features(frame: pd.DataFrame, target: str, corr_limit: float = 0.8, tree_method: str | None = "exact") -> Selection`. SULOV runs on all non-target columns in column order, then Recursive XGBoost runs on `sulov.kept` **in that order**, as `featurewiz()` does.
  - `shuffled_order(features: list[str], seed: int = 0) -> list[str]`.
  - `SignalGroupScore` (frozen): `covered: list[str]`, `missed: list[str]`, `redundant_extras: list[str]`, `noise_kept: list[str]`, property `perfect -> bool`.
  - `score_by_signal_group(selected: list[str], dataset: SyntheticDataset) -> SignalGroupScore`.
  - `RelevanceScore` (frozen): `relevant_kept: list[str]`, `relevant_total: int`, `noise_kept: list[str]`, `noise_total: int`.
  - `score_relevant_vs_noise(selected: list[str], dataset: SyntheticDataset) -> RelevanceScore`.

- [ ] **Step 1: Write the failing tests**

`tests/test_pipeline_and_scoring.py`:

```python
import numpy as np
import pandas as pd
import pytest

import featurewiz_explained as fwx


def test_select_features_feeds_sulov_output_order_into_recursive_xgboost():
    data = fwx.make_canonical_dataset()
    selection = fwx.select_features(data.frame, data.target)
    assert selection.xgb.rounds[0].window == selection.sulov.kept
    assert selection.selected == ["x1", "x2", "x3_copy"]


def test_select_features_rejects_column_names_with_spaces():
    frame = fwx.make_canonical_dataset().frame.rename(columns={"x1": "x 1"})
    with pytest.raises(ValueError, match="spaces"):
        fwx.select_features(frame, "target")


def test_select_features_rejects_missing_values():
    frame = fwx.make_canonical_dataset().frame
    frame.loc[0, "x2"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        fwx.select_features(frame, "target")


def test_select_features_rejects_non_numeric_columns():
    frame = fwx.make_canonical_dataset().frame.assign(colour="red")
    with pytest.raises(ValueError, match="numeric"):
        fwx.select_features(frame, "target")


def test_shuffled_order_is_a_deterministic_permutation_of_plain_strings():
    features = ["a", "b", "c", "d", "e"]
    order = fwx.shuffled_order(features, seed=0)
    assert sorted(order) == features and order != features
    assert order == fwx.shuffled_order(features, seed=0)
    assert all(type(f) is str for f in order)


def test_signal_group_score_does_not_punish_keeping_a_copy():
    data = fwx.make_canonical_dataset()
    score = fwx.score_by_signal_group(["x1", "x2", "x3_copy"], data)
    assert score.covered == ["x1", "x2", "x3"]
    assert score.missed == ["x4"]
    assert score.redundant_extras == [] and score.noise_kept == []
    assert not score.perfect


def test_signal_group_score_counts_extras_and_noise():
    data = fwx.make_canonical_dataset()
    score = fwx.score_by_signal_group(["x1", "x1_copy", "x2", "x3", "x4_squared", "noise_2"], data)
    assert score.missed == [] and score.redundant_extras == ["x1_copy"] and score.noise_kept == ["noise_2"]
    perfect = fwx.score_by_signal_group(["x1", "x2_copy", "x3", "x4_squared"], data)
    assert perfect.perfect


def test_signal_group_score_needs_groups():
    with pytest.raises(ValueError, match="score_relevant_vs_noise"):
        fwx.score_by_signal_group(["f000"], fwx.make_madelon_style_dataset(n_rows=50))


def test_relevance_score():
    data = fwx.make_madelon_style_dataset(n_rows=50)
    score = fwx.score_relevant_vs_noise(["f000", "f007", "f150", "f299"], data)
    assert score.relevant_kept == ["f000", "f007"] and score.relevant_total == 20
    assert score.noise_kept == ["f150", "f299"] and score.noise_total == 280
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pipeline_and_scoring.py -v`
Expected: FAIL with `AttributeError: ... no attribute 'select_features'`.

- [ ] **Step 3: Write the implementation**

Append:

```python
# --------------------------------------------------------------------------- pipeline

@dataclass(frozen=True)
class Selection:
    sulov: SulovResult
    xgb: RecursiveXgbResult

    @property
    def selected(self) -> list[str]:
        return self.xgb.selected


def _check_supported(frame: pd.DataFrame, features: list[str]) -> None:
    """Refuse inputs where real featurewiz() would transform the data first."""
    spaced = [f for f in features if " " in f]
    if spaced:
        raise ValueError(
            f"column names contain spaces: {spaced}. featurewiz strips spaces from names, "
            "so rename them (e.g. with underscores) to compare against featurewiz."
        )
    non_numeric = [f for f in features if not pd.api.types.is_numeric_dtype(frame[f])]
    if non_numeric:
        raise ValueError(
            f"non-numeric columns: {non_numeric}. featurewiz encodes categoricals first; "
            "that step is out of scope here."
        )
    missing = [f for f in features if frame[f].isna().any()]
    if missing:
        raise ValueError(
            f"columns with missing values: {missing}. featurewiz adds missing-value flag "
            "columns first; that step is out of scope here."
        )


def select_features(
    frame: pd.DataFrame, target: str, corr_limit: float = 0.8, tree_method: str | None = "exact"
) -> Selection:
    """featurewiz()'s selection: SULOV on every column, then Recursive XGBoost on
    SULOV's kept list in SULOV's output order."""
    features = [c for c in frame.columns if c != target]
    _check_supported(frame, features)
    sulov_result = sulov(frame, features, target, corr_limit)
    return Selection(sulov_result, recursive_xgboost(frame, sulov_result.kept, target, tree_method))


def shuffled_order(features: list[str], seed: int = 0) -> list[str]:
    """A reproducible reordering, used to show that Recursive XGBoost depends on column order."""
    return [str(f) for f in np.random.default_rng(seed).permutation(list(features))]


# --------------------------------------------------------------------------- scoring

@dataclass(frozen=True)
class SignalGroupScore:
    covered: list[str]
    missed: list[str]
    redundant_extras: list[str]
    noise_kept: list[str]

    @property
    def perfect(self) -> bool:
        return not (self.missed or self.redundant_extras or self.noise_kept)


def score_by_signal_group(selected: list[str], dataset: SyntheticDataset) -> SignalGroupScore:
    """Perfect = exactly one feature per Signal group and no noise. Which member doesn't matter."""
    if not dataset.groups:
        raise ValueError("this dataset has no Signal groups; use score_relevant_vs_noise instead")
    group_names = list(dict.fromkeys(dataset.groups.values()))
    kept = {g: [f for f in selected if dataset.groups.get(f) == g] for g in group_names}
    return SignalGroupScore(
        covered=[g for g in group_names if kept[g]],
        missed=[g for g in group_names if not kept[g]],
        redundant_extras=[f for g in group_names for f in kept[g][1:]],
        noise_kept=[f for f in selected if dataset.roles.get(f) == "noise"],
    )


@dataclass(frozen=True)
class RelevanceScore:
    relevant_kept: list[str]
    relevant_total: int
    noise_kept: list[str]
    noise_total: int


def score_relevant_vs_noise(selected: list[str], dataset: SyntheticDataset) -> RelevanceScore:
    """For datasets whose redundant features mix several informative ones (Madelon-style)."""
    relevant = {f for f, role in dataset.roles.items() if role != "noise"}
    return RelevanceScore(
        relevant_kept=[f for f in selected if f in relevant],
        relevant_total=len(relevant),
        noise_kept=[f for f in selected if f not in relevant],
        noise_total=len(dataset.roles) - len(relevant),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pipeline_and_scoring.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/featurewiz_explained.py tests/test_pipeline_and_scoring.py
git commit -m "feat: selection pipeline, input checks and ground-truth scoring"
```

---

### Task 6: Reference runs and comparison

**Files:**
- Create: `requirements-reference.txt`, `requirements-pinned.txt`, `scripts/make_reference_runs.py`, `reference_runs/reference_{canonical,canonical_shuffled,breast_cancer,madelon_style}.json` and `reference_runs/manifest.json` (generated), `tests/test_reference.py`, `tests/test_matches_reference.py`
- Modify: `src/featurewiz_explained.py` (new section at the end)

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces:
  - `REFERENCE_RUNS = ("canonical", "canonical_shuffled", "breast_cancer", "madelon_style")`.
  - `reference_frame(name: str) -> tuple[pd.DataFrame, str]`, returning the input data and target for a Reference run.
  - `ReferenceRun` (frozen): `name: str`, `stage: str` (`"pipeline"` or `"recursive_xgboost"`), `versions: dict[str, str]`, `input_features: list[str]`, `sulov_kept: list[str] | None`, `rounds: list[dict]` (each `{"window": list[str], "gains": dict[str, float]}`), `selected: list[str]`.
  - `load_reference(name: str) -> ReferenceRun`.
  - `run_like_reference(reference: ReferenceRun, tree_method: str | None = "exact") -> Selection | RecursiveXgbResult`.
  - `matches_reference_environment(reference: ReferenceRun) -> bool`.
  - `ReferenceComparison` (frozen): `sulov_matches: bool | None`, `windows_match: bool`, `max_relative_gain_diff: float`, `selected_matches: bool`, `only_ours: list[str]`, `only_reference: list[str]`, property `exact -> bool`.
  - `compare_to_reference(ours: Selection | RecursiveXgbResult, reference: ReferenceRun) -> ReferenceComparison`.

- [ ] **Step 1: Write the environment files**

`requirements-reference.txt` (pinned environment *with* the real library, Python 3.11):

```
featurewiz==0.6.1
xgboost==1.7.6
scikit-learn==1.5.2
numpy==1.26.4
pandas==2.2.3
pyarrow==16.1.0
scipy==1.17.1
plotly
```

`requirements-pinned.txt` (CI exact-match job; the same numerics, without featurewiz's heavy dependencies):

```
xgboost==1.7.6
scikit-learn==1.5.2
numpy==1.26.4
pandas==2.2.3
scipy==1.17.1
joblib==1.6.0
threadpoolctl==3.7.0
plotly
pytest
```

- [ ] **Step 2: Write the failing unit tests for loading and comparison**

`tests/test_reference.py`:

```python
import json

import pytest

import featurewiz_explained as fwx


def _write(tmp_path, **overrides):
    data = {
        "name": "toy", "stage": "recursive_xgboost",
        "versions": {"featurewiz": "0.6.1", "xgboost": "0.0.0", "scikit-learn": "0.0.0"},
        "input_features": ["a", "b"], "sulov_kept": None,
        "rounds": [{"window": ["a", "b"], "gains": {"a": 2.0, "b": 1.0}}],
        "selected": ["a", "b"],
    } | overrides
    (tmp_path / "reference_toy.json").write_text(json.dumps(data))
    (tmp_path / "breast_cancer.csv").write_text("x\n")
    return data


def test_load_reference_round_trips_json(monkeypatch, tmp_path):
    _write(tmp_path)
    monkeypatch.setenv("FWX_REFERENCE_DIR", str(tmp_path))
    ref = fwx.load_reference("toy")
    assert ref.stage == "recursive_xgboost" and ref.rounds[0]["gains"] == {"a": 2.0, "b": 1.0}
    assert not fwx.matches_reference_environment(ref)


def test_compare_exact_and_different(monkeypatch, tmp_path):
    _write(tmp_path)
    monkeypatch.setenv("FWX_REFERENCE_DIR", str(tmp_path))
    ref = fwx.load_reference("toy")
    same = fwx.RecursiveXgbResult(2, 2, [fwx.XgbRound(["a", "b"], {"a": 2.0, "b": 1.0}, ["a", "b"])], ["a", "b"])
    assert fwx.compare_to_reference(same, ref).exact

    other = fwx.RecursiveXgbResult(2, 2, [fwx.XgbRound(["a", "b"], {"a": 3.0}, ["a", "c"])], ["a", "c"])
    comparison = fwx.compare_to_reference(other, ref)
    assert not comparison.exact
    assert comparison.windows_match and not comparison.selected_matches
    assert comparison.max_relative_gain_diff == pytest.approx(1.0)  # b: 0 vs 1
    assert comparison.only_ours == ["c"] and comparison.only_reference == ["b"]
    assert comparison.sulov_matches is None


def test_every_reference_run_is_present_and_pinned_to_featurewiz_0_6_1():
    for name in fwx.REFERENCE_RUNS:
        ref = fwx.load_reference(name)
        assert ref.versions["featurewiz"] == "0.6.1"
        assert ref.versions["xgboost"] == "1.7.6"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_reference.py -v`
Expected: FAIL with `AttributeError: ... no attribute 'load_reference'`.

- [ ] **Step 4: Write the module implementation**

Add `import json` at the top, then append:

```python
# --------------------------------------------------------------------------- Reference runs

REFERENCE_RUNS = ("canonical", "canonical_shuffled", "breast_cancer", "madelon_style")


def reference_frame(name: str) -> tuple[pd.DataFrame, str]:
    """The exact data a Reference run was computed on."""
    if name in ("canonical", "canonical_shuffled"):
        data = make_canonical_dataset()
        return data.frame, data.target
    if name == "madelon_style":
        data = make_madelon_style_dataset()
        return data.frame, data.target
    if name == "breast_cancer":
        return load_breast_cancer(), "diagnosis"
    raise ValueError(f"unknown Reference run {name!r}; expected one of {REFERENCE_RUNS}")


@dataclass(frozen=True)
class ReferenceRun:
    name: str
    stage: str  # "pipeline" (featurewiz()) or "recursive_xgboost" (that stage alone)
    versions: dict[str, str]
    input_features: list[str]
    sulov_kept: list[str] | None
    rounds: list[dict]
    selected: list[str]


def load_reference(name: str) -> ReferenceRun:
    return ReferenceRun(**json.loads((reference_dir() / f"reference_{name}.json").read_text()))


def run_like_reference(reference: ReferenceRun, tree_method: str | None = "exact") -> Selection | RecursiveXgbResult:
    """Run the Re-implementation on the same input, in the same column order, as a Reference run."""
    frame, target = reference_frame(reference.name)
    if reference.stage == "pipeline":
        return select_features(frame[[*reference.input_features, target]], target, tree_method=tree_method)
    return recursive_xgboost(frame, reference.input_features, target, tree_method)


def matches_reference_environment(reference: ReferenceRun) -> bool:
    """True when xgboost and scikit-learn are the versions the Reference run used,
    which is the only setting where an exact match is expected."""
    import sklearn

    return (xgb.__version__, sklearn.__version__) == (
        reference.versions["xgboost"], reference.versions["scikit-learn"]
    )


@dataclass(frozen=True)
class ReferenceComparison:
    sulov_matches: bool | None  # None when the Reference run has no SULOV stage
    windows_match: bool
    max_relative_gain_diff: float
    selected_matches: bool
    only_ours: list[str]
    only_reference: list[str]

    @property
    def exact(self) -> bool:
        return (
            self.sulov_matches is not False
            and self.windows_match
            and self.max_relative_gain_diff == 0.0
            and self.selected_matches
        )


def compare_to_reference(ours: Selection | RecursiveXgbResult, reference: ReferenceRun) -> ReferenceComparison:
    xgb_result = ours.xgb if isinstance(ours, Selection) else ours
    sulov_matches = None
    if reference.sulov_kept is not None:
        sulov_matches = isinstance(ours, Selection) and ours.sulov.kept == reference.sulov_kept

    windows_match = [r.window for r in xgb_result.rounds] == [r["window"] for r in reference.rounds]
    diffs = [0.0]
    for mine, theirs in zip(xgb_result.rounds, reference.rounds):
        for feature in set(mine.gains) | set(theirs["gains"]):
            a, b = mine.gains.get(feature, 0.0), theirs["gains"].get(feature, 0.0)
            if a != b:
                diffs.append(abs(a - b) / max(abs(b), 1e-12))
    return ReferenceComparison(
        sulov_matches=sulov_matches,
        windows_match=windows_match,
        max_relative_gain_diff=max(diffs),
        selected_matches=xgb_result.selected == reference.selected,
        only_ours=[f for f in xgb_result.selected if f not in reference.selected],
        only_reference=[f for f in reference.selected if f not in xgb_result.selected],
    )
```

- [ ] **Step 5: Write the Reference-run script**

`scripts/make_reference_runs.py`:

```python
"""Produce Reference runs: run the REAL featurewiz 0.6.1 and record what it did.

Only run this in the pinned environment (Python 3.11):
    uv venv -p 3.11 .venv-reference
    uv pip install -p .venv-reference -r requirements-reference.txt
    uv pip install -p .venv-reference -e . --no-deps
    .venv-reference/bin/python scripts/make_reference_runs.py

featurewiz returns only final lists, so we record intermediates by wrapping
its SULOV function and xgboost.train while featurewiz runs (ADR 0001, 0002).
"""
import contextlib
import io
import json
import sys
import warnings
from importlib.metadata import version

import numpy
import pandas
import sklearn
import xgboost

import featurewiz  # noqa: E402  (registers featurewiz.featurewiz in sys.modules)
from featurewiz import settings

import featurewiz_explained as fwx

# `import featurewiz.featurewiz` resolves to the featurewiz() *function*; we need the module.
FW = sys.modules["featurewiz.featurewiz"]
VERSIONS = {
    "featurewiz": version("featurewiz"),
    "xgboost": xgboost.__version__,
    "scikit-learn": sklearn.__version__,
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
}


class Recorder:
    """Wrap featurewiz's SULOV function and xgboost.train to capture what happens inside."""

    def __enter__(self):
        self.sulov_kept, self.rounds = None, []
        self._sulov, self._train = FW.FE_remove_variables_using_SULOV_method, FW.xgb.train

        def sulov(*args, **kwargs):
            out = self._sulov(*args, **kwargs)
            self.sulov_kept = [str(f) for f in out]
            return out

        def train(params, dtrain, *args, **kwargs):
            booster = self._train(params, dtrain, *args, **kwargs)
            self.rounds.append({
                "window": list(dtrain.feature_names),
                "gains": booster.get_score(fmap="", importance_type="total_gain"),
            })
            return booster

        FW.FE_remove_variables_using_SULOV_method, FW.xgb.train = sulov, train
        return self

    def __exit__(self, *exc):
        FW.FE_remove_variables_using_SULOV_method, FW.xgb.train = self._sulov, self._train


def quietly(fn, *args, **kwargs):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def pipeline_run(name):
    frame, target = fwx.reference_frame(name)
    with Recorder() as rec:
        # featurewiz() returns (features, train_frame) when no test data is given
        selected = quietly(featurewiz.featurewiz, frame.copy(), target, corr_limit=0.8, verbose=0)[0]
    return {
        "name": name, "stage": "pipeline", "versions": VERSIONS,
        "input_features": [c for c in frame.columns if c != target],
        "sulov_kept": rec.sulov_kept, "rounds": rec.rounds, "selected": [str(f) for f in selected],
    }


def recursive_xgboost_run(name, features):
    frame, target = fwx.reference_frame(name)
    settings.multi_label = False
    with Recorder() as rec:
        selected = quietly(
            FW.FE_perform_recursive_xgboost, frame[[*features, target]], target, "Binary_Classification", False
        )
    return {
        "name": name, "stage": "recursive_xgboost", "versions": VERSIONS,
        "input_features": features, "sulov_kept": None, "rounds": rec.rounds,
        "selected": [str(f) for f in selected],
    }


def main():
    out = fwx.reference_dir()
    runs = {name: pipeline_run(name) for name in ("canonical", "breast_cancer", "madelon_style")}
    # The column-order Quirk: real featurewiz's own SULOV output for the Canonical dataset, shuffled.
    runs["canonical_shuffled"] = recursive_xgboost_run(
        "canonical_shuffled", fwx.shuffled_order(runs["canonical"]["sulov_kept"], seed=0)
    )
    for name, run in runs.items():
        (out / f"reference_{name}.json").write_text(json.dumps(run, indent=1) + "\n")
        print(name, "->", run["selected"][:10], f"({len(run['selected'])} selected)")
    (out / "manifest.json").write_text(json.dumps({"versions": VERSIONS, "runs": sorted(runs)}, indent=1) + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Generate the Reference runs**

Run:
```bash
uv venv -p 3.11 .venv-reference
uv pip install -p .venv-reference -r requirements-reference.txt
uv pip install -p .venv-reference -e . --no-deps
.venv-reference/bin/python scripts/make_reference_runs.py
```
Expected (this takes about 2 minutes; Madelon-style dominates):
```
canonical -> ['x1', 'x2', 'x3_copy'] (3 selected)
breast_cancer -> ['perimeter_worst', 'smoothness_worst', 'texture_worst', 'area_se', 'compactness_worst', 'fractal_dimension_se'] (6 selected)
madelon_style -> [...] (161 selected)
canonical_shuffled -> ['x1', 'x3_copy', 'x2', 'noise_2', 'x4', 'x4_squared'] (6 selected)
```
These values were observed in the prototype on 2026-10-22. If `canonical` differs, stop and investigate before continuing.

- [ ] **Step 7: Write the reference-match tests**

`tests/test_matches_reference.py`:

```python
"""The Re-implementation against Reference runs of real featurewiz 0.6.1 (ADR 0002).

Pinned environment: must match exactly. Current environment: write a report, never fail.
"""
from pathlib import Path

import pytest
import sklearn
import xgboost

import featurewiz_explained as fwx

REPORT = Path(__file__).resolve().parents[1] / "build" / "reference-diff.md"


@pytest.mark.parametrize("name", fwx.REFERENCE_RUNS)
def test_matches_reference_exactly_in_pinned_environment(name):
    reference = fwx.load_reference(name)
    if not fwx.matches_reference_environment(reference):
        pytest.skip("not the pinned environment; differences are reported instead")
    comparison = fwx.compare_to_reference(fwx.run_like_reference(reference), reference)
    assert comparison.exact, comparison


def test_report_differences_in_current_environment():
    references = [fwx.load_reference(name) for name in fwx.REFERENCE_RUNS]
    if all(fwx.matches_reference_environment(r) for r in references):
        pytest.skip("pinned environment; exact matching is tested instead")
    lines = [
        "# Re-implementation vs Reference runs (current environment)",
        "",
        f"xgboost {xgboost.__version__}, scikit-learn {sklearn.__version__}; "
        f"Reference runs used xgboost {references[0].versions['xgboost']}, "
        f"scikit-learn {references[0].versions['scikit-learn']}.",
        "",
        "| run | SULOV matches | windows match | max relative gain diff | selection matches | only ours | only reference |",
        "|---|---|---|---|---|---|---|",
    ]
    for reference in references:
        c = fwx.compare_to_reference(fwx.run_like_reference(reference), reference)
        lines.append(
            f"| {reference.name} | {c.sulov_matches} | {c.windows_match} | {c.max_relative_gain_diff:.3g} "
            f"| {c.selected_matches} | {len(c.only_ours)} | {len(c.only_reference)} |"
        )
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
```

- [ ] **Step 8: Run the tests in both environments**

Run: `.venv/bin/pytest tests/test_reference.py tests/test_matches_reference.py -v && cat build/reference-diff.md`
Expected: `test_reference.py` 3 passed. The 4 exact tests SKIPPED; the report test passes. In the report, `canonical` and `canonical_shuffled` show `selection matches = True`, and `madelon_style` shows `False` (prototype: xgboost 3.x picks a slightly different set).

Then:
```bash
uv venv -p 3.11 .venv-pinned
uv pip install -p .venv-pinned -r requirements-pinned.txt
uv pip install -p .venv-pinned -e . --no-deps
.venv-pinned/bin/pytest -v
```
Expected: every test passes, including the 4 exact matches; the report test is skipped.

- [ ] **Step 9: Commit**

```bash
git add requirements-reference.txt requirements-pinned.txt scripts/make_reference_runs.py reference_runs/ src/featurewiz_explained.py tests/test_reference.py tests/test_matches_reference.py
git commit -m "feat: Reference runs of featurewiz 0.6.1 and exact-match comparison"
```

---

### Task 7: Plots module

**Files:**
- Create: `src/featurewiz_explained_plots.py`, `tests/test_plots.py`

**Interfaces:**
- Consumes: `SulovResult`, `RecursiveXgbResult`, `SyntheticDataset`.
- Produces (all return `plotly.graph_objects.Figure`):
  - `correlation_heatmap(frame: pd.DataFrame, features: list[str], corr_limit: float = 0.8)`
  - `sulov_graph(result: SulovResult, frame: pd.DataFrame)`
  - `rounds_heatmap(result: RecursiveXgbResult)`
  - `selection_table(selected: list[str], dataset: SyntheticDataset)`

Design rules (dataviz reference palette):
- Sequential magnitude (|corr|, gain share) uses the single-hue `Blues` scale.
- Kept vs dropped uses blue `#2a78d6` vs neutral `#a3a29c`, and every node is also *labelled* "(kept)"/"(dropped)".
- Roles use informative `#2a78d6`, redundant `#eb6834`, noise `#1baf7a`, always next to the role name.
- Hover shows the exact value. Template `plotly_white`.

- [ ] **Step 1: Write the failing tests**

`tests/test_plots.py`:

```python
import plotly.graph_objects as go

import featurewiz_explained as fwx
import featurewiz_explained_plots as plots


def test_correlation_heatmap_is_absolute_and_square():
    data = fwx.make_canonical_dataset()
    fig = plots.correlation_heatmap(data.frame, data.features)
    assert isinstance(fig, go.Figure)
    heat = fig.data[0]
    assert heat.type == "heatmap" and len(heat.z) == 12 and len(heat.z[0]) == 12
    assert min(min(row) for row in heat.z) >= 0


def test_sulov_graph_has_one_edge_per_pair_and_labels_every_node():
    data = fwx.make_canonical_dataset()
    result = fwx.sulov(data.frame, data.features, data.target)
    fig = plots.sulov_graph(result, data.frame)
    edges = [t for t in fig.data if t.mode == "lines"]
    nodes = [t for t in fig.data if t.mode == "markers+text"]
    assert len(edges) == len(result.correlated_pairs)
    labels = [label for t in nodes for label in t.text]
    assert sorted(labels) == sorted(
        [f"{f} (kept)" for f in result.mutual_info if f in result.kept]
        + [f"{f} (dropped)" for f in result.mutual_info if f in result.dropped]
    )


def test_rounds_heatmap_marks_nominations():
    data = fwx.make_canonical_dataset()
    kept = fwx.sulov(data.frame, data.features, data.target).kept
    result = fwx.recursive_xgboost(data.frame, kept, data.target)
    heat = plots.rounds_heatmap(result).data[0]
    assert list(heat.y) == kept
    assert len(heat.x) == len(result.rounds)
    assert sum(cell == "★" for row in heat.text for cell in row) == sum(len(r.nominated) for r in result.rounds)


def test_selection_table_lists_every_feature_with_its_role():
    data = fwx.make_canonical_dataset()
    table = plots.selection_table(["x1", "x2"], data).data[0]
    feature_col, role_col, kept_col = table.cells.values[0], table.cells.values[1], table.cells.values[3]
    assert list(feature_col) == data.features
    assert role_col[0] == "informative"
    assert kept_col[0] == "✓ kept" and kept_col[2] == "—"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_plots.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'featurewiz_explained_plots'`.

- [ ] **Step 3: Write the implementation**

`src/featurewiz_explained_plots.py`:

```python
"""Plotly figures for the featurewiz, explained Lessons.

Static Plotly only (no widgets), so charts work in Kaggle's read-only view.
Colours follow the dataviz reference palette; identity always has a text label too.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from featurewiz_explained import RecursiveXgbResult, SulovResult, SyntheticDataset

ROLE_COLOURS = {"informative": "#2a78d6", "redundant": "#eb6834", "noise": "#1baf7a"}
KEPT, DROPPED = "#2a78d6", "#a3a29c"
TEMPLATE = "plotly_white"


def correlation_heatmap(frame: pd.DataFrame, features: list[str], corr_limit: float = 0.8) -> go.Figure:
    corr = frame[features].corr(method="pearson").abs()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=features, y=features, zmin=0, zmax=1, colorscale="Blues",
        colorbar=dict(title="|corr|"),
        hovertemplate="%{y} vs %{x}<br>|corr| = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        template=TEMPLATE, title=f"Absolute Pearson correlation (SULOV pairs need ≥ {corr_limit})",
        yaxis=dict(autorange="reversed"), height=520,
    )
    return fig


def sulov_graph(result: SulovResult, frame: pd.DataFrame) -> go.Figure:
    """Features that are in a correlated pair, on a circle. Edges = pairs. Size = mutual information."""
    nodes = list(result.mutual_info)
    angle = {f: 2 * math.pi * i / max(len(nodes), 1) for i, f in enumerate(nodes)}
    pos = {f: (math.cos(a), math.sin(a)) for f, a in angle.items()}
    corr = frame[nodes].corr(method="pearson").abs() if nodes else None
    fig = go.Figure()
    for a, b in result.correlated_pairs:
        c = corr.loc[a, b]
        fig.add_trace(go.Scatter(
            x=[pos[a][0], pos[b][0]], y=[pos[a][1], pos[b][1]], mode="lines",
            line=dict(color="#c3c2b7", width=1 + 4 * c), hoverinfo="text",
            text=f"{a} – {b}: |corr| = {c:.3f}", showlegend=False,
        ))
    biggest = max(result.mutual_info.values(), default=1) or 1
    for label, colour, members in (
        ("kept", KEPT, [f for f in nodes if f in result.kept]),
        ("dropped", DROPPED, [f for f in nodes if f in result.dropped]),
    ):
        fig.add_trace(go.Scatter(
            x=[pos[f][0] for f in members], y=[pos[f][1] for f in members], mode="markers+text",
            name=label, text=[f"{f} ({label})" for f in members], textposition="top center",
            marker=dict(size=[14 + 36 * result.mutual_info[f] / biggest for f in members],
                        color=colour, line=dict(color="white", width=2)),
            customdata=[result.mutual_info[f] for f in members],
            hovertemplate="%{text}<br>mutual information = %{customdata:.4f}<extra></extra>",
        ))
    fig.update_layout(
        template=TEMPLATE, title="SULOV: bigger circle = more mutual information; thicker line = stronger correlation",
        xaxis=dict(visible=False, range=[-1.5, 1.5]), yaxis=dict(visible=False, range=[-1.4, 1.5], scaleanchor="x"),
        height=560,
    )
    return fig


def rounds_heatmap(result: RecursiveXgbResult) -> go.Figure:
    """Rows = features (input order), columns = Rounds. Cell = share of that Round's total gain.
    Blank = not in that Round's window. ★ = nominated."""
    features = result.rounds[0].window if result.rounds else []
    z, text = [], []
    for f in features:
        row_z, row_t = [], []
        for r in result.rounds:
            total = sum(r.gains.values()) or 1.0
            row_z.append(r.gains.get(f, 0.0) / total if f in r.window else np.nan)
            row_t.append("★" if f in r.nominated else "")
        z.append(row_z)
        text.append(row_t)
    labels = [f"Round {i + 1}" for i in range(len(result.rounds))]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=features, text=text, texttemplate="%{text}", colorscale="Blues", zmin=0,
        colorbar=dict(title="gain share"),
        hovertemplate="%{y}, %{x}<br>share of gain = %{z:.1%}<extra></extra>",
    ))
    fig.update_layout(
        template=TEMPLATE, height=max(320, 26 * len(features)),
        title=f"Recursive XGBoost: step {result.step}, top {result.top_num} per Round (★ = nominated, blank = outside window)",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def selection_table(selected: list[str], dataset: SyntheticDataset) -> go.Figure:
    features = dataset.features
    roles = [dataset.roles[f] for f in features]
    fig = go.Figure(go.Table(
        header=dict(values=["feature", "planted role", "signal group", "featurewiz"], align="left"),
        cells=dict(
            values=[features, roles, [dataset.groups.get(f, "") for f in features],
                    ["✓ kept" if f in selected else "—" for f in features]],
            fill_color=[["white"] * len(features), [ROLE_COLOURS[r] for r in roles]],
            font=dict(color=[["#0b0b0b"] * len(features), ["white"] * len(features)]),
            align="left",
        ),
    ))
    fig.update_layout(template=TEMPLATE, height=60 + 28 * len(features), margin=dict(t=10, b=10))
    return fig
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_plots.py -v`
Expected: 4 passed.

- [ ] **Step 5: Look at the charts**

Run: `.venv/bin/python -c "import featurewiz_explained as fwx, featurewiz_explained_plots as p; d=fwx.make_canonical_dataset(); s=fwx.sulov(d.frame,d.features,d.target); p.sulov_graph(s,d.frame).write_html('build/sulov.html'); p.rounds_heatmap(fwx.recursive_xgboost(d.frame,s.kept,d.target)).write_html('build/rounds.html')"` (create `build/` first). Open both files in a browser and check that labels don't collide and hover works. Fix only layout numbers (ranges, heights, text positions) if needed.

- [ ] **Step 6: Commit**

```bash
git add src/featurewiz_explained_plots.py tests/test_plots.py
git commit -m "feat: Plotly figures for the Lessons"
```

---

### Lesson authoring conventions (applies to Tasks 8–11)

- Author each Lesson as a jupytext *percent* file in `build/`, then convert it and commit only the `.ipynb`:
  `.venv/bin/jupytext --to ipynb --output notebooks/NN-name.ipynb build/NN-name.py`
- After any edit, strip outputs with `.venv/bin/nbstripout notebooks/NN-name.ipynb`. Run `.venv/bin/nbstripout --install --attributes .gitattributes` once, in Task 8, and commit `.gitattributes`.
- Every Lesson starts with the **same bootstrap cell**, which finds the module in an attached Kaggle dataset wherever Kaggle mounts it. Locally, the editable install makes the import work anyway:

```python
import pathlib, sys

for module_file in pathlib.Path("/kaggle/input").glob("**/featurewiz_explained.py"):
    sys.path.insert(0, str(module_file.parent))
    break

import featurewiz_explained as fwx
import featurewiz_explained_plots as plots
```

- The second cell is always this markdown, with the topic and links filled in for that Lesson:

```markdown
> **featurewiz, explained**: a four-part series that rebuilds featurewiz 0.6.1's feature selection step by step.
> 1 Intuition · 2 SULOV · 3 Recursive XGBoost · 4 Full pipeline.
> Everything here runs with Internet **off**. Results are checked against Reference runs of the real featurewiz 0.6.1.
```

- Narrative markdown states numbers **only after you've run the cell and seen them**. The numbers quoted below come from the prototype; if your run differs, write what you observe.
- Execute each Lesson to check it: `.venv/bin/pytest --nbmake --nbmake-timeout=900 notebooks/NN-name.ipynb`.

---

### Task 8: Lesson 1, Intuition

**Files:**
- Create: `notebooks/01-intuition.ipynb`, `.gitattributes`

**Interfaces:**
- Consumes: `make_canonical_dataset`, `select_features`, `plots.correlation_heatmap`, `plots.selection_table`.

- [ ] **Step 1: Write `build/01-intuition.py`**

```python
# %% [markdown]
# # featurewiz, explained 1/4: Intuition
# featurewiz picks features in two stages: **SULOV** throws out redundant features,
# then **Recursive XGBoost** keeps the ones a model actually uses. Before opening
# either stage, let's see the problem they solve, on data where we *know* the answer.

# %%
# (bootstrap cell, exactly as in the Lesson authoring conventions)

# %% [markdown]
# (series banner, exactly as in the Lesson authoring conventions)

# %% [markdown]
# ## The Canonical dataset
# We'll use this dataset in every Lesson. Its features were built on purpose, so each one has a
# **planted role**:
# - `x1`–`x4` are **informative**: the target depends on them.
# - `x1_copy`–`x3_copy` are **redundant**: the original plus a little noise.
# - `x4_squared` is **redundant** too, but a *nonlinear* copy. Remember it.
# - `noise_1`–`noise_4` are **noise**: unrelated to the target.

# %%
data = fwx.make_canonical_dataset()
plots.selection_table([], data).show()
data.frame.head()

# %% [markdown]
# ## Redundant features show up as bright off-diagonal squares

# %%
plots.correlation_heatmap(data.frame, data.features).show()

# %% [markdown]
# Each `x` and its `_copy` correlate above 0.9. `x4` and `x4_squared` barely correlate
# linearly, even though one is computed from the other.
#
# ## Why bother? Redundancy splits importance
# Train one XGBoost model on everything and look at how it spreads importance:

# %%
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score

model = xgb.XGBClassifier(n_estimators=200, max_depth=3, random_state=0, tree_method="exact")
model.fit(data.frame[data.features], data.frame[data.target])
gain = pd.Series(model.get_booster().get_score(importance_type="total_gain")).reindex(data.features, fill_value=0)
(gain / gain.sum()).round(3).sort_values(ascending=False).to_frame("share of gain")

# %% [markdown]
# `x1`'s importance is shared with `x1_copy`, so a reader of this chart would underrate both.
# Feature selection should keep **one member of each group** and drop the noise.
#
# ## Why bother? Noise and redundancy cost accuracy on small data

# %%
small = data.frame.sample(300, random_state=0)
informative = [f for f in data.features if data.roles[f] == "informative"]
for name, cols in [("all 12 features", data.features), ("only the 4 informative", informative)]:
    scores = cross_val_score(
        xgb.XGBClassifier(n_estimators=200, max_depth=3, random_state=0, tree_method="exact"),
        small[cols], small[data.target], cv=5, scoring="roc_auc",
    )
    print(f"{name:>24}: AUC {scores.mean():.3f} ± {scores.std():.3f}")

# %% [markdown]
# (Write what you observed in the previous cell: the AUC gap and the spread. Don't claim a gap you didn't see.)
#
# ## What featurewiz does about it
# The next three Lessons rebuild it piece by piece. Here's the whole thing in one call,
# using our Re-implementation (checked against the real featurewiz 0.6.1):

# %%
selection = fwx.select_features(data.frame, data.target)
print(f"{len(data.features)} features -> SULOV keeps {len(selection.sulov.kept)} -> Recursive XGBoost keeps {len(selection.selected)}")
plots.selection_table(selection.selected, data).show()

# %% [markdown]
# It kept `x1`, `x2` and `x3_copy` (a copy is fine: it carries the same signal), and **missed `x4`**.
# Lesson 2 shows how SULOV chose between originals and copies. Lesson 3 shows why
# Recursive XGBoost let `x4` slip away.
```

Replace the two placeholder cells (bootstrap, banner) with the exact content from the conventions section before converting. Replace the "Write what you observed" line with real observations after running.

- [ ] **Step 2: Convert, run and strip**

Run:
```bash
.venv/bin/nbstripout --install --attributes .gitattributes
.venv/bin/jupytext --to ipynb --output notebooks/01-intuition.ipynb build/01-intuition.py
.venv/bin/pytest --nbmake --nbmake-timeout=900 notebooks/01-intuition.ipynb
```
Expected: `1 passed`. Then open the notebook (`.venv/bin/jupyter lab` or VS Code), run it, write the observed-AUC sentence, and save. Run `.venv/bin/nbstripout notebooks/01-intuition.ipynb` and the nbmake command again; it should pass.

- [ ] **Step 3: Commit**

```bash
git add .gitattributes notebooks/01-intuition.ipynb
git commit -m "docs: Lesson 1 Intuition"
```

---

### Task 9: Lesson 2, SULOV step by step

**Files:**
- Create: `notebooks/02-sulov.ipynb`

**Interfaces:**
- Consumes: `make_canonical_dataset`, `sulov`, `load_reference`, `load_breast_cancer`, `BREAST_CANCER_FEATURES`, `plots.sulov_graph`, `plots.correlation_heatmap`.

- [ ] **Step 1: Write `build/02-sulov.py`**

```python
# %% [markdown]
# # featurewiz, explained 2/4: SULOV, step by step
# **SULOV** ("Searching for Uncorrelated List Of Variables", created by Ram Seshadri in 2018)
# is featurewiz's first stage. In one sentence: *for every pair of highly correlated features, keep
# the one that tells you more about the target.* We'll rebuild it in three steps and check
# each step against the real featurewiz 0.6.1.

# %%
# (bootstrap cell)

# %% [markdown]
# (series banner)

# %%
data = fwx.make_canonical_dataset()
features, target = data.features, data.target

# %% [markdown]
# ## Step 1: find correlated pairs
# A pair counts when the absolute Pearson correlation is **≥ 0.8**. Two Quirks:
# - featurewiz() uses 0.8, but the SULOV function's own default is 0.70. Which you get depends on how you call it.
# - The pairs are **sorted by name**, which affects ties later.

# %%
corr = data.frame[features].corr().abs()
pairs = sorted(
    (features[i], features[j])
    for i in range(len(features)) for j in range(i + 1, len(features))
    if corr.iloc[i, j] >= 0.8
)
pairs

# %% [markdown]
# Three pairs, one per linear copy. `x4`–`x4_squared` isn't a pair: its correlation is

# %%
round(corr.loc["x4", "x4_squared"], 3)

# %% [markdown]
# Pearson correlation only sees *straight-line* relationships, so SULOV can't catch a nonlinear copy.
#
# ## Step 2: score each paired feature by mutual information with the target
# Mutual information measures how much knowing a feature tells you about the target, including nonlinear
# relationships. featurewiz computes it only for features that are in some pair, with a fixed seed.

# %%
from sklearn.feature_selection import mutual_info_classif

paired = list(dict.fromkeys(f for pair in pairs for f in pair))
mi = dict(zip(paired, mutual_info_classif(
    data.frame[paired], data.frame[target], n_neighbors=5, discrete_features=False, random_state=42)))
{f: round(v, 4) for f, v in sorted(mi.items(), key=lambda kv: -kv[1])}

# %% [markdown]
# Look at `x3` and `x3_copy`: the **copy** scores slightly higher. The estimate is noisy, and the copy wins by chance.
#
# ## Step 3: walk down the ranking, knocking out partners
# Take the highest-scoring feature, keep it, and remove its partners from the list. Repeat.

# %%
walk = sorted(mi, key=mi.get, reverse=True)
kept_paired, analysed = [], set()
for feature in walk:  # featurewiz removes items from the list while walking it; so do we
    kept_paired.append(feature)
    for a, b in pairs:
        pair = tuple(sorted((a, b)))
        if feature in (a, b) and pair not in analysed:
            analysed.add(pair)
            partner = b if feature == a else a
            if partner in walk:
                walk.remove(partner)
                print(f"keep {feature:8} (MI {mi[feature]:.4f}) -> knock out {partner} (MI {mi[partner]:.4f})")

kept = [f for f in features if f not in paired] + kept_paired
kept

# %% [markdown]
# Features in no pair go first, in their original order, then the winners in ranking order.
# **This order matters in Lesson 3.**
#
# ## Check: our inline code, the module, and the real featurewiz

# %%
reference = fwx.load_reference("canonical")
module_result = fwx.sulov(data.frame, features, target)
assert kept == module_result.kept == reference.sulov_kept
print("inline == module == featurewiz 0.6.1:", kept)

# %%
plots.sulov_graph(module_result, data.frame).show()

# %% [markdown]
# ## On real data: Breast Cancer Wisconsin
# 30 measurements of cell nuclei. Radius, perimeter and area are nearly the same measurement three times over.
# One more Quirk: featurewiz **strips spaces from column names**, so our copy of this dataset uses
# `concave_points_mean` instead of `concave points_mean`.

# %%
bc = fwx.load_breast_cancer()
plots.correlation_heatmap(bc, fwx.BREAST_CANCER_FEATURES).show()

# %%
bc_result = fwx.sulov(bc, fwx.BREAST_CANCER_FEATURES, "diagnosis")
assert bc_result.kept == fwx.load_reference("breast_cancer").sulov_kept
print(f"SULOV keeps {len(bc_result.kept)} of 30:", bc_result.kept)
plots.sulov_graph(bc_result, bc).show()

# %% [markdown]
# ## What to remember
# - SULOV only sees **linear** redundancy (`x4_squared` got through).
# - Between copies it keeps the higher mutual information, which can be **the copy** (`x3_copy`).
# - Its output order (unpaired first, then by mutual information) becomes Recursive XGBoost's column order in Lesson 3.
```

Replace the bootstrap and banner placeholder cells with their exact content.

- [ ] **Step 2: Convert, run and strip**

Run:
```bash
.venv/bin/jupytext --to ipynb --output notebooks/02-sulov.ipynb build/02-sulov.py
.venv/bin/pytest --nbmake --nbmake-timeout=900 notebooks/02-sulov.ipynb
```
Expected: `1 passed`. The asserts inside the notebook *are* this Lesson's tests: they fail the run if the inline code, the module or the Reference run disagree.

- [ ] **Step 3: Commit**

```bash
git add notebooks/02-sulov.ipynb
git commit -m "docs: Lesson 2 SULOV step by step"
```

---

### Task 10: Lesson 3, Recursive XGBoost step by step

**Files:**
- Create: `notebooks/03-recursive-xgboost.ipynb`

**Interfaces:**
- Consumes: `make_canonical_dataset`, `sulov`, `round_plan`, `nominate`, `recursive_xgboost`, `load_reference`, `compare_to_reference`, `matches_reference_environment`, `plots.rounds_heatmap`.

- [ ] **Step 1: Write `build/03-recursive-xgboost.py`**

```python
# %% [markdown]
# # featurewiz, explained 3/4: Recursive XGBoost, step by step
# The second stage takes what SULOV kept and asks XGBoost which features it actually uses.
# It doesn't ask once. It runs several **Rounds**, each on a shrinking **window** of columns,
# and pools every Round's top features.

# %%
# (bootstrap cell)

# %% [markdown]
# (series banner)

# %%
data = fwx.make_canonical_dataset()
target = data.target
features = fwx.sulov(data.frame, data.features, target).kept  # SULOV's output order, as featurewiz passes it on
features

# %% [markdown]
# ## Step 1: plan the Rounds
# - **step**: how many columns each Round slides past (2 if there are ≤ 10 features, otherwise about n/5).
# - **top_num**: how many features each Round nominates at least (25% of n if n ≤ 50, otherwise 20%).

# %%
n = len(features)
step = 2 if n <= 10 else int(n / 5 + 0.5)
top_num = int(max(2, n * 0.25)) if n <= 50 else int(n * 0.20)
assert (step, top_num) == fwx.round_plan(n)
print(f"{n} features -> step {step}, top {top_num} per Round")

# %% [markdown]
# ## Step 2: run the Rounds
# Round *k* trains on `features[k*step:]`, a **suffix**. The window drops columns from the *front*, so the
# first columns are seen in one Round and the last ones in every Round. Hold on to that.
#
# Which features does a Round nominate? Those with at least **half the top feature's gain**, unless that
# gives fewer than `top_num`, in which case it takes the top `top_num` by gain.

# %%
import pandas as pd
import xgboost as xgb

params = {"objective": "binary:logistic", "num_class": 1, "min_child_weight": 0.5, "tree_method": "exact"}
selected, rounds = [], []
for i in range(0, n, step):
    window = features[i:]
    if len(window) < 2:
        continue
    booster = xgb.train(params, xgb.DMatrix(data.frame[window], data.frame[target], feature_names=window),
                        num_boost_round=100)
    gains = booster.get_score(importance_type="total_gain")
    nominated = fwx.nominate(gains, top_num)
    selected = list(dict.fromkeys(selected + nominated))
    rounds.append(window)
    print(f"Round {len(rounds)}: window starts at {window[0]!r:14} nominates {nominated}")
selected

# %%
result = fwx.recursive_xgboost(data.frame, features, target)
assert selected == result.selected
plots.rounds_heatmap(result).show()

# %% [markdown]
# **Why was `x4` missed?** `x4` and `x4_squared` sit at the *front* of the column order (SULOV puts unpaired
# features first), so they're only in Round 1. There, `x1`, `x2` and `x3_copy` all clear "half of the top gain",
# so that's three nominations, more than `top_num = 2`, and `x4` isn't one of them. In later Rounds `x4` isn't
# in the window at all.
#
# ## The Quirk: shuffle the columns, change the answer
# Same data, same features, same model. Only the column order differs:

# %%
shuffled = fwx.shuffled_order(features, seed=0)
shuffled_result = fwx.recursive_xgboost(data.frame, shuffled, target)
print("order:   ", shuffled)
print("selected:", shuffled_result.selected)
plots.rounds_heatmap(shuffled_result).show()

# %% [markdown]
# Now `x4` is found, along with `x4_squared` and `noise_2`. The real featurewiz does exactly the same:

# %%
reference = fwx.load_reference("canonical_shuffled")
assert reference.input_features == shuffled
comparison = fwx.compare_to_reference(shuffled_result, reference)
print("matches featurewiz 0.6.1 exactly:", comparison.exact)
comparison

# %% [markdown]
# ## A version note: why `tree_method="exact"`?
# The Reference runs used xgboost 1.7.6, where small datasets are trained with the *exact* tree method.
# xgboost 2.0 changed the default to *hist*. If you don't pin it, gains shift, and sometimes nominations do too:

# %%
import sklearn

print("this notebook: xgboost", xgb.__version__, "scikit-learn", sklearn.__version__)
print("reference runs:", reference.versions)
print("pinned environment:", fwx.matches_reference_environment(reference))
default_result = fwx.recursive_xgboost(data.frame, features, target, tree_method=None)
print("with this xgboost's default tree method:", default_result.selected)

# %% [markdown]
# (Write what you observed: did the default tree method change the selection here?)
#
# ## What to remember
# - Each Round sees a **suffix** of the columns, so **column order changes the result**.
# - Every Round nominates **at least `top_num`** features, even when they're noise.
# - Results depend on the xgboost version. Pin it if you need reproducibility.
```

Replace the placeholders as before. Write the observation line after running.

- [ ] **Step 2: Convert, run and strip**

Run:
```bash
.venv/bin/jupytext --to ipynb --output notebooks/03-recursive-xgboost.ipynb build/03-recursive-xgboost.py
.venv/bin/pytest --nbmake --nbmake-timeout=900 notebooks/03-recursive-xgboost.ipynb
```
Expected: `1 passed`. If `comparison.exact` prints `False` in the current environment, that's allowed (ADR 0002), and the markdown above the cell must say so. In the pinned environment it must print `True`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/03-recursive-xgboost.ipynb
git commit -m "docs: Lesson 3 Recursive XGBoost step by step"
```

---

### Task 11: Lesson 4, Full pipeline

**Files:**
- Create: `notebooks/04-full-pipeline.ipynb`

**Interfaces:**
- Consumes: `select_features`, `load_reference`, `run_like_reference`, `compare_to_reference`, `matches_reference_environment`, `score_by_signal_group`, `score_relevant_vs_noise`, `make_madelon_style_dataset`, `load_breast_cancer`, `plots.selection_table`, `plots.rounds_heatmap`.

- [ ] **Step 1: Write `build/04-full-pipeline.py`**

```python
# %% [markdown]
# # featurewiz, explained 4/4: Full pipeline
# SULOV, then Recursive XGBoost, as `featurewiz()` runs them. We'll check our version against the
# real featurewiz, score it against the planted truth, and finish on a much harder dataset.

# %%
# (bootstrap cell)

# %% [markdown]
# (series banner)

# %% [markdown]
# ## Our pipeline vs the real featurewiz 0.6.1

# %%
data = fwx.make_canonical_dataset()
selection = fwx.select_features(data.frame, data.target)
reference = fwx.load_reference("canonical")
comparison = fwx.compare_to_reference(selection, reference)
print("ours:      ", selection.selected)
print("featurewiz:", reference.selected)
print("pinned environment:", fwx.matches_reference_environment(reference), "| exact match:", comparison.exact)

# %% [markdown]
# The Reference runs were computed once with the real featurewiz 0.6.1 (and xgboost 1.7.6), because
# featurewiz's dependency pins clash with Kaggle's current image. In that pinned environment our version
# matches it **exactly**: every Round's gains to the last bit. Here on Kaggle, xgboost is newer, so gains
# can drift; the selection above shows whether it drifted for this dataset.
#
# ## Scoring against the planted truth, by Signal group
# Keeping `x3_copy` instead of `x3` isn't a mistake: both carry the same signal. So we score by
# **Signal group** (an informative feature plus its copies): a perfect selection keeps exactly one
# feature from each group and no noise.

# %%
score = fwx.score_by_signal_group(selection.selected, data)
score

# %%
plots.selection_table(selection.selected, data).show()

# %% [markdown]
# 3 of 4 groups covered, no extras, no noise, and `x4` missed (Lesson 3 explains why).
#
# ## Breast Cancer Wisconsin

# %%
bc = fwx.load_breast_cancer()
bc_selection = fwx.select_features(bc, "diagnosis")
bc_reference = fwx.load_reference("breast_cancer")
print(f"30 -> SULOV {len(bc_selection.sulov.kept)} -> {len(bc_selection.selected)}:", bc_selection.selected)
print("featurewiz 0.6.1:", bc_reference.selected)
fwx.compare_to_reference(bc_selection, bc_reference)

# %% [markdown]
# No ground truth here: this is real data, so we can only compare with the Reference run.
#
# ## Capstone: a Madelon-style dataset
# The NIPS 2003 feature-selection challenge used **Madelon**: 5 informative features, 15 redundant
# mixtures of them, and hundreds of noise features. Real Madelon never published which columns are which,
# so we build one with the same recipe (scikit-learn's `make_classification` *is* that generator):
# 5 informative, 15 redundant, 280 noise. This cell takes a few minutes on Kaggle's CPUs.

# %%
madelon = fwx.make_madelon_style_dataset()
m_selection = fwx.select_features(madelon.frame, madelon.target)
m_score = fwx.score_relevant_vs_noise(m_selection.selected, madelon)
print(f"SULOV dropped: {m_selection.sulov.dropped}")
print(f"selected {len(m_selection.selected)}: relevant {len(m_score.relevant_kept)}/{m_score.relevant_total}, "
      f"noise {len(m_score.noise_kept)}/{m_score.noise_total}")

# %%
plots.rounds_heatmap(m_selection.xgb).show()

# %%
m_reference = fwx.load_reference("madelon_style")
fwx.compare_to_reference(m_selection, m_reference)

# %% [markdown]
# Most of the selection is **noise**. That follows from Lesson 3: with 295 features each Round must nominate
# at least `top_num = 59`, and there are only 20 relevant features to go round, so the rest of each quota
# is filled with noise that got a little gain by chance. (Write the exact numbers you observed above.)
#
# ## Sidebar: regression
# For a numeric target featurewiz swaps in `mutual_info_regression` and an XGBoost regressor
# (`reg:squarederror`). Everything else is identical: the pairs, the knockout walk, the windows,
# the nomination rule. This series sticks to classification.
#
# ## Optional: rerun the real featurewiz yourself
# This needs **Internet on** (phone-verified Kaggle account) and installs featurewiz's old pins into a
# separate virtual environment, so your notebook's packages stay untouched. Copy it into a cell if you want to try:
#
# ```python
# !python -m venv /tmp/fw && /tmp/fw/bin/pip -q install featurewiz==0.6.1 xgboost==1.7.6 scikit-learn==1.5.2 "numpy<2" "pandas<3" "pyarrow<17"
# data.frame.to_csv("/tmp/canonical.csv", index=False)
# !/tmp/fw/bin/python -c "import pandas as pd, featurewiz as fw; print(fw.featurewiz(pd.read_csv('/tmp/canonical.csv'), 'target', corr_limit=0.8, verbose=0)[0])"
# ```
#
# ## The whole series in five lines
# 1. SULOV removes **linearly** redundant features, keeping the one with more mutual information.
# 2. Recursive XGBoost trains on **suffixes** of the column list and pools each Round's top features.
# 3. **Column order** (SULOV's output order) changes what Recursive XGBoost finds.
# 4. Each Round nominates a **fixed minimum**, so with many features, noise gets in.
# 5. Pin your xgboost version if you need reproducible selections.
```

Replace the placeholders. Write the observed numbers after running.

- [ ] **Step 2: Convert, run and strip**

Run:
```bash
.venv/bin/jupytext --to ipynb --output notebooks/04-full-pipeline.ipynb build/04-full-pipeline.py
.venv/bin/pytest --nbmake --nbmake-timeout=900 notebooks/04-full-pipeline.ipynb
```
Expected: `1 passed` (around 1–3 minutes).

- [ ] **Step 3: Commit**

```bash
git add notebooks/04-full-pipeline.ipynb
git commit -m "docs: Lesson 4 Full pipeline"
```

---

### Task 12: Notebook hygiene tests and CI

**Files:**
- Create: `tests/test_notebooks.py`, `.github/workflows/ci.yml`

- [ ] **Step 1: Write the failing hygiene tests**

`tests/test_notebooks.py`:

```python
"""Lessons must run with Internet off and be committed without outputs."""
import json
import re
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))
NEEDS_INTERNET = re.compile(r"pip install|urlopen|requests\.get|https?://|kaggle datasets download")


def test_there_are_four_lessons():
    assert [p.name for p in NOTEBOOKS] == [
        "01-intuition.ipynb", "02-sulov.ipynb", "03-recursive-xgboost.ipynb", "04-full-pipeline.ipynb",
    ]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_code_cells_never_need_the_internet(path):
    cells = json.loads(path.read_text())["cells"]
    for cell in cells:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            assert not NEEDS_INTERNET.search(source), f"{path.name}: {source[:80]!r}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_outputs_are_stripped(path):
    for cell in json.loads(path.read_text())["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == [] and cell.get("execution_count") is None


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_lesson_starts_with_the_bootstrap_cell(path):
    first_code = next(c for c in json.loads(path.read_text())["cells"] if c["cell_type"] == "code")
    assert 'glob("**/featurewiz_explained.py")' in "".join(first_code["source"])
```

- [ ] **Step 2: Run them**

Run: `.venv/bin/pytest tests/test_notebooks.py -v`
Expected: all pass (13 tests). If one fails, fix the Lesson, not the test. To check that the Internet test bites, temporarily add a code cell containing `!pip install x` to one Lesson, see the test FAIL, then remove the cell.

- [ ] **Step 3: Write the workflow**

`.github/workflows/ci.yml`:

```yaml
name: ci
on: [push, pull_request]

jobs:
  pinned:
    # The Re-implementation must match the Reference runs exactly (ADR 0002).
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements-pinned.txt && pip install -e . --no-deps
      - run: pytest -v

  current:
    # Stands in for Kaggle's notebook image: latest packages. Differences are reported, not failed.
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -e ".[dev]"
      - run: nbstripout --verify notebooks/*.ipynb
      - run: pytest -v
      - run: pytest --nbmake --nbmake-timeout=900 notebooks/
      - if: always()
        uses: actions/upload-artifact@v4
        with:
          name: reference-diff
          path: build/reference-diff.md
```

- [ ] **Step 4: Check the workflow locally as far as possible**

Run: `.venv/bin/nbstripout --verify notebooks/*.ipynb && .venv/bin/pytest -v && .venv/bin/pytest --nbmake --nbmake-timeout=900 notebooks/`
Expected: no output from `--verify` (exit 0), and all tests pass. If your nbstripout version lacks `--verify`, use `--dry-run` piped to `test -z "$(...)"` in both places instead.

- [ ] **Step 5: Commit, push and watch CI**

```bash
git add tests/test_notebooks.py .github/workflows/ci.yml
git commit -m "ci: pinned exact-match and current-environment jobs"
git push origin main
gh run watch
```
Expected: both jobs green. Download the `reference-diff` artifact (`gh run download -n reference-diff`) and read it.

---

### Task 13: Kaggle packaging, then a private publish (needs the owner's go-ahead)

**Files:**
- Create: `scripts/build_kaggle.py`, `tests/test_build_kaggle.py`

**Interfaces:**
- Consumes: the module files, `reference_runs/`, `notebooks/`.
- Produces: `build_kaggle(out: Path, owner: str = "amayzes") -> None`, which writes `out/dataset/` (flat) and `out/lesson-N/` (notebook plus `kernel-metadata.json`).

- [ ] **Step 1: Write the failing test**

`tests/test_build_kaggle.py`:

```python
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_kaggle", ROOT / "scripts" / "build_kaggle.py")
build_kaggle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_kaggle)


def test_dataset_is_flat_and_complete(tmp_path):
    build_kaggle.build_kaggle(tmp_path)
    names = {p.name for p in (tmp_path / "dataset").iterdir()}
    assert {"featurewiz_explained.py", "featurewiz_explained_plots.py", "breast_cancer.csv",
            "manifest.json", "LICENSE", "NOTICE", "dataset-metadata.json"} <= names
    assert {f"reference_{n}.json" for n in ("canonical", "canonical_shuffled", "breast_cancer", "madelon_style")} <= names
    assert not any(p.is_dir() for p in (tmp_path / "dataset").iterdir())  # Kaggle skips subfolders by default
    meta = json.loads((tmp_path / "dataset" / "dataset-metadata.json").read_text())
    assert meta["id"] == "amayzes/featurewiz-explained-reference-runs"


def test_lessons_are_private_offline_and_attach_the_dataset(tmp_path):
    build_kaggle.build_kaggle(tmp_path)
    lessons = sorted(tmp_path.glob("lesson-*"))
    assert len(lessons) == 4
    meta = json.loads((lessons[1] / "kernel-metadata.json").read_text())
    assert meta["title"] == "featurewiz, explained 2/4: SULOV, step by step"
    assert meta["id"] == "amayzes/featurewiz-explained-2-4-sulov-step-by-step"
    assert meta["is_private"] is True and meta["enable_internet"] is False
    assert meta["dataset_sources"] == ["amayzes/featurewiz-explained-reference-runs"]
    assert (lessons[1] / meta["code_file"]).is_file()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_build_kaggle.py -v`
Expected: FAIL. `FileNotFoundError` for `scripts/build_kaggle.py`.

- [ ] **Step 3: Write the implementation**

`scripts/build_kaggle.py`:

```python
"""Assemble what gets pushed to Kaggle into build/kaggle/.

    .venv/bin/python scripts/build_kaggle.py
    kaggle datasets create -p build/kaggle/dataset        # first time (then: kaggle datasets version -p ... -m "msg")
    for d in build/kaggle/lesson-*; do kaggle kernels push -p "$d"; done

The dataset is flat (Kaggle skips subfolders by default). Lessons are pushed private, with Internet off.
"""
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_SLUG = "featurewiz-explained-reference-runs"
LESSONS = [
    ("01-intuition.ipynb", "Intuition"),
    ("02-sulov.ipynb", "SULOV, step by step"),
    ("03-recursive-xgboost.ipynb", "Recursive XGBoost, step by step"),
    ("04-full-pipeline.ipynb", "Full pipeline"),
]


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def build_kaggle(out: Path, owner: str = "amayzes") -> None:
    if out.exists():
        shutil.rmtree(out)
    dataset = out / "dataset"
    dataset.mkdir(parents=True)
    for name in ("featurewiz_explained.py", "featurewiz_explained_plots.py"):
        shutil.copy(ROOT / "src" / name, dataset / name)
    for path in (ROOT / "reference_runs").iterdir():
        if path.is_file():
            shutil.copy(path, dataset / path.name)
    for name in ("LICENSE", "NOTICE"):
        shutil.copy(ROOT / name, dataset / name)
    (dataset / "dataset-metadata.json").write_text(json.dumps({
        "title": "featurewiz, explained: reference runs",
        "id": f"{owner}/{DATASET_SLUG}",
        "licenses": [{"name": "CC-BY-4.0"}],
    }, indent=1))

    for number, (notebook, topic) in enumerate(LESSONS, start=1):
        title = f"featurewiz, explained {number}/{len(LESSONS)}: {topic}"
        folder = out / f"lesson-{number}"
        folder.mkdir()
        shutil.copy(ROOT / "notebooks" / notebook, folder / notebook)
        (folder / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{owner}/{_slug(title)}",
            "title": title,
            "code_file": notebook,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,
            "enable_internet": False,
            "dataset_sources": [f"{owner}/{DATASET_SLUG}"],
            "competition_sources": [],
            "kernel_sources": [],
        }, indent=1))


if __name__ == "__main__":
    build_kaggle(ROOT / "build" / "kaggle")
    print("built", ROOT / "build" / "kaggle")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_build_kaggle.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_kaggle.py tests/test_build_kaggle.py
git commit -m "feat: Kaggle packaging for dataset and Lessons"
```

- [ ] **Step 6: STOP and ask the owner before publishing**

Publishing is outward-facing. Ask: "Ready to push the reference-runs dataset and the four Lessons to Kaggle as **private** under amayzes?" Continue only on an explicit yes.

- [ ] **Step 7: Publish privately and verify on Kaggle**

Run:
```bash
.venv/bin/pip install kaggle
.venv/bin/python scripts/build_kaggle.py
.venv/bin/kaggle datasets create -p build/kaggle/dataset    # creates it private by default
for d in build/kaggle/lesson-*; do .venv/bin/kaggle kernels push -p "$d"; done
for n in 1-4-intuition 2-4-sulov-step-by-step 3-4-recursive-xgboost-step-by-step 4-4-full-pipeline; do
  .venv/bin/kaggle kernels status amayzes/featurewiz-explained-$n
done
```
Expected: every kernel eventually reports `complete`. If one reports `error`, run `kaggle kernels output amayzes/<slug> -p build/kaggle/out` and read the log. The most likely cause is the dataset mount path, which the bootstrap cell's `glob("**/...")` is meant to absorb. Report the four URLs to the owner, who decides when to make them public.

---

## Self-review notes

- Spec coverage:
  - Scope (SULOV + Recursive XGBoost, classification): Tasks 3–5.
  - Faithful Re-implementation with quirks: Tasks 3–4.
  - Reference runs, pinned vs current: Task 6 and CI in Task 12.
  - Four Lessons: Tasks 8–11.
  - Canonical, Madelon-style and Breast Cancer datasets: Tasks 1–2.
  - Signal-group and relevance scoring: Task 5.
  - Plotly, no widgets: Task 7.
  - nbstripout: Tasks 8 and 12.
  - Module shipped in the dataset: Task 13.
  - Private publish: Task 13.
  - License and NOTICE: Task 1.
  - Regression sidebar and optional live rerun: Task 11.
- The SULOV attribution text is in the `sulov` docstring (Task 3).
- Each Review Focus item has a test: spaces, missing values and non-numeric columns (Task 5); bad labels (Task 4); Internet use and stripped outputs (Task 12); runtime (nbmake timeout, Task 12).
