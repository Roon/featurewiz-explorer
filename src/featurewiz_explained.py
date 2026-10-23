"""featurewiz, explained: a faithful, readable Re-implementation of featurewiz 0.6.1's
feature selection (SULOV, then Recursive XGBoost), for learning.

Vocabulary is defined in CONTEXT.md. Every Quirk reproduced here is deliberate
(docs/adr/0001-reimplement-featurewiz-stages.md): the Re-implementation must match a
Reference run of the real library exactly, so don't "fix" anything here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

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
