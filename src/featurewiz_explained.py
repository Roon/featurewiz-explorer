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
from sklearn.feature_selection import mutual_info_classif

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
