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
