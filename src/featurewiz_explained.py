"""featurewiz, explained: a faithful, readable Re-implementation of featurewiz 0.6.1's
feature selection (SULOV, then Recursive XGBoost), for learning.

Vocabulary is defined in CONTEXT.md. Every Quirk reproduced here is deliberate
(docs/adr/0001-reimplement-featurewiz-stages.md): the Re-implementation must match a
Reference run of the real library exactly, so don't "fix" anything here.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
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
