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
