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
