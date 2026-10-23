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
