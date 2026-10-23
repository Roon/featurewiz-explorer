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


@pytest.mark.parametrize("name", ["x-1", "x.1", "ratio(%)", "a[0]"])
def test_select_features_rejects_names_featurewiz_would_rename(name):
    frame = fwx.make_canonical_dataset().frame.rename(columns={"x1": name})
    with pytest.raises(ValueError, match="A-Za-z0-9_"):
        fwx.select_features(frame, "target")


def test_select_features_rejects_non_finite_values():
    frame = fwx.make_canonical_dataset().frame
    frame.loc[0, "x2"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        fwx.select_features(frame, "target")


def test_select_features_rejects_constant_columns():
    frame = fwx.make_canonical_dataset().frame.assign(const=1.0)
    with pytest.raises(ValueError, match="constant"):
        fwx.select_features(frame, "target")


def test_select_features_rejects_integer_columns():
    frame = fwx.make_canonical_dataset().frame
    frame["row_id"] = np.arange(len(frame))
    with pytest.raises(ValueError, match="integer"):
        fwx.select_features(frame, "target")


def test_select_features_rejects_low_cardinality_floats():
    frame = fwx.make_canonical_dataset().frame
    frame["level"] = np.resize([0.5, 1.5, 2.5], len(frame))
    with pytest.raises(ValueError, match="categorical"):
        fwx.select_features(frame, "target")


def test_select_features_rejects_data_featurewiz_would_sample(monkeypatch):
    monkeypatch.setattr(fwx, "SULOV_SAMPLING_CELLS", 1000)
    with pytest.raises(ValueError, match="sample"):
        fwx.select_features(fwx.make_canonical_dataset().frame, "target")


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
