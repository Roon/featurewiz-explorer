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


def test_targets_featurewiz_would_treat_as_regression_are_rejected():
    frame = _small_frame()
    frame["y"] = np.arange(len(frame)) % 40  # 40 integer classes: featurewiz switches to Regression
    with pytest.raises(ValueError, match="regression"):
        fwx.recursive_xgboost(frame, ["f0", "f1", "f2"], "y")


def test_string_targets_are_rejected():
    frame = _small_frame()
    frame["y"] = frame["y"].map({0: "B", 1: "M"})
    with pytest.raises(ValueError, match="0..k-1"):
        fwx.recursive_xgboost(frame, ["f0", "f1", "f2"], "y")


def test_a_round_with_no_gains_falls_back_to_its_window_like_featurewiz():
    # featurewiz wraps the whole loop in try/except; when a Round's gains are empty,
    # nominating raises and featurewiz returns that Round's window, discarding earlier
    # nominations (a Quirk). Here Round 2's window holds only constant columns.
    frame = _small_frame()[["f0", "f1", "y"]].assign(c1=1.0, c2=2.0)
    result = fwx.recursive_xgboost(frame, ["f0", "f1", "c1", "c2"], "y")
    assert [r.window for r in result.rounds] == [["f0", "f1", "c1", "c2"], ["c1", "c2"]]
    assert result.rounds[1].gains == {} and result.rounds[1].nominated == []
    assert result.selected == ["c1", "c2"]


def test_canonical_dataset_after_sulov_misses_x4():
    data = fwx.make_canonical_dataset()
    kept = fwx.sulov(data.frame, data.features, data.target).kept
    result = fwx.recursive_xgboost(data.frame, kept, data.target)
    assert result.selected == ["x1", "x2", "x3_copy"]
