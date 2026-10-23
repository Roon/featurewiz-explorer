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
