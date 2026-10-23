import numpy as np

import featurewiz_explained as fwx


def test_canonical_dataset_shape_and_columns():
    data = fwx.make_canonical_dataset()
    assert data.target == "target"
    assert data.features == [
        "x1", "x2", "x3", "x4",
        "x1_copy", "x2_copy", "x3_copy", "x4_squared",
        "noise_1", "noise_2", "noise_3", "noise_4",
    ]
    assert data.frame.shape == (2000, 13)
    assert set(data.frame["target"].unique()) == {0, 1}


def test_canonical_dataset_roles_and_signal_groups():
    data = fwx.make_canonical_dataset()
    assert [f for f in data.features if data.roles[f] == "informative"] == ["x1", "x2", "x3", "x4"]
    assert [f for f in data.features if data.roles[f] == "redundant"] == ["x1_copy", "x2_copy", "x3_copy", "x4_squared"]
    assert [f for f in data.features if data.roles[f] == "noise"] == ["noise_1", "noise_2", "noise_3", "noise_4"]
    assert data.groups == {
        "x1": "x1", "x2": "x2", "x3": "x3", "x4": "x4",
        "x1_copy": "x1", "x2_copy": "x2", "x3_copy": "x3", "x4_squared": "x4",
    }


def test_canonical_dataset_plants_linear_copies_and_a_nonlinear_copy():
    frame = fwx.make_canonical_dataset().frame
    for i in (1, 2, 3):
        assert frame[f"x{i}"].corr(frame[f"x{i}_copy"]) > 0.9  # SULOV (limit 0.8) will see these
    assert abs(frame["x4"].corr(frame["x4_squared"])) < 0.2  # ...but not this one


def test_canonical_dataset_is_deterministic_per_seed():
    a, b = fwx.make_canonical_dataset(seed=7), fwx.make_canonical_dataset(seed=7)
    assert a.frame.equals(b.frame)
    assert not np.allclose(a.frame["x1"], fwx.make_canonical_dataset(seed=8).frame["x1"])
