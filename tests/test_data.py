import numpy as np
import pytest
from xai_mini_research import generate_time_data

def test_generated_data_has_expected_shapes():
    n_samples = 100
    data = generate_time_data(n_samples=n_samples)

    assert data["time"].shape == (n_samples,)
    assert data["true_target"].shape == (n_samples,)
    assert data["target"].shape == (n_samples,)
    assert data["split"].shape == (n_samples,)

def test_features_matrix_has_expected_shape():
    n_samples = 100
    data = generate_time_data(n_samples=n_samples)

    assert data["features"].shape == (n_samples, 3)

def test_same_seed_generates_same_data():
    data_a = generate_time_data(seed=3)
    data_b = generate_time_data(seed=3)

    np.testing.assert_array_equal(data_a["target"], data_b["target"])
    np.testing.assert_array_equal(data_a["features"], data_b["features"])

def test_split_order_is_correct():
    data = generate_time_data(n_samples=100)

    train_time = data["train"]["time"]
    val_time = data["val"]["time"]
    test_time = data["test"]["time"]

    assert train_time[-1] < val_time[0]
    assert val_time[-1] < test_time[0]

def test_split_sizes_are_correct():
    data = generate_time_data(n_samples=100)
    split = data["split"]

    assert np.sum(split == "train") == 70
    assert np.sum(split == "val") == 15
    assert np.sum(split == "test") == 15

def test_invalid_split_ratios_raises_error():
    with pytest.raises(ValueError):
        generate_time_data(training_ratio=0.8, val_ratio=0.15, test_ratio=0.15)
