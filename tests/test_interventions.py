import numpy as np
from xai_mini_research import intervene_scaled_shortcut, preprocess, generate_time_data

def create_intervened_dicts(processed_data, split="test", seed=42):
    """
    Returns 4 dicts of intervened shortcut results on selected split of all condition types: zeroed, noise, reversed, permuted.
    """
    return (intervene_scaled_shortcut(processed_data=processed_data, split=split, condition="zeroed", seed=seed), 
            intervene_scaled_shortcut(processed_data=processed_data, split=split, condition="noise", seed=seed), 
            intervene_scaled_shortcut(processed_data=processed_data, split=split, condition="reversed", seed=seed), 
            intervene_scaled_shortcut(processed_data=processed_data, split=split, condition="permuted", seed=seed))

def test_returned_X_has_same_shape_as_original():
    processed_data = preprocess(generate_time_data(n_samples=100, shortcut=True))

    for split in ["train", "val", "test"]:
        zeroed, noise, reversed, permuted = create_intervened_dicts(processed_data=processed_data,split=split, seed=42)

        assert processed_data[split]["X_scaled"].shape == zeroed["X_scaled"].shape == noise["X_scaled"].shape == permuted["X_scaled"].shape == reversed["X_scaled"].shape

def test_processed_data_X_scaled_remains_unchanged():
    processed_data = preprocess(generate_time_data(n_samples=100, shortcut=True))
    X_scaled = processed_data["test"]["X_scaled"].copy()
    intervene_scaled_shortcut(processed_data=processed_data, split="test", condition="zeroed", seed=42)

    np.testing.assert_equal(X_scaled, processed_data["test"]["X_scaled"])

def test_zero_condition_creates_all_zeroes_in_split():
    processed_data = preprocess(generate_time_data(n_samples=100, shortcut=True))
    shortcut_idx = processed_data["metadata"]["feature_names"].index("shortcut_polynomial")
    zeroed = intervene_scaled_shortcut(processed_data=processed_data, split="test", condition="zeroed", seed=42)

    assert np.all(zeroed["X_scaled"][:, shortcut_idx] == 0.0)

def test_reversed_shortcut_is_shortcut_times_minus_one():
    processed_data = preprocess(generate_time_data(n_samples=100, shortcut=True))
    shortcut_idx = processed_data["metadata"]["feature_names"].index("shortcut_polynomial")
    reversed = intervene_scaled_shortcut(processed_data=processed_data, split="test", condition="reversed", seed=42)

    np.testing.assert_equal(processed_data["test"]["X_scaled"][:, shortcut_idx], -reversed["X_scaled"][:, shortcut_idx])

def test_noise_generation_is_equal_for_same_seed():
    processed_data = preprocess(generate_time_data(n_samples=100, shortcut=True))
    shortcut_idx = processed_data["metadata"]["feature_names"].index("shortcut_polynomial")
    noise_a = intervene_scaled_shortcut(processed_data=processed_data, split="test", condition="noise", seed=42)
    noise_b = intervene_scaled_shortcut(processed_data=processed_data, split="test", condition="noise", seed=42)

    np.testing.assert_equal(noise_a["X_scaled"][:, shortcut_idx], noise_b["X_scaled"][:, shortcut_idx])

def test_permuted_shortcuts_retain_same_values_after_shuffle():
    processed_data = preprocess(generate_time_data(n_samples=100, shortcut=True))
    shortcut_idx = processed_data["metadata"]["feature_names"].index("shortcut_polynomial")
    permuted = intervene_scaled_shortcut(processed_data=processed_data, split="test", condition="permuted", seed=42)

    np.testing.assert_equal(np.sort(processed_data["test"]["X_scaled"][:, shortcut_idx]), np.sort(permuted["X_scaled"][:, shortcut_idx]))

def test_non_intervened_features_remain_unchanged():
    processed_data = preprocess(generate_time_data(n_samples=100, shortcut=True))
    shortcut_idx = processed_data["metadata"]["feature_names"].index("shortcut_polynomial")
    orig_without_short = np.delete(processed_data["test"]["X_scaled"], shortcut_idx, axis=1)

    for condition in ["zeroed", "noise", "reversed", "permuted"]:
        intervened_data = intervene_scaled_shortcut(processed_data=processed_data, split="test", condition=condition, seed=42)
        intervened_without_short = np.delete(intervened_data["X_scaled"], shortcut_idx, axis=1)

        np.testing.assert_equal(orig_without_short, intervened_without_short)
