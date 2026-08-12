import numpy as np
from xai_mini_research import preprocess, generate_time_data

def test_X_scaled_exists_in_all_splits():
    processed_data = preprocess(generate_time_data())

    assert "X_scaled" in processed_data["train"]
    assert "X_scaled" in processed_data["val"]
    assert "X_scaled" in processed_data["test"]

def test_training_data_is_centered():
    processed_data = preprocess(generate_time_data())
    X_train_scaled = processed_data["train"]["X_scaled"]

    np.testing.assert_allclose(X_train_scaled.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(X_train_scaled.std(axis=0), 1, atol=1e-12)

def test_val_and_test_splits_transformed_correctly():
    processed_data = preprocess(generate_time_data())
    scaler = processed_data["metadata"]["scaler"]

    X_train = processed_data["train"]["X"]
    X_val = processed_data["val"]["X"]
    X_test = processed_data["test"]["X"]

    np.testing.assert_allclose(scaler.transform(X_train), processed_data["train"]["X_scaled"])
    np.testing.assert_allclose(scaler.transform(X_val), processed_data["val"]["X_scaled"])
    np.testing.assert_allclose(scaler.transform(X_test), processed_data["test"]["X_scaled"])

def test_preprocess_does_not_mutate_original_split_dicts():
    data = generate_time_data()
    processed_data = preprocess(data)

    assert data is not processed_data
    assert data["train"] is not processed_data["train"]
    assert data["val"] is not processed_data["val"]
    assert data["test"] is not processed_data["test"]

    assert "X_scaled" not in data["train"]
    assert "X_scaled" not in data["val"]
    assert "X_scaled" not in data["test"]

    assert "X_scaled" in processed_data["train"]
    assert "X_scaled" in processed_data["val"]
    assert "X_scaled" in processed_data["test"]
