import math
from xai_mini_research.models import train_linear_model, predict_splits
from xai_mini_research import preprocess, generate_time_data, regression_metrics, regression_metrics_all_splits

def test_metrics_for_all_splits_dictionary_contains_all_split_keys():
    processed_data = preprocess(generate_time_data())
    model = train_linear_model(processed_data)
    predictions = predict_splits(model, processed_data)
    reg_metrics_all = regression_metrics_all_splits(processed_data, predictions)

    assert "train" in reg_metrics_all
    assert "val" in reg_metrics_all
    assert "test" in reg_metrics_all

def test_metrics_dictionary_for_one_split_contains_all_scores():
    processed_data = preprocess(generate_time_data())
    model = train_linear_model(processed_data)
    predictions = predict_splits(model, processed_data)

    train_metrics = regression_metrics(processed_data["train"]["y"], predictions["train"])
    val_metrics = regression_metrics(processed_data["val"]["y"], predictions["val"])
    test_metrics = regression_metrics(processed_data["test"]["y"], predictions["test"])

    assert "mae" in train_metrics
    assert "rmse" in train_metrics
    assert "r2" in train_metrics

    assert "mae" in val_metrics
    assert "rmse" in val_metrics
    assert "r2" in val_metrics

    assert "mae" in test_metrics
    assert "rmse" in test_metrics
    assert "r2" in test_metrics

def test_regression_metrics_returns_finite_values():
    metrics = regression_metrics(
        y_target=[1, 2, 3],
        y_pred=[1, 2, 2.5],
    )

    assert math.isfinite(metrics["mae"])
    assert math.isfinite(metrics["rmse"])
    assert math.isfinite(metrics["r2"])

def test_regression_metrics_calculates_expected_values():
    metrics = regression_metrics(
        y_target=[1, 2, 3],
        y_pred=[1, 2, 2],
    )

    assert math.isclose(metrics["mae"], 1 / 3)
    assert math.isclose(metrics["rmse"], (1 / 3) ** 0.5)
    assert math.isclose(metrics["r2"], 0.5)
