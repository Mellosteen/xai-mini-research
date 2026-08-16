import torch
import math
from sklearn.dummy import DummyRegressor
from xai_mini_research import (
    generate_time_data,
    preprocess,
    regression_metrics,
    regression_metrics_all_splits,
)
from xai_mini_research.models import MLPRegressor, predict_mlp_splits, set_torch_seed, train_mlp

def train_test_mlp(seed=42):
    processed_data = preprocess(generate_time_data(n_samples=300))
    set_torch_seed(seed)
    model = MLPRegressor(input_dim=processed_data["train"]["X_scaled"].shape[1])
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    training_metrics = train_mlp(model=model, processed_data=processed_data, optimizer=optimizer, criterion=criterion, epochs=5, seed=seed)

    predictions = predict_mlp_splits(model=model, processed_data=processed_data)

    return processed_data, model, training_metrics, predictions

def test_mlp_predicts_expected_shape():
    processed_data, _, _, predictions = train_test_mlp()

    assert predictions["train"].shape == processed_data["train"]["y"].shape
    assert predictions["val"].shape == processed_data["val"]["y"].shape
    assert predictions["test"].shape == processed_data["test"]["y"].shape

def test_mlp_has_higher_validation_r2_than_mean_predictor():
    processed_data, _, _, predictions = train_test_mlp()
    mean_predictor = DummyRegressor(strategy="mean").fit(processed_data["train"]["X_scaled"], processed_data["train"]["y"])
    mlp_metrics = regression_metrics(processed_data["val"]["y"], predictions["val"])

    mean_score = mean_predictor.score(processed_data["val"]["X_scaled"], processed_data["val"]["y"])
    mlp_score = mlp_metrics["r2"]

    assert mlp_score > mean_score

def test_mlp_training_records_one_loss_per_epoch():
    _, _, training_metrics, _ = train_test_mlp()

    assert len(training_metrics.train_losses) == 5
    assert len(training_metrics.val_losses) == 5

def test_mlp_training_losses_are_finite():
    _, _, training_metrics, _ = train_test_mlp()

    for loss in training_metrics.train_losses:
        assert math.isfinite(loss)

    for loss in training_metrics.val_losses:
        assert math.isfinite(loss)

def test_mlp_metrics_are_finite_for_all_splits():
    processed_data, _, _, predictions = train_test_mlp()
    metrics = regression_metrics_all_splits(processed_data, predictions)

    for split_metrics in metrics.values():
        assert math.isfinite(split_metrics["mae"])
        assert math.isfinite(split_metrics["rmse"])
        assert math.isfinite(split_metrics["r2"])

def test_mlp_training_is_reproducible_for_same_seed():
    _, _, training_metrics_a, predictions_a = train_test_mlp(seed=42)
    _, _, training_metrics_b, predictions_b = train_test_mlp(seed=42)

    assert training_metrics_a.train_losses == training_metrics_b.train_losses
    assert training_metrics_a.val_losses == training_metrics_b.val_losses
    torch.testing.assert_close(torch.from_numpy(predictions_a["val"]), torch.from_numpy(predictions_b["val"]))

def test_mlp_early_stopping_stops_before_max_epochs():
    processed_data = preprocess(generate_time_data(n_samples=300))
    set_torch_seed(42)

    model = MLPRegressor(input_dim=processed_data["train"]["X_scaled"].shape[1])
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    training_metrics = train_mlp(model=model, processed_data=processed_data, optimizer=optimizer, criterion=criterion, epochs=20, patience=1, min_delta=1e12, seed=42)

    assert len(training_metrics.train_losses) < 20
    assert len(training_metrics.val_losses) < 20
