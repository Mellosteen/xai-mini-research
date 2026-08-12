from sklearn.dummy import DummyRegressor
from xai_mini_research import generate_time_data, preprocess
from xai_mini_research.models import train_linear_model, predict_splits

def test_linear_regression_predicts_expected_shape():
    processed_data = preprocess(generate_time_data())
    model = train_linear_model(processed_data)
    predictions = predict_splits(model, processed_data)

    assert predictions["train"].shape == processed_data["train"]["y"].shape
    assert predictions["val"].shape == processed_data["val"]["y"].shape
    assert predictions["test"].shape == processed_data["test"]["y"].shape

def test_linear_regression_model_has_higher_validation_r2_than_mean_predictor():
    processed_data = preprocess(generate_time_data())
    linear_model = train_linear_model(processed_data)
    mean_predictor = DummyRegressor(strategy="mean").fit(processed_data["train"]["X_scaled"], processed_data["train"]["y"])

    linear_score = linear_model.score(processed_data["val"]["X_scaled"], processed_data["val"]["y"])
    mean_score = mean_predictor.score(processed_data["val"]["X_scaled"], processed_data["val"]["y"])

    assert linear_score > mean_score

def test_linear_model_satisfies_benchmarks_rmse_and_r2():
    # TODO: Add benchmark test after regression_metrics in metrics.py is implemented.
    # Expected validation RMSE <= 3.0 and R2 >= 0.70.
    ...