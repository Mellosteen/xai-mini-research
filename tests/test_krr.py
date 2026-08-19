import math
from sklearn.dummy import DummyRegressor
from xai_mini_research import generate_time_data, preprocess, regression_metrics_all_splits, regression_metrics
from xai_mini_research.models import train_krr_model, predict_krr_splits

def train_test_krr():
    processed_data = preprocess(generate_time_data(100))
    krr = train_krr_model(processed_data=processed_data, alpha=0.1, gamma=0.1)
    predictions = predict_krr_splits(model=krr, processed_data=processed_data)

    return processed_data, predictions

def test_krr_predicts_expected_shape():
    processed_data, predictions = train_test_krr()

    assert predictions["train"].shape == processed_data["train"]["y"].shape
    assert predictions["val"].shape == processed_data["val"]["y"].shape
    assert predictions["test"].shape == processed_data["test"]["y"].shape

def test_prediction_metrics_are_finite():
    processed_data, predictions = train_test_krr()
    metrics = regression_metrics_all_splits(processed_data=processed_data, predictions=predictions)

    for split in ["train", "val", "test"]:
        for metric in ["mae", "rmse", "r2"]:
            assert math.isfinite(metrics[split][metric])

def test_krr_has_higher_validation_r2_than_mean_predictor():
    processed_data, krr_predictions = train_test_krr()
    X_train = processed_data["train"]["X_scaled"]
    y_train = processed_data["train"]["y"]
    X_val = processed_data["val"]["X_scaled"]
    y_val = processed_data["val"]["y"]

    mean_predictor = DummyRegressor(strategy="mean")
    mean_predictor.fit(X_train, y_train)
    mean_score = mean_predictor.score(X_val,y_val)
    krr_val_metrics = regression_metrics(y_val, krr_predictions["val"])

    assert krr_val_metrics["r2"] > mean_score
