from sklearn.dummy import DummyRegressor
from xai_mini_research import generate_time_data, preprocess, regression_metrics
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

def test_linear_model_satisfies_benchmarks_rmse_and_r2_on_validation_split():
    # Expected validation RMSE <= 3.0 and R2 >= 0.70.
    processed_data = preprocess(generate_time_data())
    model = train_linear_model(processed_data)
    predictions = predict_splits(model, processed_data)
    reg_metrics_val = regression_metrics(processed_data["val"]["y"], predictions["val"])

    assert reg_metrics_val["rmse"] <= 3.0
    assert reg_metrics_val["r2"] >= 0.70

def test_linear_train_only_shortcut_fails_during_validation():
    processed_data = preprocess(generate_time_data())
    processed_shortcut_data = preprocess(generate_time_data(shortcut=True, shortcut_fit_split="train"))
    model = train_linear_model(processed_data=processed_data)
    shortcut_model = train_linear_model(processed_data=processed_shortcut_data)

    predictions = predict_splits(model=model, processed_data=processed_data)
    shortcut_predictions = predict_splits(model=shortcut_model, processed_data=processed_shortcut_data)

    metrics_val = regression_metrics(y_target=processed_data["val"]["y"], y_pred=predictions["val"])
    metrics_shortcut_val = regression_metrics(y_target=processed_shortcut_data["val"]["y"], y_pred=shortcut_predictions["val"])

    assert metrics_val["rmse"] < metrics_shortcut_val["rmse"]

def test_linear_train_val_shortcut_fails_on_test():
    processed_data = preprocess(generate_time_data())
    processed_shortcut_data = preprocess(generate_time_data(shortcut=True, shortcut_fit_split="train_val"))
    model = train_linear_model(processed_data=processed_data)
    shortcut_model = train_linear_model(processed_data=processed_shortcut_data)

    predictions = predict_splits(model=model, processed_data=processed_data)
    shortcut_predictions = predict_splits(model=shortcut_model, processed_data=processed_shortcut_data)

    metrics_val = regression_metrics(y_target=processed_data["val"]["y"], y_pred=predictions["val"])
    metrics_shortcut_val = regression_metrics(y_target=processed_shortcut_data["val"]["y"], y_pred=shortcut_predictions["val"])
    metrics_test = regression_metrics(y_target=processed_data["test"]["y"], y_pred=predictions["test"])
    metrics_shortcut_test = regression_metrics(y_target=processed_shortcut_data["test"]["y"], y_pred=shortcut_predictions["test"]) 

    assert metrics_val["rmse"] > metrics_shortcut_val["rmse"]   # Shortcut val metric should improve since data is directly seen
    assert metrics_test["rmse"] < metrics_shortcut_test["rmse"]
