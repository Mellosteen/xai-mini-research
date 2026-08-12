from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def regression_metrics(y_target, y_pred):
    """
    Calculates and returns MAE, RMSE, and R2 metrics on one split (train, validation, test) of a regression model.

    Args:
        y_target (np.array): array of true output values of a split; shape (n_samples,).
        y_pred (np.array): array of predicted outputs values of the same split; shape (n_samples,).

    Returns:
        reg_metrics (Dict): Dictionary containing all 3 metrics for the given split.
    """

    reg_metrics = {
        "mae" : mean_absolute_error(y_target, y_pred),
        "rmse" : root_mean_squared_error(y_target, y_pred),
        "r2" : r2_score(y_target, y_pred),
    }

    return reg_metrics

def regression_metrics_all_splits(processed_data, predictions):
    """
    Calculates and returns MAE, RMSE, and R2 metrics based on predictions from a regression model on all splits.

    Args:
        processed_data (Dict): Dictionary containing all relevant data.
        predictions (Dict): Dictionary containing model predictions on all splits.

    Returns:
        reg_metrics_all (Dict): Dictionary containing all 3 metrics for all splits.
    """
    y_train_true = processed_data["train"]["y"]
    y_val_true = processed_data["val"]["y"]
    y_test_true = processed_data["test"]["y"]

    reg_metrics_all = {
        "train" : regression_metrics(y_train_true, predictions["train"]),
        "val" : regression_metrics(y_val_true, predictions["val"]),
        "test" : regression_metrics(y_test_true, predictions["test"]),
    }

    return reg_metrics_all
