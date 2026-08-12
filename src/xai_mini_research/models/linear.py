from sklearn.linear_model import LinearRegression

def train_linear_model(processed_data):
    """
    Fits a linear regression model based on centered inputs X and outputs y.

    Args:
        processed_data (Dict): Dictionary of data which include processed data.

    Returns:
        model (LinearRegression): a model fit to the args.
    """
    model = LinearRegression()

    X_train = processed_data["train"]["X_scaled"]
    y_train = processed_data["train"]["y"]

    model.fit(X_train, y_train)

    return model

def predict_splits(model, processed_data):
    """
    Returns predicted outputs for a given input.

    Args:
        model (LinearRegression): trained linear regression model.
        processed_data (Dict): Dictionary of data which include processed data.

    Returns:
        predictions (Dict): A dictionary of predictions on all splits' inputs.
    """
    X_train = processed_data["train"]["X_scaled"]
    X_val = processed_data["val"]["X_scaled"]
    X_test = processed_data["test"]["X_scaled"]

    predictions = {
        "train" : model.predict(X_train),
        "val" : model.predict(X_val),
        "test" : model.predict(X_test),
    }

    return predictions
