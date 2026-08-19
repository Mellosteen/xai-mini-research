from sklearn.kernel_ridge import KernelRidge

def train_krr_model(processed_data, alpha, gamma):
    """
    Trains an RBF kernel ridge regression model with the given hyperparameters.

    Args:
        processed_data (Dict): Dictionary of all important data.
        alpha (float): Ridge/lambda-like regularization strength for the fitted function.
        gamma (float): RBF kernel locality parameter. Larger values make the kernel more local.

    Returns: 
        model (KernelRidge): A fitted kRR model based on training data in processed_data.
    """
    X_train = processed_data["train"]["X_scaled"]
    y_train = processed_data["train"]["y"]

    model = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
    model.fit(X_train, y_train)

    return model

def predict_krr_splits(model : KernelRidge, processed_data):
    """
    Predicts outputs of a trained kRR model on all splits of data.

    Args:
        model (KernelRidge): A trained kRR model with the RBF kernel.
        processed_data (Dict): Dictionary of all important data.

    Returns:
        predictions (Dict): A dictionary of predictions on each split.
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
