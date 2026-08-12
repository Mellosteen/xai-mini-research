from sklearn.preprocessing import StandardScaler

def preprocess(data):
    """
    Preprocess data by centering training features. The scaler used to center training data should 
    be applied to validation and test splits since we treat it as new, unseen data. 

    Args:
        data (Dict): Dictionary output of generate_time_data function  with relevant information of time-series synthetic data.

    Returns:
        processed_data (Dict): Copy of data Dict with added centered training features and applied scaling on validation & test splits.
    """
    scaler = StandardScaler()
    processed_data = {
            **data,
            "train" : {**data["train"]},
            "val" : {**data["val"]},
            "test" : {**data["test"]},
            "metadata" : {**data["metadata"]},
        }   # Copy data into a new dict to add processed X
    
    X_train = processed_data["train"]["X"]
    X_val = processed_data["val"]["X"]
    X_test = processed_data["test"]["X"]
    
    X_train_scaled = scaler.fit_transform(X_train)  # Fit to training data
    X_val_scaled = scaler.transform(X_val)          # Apply same transform to other splits
    X_test_scaled = scaler.transform(X_test)

    # Add scaled training features to each split
    processed_data["train"]["X_scaled"] = X_train_scaled
    processed_data["val"]["X_scaled"] = X_val_scaled
    processed_data["test"]["X_scaled"] = X_test_scaled

    # Add scaler to metadata
    processed_data["metadata"]["scaler"] = scaler

    return processed_data
