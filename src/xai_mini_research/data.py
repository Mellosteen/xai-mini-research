import numpy as np

# --------------------------- Visualization --------------------------------------
def plot_time_data(data):
    import matplotlib.pyplot as plt

    time = data["time"]

    fig, ax = plt.subplots()
    ax.plot(time, data["target"])
    ax.plot(time, data["true_target"])
    ax.plot(data["train"]["time"], data["train"]["y"])
    ax.plot(data["val"]["time"], data["val"]["y"])
    ax.plot(data["test"]["time"], data["test"]["y"])
    plt.show()
# --------------------------- End Visualization ----------------------------------

def generate_time_data(n_samples=3650, seed=42, noise_level=0.05, shortcut=False, training_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Generate synthetic time series data based on annual seasonality function modelled after one from YData. 
    The shortcut parameter will be utilized later after a functional model has been implemented.

    Args:
        n_samples (Integer): Number of training, validation, and testing data points.
        seed (Integer): Seed for deterministic data generation.
        noise_level (Float): A value that represents the standard deviation of added Gaussian noise to target function.
        shortcut (Bool): Toggle for injecting artifical shortcut to data. 
        training_ratio (Float): A ratio that determines size of training data split.
        val_ratio (Float): A ratio that determines size of validation data split.
        test_ratio (Float): A ratio that determines the size of test data split.

    Returns:
        Dict: All relevant data for visualization and training of a regression model, which include: 
              time, true function, target function, all splits, features (linear, annual seasonality), and ending timestamps
              of splits.
    """

    # Noise array
    rng = np.random.default_rng(seed=seed)
    f_noise = rng.normal(loc=0, scale=noise_level, size=n_samples)

    # Initialize linear + annual seasonality target function
    time = np.arange(n_samples)
    f_linear = time*0.02

    dist_from_day182 = (time % 365) - 182
    normalized_dfd182 = dist_from_day182 * np.pi / 182
    annual_seasonality = np.sin(normalized_dfd182)

    f_true = f_linear + annual_seasonality * time / 400

    # Synthetic data, target + noise
    f_target = f_true + f_noise

    # Construct features vector
    features = np.column_stack((time, f_linear, annual_seasonality))

    # Split data
    # Check valid ratios
    if not np.isclose(training_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0.")

    # Determine ending indices for each data split, remainder go to test split.
    train_end_index = int(n_samples * training_ratio)
    validation_end_index = train_end_index + int(n_samples * val_ratio)

    # Create tabular column for table
    split = np.empty(n_samples, dtype=object)
    split[:train_end_index] = "train"
    split[train_end_index:validation_end_index] = "val"
    split[validation_end_index:] = "test"

    return {    
        "time" : time,
        "true_target" : f_true,
        "target" : f_target,
        "split" : split,
        "features" : features,

        "train" : {
            "time" : time[:train_end_index],
            "X" : features[:train_end_index],
            "y" : f_target[:train_end_index],
        },

        "val" : {
            "time" : time[train_end_index:validation_end_index],
            "X" : features[train_end_index:validation_end_index],
            "y" : f_target[train_end_index:validation_end_index],
        },

        "test" : {
            "time" : time[validation_end_index:],
            "X" : features[validation_end_index:],
            "y" : f_target[validation_end_index:],
        },

        "metadata" : {
            "train_end" : train_end_index,
            "val_end" : validation_end_index,
            "feature_names" : ["time", "f_linear", "annual_seasonality"],
        },
    }   # Return 
