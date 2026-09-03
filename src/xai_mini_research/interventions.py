import numpy as np

def intervene_scaled_shortcut(processed_data, split="test", condition="zeroed", seed=42):
    """
    Intervention validation method to change shortcut feature on the given split. 
    Applies zeroed, noise, reversed, or permuted shortcuts.

    Args:
        processed_data (Dict): Dictionary containing all data.
        split (str): Given split to apply the intervention on. Allowed are 'train', 'val', and 'test'. Test split by default.
        condition (str): The method of intervention to apply to X_scaled split data. Allowed are 'zeroed', 'noise', 'reversed', and 'permuted'. Default is 'zeroed'.
        seed (int): Applied seed for the noise and permuted shortcut. Default seed is 42.

    Returns:
        X_scaled_intervened (dict): Intervention results for the selected split contained in a dictionary with corresponding params as metadata.
    """
    X_scaled_intervened = processed_data[split]["X_scaled"].copy()
    shortcut_idx = processed_data["metadata"]["feature_names"].index("shortcut_polynomial")
    rng = np.random.default_rng(seed)

    if condition == "zeroed":
        X_scaled_intervened[:, shortcut_idx] = 0.0
    elif condition == "noise":
        X_scaled_intervened[:, shortcut_idx] = rng.normal(loc=0.0, scale=1.0, size=X_scaled_intervened.shape[0])
    elif condition == "reversed":
        X_scaled_intervened[:, shortcut_idx] = -X_scaled_intervened[:, shortcut_idx]
    elif condition == "permuted":
        X_scaled_intervened[:, shortcut_idx] = rng.permutation(X_scaled_intervened[:, shortcut_idx])
    else:
        raise ValueError("Invalid condition. Please provide 'zeroed', 'noise', 'reversed', or 'permuted'.")
    
    return {
        "condition": condition,
        "split": split,
        "X_scaled": X_scaled_intervened,
        "changed_feature": "shortcut_polynomial",
    }
