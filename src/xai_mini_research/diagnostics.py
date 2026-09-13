"""
Helper functions for checking whether hidden units are active in a given data split.
The measurements are taken after training, using the model's final restored weights.
"""
import numpy as np
import torch


def summarize_hidden_layer(preactivation, activation, near_zero=1e-8):
    """
    Summarize the values before and after one hidden layer's activation function.

    Args:
        preactivation (np.array): Linear layer outputs, with shape (n_samples, n_units).
        activation (np.array): Values after applying ReLU, LogSigmoid, or GELU.
        near_zero (Float): Threshold for counting small absolute values separately from exact zeros.

    Returns:
        summary (Dict): Fractions of zero and near-zero values, including which units
            are zero for every sample in the supplied data.
    """
    nonpositive_inputs = preactivation <= 0
    zero_outputs = activation == 0
    near_zero_outputs = np.abs(activation) <= near_zero

    # axis=1 checks units within each sample; axis=0 checks samples for each unit.
    all_units_zero = zero_outputs.all(axis=1)
    unit_always_zero = zero_outputs.all(axis=0)
    zero_unit_indices = np.flatnonzero(unit_always_zero)
    zero_fraction_per_unit = zero_outputs.mean(axis=0)

    summary = {
        "pre_nonpositive_fraction": float(nonpositive_inputs.mean()),
        "exact_zero_fraction": float(zero_outputs.mean()),
        "near_zero_fraction": float(near_zero_outputs.mean()),
        "all_units_zero_row_fraction": float(all_units_zero.mean()),
        "units_zero_on_all_rows": zero_unit_indices.tolist(),
        "zero_fraction_per_unit": zero_fraction_per_unit.tolist(),
    }
    return summary


def inspect_hidden_activations(model, X, time, near_zero=1e-8):
    """
    Pass inputs through the MLP one layer at a time and keep the intermediate values.

    This follows the same forward pass as MLPRegressor. Keeping the values before
    and after each activation helps check whether ReLU causes the flat prediction
    region. A unit that is zero on these inputs was not necessarily zero during training.

    Args:
        model (MLPRegressor): Trained model with two hidden layers.
        X (np.array or torch.Tensor): Scaled input features, with shape (n_samples, n_features).
        time (np.array): Time indices matching the rows of X.
        near_zero (Float): Threshold used to count small absolute activation values.

    Returns:
        summary (Dict): Hidden-layer statistics and a check against the output bias.
        arrays (Dict): Time, preactivation and activation arrays, and predictions.
            The saved keys use 'pre' for before activation and 'post' for after activation.
    """
    was_training = model.training
    model.eval()

    # Restore the original model mode even if a measurement raises an error.
    try:
        with torch.no_grad():
            inputs = torch.as_tensor(X, dtype=torch.float32)

            # First hidden layer: linear transformation, followed by activation.
            hidden_1_pre = model.network[0](inputs)
            hidden_1_post = model.network[1](hidden_1_pre)

            # Second hidden layer receives the activated values from the first.
            hidden_2_pre = model.network[2](hidden_1_post)
            hidden_2_post = model.network[3](hidden_2_pre)

            # The output layer is linear and has no further activation function.
            predictions = model.network[4](hidden_2_post).squeeze(-1)

        arrays = {
            "time": np.asarray(time),
            "layer_1_pre": hidden_1_pre.cpu().numpy(),
            "layer_1_post": hidden_1_post.cpu().numpy(),
            "layer_2_pre": hidden_2_pre.cpu().numpy(),
            "layer_2_post": hidden_2_post.cpu().numpy(),
            "prediction": predictions.cpu().numpy(),
        }
        layer_1_summary = summarize_hidden_layer(arrays["layer_1_pre"], arrays["layer_1_post"], near_zero)
        layer_2_summary = summarize_hidden_layer(arrays["layer_2_pre"], arrays["layer_2_post"], near_zero)

        # If every value in the second hidden layer is zero, output = weights * 0 + bias.
        last_layer_zero = (arrays["layer_2_post"] == 0).all(axis=1)
        output_bias = float(model.network[4].bias.item())
        zero_times = arrays["time"][last_layer_zero]
        if last_layer_zero.any():
            zero_predictions = arrays["prediction"][last_layer_zero]
            predictions_equal_bias = bool(np.all(zero_predictions == output_bias))
        else:
            predictions_equal_bias = None  # There are no all-zero rows to check.

        summary = {
            "n_samples": len(time),
            "near_zero_threshold": near_zero,
            "layers": {"layer_1": layer_1_summary, "layer_2": layer_2_summary},
            "output_bias": output_bias,
            "last_hidden_all_zero_times": zero_times.tolist(),
            "prediction_equals_bias_when_last_hidden_zero": predictions_equal_bias,
        }
    finally:
        model.train(was_training)

    return summary, arrays


# --------------------------- Visualization --------------------------------------
def plot_activation_diagnostics(arrays, relevance, target, title):
    """
    Plot predictions, hidden-unit inactivity, and raw relevance over the test split.

    Args:
        arrays (Dict): Forward-pass arrays returned by inspect_hidden_activations.
        relevance (np.array): Input relevance for the shortcut model. The last column
            is the shortcut feature, matching the feature order in generate_time_data.
        target (np.array): True test targets in the same order as the time indices.
        title (String): Figure title, including the activation being tested.

    Returns:
        fig (Figure): Three vertically arranged plots with a shared time axis.
    """
    import matplotlib.pyplot as plt

    time = arrays["time"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Prediction plot: symlog keeps both small and extreme predictions visible.
    axes[0].plot(time, target, color="black", label="Target")
    axes[0].plot(time, arrays["prediction"], label="Prediction")
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].set_ylabel("Prediction (symlog)")

    # Hidden-layer plot: 1.0 means that every unit is zero for that sample.
    for layer in (1, 2):
        hidden_outputs = arrays[f"layer_{layer}_post"]
        zero_fraction = (hidden_outputs == 0).mean(axis=1)
        axes[1].plot(time, zero_fraction, label=f"Hidden layer {layer}")
    axes[1].set_ylabel("Fraction exactly zero")
    axes[1].set_ylim(-0.05, 1.05)

    # Relevance plot: use raw absolute values rather than normalized shares.
    absolute_relevance = np.abs(relevance)
    total_relevance = absolute_relevance.sum(axis=1)
    shortcut_relevance = absolute_relevance[:, -1]
    axes[2].plot(time, total_relevance, label="Total absolute relevance")
    axes[2].plot(time, shortcut_relevance, label="Absolute shortcut relevance")
    axes[2].set_yscale("symlog", linthresh=1e-6)
    axes[2].set_ylabel("Raw relevance (symlog)")
    axes[2].set_xlabel("Time index")

    for ax in axes:
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig
