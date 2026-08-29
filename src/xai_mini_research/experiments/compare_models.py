"""
This file is created and intended for a quick test & comparison of the metrics
of the linear regression model vs. kernel ridge regression model vs. MLP regression model.
"""
import torch
import matplotlib.pyplot as plt
from xai_mini_research import generate_time_data, preprocess, regression_metrics_all_splits, save_results, summarize_mlp_lrp, explain_mlp_lrp
from xai_mini_research.models import train_mlp, train_linear_model, predict_mlp_splits, predict_splits, MLPRegressor, set_torch_seed, train_krr_model, predict_krr_splits

def print_metrics(name: str, metrics: dict):
    print(f"\n{name}")
    print("split   MAE      RMSE     R2")
    for split, split_metrics in metrics.items():
        print(
            f"{split:<6} "
            f"{split_metrics['mae']:.4f}  "
            f"{split_metrics['rmse']:.4f}  "
            f"{split_metrics['r2']:.4f}"
        )

def plot_model_comparison(processed_data, linear_predictions, krr_predictions, mlp_predictions, title="Linear Regression vs kRR Regression vs MLP Predictions", show=True, zoom_to_target=False):
    fig, ax = plt.subplots(figsize=(12, 5))
    train_end = processed_data["metadata"]["train_end"]
    val_end = processed_data["metadata"]["val_end"]
    final_time = processed_data["time"][-1]

    ax.axvspan(
        train_end,
        val_end,
        color="gray",
        alpha=0.08,
        label="Validation split",
    )
    ax.axvspan(val_end, final_time, color="gray", alpha=0.14, label="Test split")

    ax.plot(processed_data["time"], processed_data["target"], label="Target", alpha=0.5)

    ax.plot(
        processed_data["train"]["time"],
        linear_predictions["train"],
        label="Linear train",
        linestyle="--",
    )
    ax.plot(
        processed_data["val"]["time"],
        linear_predictions["val"],
        label="Linear val",
        linestyle="--",
    )
    ax.plot(
        processed_data["test"]["time"],
        linear_predictions["test"],
        label="Linear test",
        linestyle="--",
    )

    ax.plot(
        processed_data["train"]["time"],
        krr_predictions["train"],
        label="kRR train",
        linestyle=":",
    )
    ax.plot(
        processed_data["val"]["time"],
        krr_predictions["val"],
        label="kRR val",
        linestyle=":",
    )
    ax.plot(
        processed_data["test"]["time"],
        krr_predictions["test"],
        label="kRR test",
        linestyle=":",
    )

    ax.plot(
        processed_data["train"]["time"],
        mlp_predictions["train"],
        label="MLP train",
        alpha=0.8,
    )
    ax.plot(
        processed_data["val"]["time"],
        mlp_predictions["val"],
        label="MLP val",
        alpha=0.8,
    )
    ax.plot(
        processed_data["test"]["time"],
        mlp_predictions["test"],
        label="MLP test",
        alpha=0.8,
    )

    ax.axvline(train_end, color="black", linestyle=":", linewidth=1)
    ax.axvline(val_end, color="black", linestyle=":", linewidth=1)

    if zoom_to_target:
        target_min = processed_data["target"].min()
        target_max = processed_data["target"].max()
        target_range = target_max - target_min
        margin = target_range * 0.1
        ax.set_ylim(target_min - margin, target_max + margin)

    y_top = ax.get_ylim()[1]
    ax.text(train_end, y_top, "validation", va="top", ha="left")
    ax.text(val_end, y_top, "test", va="top", ha="left")

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Target")
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax

def plot_lrp_relevance_heatmap(feature_names, relevances : torch.Tensor, title="LRP Relevance Heatmap", show=True):
    fig, ax = plt.subplots(figsize=(12, 5))
    abs_relevances = relevances.detach().cpu().abs()   # Shape n_samples, n_feats
    row_sums = abs_relevances.sum(dim=1, keepdim=True).clamp_min(1e-12)
    relevances_norm = (abs_relevances / row_sums).numpy()

    image = ax.imshow(
        relevances_norm,
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    ax.set_title(title)
    ax.set_xlabel("Features")
    ax.set_ylabel("Test samples")

    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=30, ha="right")

    fig.colorbar(image, ax=ax, label="Normalized absolute LRP relevance share")
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax

def plot_lrp_relevance_lines(time, feature_names, relevances : torch.Tensor, title="LRP Relevance Over Time", show=True):
    abs_relevances = relevances.detach().cpu().abs()
    row_sums = abs_relevances.sum(dim=1, keepdim=True).clamp_min(1e-12)
    relevance_norm = (abs_relevances / row_sums).numpy()

    fig, ax = plt.subplots(figsize=(12, 5))

    for feature_idx, feature_name in enumerate(feature_names):
        ax.plot(
            time,
            relevance_norm[:, feature_idx],
            label=feature_name,
            linewidth=1
        )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized absolute LRP relevance share")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax

def plot_lrp_prediction_and_shortcut_relevance_scatter(time, predictions : torch.Tensor, relevances : torch.Tensor, feature_names, target=None, title="LRP Prediction vs. Relevance Scatter", show=True, symlog=True, linthresh=10):
    predictions = predictions.detach().cpu().numpy()
    abs_relevances = relevances.detach().cpu().abs()
    row_sums = abs_relevances.sum(dim=1, keepdim=True).clamp_min(1e-12)
    relevance_norm = (abs_relevances / row_sums).numpy()

    shortcut_idx = feature_names.index("shortcut_polynomial")
    shortcut_relevance = relevance_norm[:, shortcut_idx]

    fig, ax = plt.subplots(figsize=(12,5))

    if target is not None:
        ax.plot(time, target, color="black", alpha=0.5, linewidth=1, label="Target")

    scatter = ax.scatter(
        time,
        predictions,
        c=shortcut_relevance,
        cmap="viridis",
        vmin=0,
        vmax=1,
        s=15,
    )
    fig.colorbar(scatter, ax=ax, label="Shortcut relevance share")

    if symlog:
        ax.set_yscale("symlog", linthresh=linthresh)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Prediction")
    if target is not None:
        ax.legend()
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax

def choose_best_krr(processed_data):
    """
    Helper function to identify best hyperparameters alpha and gamma from a given list (grid search). 
    Used metric is validation RMSE.

    Returns:
        best_model (KernelRidge): best performing model on validation RMSE.
        best_params (Dict): Dictionary containing values of alpha and gamma for best model.
        best_metrics (Dict): The metrics on all splits of the best model.
        best_predictions (Dict): Predictions of best model on all splits.
    """
    alphas = [0.001, 0.005, 0.01, 0.1, 1.0]
    gammas = [0.001, 0.005, 0.01, 0.1, 1.0]
    best_val_rmse = float("inf")

    for alpha in alphas:
        for gamma in gammas:
            model = train_krr_model(processed_data, alpha=alpha, gamma=gamma)
            predictions = predict_krr_splits(model, processed_data)
            metrics = regression_metrics_all_splits(processed_data, predictions)

            if metrics["val"]["rmse"] < best_val_rmse:
                best_val_rmse = metrics["val"]["rmse"]
                best_model = model
                best_params = {"alpha": alpha, "gamma": gamma}
                best_metrics = metrics
                best_predictions = predictions

    return best_model, best_params, best_metrics, best_predictions
                

def main():
    data_seed = 42
    n_samples = 3650
    noise_level = 0.05
    shortcut = True
    shortcut_fit_split = "train_val"
    shortcut_deg = 20
    training_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    processed_data = preprocess(generate_time_data(n_samples=n_samples, seed=data_seed, noise_level=noise_level, training_ratio=training_ratio, val_ratio=val_ratio, test_ratio=test_ratio))
    processed_shortcut_data = preprocess(generate_time_data(n_samples=n_samples, seed=data_seed, noise_level=noise_level, shortcut=shortcut, shortcut_fit_split=shortcut_fit_split, shortcut_deg=shortcut_deg, training_ratio=training_ratio, val_ratio=val_ratio, test_ratio=test_ratio))

    # Linear base model
    lin_model = train_linear_model(processed_data)
    lin_predictions = predict_splits(lin_model, processed_data)
    lin_metrics = regression_metrics_all_splits(processed_data, lin_predictions)

    # Linear shortcut model
    lin_shortcut_model = train_linear_model(processed_shortcut_data)
    lin_shortcut_predictions = predict_splits(lin_shortcut_model, processed_shortcut_data)
    lin_shortcut_metrics = regression_metrics_all_splits(processed_shortcut_data, lin_shortcut_predictions)

    # kRR base
    krr_model, krr_params, krr_metrics, krr_predictions = choose_best_krr(processed_data=processed_data)

    # kRR shortcut
    krr_shortcut_model, krr_shortcut_params, krr_shortcut_metrics, krr_shortcut_predictions = choose_best_krr(processed_data=processed_shortcut_data)

    # MLP base
    seed = 42
    lr = 0.01
    epochs = 50
    patience = 5
    set_torch_seed(seed=seed)
    mlp_model = MLPRegressor(input_dim=processed_data["train"]["X_scaled"].shape[1])
    train_mlp(model=mlp_model, processed_data=processed_data, optimizer=torch.optim.Adam(mlp_model.parameters(), lr=lr), criterion=torch.nn.MSELoss(), epochs=epochs, patience=patience, seed=seed)
    mlp_predictions = predict_mlp_splits(mlp_model, processed_data)
    mlp_metrics = regression_metrics_all_splits(processed_data, mlp_predictions)

    # MLP shortcut
    set_torch_seed(seed=seed)
    mlp_shortcut_model = MLPRegressor(input_dim=processed_shortcut_data["train"]["X_scaled"].shape[1])
    train_mlp(model=mlp_shortcut_model, processed_data=processed_shortcut_data, optimizer=torch.optim.Adam(mlp_shortcut_model.parameters(), lr=lr), criterion=torch.nn.MSELoss(), epochs=epochs, patience=patience, seed=seed)
    mlp_shortcut_predictions = predict_mlp_splits(mlp_shortcut_model, processed_shortcut_data)
    mlp_shortcut_metrics = regression_metrics_all_splits(processed_shortcut_data, mlp_shortcut_predictions)

    mlp_lrp_summary = summarize_mlp_lrp(mlp_model, processed_data, split="test")
    mlp_shortcut_lrp_summary = summarize_mlp_lrp(mlp_shortcut_model, processed_shortcut_data, split="test")

    print_metrics("Linear Regression", lin_metrics)
    print_metrics("Linear Regression Shortcut", lin_shortcut_metrics)
    print_metrics(f"kRR alpha = {krr_params['alpha']} gamma = {krr_params['gamma']}", krr_metrics)
    print_metrics(f"kRR Shortcut alpha = {krr_shortcut_params['alpha']} gamma = {krr_shortcut_params['gamma']}", krr_shortcut_metrics)
    print_metrics("MLP", mlp_metrics)
    print_metrics("MLP Shortcut", mlp_shortcut_metrics)

    # Saving results under results/
    results = {
        "time_data" : {
            "n_samples" : n_samples,
            "noise_level" : noise_level,
            "training_ratio" : training_ratio,
            "val_ratio" : val_ratio,
            "test_ratio" : test_ratio,
            "data_seed" : data_seed,
            "shortcut" : shortcut,
            "shortcut_fit_split" : shortcut_fit_split,
            "shortcut_deg" : shortcut_deg,
        },
        "models" : {
            "linear_regression" : {
                "metrics" : lin_metrics,
                "shortcut_metrics" : lin_shortcut_metrics,
            },
            "kernel_rr" : {
                "params" : krr_params,
                "metrics" : krr_metrics,
                "shortcut_params" : krr_shortcut_params,
                "shortcut_metrics" : krr_shortcut_metrics,
            },
            "mlp" : {
                "params" : {
                    "learning_rate" : lr,
                    "epochs" : epochs,
                    "patience" : patience,
                    "seed" : seed,
                },
                "metrics" : mlp_metrics,
                "shortcut_metrics" : mlp_shortcut_metrics,
                "lrp" : {
                    "baseline" : mlp_lrp_summary,
                    "shortcut" : mlp_shortcut_lrp_summary,
                },
            },
        }
    }

    save_results(results=results)

    plot_model_comparison(
        processed_data,
        lin_predictions,
        krr_predictions,
        mlp_predictions,
        title="Baseline Model Predictions",
        show=False,
    )
    plot_model_comparison(
        processed_shortcut_data,
        lin_shortcut_predictions,
        krr_shortcut_predictions,
        mlp_shortcut_predictions,
        title=f"Shortcut Model Predictions ({shortcut_fit_split}, degree {shortcut_deg})",
        zoom_to_target=True,
        show=True,
    )

    _, relevances = explain_mlp_lrp(model=mlp_model, X=processed_data["test"]["X_scaled"])
    shortcut_outputs, shortcut_relevances = explain_mlp_lrp(model=mlp_shortcut_model, X=processed_shortcut_data["test"]["X_scaled"])

    plot_lrp_relevance_heatmap(  # Default heatmap
        feature_names=processed_data["metadata"]["feature_names"],
        relevances=relevances,
        title="LRP Relevance Heatmap",
        show=True
    )

    plot_lrp_relevance_heatmap(  # Shortcut heatmap
        feature_names=processed_shortcut_data["metadata"]["feature_names"],
        relevances=shortcut_relevances,
        title="LRP Relevance Heatmap w/ Shortcut",
        show=True
    )

    plot_lrp_relevance_lines(   # Default line map
        time=processed_data["test"]["time"],
        feature_names=processed_data["metadata"]["feature_names"],
        relevances=relevances,
        title="LRP Relevance Line Map",
        show=True
    )

    plot_lrp_relevance_lines(   # Shortcut line map
        time=processed_shortcut_data["test"]["time"],
        feature_names=processed_shortcut_data["metadata"]["feature_names"],
        relevances=shortcut_relevances,
        title="LRP Relevance Line Map w/ Shortcut",
        show=True
    )

    plot_lrp_prediction_and_shortcut_relevance_scatter( # Prediction v. Relevance scatter for shortcut model
        time=processed_shortcut_data["test"]["time"],
        predictions=shortcut_outputs,
        relevances=shortcut_relevances,
        feature_names=processed_shortcut_data["metadata"]["feature_names"],
        target=processed_shortcut_data["test"]["y"],
        title="LRP Prediction vs. Relevance Scatter",
        show=True
    )

if __name__ == "__main__":
    main()
