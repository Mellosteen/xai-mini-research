"""
This file is created and intended for a quick test & comparison of the metrics
of the linear regression model vs. MLP regression model.
"""
import torch
from xai_mini_research import (
    generate_time_data,
    preprocess,
    regression_metrics_all_splits,
)
from xai_mini_research.models import (
    train_mlp,
    train_linear_model,
    predict_mlp_splits,
    predict_splits,
    MLPRegressor,
    set_torch_seed,
)

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

def plot_model_comparison(processed_data, linear_predictions, mlp_predictions):
    import matplotlib.pyplot as plt

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

    y_top = ax.get_ylim()[1]
    ax.text(train_end, y_top, "validation", va="top", ha="left")
    ax.text(val_end, y_top, "test", va="top", ha="left")

    ax.set_title("Linear Regression vs MLP Predictions")
    ax.set_xlabel("Time")
    ax.set_ylabel("Target")
    ax.legend()
    fig.tight_layout()

    plt.show()

def main():
    processed_data = preprocess(generate_time_data())

    # Linear model
    lin_model = train_linear_model(processed_data)
    lin_predictions = predict_splits(lin_model, processed_data)
    lin_metrics = regression_metrics_all_splits(processed_data, lin_predictions)

    # MLP
    set_torch_seed(42)
    mlp_model = MLPRegressor(input_dim=processed_data["train"]["X_scaled"].shape[1])
    train_mlp(model=mlp_model, processed_data=processed_data, optimizer=torch.optim.Adam(mlp_model.parameters(), lr=0.01), criterion=torch.nn.MSELoss(), epochs=50, patience=5, seed=42)
    mlp_predictions = predict_mlp_splits(mlp_model, processed_data)
    mlp_metrics = regression_metrics_all_splits(processed_data, mlp_predictions)

    print_metrics("Linear Regression", lin_metrics)
    print_metrics("MLP", mlp_metrics)

    plot_model_comparison(processed_data, lin_predictions, mlp_predictions)

if __name__ == "__main__":
    main()
