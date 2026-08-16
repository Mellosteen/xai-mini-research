import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from dataclasses import dataclass
import copy

@dataclass(frozen=True)
class MLPTrainingMetrics:
    train_losses: list[float]
    val_losses: list[float]

def set_torch_seed(seed: int):
    torch.manual_seed(seed)

class MLPRegressor(nn.Module):
    """
    MLP for synthetic time series data

    Passes an input Tensor with 3 features through two hidden layers before returning a regression output, a singular value. The model learns
    8 features upon passing into the first hidden layer, 4 features upon passing into the second before converging the outputs
    into one in the output layer.

    Attributes:
        network (nn.Sequential): The construction of the entire neural network. The number of input dimensions begins with 3, but remains
        required for when shortcuts are added.
    """
    def __init__(self, input_dim, hidden_dim_1=8, hidden_dim_2=4, ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim_1),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim_1, out_features=hidden_dim_2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim_2, out_features=1)
        )

    def forward(self, X):
        return self.network(X).squeeze(-1)  # Remove last dimension to match y in dict if dim exists; shape (n,)

def train_mlp(model: nn.Module, processed_data, optimizer: torch.optim.Optimizer, criterion: nn.Module, epochs: int, patience: int | None = None, min_delta: float = 1e-4, seed: int = 42) -> MLPTrainingMetrics:
    """
    Trains the given model for n epochs and returns evaluation metrics. Early stopping is togglable with the entry of the patience param upon function call.
    The best model state is then reloaded after the loop is finished after early stopping is activated.

    Args:
        model (nn.Module): The model that should be used for training.
        processed_data (Dict): Dictionary of all relevant data after preprocessing.
        optimizer (torch.optim.Optimizer): Tool for updating parameters based on gradients.
        criterion (nn.Module): Tool for calculating the loss based on forward pass results.
        epochs (Integer): Number of epochs to run the training loop.
        seed (Integer): Seed used for deterministic DataLoader shuffling.
        patience (Integer or None, default = None): Number of allowed epochs without significant improvement before early-stopping. Early-stopping is disabled when patience is None.
        min_delta (Float): The minimum validation loss improvement needed to reset early-stopping.

    Returns:
        MLPTrainingMetrics: Dataclass containing average loss of training and validation splits.
    """
    # Lists for dataclass attributes
    train_losses = []
    val_losses = []

    # Setup early stopping
    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    # Create DataLoaders for training and validation splits
    X_train = torch.from_numpy(processed_data["train"]["X_scaled"]).float()
    y_train = torch.from_numpy(processed_data["train"]["y"]).float()

    train_set = TensorDataset(X_train, y_train)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, generator=generator)

    X_val = torch.from_numpy(processed_data["val"]["X_scaled"]).float()
    y_val = torch.from_numpy(processed_data["val"]["y"]).float()

    val_set = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    for _ in range(epochs): # One training epoch
        # Begin training split
        train_loss = 0.0
        model.train()

        for features, targets in train_loader:  # One training loop
            optimizer.zero_grad()

            predictions = model(features)
            loss = criterion(predictions, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(targets)

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        # End training split

        # Begin validation split
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for features, targets in val_loader:
                predictions = model(features)
                loss = criterion(predictions, targets)
                val_loss += loss.item() * len(targets)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        # End validation split

        # Check for early stopping
        if patience is not None:
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return MLPTrainingMetrics(
        train_losses=train_losses,
        val_losses=val_losses
    )

def predict_mlp_splits(model : nn.Module, processed_data):
    """
    Returns predicted outputs from MLP with given inputs from processed data.

    Args:
        model (nn.Module): The trained torch.nn model.
        processed_data (Dict): Dictionary containing all relevant information of processed data.

    Returns:
        predictions (Dict): A dictionary of predictions across all splits.
    """
    model.eval()

    X_train = torch.from_numpy(processed_data["train"]["X_scaled"]).float()
    X_val = torch.from_numpy(processed_data["val"]["X_scaled"]).float()
    X_test = torch.from_numpy(processed_data["test"]["X_scaled"]).float()

    with torch.no_grad():
        predictions = {
            "train": model(X_train).numpy(),
            "val": model(X_val).numpy(),
            "test": model(X_test).numpy(),
        }   # Convert results to numpy since metrics.py accepts NumPy dtype

    return predictions
