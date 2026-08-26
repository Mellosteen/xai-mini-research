import torch
import torch.nn as nn
from xai_mini_research import generate_time_data, preprocess, explain_mlp_lrp
from xai_mini_research.models import MLPRegressor, train_mlp, set_torch_seed

def train_test_mlp(shortcut=False):
    processed_data = preprocess(generate_time_data(n_samples=300, shortcut=shortcut))
    set_torch_seed(42)
    model = MLPRegressor(input_dim=processed_data["train"]["X_scaled"].shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    train_mlp(model=model, processed_data=processed_data, optimizer=optimizer, criterion=criterion, epochs=10)

    return model, processed_data

def test_lrp_output_and_relevance_have_expected_shape():
    model, processed_data = train_test_mlp()
    shortcut_model, processed_shortcut_data = train_test_mlp(shortcut=True)

    X = processed_data["test"]["X_scaled"]
    X_shortcut = processed_shortcut_data["test"]["X_scaled"]

    output, relevance = explain_mlp_lrp(model=model, X=X)
    shortcut_output, shortcut_relevance = explain_mlp_lrp(model=shortcut_model, X=X_shortcut)

    # Without shortcut
    assert output.shape == (X.shape[0],)
    assert relevance.shape == X.shape

    # With shortcut
    assert shortcut_output.shape == (X_shortcut.shape[0],)
    assert shortcut_relevance.shape == X_shortcut.shape

def test_lrp_finite_relevance_values():
    model, processed_data = train_test_mlp()
    shortcut_model, processed_shortcut_data = train_test_mlp(shortcut=True)

    X = processed_data["test"]["X_scaled"]
    X_shortcut = processed_shortcut_data["test"]["X_scaled"]

    _, relevance = explain_mlp_lrp(model=model, X=X)
    _, shortcut_relevance = explain_mlp_lrp(model=shortcut_model, X=X_shortcut)

    assert torch.isfinite(relevance).all()
    assert torch.isfinite(shortcut_relevance).all()
