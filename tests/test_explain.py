import pytest
import torch
import torch.nn as nn
from xai_mini_research import generate_time_data, preprocess, explain_mlp_lrp, summarize_lrp_relevance, summarize_mlp_lrp
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

def test_lrp_summary_contains_feature_relevance_shares():
    model, processed_data = train_test_mlp()

    summary = summarize_mlp_lrp(model=model, processed_data=processed_data)

    assert summary["split"] == "test"
    assert summary["feature_names"] == processed_data["metadata"]["feature_names"]
    assert summary["n_samples"] == processed_data["test"]["X_scaled"].shape[0]
    assert summary["shortcut_relevance_share"] is None

    for feature_name in processed_data["metadata"]["feature_names"]:
        assert feature_name in summary["by_feature"]
        assert "mean_abs_relevance" in summary["by_feature"][feature_name]
        assert "mean_signed_relevance" in summary["by_feature"][feature_name]
        assert "relevance_share" in summary["by_feature"][feature_name]

def test_shortcut_lrp_summary_reports_shortcut_share():
    shortcut_model, processed_shortcut_data = train_test_mlp(shortcut=True)

    summary = summarize_mlp_lrp(model=shortcut_model, processed_data=processed_shortcut_data)

    assert summary["shortcut_relevance_share"] == summary["by_feature"]["shortcut_polynomial"]["relevance_share"]
    assert 0.0 <= summary["shortcut_relevance_share"] <= 1.0
    assert torch.isclose(
        torch.tensor(sum(feature["relevance_share"] for feature in summary["by_feature"].values())),
        torch.tensor(1.0),
    )

def test_lrp_relevance_summary_accepts_precomputed_relevance():
    shortcut_model, processed_shortcut_data = train_test_mlp(shortcut=True)
    output, relevance = explain_mlp_lrp(
        model=shortcut_model,
        X=processed_shortcut_data["test"]["X_scaled"],
    )

    summary = summarize_lrp_relevance(
        output=output,
        relevance=relevance,
        feature_names=processed_shortcut_data["metadata"]["feature_names"],
        split="test",
        condition="zeroed",
    )

    assert summary["condition"] == "zeroed"
    assert summary["split"] == "test"
    assert summary["n_samples"] == relevance.shape[0]
    assert summary["shortcut_relevance_share"] == summary["by_feature"]["shortcut_polynomial"]["relevance_share"]

def test_lrp_relevance_summary_requires_matching_feature_names():
    output = torch.ones(3)
    relevance = torch.ones((3, 2))

    with pytest.raises(ValueError):
        summarize_lrp_relevance(
            output=output,
            relevance=relevance,
            feature_names=["only_one_feature"],
        )
