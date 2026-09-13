import numpy as np
import pytest
import torch
from zennit.composites import EpsilonPlus

from xai_mini_research.diagnostics import inspect_hidden_activations
from xai_mini_research.models import MLPRegressor
from xai_mini_research.explain import explain_mlp_lrp


@pytest.mark.parametrize("activation,cls", [
    ("relu", torch.nn.ReLU), ("lgsigmoid", torch.nn.LogSigmoid), ("gelu", torch.nn.GELU),
])
def test_activation_selection_forward_and_lrp(activation, cls):
    torch.manual_seed(42)
    model = MLPRegressor(3, activation=activation)
    assert isinstance(model.network[1], cls)
    assert isinstance(model.network[3], cls)
    X = np.array([[-1., 0., 1.], [1., 0., -1.]], dtype=np.float32)
    model.train()
    summary, arrays = inspect_hidden_activations(model, X, np.arange(2))
    assert model.training  # Inspection restores caller's state.
    torch.testing.assert_close(torch.from_numpy(arrays["prediction"]), model(torch.from_numpy(X)))
    output, relevance = explain_mlp_lrp(model, X)
    torch.testing.assert_close(output, torch.from_numpy(arrays["prediction"]))
    assert torch.isfinite(relevance).all()
    assert relevance.shape == X.shape
    assert type(EpsilonPlus().module_map({}, "activation", model.network[1])).__name__ == "Pass"
    assert summary["n_samples"] == 2


def test_relu_regional_inactivity_is_distinct_from_dataset_inactivity():
    model = MLPRegressor(1, hidden_dim_1=1, hidden_dim_2=1, activation="relu")
    with torch.no_grad():
        for idx in (0, 2, 4):
            model.network[idx].weight.fill_(1)
            model.network[idx].bias.zero_()
        model.network[4].bias.fill_(3)
    X = np.array([[-2.], [-1.], [2.]], dtype=np.float32)
    summary, arrays = inspect_hidden_activations(model, X, np.array([10, 11, 12]))
    assert summary["last_hidden_all_zero_times"] == [10, 11]
    assert summary["prediction_equals_bias_when_last_hidden_zero"] is True
    assert summary["layers"]["layer_2"]["units_zero_on_all_rows"] == []
    np.testing.assert_array_equal(arrays["prediction"], [3., 3., 5.])
    regional, _ = inspect_hidden_activations(model, X[:2], np.array([10, 11]))
    assert regional["layers"]["layer_2"]["units_zero_on_all_rows"] == [0]


def test_near_zero_is_not_exact_inactivity():
    model = MLPRegressor(1, hidden_dim_1=1, hidden_dim_2=1, activation="lgsigmoid")
    with torch.no_grad():
        model.network[0].weight.zero_()
        model.network[0].bias.fill_(20)
    summary, _ = inspect_hidden_activations(model, np.ones((2, 1)), np.arange(2))
    layer = summary["layers"]["layer_1"]
    assert layer["near_zero_fraction"] == 1
    assert layer["exact_zero_fraction"] == 0


def test_unknown_activation_rejected():
    with pytest.raises(ValueError, match="Unknown activation"):
        MLPRegressor(3, activation="sigmoid")
