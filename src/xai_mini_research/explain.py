from zennit.composites import EpsilonPlus
from zennit.attribution import Gradient
import torch
import torch.nn as nn

def explain_mlp_lrp(model : nn.Module, X, epsilon=1e-6):
    """
    Method for performing layer-wise relevance propagation. Epsilon rule is applied for LRP on a standard gradient attributor.

    Args:
        model (nn.Module): A trained MLP model (class MLPRegressor)
        X (np.array or torch.Tensor): The input matrix with features. Shape (n_samples, n_feats), where n-feats may include the shortcut feature.
        epsilon (float): A float value that defines epsilon for the epsilon LRP-rule. Default 1e-6.

    Returns:
        output (torch.Tensor): Model output of each sample. Shape (n_samples,).
        relevance (torch.Tensor): Tensor containing the relevances of the input features for each sample. Shape (n_samples, n_feats).
            The columns represent each feature in order of the feature array (Typically time, f_linear, annual_seasonality, shortcut_feature).
    """
    composite = EpsilonPlus(epsilon=epsilon)    # Establish rule for LRP (epsilon)
    if not isinstance(X, torch.Tensor):     # Convert X into Tensor
        X = torch.from_numpy(X)

    X = X.detach().clone().float()
    X.requires_grad = True

    model.eval()

    with Gradient(model=model, composite=composite, attr_output=torch.ones_like) as attributor:
        output, relevance = attributor(X)   # Attributor executes the pass and relevance calculation based on rule

    return output.detach(), relevance.detach()

def summarize_mlp_lrp(model: nn.Module, processed_data, split="test", epsilon=1e-6):
    """
    Summarize MLP LRP relevance values for one processed data split.

    Args:
        model (nn.Module): A trained MLP model.
        processed_data (dict): Dictionary returned by preprocess.
        split (str): Data split to explain. Default "test".
        epsilon (float): Stabilizer used by the Zennit EpsilonPlus composite.

    Returns:
        dict: relevance summary by feature.
    """
    X = processed_data[split]["X_scaled"]
    feature_names = processed_data["metadata"]["feature_names"]
    output, relevance = explain_mlp_lrp(model=model, X=X, epsilon=epsilon)

    mean_abs_relevance = relevance.abs().mean(dim=0)
    mean_signed_relevance = relevance.mean(dim=0)
    total_mean_abs_relevance = mean_abs_relevance.sum()

    if total_mean_abs_relevance.item() == 0:
        relevance_share = torch.zeros_like(mean_abs_relevance)
    else:
        relevance_share = mean_abs_relevance / total_mean_abs_relevance

    by_feature = {}
    for name, mean_abs, mean_signed, share in zip(
        feature_names,
        mean_abs_relevance,
        mean_signed_relevance,
        relevance_share,
    ):
        by_feature[name] = {
            "mean_abs_relevance": float(mean_abs.item()),
            "mean_signed_relevance": float(mean_signed.item()),
            "relevance_share": float(share.item()),
        }

    shortcut_feature = "shortcut_polynomial"
    shortcut_relevance_share = None
    if shortcut_feature in by_feature:
        shortcut_relevance_share = by_feature[shortcut_feature]["relevance_share"]

    return {
        "method": "zennit_lrp_epsilon_plus",
        "split": split,
        "epsilon": epsilon,
        "n_samples": int(X.shape[0]),
        "feature_names": list(feature_names),
        "output_mean": float(output.mean().item()),
        "output_std": float(output.std(unbiased=False).item()),
        "total_mean_abs_relevance": float(total_mean_abs_relevance.item()),
        "shortcut_relevance_share": shortcut_relevance_share,
        "by_feature": by_feature,
    }
