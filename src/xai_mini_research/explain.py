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
