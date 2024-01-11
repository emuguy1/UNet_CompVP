import torch
import numpy as np

def depth_error_metric(pred, target, metric='rmse'):
    """
    Compute depth error metrics between prediction and target.

    Args:
    - pred (torch.Tensor): Predicted depth maps.
    - target (torch.Tensor): Ground truth depth maps.
    - metric (str): The error metric to compute ('rmse', 'mae', 'msle').

    Returns:
    - float: Computed error metric.
    """
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    if metric == 'rmse':
        return np.sqrt(np.mean((pred - target) ** 2))
    elif metric == 'mae':
        return np.mean(np.abs(pred - target))
    elif metric == 'msle':
        # Adding a small constant to avoid log(0)
        return np.mean((np.log(pred + 1) - np.log(target + 1)) ** 2)
    else:
        raise ValueError(f"Unsupported metric: {metric}")