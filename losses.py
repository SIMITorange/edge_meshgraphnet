"""
Module purpose:
    Composite loss functions emphasizing both node-wise accuracy and edge-wise gradients.
Inputs:
    Predicted and target tensors, edge_index for gradient terms, optional boundary mask.
Outputs:
    Total loss tensor and dictionary of component losses for logging.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F

import config


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    edge_index: torch.Tensor,
    boundary_mask: Optional[torch.Tensor] = None,
    node_weight: float = config.NODE_LOSS_WEIGHT,
    grad_weight: float = config.GRAD_LOSS_WEIGHT,
    boundary_weight: float = config.BOUNDARY_LOSS_WEIGHT,
) -> (torch.Tensor, Dict[str, float]):
    """
    Compute composite loss.
    Inputs:
        pred: [N, 1] predicted normalized values.
        target: [N, 1] ground truth normalized values.
        edge_index: [2, E] edge indices.
        boundary_mask: Optional [N] float mask indicating boundary nodes (1 means boundary).
        node_weight, grad_weight, boundary_weight: Scalars weighting components.
    Outputs:
        total_loss: Scalar tensor.
        components: Dict of scalar floats.
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    node_loss_all = F.smooth_l1_loss(pred_flat, target_flat, reduction="none")
    node_loss = node_loss_all.mean()

    src, dst = edge_index
    pred_diff = pred_flat[src] - pred_flat[dst]
    target_diff = target_flat[src] - target_flat[dst]
    grad_loss_all = F.smooth_l1_loss(pred_diff, target_diff, reduction="none")
    grad_loss = grad_loss_all.mean()

    if boundary_mask is not None and boundary_weight > 0:
        boundary_mask = boundary_mask.to(pred_flat.device)
        weighted = node_loss_all * boundary_mask
        # Avoid division by zero
        denom = boundary_mask.sum().clamp(min=1.0)
        boundary_term = weighted.sum() / denom
    else:
        boundary_term = torch.tensor(0.0, device=pred.device)

    total = node_weight * node_loss + grad_weight * grad_loss + boundary_weight * boundary_term
    components = {
        "node_loss": float(node_loss.detach().cpu().item()),
        "grad_loss": float(grad_loss.detach().cpu().item()),
        "boundary_loss": float(boundary_term.detach().cpu().item()),
        "total": float(total.detach().cpu().item()),
    }
    return total, components

