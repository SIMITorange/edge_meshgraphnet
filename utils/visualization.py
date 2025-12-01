"""
Module purpose:
    Visualization helpers for training curves and field predictions vs. ground truth.
Inputs:
    Metrics dictionaries and tensor/array data for plotting.
Outputs:
    Saved matplotlib figures to configured directories.
"""

from pathlib import Path
from typing import Dict, Sequence, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_curves(
    history: Dict[str, Sequence[float]],
    save_path: Path,
    title: str = "Training Curves",
) -> None:
    """
    Plot loss curves.
    Inputs:
        history: Dict mapping metric name to list of values per epoch.
        save_path: Destination PNG path.
        title: Plot title.
    Outputs:
        Figure saved to disk.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for key, values in history.items():
        plt.plot(values, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def scatter_field_comparison(
    pos: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    save_prefix: Path,
    title_prefix: str = "",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Create side-by-side scatter plots for predicted, ground truth, and error fields.
    Inputs:
        pos: Tensor [N, 2] with node coordinates (unnormalized if available).
        pred: Tensor [N] predicted field in physical units.
        target: Tensor [N] ground truth field in physical units.
        save_prefix: Path prefix; will create three PNGs with suffixes.
        title_prefix: Optional string prefix for titles.
        vmin, vmax: Optional limits for color scale.
    Outputs:
        Three PNG files saved: pred, target, error.
    """
    save_prefix.parent.mkdir(parents=True, exist_ok=True)
    pos_np = pos.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    error_np = pred_np - target_np

    titles = ["Prediction", "Ground Truth", "Error"]
    data_list = [pred_np, target_np, error_np]
    suffixes = ["pred", "true", "error"]

    for title, data, suffix in zip(titles, data_list, suffixes):
        plt.figure(figsize=(6, 5))
        sc = plt.scatter(
            pos_np[:, 0],
            pos_np[:, 1],
            c=data,
            cmap="coolwarm",
            s=5,
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(sc, label=title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"{title_prefix} {title}".strip())
        plt.tight_layout()
        plt.savefig(save_prefix.with_name(f"{save_prefix.name}_{suffix}.png"), dpi=150)
        plt.close()

