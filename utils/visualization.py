"""
Module purpose:
    Visualization helpers for training curves and field predictions vs. ground truth.
    Style: Asymmetric Linear Scale with Heatmap Colors (RdYlBu_r).
"""

from pathlib import Path
from typing import Dict, Sequence, Optional

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm, Normalize
import numpy as np
import torch

import config


def plot_training_curves(
    history: Dict[str, Sequence[float]],
    save_path: Path,
    title: str = "Training Curves",
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (key, values) in enumerate(history.items()):
        plt.plot(values, label=key, linewidth=1.5, color=colors[i % len(colors)])
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.yscale('log')
    plt.title(title)
    plt.legend(frameon=True)
    plt.grid(True, which="both", ls="-", alpha=0.2)
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
    edge_index: Optional[torch.Tensor] = None,
    use_mesh: bool = True,
    field_name: Optional[str] = None,
) -> None:
    """
    Create side-by-side plots using Asymmetric Linear Scale and Heatmap Colors.
    
    Key Features:
    - Range: Determined by 2nd and 98th percentile of the ACTUAL data (Pred + True).
      This removes outliers but keeps the asymmetry (e.g., high P+, low N-).
    - Zero Alignment: Uses TwoSlopeNorm to ensure 0 is always Yellow/White, 
      even if the positive range is much larger than the negative range.
    - Colormap: 'RdYlBu_r'.
    """
    save_prefix.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Prepare Data
    pos_np = pos.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy().flatten()
    target_np = target.detach().cpu().numpy().flatten()
    error_np = pred_np - target_np
    
    # 2. Determine Comparison Limits
    # Combine data to find a common scale for Pred and Truth
    combined_data = np.concatenate([pred_np, target_np])

    # Use percentiles to determine the robust range (approx 95% coverage)
    v_min_robust = np.percentile(combined_data, 2)  # Bottom 2%
    v_max_robust = np.percentile(combined_data, 98) # Top 2%

    # Safety check: if data is flat (all zeros), set dummy range
    if v_max_robust == v_min_robust:
        v_max_robust += 1e-12
        v_min_robust -= 1e-12

    # 3. Create Norm based on field
    if (field_name or config.OUTPUT_FIELD) == "SpaceCharge":
        # Preserve asymmetric range for SpaceCharge (as before)
        if v_min_robust < 0 and v_max_robust > 0:
            data_norm = TwoSlopeNorm(vmin=v_min_robust, vcenter=0, vmax=v_max_robust)
        elif v_min_robust >= 0:
            data_norm = Normalize(vmin=0, vmax=v_max_robust)
        else: # v_max_robust <= 0
            data_norm = Normalize(vmin=v_min_robust, vmax=0)
    else:
        # For other fields, use symmetric range about zero with max absolute bound
        max_abs = max(abs(v_min_robust), abs(v_max_robust))
        if max_abs == 0:
            max_abs = 1e-12
        v_min_sym, v_max_sym = -max_abs, max_abs
        data_norm = TwoSlopeNorm(vmin=v_min_sym, vcenter=0, vmax=v_max_sym)

    # 4. Error Limits (Symmetric centered at 0)
    # Error should still be symmetric to judge bias
    err_limit = np.percentile(np.abs(error_np), 98) # 98th percentile of error magnitude
    if err_limit == 0: err_limit = 1e-12
    error_norm = TwoSlopeNorm(vmin=-err_limit, vcenter=0, vmax=err_limit)

    # 5. Triangulation
    tri_obj = mtri.Triangulation(pos_np[:, 0], pos_np[:, 1])

    # 6. Plot Config
    # --- Font Configuration ---
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'

    field_label = field_name or config.OUTPUT_FIELD
    cmap_main = "RdYlBu_r" # Red(+) - Yellow(0) - Blue(-)
    # cmap_error = "bwr"     # Blue(-) - White(0) - Red(+) (Unused in shared-legend mode)

    # --- Grouped Plot Configuration ---
    # Adjust 'nrows' and 'ncols' to change layout (e.g., 1 row, 3 cols).
    nrows, ncols = 3, 1
    # Adjust 'figsize' to change overall image size (width, height).
    figsize = (6, 3) 
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)
    if nrows * ncols == 1: axes = [axes] # Handle single plot case if needed

    # Data specifications
    specs = [
        {"data": pred_np,   "ax": axes[0], "label": "Prediction"},
        {"data": target_np, "ax": axes[1], "label": "Ground Truth"},
        {"data": error_np,  "ax": axes[2], "label": "Error"},
    ]

    mappable = None
    for i, spec in enumerate(specs):
        ax = spec["ax"]
        
        # --- Plot Field ---
        # Note: x and y axes are swapped (pos_np[:, 1] is horizontal y, pos_np[:, 0] is vertical x)
        if use_mesh:
            im = ax.tripcolor(pos_np[:, 1], pos_np[:, 0], spec["data"], 
                              cmap=cmap_main, norm=data_norm, shading='gouraud', alpha=0.95)
        else:
            im = ax.scatter(pos_np[:, 1], pos_np[:, 0], c=spec["data"], 
                            cmap=cmap_main, norm=data_norm, s=5, alpha=0.9)
        
        mappable = im # Save for colorbar

        # Layout
        ax.set_ylabel("x (um)", fontsize=12)
        # Only show x-label on the bottom-most plot
        if i == len(specs) - 1:
            ax.set_xlabel("y (um)", fontsize=12)
        
        # Axis adjustments
        ax.set_aspect("equal")
        ax.tick_params(labelsize=10, width=2)

    # Invert y-axis (shared) once to ensure 10 -> 0 direction
    axes[0].invert_yaxis()

    # Adjust vertical spacing between subplots
    plt.subplots_adjust(hspace=0.3)

    # Save main grouped figure without colorbar
    save_file = save_prefix.with_name(f"{save_prefix.name}_combined.png")
    plt.savefig(save_file, dpi=1080, bbox_inches='tight')
    plt.close()

    # --- Separate Colorbar Figure ---
    # Create a new figure just for the colorbar
    fig_cbar, ax_cbar = plt.subplots(figsize=(6, 0.5))
    
    cbar = fig_cbar.colorbar(mappable, cax=ax_cbar, orientation='horizontal')
    cbar.set_label(field_label, fontsize=12)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=10)
    
    # Save colorbar figure
    cbar_file = save_prefix.with_name(f"{save_prefix.name}_colorbar.png")
    fig_cbar.savefig(cbar_file, dpi=300, bbox_inches='tight')
    plt.close(fig_cbar)