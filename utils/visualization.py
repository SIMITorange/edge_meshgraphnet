"""
Module purpose:
    Visualization helpers for training curves and field predictions vs. ground truth.
    Style: Asymmetric Linear Scale with Heatmap Colors (RdYlBu_r).
"""

from pathlib import Path
from typing import Dict, Sequence, Optional

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.lines as mlines
from matplotlib.colors import TwoSlopeNorm, Normalize
import numpy as np
import torch


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
    
    # 2. Determine Comparison Limits (Asymmetric)
    # Combine data to find a common scale for Pred and Truth
    combined_data = np.concatenate([pred_np, target_np])
    
    # Use percentiles to determine the robust range (approx 95% coverage)
    # This filters out single-pixel spikes while respecting the actual data distribution.
    # e.g., v_min might be -1e15, v_max might be +1e19.
    v_min_robust = np.percentile(combined_data, 2)  # Bottom 2%
    v_max_robust = np.percentile(combined_data, 98) # Top 2%
    
    # Safety check: if data is flat (all zeros), set dummy range
    if v_max_robust == v_min_robust:
        v_max_robust += 1e-12
        v_min_robust -= 1e-12

    # 3. Create Norm based on Data Sign
    # We must handle 3 cases to keep physics intuitive:
    # A) Crossing Zero: min < 0 < max. Use TwoSlopeNorm(vcenter=0).
    # B) All Positive: min > 0. Use Normalize(vmin=0, vmax=max).
    # C) All Negative: max < 0. Use Normalize(vmin=min, vmax=0).
    
    if v_min_robust < 0 and v_max_robust > 0:
        # This allows the 'Red' side to scale to 1e19 and 'Blue' side to -1e15 independently
        # while keeping 0 exactly at the "Yellow" center.
        data_norm = TwoSlopeNorm(vmin=v_min_robust, vcenter=0, vmax=v_max_robust)
    elif v_min_robust >= 0:
        data_norm = Normalize(vmin=0, vmax=v_max_robust)
    else: # v_max_robust <= 0
        data_norm = Normalize(vmin=v_min_robust, vmax=0)

    # 4. Error Limits (Symmetric centered at 0)
    # Error should still be symmetric to judge bias
    err_limit = np.percentile(np.abs(error_np), 98) # 98th percentile of error magnitude
    if err_limit == 0: err_limit = 1e-12
    error_norm = TwoSlopeNorm(vmin=-err_limit, vcenter=0, vmax=err_limit)

    # 5. Triangulation
    tri_obj = mtri.Triangulation(pos_np[:, 0], pos_np[:, 1])

    # 6. Plot Config
    cmap_main = "RdYlBu_r" # Red(+) - Yellow(0) - Blue(-)
    cmap_error = "bwr"     # Blue(-) - White(0) - Red(+)

    specs = [
        {"data": pred_np,   "title": "Prediction",   "suffix": "pred",  "norm": data_norm,  "cmap": cmap_main,  "is_err": False},
        {"data": target_np, "title": "Ground Truth", "suffix": "true",  "norm": data_norm,  "cmap": cmap_main,  "is_err": False},
        {"data": error_np,  "title": "Error",        "suffix": "error", "norm": error_norm, "cmap": cmap_error, "is_err": True},
    ]

    for spec in specs:
        fig, ax = plt.subplots(figsize=(7, 6))
        
        # --- Plot Field ---
        if use_mesh:
            im = ax.tripcolor(tri_obj, spec["data"], cmap=spec["cmap"], norm=spec["norm"], 
                              shading='gouraud', alpha=0.95)
        else:
            im = ax.scatter(pos_np[:, 0], pos_np[:, 1], c=spec["data"], 
                            cmap=spec["cmap"], norm=spec["norm"], s=5, alpha=0.9)

        # --- Draw Depletion Boundary ---
        if not spec["is_err"]:
            # Threshold: 5% of the larger absolute boundary
            # This helps ignore the noise floor near 0
            max_abs_val = max(abs(v_min_robust), abs(v_max_robust))
            thresh = max_abs_val * 0.05
            
            try:
                # Draw lines
                ax.tricontour(tri_obj, spec["data"], levels=[thresh], 
                              colors='black', linestyles='solid', linewidths=0.5, alpha=0.6)
                ax.tricontour(tri_obj, spec["data"], levels=[-thresh], 
                              colors='black', linestyles='dashed', linewidths=0.5, alpha=0.6)
                
                # Proxy Legend
                line_proxy = mlines.Line2D([], [], color='black', linestyle='-', 
                                         linewidth=1, label='Depletion Edge')
                ax.legend(handles=[line_proxy], loc='upper right', fontsize=8, framealpha=0.8)
            except Exception:
                pass

        # --- Colorbar ---
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Space Charge Density")
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()

        # Layout
        ax.set_title(f"{title_prefix} {spec['title']}", fontsize=13, fontweight="bold")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        
        save_file = save_prefix.with_name(f"{save_prefix.name}_{spec['suffix']}.png")
        plt.tight_layout()
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close()