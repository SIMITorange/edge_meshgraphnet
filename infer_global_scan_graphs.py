"""\
Inference + CV analysis + 3D visualization for global ring/space scan graphs.

This is the paired script for `generate_global_width_space_scan_graphs.py`.

What it does
1) Load all `.npz` graphs in a folder (default: outputs/generated_graphs_global_scan).
2) Build node features using the fitted Normalizer: [x_norm, y_norm, doping_norm, vds_norm].
3) Run checkpointed MeshGraphNet to predict config.OUTPUT_FIELD (default ElectrostaticPotential).
4) For each graph, compute cutline gradient uniformity CV at x = x_cut.
5) Write a table (txt) with (ring_width, space_width, cv, ...).
6) Plot a 3D surface/scatter: X=ring_width, Y=space_width, Z=CV, colormap='RdYlBu_r'.
7) Save per-graph predictions (.npz) and quicklook visualization PNGs.

Outputs (default under outputs/sensitivity_global_scan)
- global_scan_results.txt
- plots/cv_3d.png
- preds/npz/*.pred.npz
- preds/figs/*_combined.png (+ colorbar)

Usage
  python infer_global_scan_graphs.py --base_group n43 --base_sheet 0 --checkpoint outputs\checkpoints\meshgraphnet_epoch_3000.pt

Notes
- CV uses |dphi/dy| mean in denominator (see infer_generated_graphs.py).
- If a graph has too few points on the cutline, CV is NaN.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import amp
from torch_geometric.data import Data

import config
from model import MeshGraphNet
from normalization import Normalizer
from utils import visualization


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer global scan graphs and compute CV")
    p.add_argument("--base_group", type=str, required=True)
    p.add_argument("--base_sheet", type=int, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--graphs_dir", type=str, default=None, help="Default outputs/generated_graphs_global_scan")
    p.add_argument("--output_dir", type=str, default=None, help="Default outputs/sensitivity_global_scan")

    p.add_argument("--x_cut", type=float, default=1.0)
    p.add_argument("--x_tol", type=float, default=0.02)
    p.add_argument("--min_points", type=int, default=50)

    p.add_argument("--max_graphs", type=int, default=None, help="Optional limit for debugging")
    return p.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device)


def load_graph(path: Path) -> Dict:
    obj = np.load(path, allow_pickle=True)
    meta = obj["meta"].item() if "meta" in obj else {}
    return {
        "pos": obj["pos"].astype(np.float32),
        "edge_index": obj["edge_index"].astype(np.int64),
        "doping": obj["doping"].astype(np.float32),
        "vds": float(np.asarray(obj["vds"]).reshape(-1)[0]),
        "meta": meta,
    }


def build_input_features(normalizer: Normalizer, pos: np.ndarray, doping: np.ndarray, vds: float) -> np.ndarray:
    x_coord = pos[:, 0]
    y_coord = pos[:, 1]
    return normalizer.transform_x(x_coord=x_coord, y_coord=y_coord, doping=doping, vds=vds)


def infer_one(
    model: MeshGraphNet,
    device: torch.device,
    use_amp: bool,
    x_feat: np.ndarray,
    pos: np.ndarray,
    edge_index: np.ndarray,
) -> np.ndarray:
    x_t = torch.from_numpy(x_feat).float().to(device)
    pos_t = torch.from_numpy(pos).float().to(device)
    edge_t = torch.from_numpy(edge_index).long().to(device)
    data = Data(x=x_t, pos=pos_t, edge_index=edge_t)

    with torch.no_grad():
        with amp.autocast(
            device_type=device.type if device.type != "meta" else "cuda",
            dtype=config.AMP_DTYPE if device.type == "cuda" else None,
            enabled=use_amp,
        ):
            pred_norm = model(data)[config.OUTPUT_FIELD]

    pred_norm = pred_norm.float() if pred_norm.dtype != torch.float32 else pred_norm
    return pred_norm.squeeze(-1).detach().cpu().numpy()


def cutline_gradient_uniformity(
    pos: np.ndarray,
    phi: np.ndarray,
    x_cut: float,
    x_tol: float,
    min_points: int,
) -> Tuple[float, Dict]:
    x = pos[:, 0]
    y = pos[:, 1]
    mask = np.abs(x - x_cut) <= x_tol
    if np.sum(mask) < min_points:
        return float("nan"), {"count": int(np.sum(mask))}

    y_sel = y[mask]
    phi_sel = phi[mask]
    order = np.argsort(y_sel)
    y_sel = y_sel[order]
    phi_sel = phi_sel[order]

    dy = np.diff(y_sel)
    dphi = np.diff(phi_sel)
    valid = np.abs(dy) > 1e-12
    if np.sum(valid) < max(5, min_points // 10):
        return float("nan"), {"count": int(np.sum(mask)), "valid": int(np.sum(valid))}

    grad = np.zeros_like(dphi)
    grad[valid] = dphi[valid] / dy[valid]
    grad = grad[valid]

    mean_abs = float(np.mean(np.abs(grad)))
    std = float(np.std(grad))
    cv = std / (mean_abs + 1e-12)

    return cv, {
        "count": int(np.sum(mask)),
        "grad_mean_abs": mean_abs,
        "grad_std": std,
        "cv": cv,
    }


def cutline_gradient_profile(
    pos: np.ndarray,
    phi: np.ndarray,
    x_cut: float,
    x_tol: float,
    min_points: int,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Return (y_mid, dphi_dy, debug) along a cutline near x_cut.

    y_mid has length M, representing midpoints between sorted y samples.
    dphi_dy has length M, computed by finite differences.
    """
    x = pos[:, 0]
    y = pos[:, 1]
    mask = np.abs(x - x_cut) <= x_tol
    count = int(np.sum(mask))
    if count < min_points:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32), {"count": count}

    y_sel = y[mask]
    phi_sel = phi[mask]
    order = np.argsort(y_sel)
    y_sel = y_sel[order]
    phi_sel = phi_sel[order]

    dy = np.diff(y_sel)
    dphi = np.diff(phi_sel)
    valid = np.abs(dy) > 1e-12
    valid_count = int(np.sum(valid))
    if valid_count < max(5, min_points // 10):
        return (
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.float32),
            {"count": count, "valid": valid_count},
        )

    y_mid = 0.5 * (y_sel[:-1] + y_sel[1:])
    grad = np.zeros_like(dphi)
    grad[valid] = dphi[valid] / dy[valid]

    y_mid = y_mid[valid].astype(np.float32)
    grad = grad[valid].astype(np.float32)

    dbg = {
        "count": count,
        "valid": valid_count,
        "y_min": float(y_sel.min()),
        "y_max": float(y_sel.max()),
    }
    return y_mid, grad, dbg


def main() -> None:
    args = parse_args()
    config.ensure_output_dirs()

    device = config.DEVICE
    use_amp = config.USE_MIXED_PRECISION and device.type == "cuda"

    graphs_dir = Path(args.graphs_dir) if args.graphs_dir else (config.OUTPUT_DIR / "generated_graphs_global_scan")
    out_dir = Path(args.output_dir) if args.output_dir else (config.OUTPUT_DIR / "sensitivity_global_scan")
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_root = out_dir / "preds"
    preds_npz_dir = preds_root / "npz"
    preds_fig_dir = preds_root / "figs"
    cutline_root = preds_root / "cutline_dy"
    cutline_txt_dir = cutline_root / "txt"
    cutline_fig_dir = cutline_root / "fig"
    preds_npz_dir.mkdir(parents=True, exist_ok=True)
    preds_fig_dir.mkdir(parents=True, exist_ok=True)
    cutline_txt_dir.mkdir(parents=True, exist_ok=True)
    cutline_fig_dir.mkdir(parents=True, exist_ok=True)

    norm_path = config.NORM_DIR / f"{config.OUTPUT_FIELD}_normalizer.npz"
    normalizer = Normalizer.load(norm_path)

    ckpt = load_checkpoint(Path(args.checkpoint), device=device)
    model = MeshGraphNet(
        input_dim=4,
        hidden_dim=config.HIDDEN_DIM,
        num_message_passing_steps=config.NUM_MESSAGE_PASSING_STEPS,
        activation=config.ACTIVATION,
        dropout=config.DROPOUT,
        target_field=config.OUTPUT_FIELD,
        use_grad_checkpoint=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Collect scan graphs
    base_prefix = f"{args.base_group}_s{args.base_sheet}_scan_"
    graph_files = sorted([p for p in graphs_dir.glob("*.npz") if p.name.startswith(base_prefix)])
    if not graph_files:
        raise FileNotFoundError(f"No scan graphs found in {graphs_dir} for prefix {base_prefix}")

    if args.max_graphs is not None:
        graph_files = graph_files[: int(args.max_graphs)]

    results: List[Dict] = []

    for path in graph_files:
        g = load_graph(path)
        meta = g.get("meta", {})
        ring_w = float(meta.get("ring_width", float("nan")))
        space_w = float(meta.get("space_width", float("nan")))

        x = build_input_features(normalizer, g["pos"], g["doping"], g["vds"])
        pred_norm = infer_one(model, device, use_amp, x, g["pos"], g["edge_index"])
        pred_phys = normalizer.inverse_transform_y(pred_norm, vds=g["vds"])

        cv, dbg = cutline_gradient_uniformity(g["pos"], pred_phys, args.x_cut, args.x_tol, args.min_points)

        # Export cutline gradient profile (dphi/dy vs y)
        y_mid, dphi_dy, prof_dbg = cutline_gradient_profile(
            g["pos"], pred_phys, args.x_cut, args.x_tol, args.min_points
        )

        rec = {
            "file": path.name,
            "ring_width": ring_w,
            "space_width": space_w,
            "cv": float(cv),
            "cutline": dbg,
            "cutline_profile": prof_dbg,
        }
        results.append(rec)

        np.savez(
            preds_npz_dir / path.with_suffix(".pred.npz").name,
            pos=g["pos"],
            edge_index=g["edge_index"],
            pred=pred_phys,
            vds=np.asarray([g["vds"]], dtype=np.float32),
            meta=meta,
            cutline=json.dumps(dbg, ensure_ascii=False),
        )

        fig_prefix = preds_fig_dir / path.with_suffix("").name
        visualization.scatter_field_comparison(
            pos=torch.from_numpy(g["pos"]),
            pred=torch.from_numpy(pred_phys),
            target=torch.from_numpy(pred_phys),
            save_prefix=fig_prefix,
            title_prefix=f"ring={ring_w:.2f} space={space_w:.2f}",
            edge_index=torch.from_numpy(g["edge_index"]),
            use_mesh=True,
            field_name=config.OUTPUT_FIELD,
        )

        # Save cutline gradient profile to txt + plot
        safe_stem = path.with_suffix("").name
        txt_out = cutline_txt_dir / f"{safe_stem}_cutline_dphi_dy.txt"
        with txt_out.open("w", encoding="utf-8") as f:
            f.write(f"file={path.name}\n")
            f.write(f"ring_width={ring_w}\n")
            f.write(f"space_width={space_w}\n")
            f.write(f"x_cut={args.x_cut} x_tol={args.x_tol} min_points={args.min_points}\n")
            f.write(json.dumps(prof_dbg, ensure_ascii=False) + "\n")
            f.write("y_mid\tdphi_dy\n")
            for yy, gg in zip(y_mid.tolist(), dphi_dy.tolist()):
                f.write(f"{yy}\t{gg}\n")

        # Plot profile
        import matplotlib.pyplot as plt

        fig2 = plt.figure(figsize=(6, 4))
        ax2 = fig2.add_subplot(111)
        if y_mid.size == 0:
            ax2.set_title(f"dphi/dy vs y (insufficient cutline points)\nring={ring_w:.2f} space={space_w:.2f}")
            ax2.set_xlabel("y (um)")
            ax2.set_ylabel("dphi/dy")
        else:
            ax2.plot(y_mid, dphi_dy, linewidth=1.2, color="k")
            ax2.set_title(f"dphi/dy vs y\nring={ring_w:.2f} space={space_w:.2f}")
            ax2.set_xlabel("y (um)")
            ax2.set_ylabel("dphi/dy")
            ax2.grid(True, alpha=0.25)
        fig2.tight_layout()
        fig_out = cutline_fig_dir / f"{safe_stem}_cutline_dphi_dy.png"
        fig2.savefig(fig_out, dpi=200)
        plt.close(fig2)

    # Write txt
    txt_path = out_dir / "global_scan_results.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"base_group={args.base_group} base_sheet={args.base_sheet}\n")
        f.write(f"x_cut={args.x_cut} x_tol={args.x_tol} min_points={args.min_points}\n")
        f.write("file\tring_width\tspace_width\tcv\tcutline_count\n")
        for r in results:
            cnt = r["cutline"].get("count", -1) if isinstance(r.get("cutline"), dict) else -1
            f.write(f"{r['file']}\t{r['ring_width']}\t{r['space_width']}\t{r['cv']}\t{cnt}\n")

    # 3D plot
    import matplotlib.pyplot as plt
    from matplotlib import cm

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    ring = np.asarray([r["ring_width"] for r in results], dtype=float)
    space = np.asarray([r["space_width"] for r in results], dtype=float)
    cv = np.asarray([r["cv"] for r in results], dtype=float)

    finite = np.isfinite(ring) & np.isfinite(space) & np.isfinite(cv)
    ring_f = ring[finite]
    space_f = space[finite]
    cv_f = cv[finite]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    if cv_f.size == 0:
        ax.set_title("CV 3D (no finite CV points)")
    else:
        norm = plt.Normalize(vmin=float(np.min(cv_f)), vmax=float(np.max(cv_f)))
        colors = cm.get_cmap("RdYlBu_r")(norm(cv_f))
        sc = ax.scatter(ring_f, space_f, cv_f, c=colors, s=25, depthshade=True)
        mappable = cm.ScalarMappable(norm=norm, cmap="RdYlBu_r")
        mappable.set_array(cv_f)
        cbar = fig.colorbar(mappable, ax=ax, pad=0.1, shrink=0.7)
        cbar.set_label("CV", rotation=90)

        ax.set_title("Cutline Gradient Uniformity (CV)")

    ax.set_xlabel("Ring width (um)")
    ax.set_ylabel("Space width (um)")
    ax.set_zlabel("CV")

    fig.tight_layout()
    fig_path = plots_dir / "cv_3d.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(f"Wrote {txt_path}")
    print(f"Saved 3D plot to {fig_path}")


if __name__ == "__main__":
    main()
