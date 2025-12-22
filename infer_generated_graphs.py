"""\
Inference + sensitivity analysis on generated (perturbed) graphs.

Pipeline
1) Load generated graphs (.npz) produced by generate_perturbed_graphs.py.
2) Convert to PyG-like Data object with x features matching training:
   [x_norm, y_norm, doping_norm, vds_norm].
3) Run model checkpoint to predict ElectrostaticPotential (or config.OUTPUT_FIELD).
4) Compare to the *base* sample prediction as reference ("old"), and quantify
   sensitivity by cutline(x=1) potential gradient uniformity.

Gradient uniformity metric (1D)
- Take nodes near x = x_cut within tolerance.
- Sort by y.
- Compute dphi/dy via finite differences.
- Use coefficient of variation: CV = std(grad) / (mean(|grad|)+eps).
  Smaller CV => more uniform.
- Sensitivity score reported as delta_CV = CV_new - CV_base.

Outputs
- outputs/sensitivity/generated_graphs_results.txt
- outputs/sensitivity/plots/*.png (per-parameter bar charts)
- outputs/generated_graphs/preds/*.npz (predictions for each graph)

Usage
  python infer_generated_graphs.py --base_group n43 --base_sheet 0 --checkpoint outputs\checkpoints\meshgraphnet_epoch_3000.pt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
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
    p = argparse.ArgumentParser(description="Infer perturbed graphs and analyze sensitivity")
    p.add_argument("--base_group", type=str, required=True)
    p.add_argument("--base_sheet", type=int, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--graphs_dir", type=str, default=None, help="Folder containing generated .npz")
    p.add_argument("--output_dir", type=str, default=None, help="Default outputs/sensitivity")
    p.add_argument("--x_cut", type=float, default=1.0)
    p.add_argument("--x_tol", type=float, default=0.02)
    p.add_argument("--min_points", type=int, default=50)
    return p.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device)


def load_generated_graph(path: Path) -> Dict:
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


def infer_one(model: MeshGraphNet, device: torch.device, use_amp: bool, x_feat: np.ndarray, pos: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
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


def cutline_gradient_uniformity(pos: np.ndarray, phi: np.ndarray, x_cut: float, x_tol: float, min_points: int) -> Tuple[float, Dict]:
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


def triangulation_edges(pos: np.ndarray) -> np.ndarray:
    import matplotlib.tri as mtri

    tri = mtri.Triangulation(pos[:, 0], pos[:, 1])
    tris = tri.triangles
    # edges from triangles
    e01 = tris[:, [0, 1]]
    e12 = tris[:, [1, 2]]
    e20 = tris[:, [2, 0]]
    edges = np.concatenate([e01, e12, e20], axis=0).astype(np.int64)

    src = edges[:, 0]
    dst = edges[:, 1]
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    src_all = np.concatenate([src, dst], axis=0)
    dst_all = np.concatenate([dst, src], axis=0)
    N = pos.shape[0]
    key = src_all * int(N) + dst_all
    order = np.argsort(key)
    key_sorted = key[order]
    uniq = np.ones_like(key_sorted, dtype=bool)
    uniq[1:] = key_sorted[1:] != key_sorted[:-1]
    return np.stack([src_all[order][uniq], dst_all[order][uniq]], axis=0)


def main() -> None:
    args = parse_args()
    config.ensure_output_dirs()

    device = config.DEVICE
    use_amp = config.USE_MIXED_PRECISION and device.type == "cuda"

    graphs_dir = Path(args.graphs_dir) if args.graphs_dir else (config.OUTPUT_DIR / "generated_graphs")
    out_dir = Path(args.output_dir) if args.output_dir else (config.OUTPUT_DIR / "sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_root = graphs_dir / "preds"
    preds_npz_dir = preds_root / "npz"
    preds_fig_dir = preds_root / "figs"
    preds_npz_dir.mkdir(parents=True, exist_ok=True)
    preds_fig_dir.mkdir(parents=True, exist_ok=True)

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

    # Base reference: find base_params file and synthesize base graph by reading a perturb file's base mapping
    base_tag = f"{args.base_group}_s{args.base_sheet}_"
    gen_files = sorted([p for p in graphs_dir.glob("*.npz") if p.name.startswith(base_tag) and "perturb" in p.name])
    if not gen_files:
        raise FileNotFoundError(f"No generated perturb graphs found in {graphs_dir} for {base_tag}")

    # Use the first perturb file's original pos/doping/vds as base (before warp is lost),
    # so we approximate base by re-loading from HDF5 via generate script? Instead, we store base via user running infer.py.
    # Here: take the unperturbed baseline as the original training sample by reading HDF5 quickly.
    import h5py

    with h5py.File(str(config.HDF5_PATH), "r") as f:
        grp = f[args.base_group]
        base_pos = grp["pos"][:].astype(np.float32)
        fields = grp["fields"][args.base_sheet].astype(np.float32)
        # Read edge_index inside the file context; some files may not store it.
        if "edge_index" in grp:
            base_edge = grp["edge_index"][:].astype(np.int64)
        else:
            base_edge = None

    base_doping = fields[:, config.FIELD_TO_INDEX["DopingConcentration"]].astype(np.float32)
    base_vds = float(fields[:, config.FIELD_TO_INDEX["ElectrostaticPotential"]].max())
    if base_edge is None:
        base_edge = triangulation_edges(base_pos)

    base_x = build_input_features(normalizer, base_pos, base_doping, base_vds)
    base_pred_norm = infer_one(model, device, use_amp, base_x, base_pos, base_edge)
    base_pred_phys = normalizer.inverse_transform_y(base_pred_norm, vds=base_vds)
    base_cv, base_dbg = cutline_gradient_uniformity(base_pos, base_pred_phys, args.x_cut, args.x_tol, args.min_points)

    # Save base prediction
    np.savez(
        preds_npz_dir / f"{args.base_group}_s{args.base_sheet}_base_pred.npz",
        pos=base_pos,
        edge_index=base_edge,
        pred=base_pred_phys,
        vds=np.asarray([base_vds], dtype=np.float32),
        cutline=json.dumps(base_dbg, ensure_ascii=False),
    )

    # Save base visualization (Prediction vs itself; mainly for consistent quicklook)
    base_save_prefix = preds_fig_dir / f"{args.base_group}_s{args.base_sheet}_base_{config.OUTPUT_FIELD}"
    visualization.scatter_field_comparison(
        pos=torch.from_numpy(base_pos),
        pred=torch.from_numpy(base_pred_phys),
        target=torch.from_numpy(base_pred_phys),
        save_prefix=base_save_prefix,
        title_prefix=f"Base {args.base_group} s{args.base_sheet}",
        edge_index=torch.from_numpy(base_edge),
        use_mesh=True,
        field_name=config.OUTPUT_FIELD,
    )

    results = []

    for path in gen_files:
        g = load_generated_graph(path)
        x = build_input_features(normalizer, g["pos"], g["doping"], g["vds"])
        pred_norm = infer_one(model, device, use_amp, x, g["pos"], g["edge_index"])
        pred_phys = normalizer.inverse_transform_y(pred_norm, vds=g["vds"])

        cv, dbg = cutline_gradient_uniformity(g["pos"], pred_phys, args.x_cut, args.x_tol, args.min_points)
        delta_cv = float(cv - base_cv) if np.isfinite(cv) and np.isfinite(base_cv) else float("nan")

        meta = g.get("meta", {})
        rec = {
            "file": path.name,
            "param": meta.get("param", "unknown"),
            "direction": meta.get("direction", "unknown"),
            "replicate": meta.get("replicate", -1),
            "cv": cv,
            "delta_cv": delta_cv,
            "cutline": dbg,
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

        # Save visualization for this perturbed graph
        fig_prefix = preds_fig_dir / path.with_suffix("").name
        visualization.scatter_field_comparison(
            pos=torch.from_numpy(g["pos"]),
            pred=torch.from_numpy(pred_phys),
            target=torch.from_numpy(base_pred_phys),
            save_prefix=fig_prefix,
            title_prefix=f"{meta.get('param','')} {meta.get('direction','')} r{meta.get('replicate','')}",
            edge_index=torch.from_numpy(g["edge_index"]),
            use_mesh=True,
            field_name=config.OUTPUT_FIELD,
        )

    # Write txt summary
    txt_path = out_dir / "generated_graphs_results.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"base_group={args.base_group} base_sheet={args.base_sheet}\n")
        f.write(f"x_cut={args.x_cut} x_tol={args.x_tol} min_points={args.min_points}\n")
        f.write(f"base_cv={base_cv}\n")
        f.write("file\tparam\tdirection\treplicate\tcv\tdelta_cv\tcutline_count\n")
        for r in results:
            cnt = r["cutline"].get("count", -1) if isinstance(r.get("cutline"), dict) else -1
            f.write(
                f"{r['file']}\t{r['param']}\t{r['direction']}\t{r['replicate']}\t"
                f"{r['cv']}\t{r['delta_cv']}\t{cnt}\n"
            )

    # Plot per-parameter bar charts (mean over replicates and directions)
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    by_param_dir = defaultdict(list)
    for r in results:
        by_param_dir[(r["param"], r["direction"])].append(r)

    # Aggregate mean delta_cv by (param, direction)
    agg: Dict[str, Dict[str, float]] = defaultdict(dict)
    for (param, direction), items in by_param_dir.items():
        vals = [it["delta_cv"] for it in items if np.isfinite(it["delta_cv"])]
        agg[param][direction] = float(np.mean(vals)) if vals else float("nan")

    params = sorted(agg.keys())
    plus_vals = [agg[p].get("plus", float("nan")) for p in params]
    minus_vals = [agg[p].get("minus", float("nan")) for p in params]

    plt.figure(figsize=(max(10, len(params) * 0.6), 4))
    x = np.arange(len(params))
    w = 0.35
    plt.bar(x - w / 2, plus_vals, width=w, label="+delta")
    plt.bar(x + w / 2, minus_vals, width=w, label="-delta")
    plt.axhline(0.0, color="k", linewidth=1)
    plt.xticks(x, params, rotation=45, ha="right")
    plt.ylabel("Delta CV (new - base)")
    plt.title("Parameter sensitivity via cutline gradient uniformity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "sensitivity_bar.png", dpi=200)
    plt.close()

    print(f"Wrote {txt_path}")
    print(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
