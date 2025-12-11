"""
Post-training sensitivity analysis for MeshGraphNet surrogate.

Features:
- Extract 1D cutline at x ~= x_cut, export per-sample H5 (y, doping, E_pred, E_true, grad_abs).
- Gradient-based saliency on doping to assess which FLR rings/spaces matter most.
- Heatmap aggregation across samples.

Usage (example):
    python sensitivity_analysis.py --checkpoint outputs/checkpoints/meshgraphnet_epoch_200.pt --x-cut 1.0 --epsilon 1e-3 --output-dir outputs/analysis
    python sensitivity_analysis.py --checkpoint outputs/checkpoints/meshgraphnet_epoch_200.pt --x-cut 1.0 --output-dir outputs/analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from data import H5MeshGraphDataset, SampleSpec, enumerate_samples
from model import MeshGraphNet
from normalization import Normalizer
from utils import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GNN gradient sensitivity and 1D export")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (.pt). Defaults to latest in outputs/checkpoints",
    )
    parser.add_argument("--x-cut", type=float, default=1.0, help="X coordinate for vertical cutline")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-3,
        help="Tolerance for selecting nodes on the cutline (abs(x - x_cut) <= epsilon)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.OUTPUT_DIR / "analysis"),
        help="Directory to save H5 exports and heatmaps",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of samples")
    parser.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Skip aggregation heatmap generation",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable fallback-to-nearest-x when no nodes are within epsilon",
    )
    return parser.parse_args()


def latest_checkpoint() -> Path:
    ckpts = sorted(config.CHECKPOINT_DIR.glob("meshgraphnet_epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError("No checkpoints found under outputs/checkpoints")
    return ckpts[-1]


def load_normalizer() -> Normalizer:
    norm_path = config.NORM_DIR / f"{config.OUTPUT_FIELD}_normalizer.npz"
    if not norm_path.exists():
        raise FileNotFoundError(f"Normalizer not found at {norm_path}. Train or provide the file first.")
    return Normalizer.load(norm_path)


def build_model(normalizer: Normalizer, device: torch.device, checkpoint_path: Path) -> MeshGraphNet:
    # Build minimal dataset entry to infer input dim from the first available sample
    samples = enumerate_samples(str(config.HDF5_PATH))
    if len(samples) == 0:
        raise RuntimeError("No samples found in HDF5; cannot build model input dim.")
    tmp_dataset = H5MeshGraphDataset(
        h5_path=str(config.HDF5_PATH),
        samples=[samples[0]],
        output_field=config.OUTPUT_FIELD,
        normalizer=normalizer,
        boundary_percentile=config.BOUNDARY_PERCENTILE,
    )
    input_dim = tmp_dataset[0].x.shape[1]
    model = MeshGraphNet(
        input_dim=input_dim,
        hidden_dim=config.HIDDEN_DIM,
        num_message_passing_steps=config.NUM_MESSAGE_PASSING_STEPS,
        activation=config.ACTIVATION,
        dropout=config.DROPOUT,
        target_field=config.OUTPUT_FIELD,
        use_grad_checkpoint=False,  # disable checkpointing for analysis clarity
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def denorm_doping(normalizer: Normalizer, doping_norm: np.ndarray) -> np.ndarray:
    """Invert doping normalization applied in Normalizer.transform_x."""
    asinh_val = doping_norm * normalizer.doping_log_std + normalizer.doping_log_mean
    return np.sinh(asinh_val) * normalizer.doping_scale


def invert_field_torch(pred_norm: torch.Tensor, normalizer: Normalizer, vds: float) -> torch.Tensor:
    """Inverse-normalize predictions for common output fields in torch for autograd."""
    pred_shifted = pred_norm * normalizer.y_std + normalizer.y_mean
    if normalizer.output_field == "ElectrostaticPotential":
        scaled = torch.sinh(pred_shifted) * normalizer.y_asinh_scale
        return scaled * vds
    if normalizer.output_field in ("ElectricField_x", "ElectricField_y"):
        return torch.sinh(pred_shifted) * normalizer.y_asinh_scale
    if normalizer.output_field == "SpaceCharge":
        return torch.sinh(pred_shifted) * (normalizer.q0 + normalizer.eps)
    # Fallback for densities or others
    return torch.sinh(pred_shifted) * normalizer.y_asinh_scale


def parse_geometry(y_coords: np.ndarray, doping_phys: np.ndarray, zero_frac: float = 0.01) -> List[Dict[str, int]]:
    """Detect P/N segments along the cutline based on doping sign changes.

    Returns a list of segments with labels Ring_i (P-type, >0) or Space_i (N-type, <0).
    Indices are returned in the sorted-by-y order for easy slicing later.
    """
    assert y_coords.ndim == 1 and doping_phys.ndim == 1
    order = np.argsort(y_coords)
    d_sorted = doping_phys[order]

    max_abs = np.max(np.abs(d_sorted)) + 1e-12
    threshold = max_abs * zero_frac
    signs = np.sign(d_sorted)
    signs[np.abs(d_sorted) < threshold] = 0.0

    segments: List[Dict[str, int]] = []
    current_sign = None
    start_idx = None
    ring_count = 0
    space_count = 0

    for idx, s in enumerate(signs):
        if s == 0:
            continue
        if current_sign is None:
            current_sign = s
            start_idx = idx
            continue
        if np.sign(current_sign) != np.sign(s):
            end_idx = idx - 1
            if current_sign > 0:
                ring_count += 1
                segments.append({"label": f"Ring_{ring_count}", "start_sorted": start_idx, "end_sorted": end_idx})
            else:
                space_count += 1
                segments.append({"label": f"Space_{space_count}", "start_sorted": start_idx, "end_sorted": end_idx})
            current_sign = s
            start_idx = idx
    if current_sign is not None and start_idx is not None:
        end_idx = len(signs) - 1
        if current_sign > 0:
            ring_count += 1
            segments.append({"label": f"Ring_{ring_count}", "start_sorted": start_idx, "end_sorted": end_idx})
        else:
            space_count += 1
            segments.append({"label": f"Space_{space_count}", "start_sorted": start_idx, "end_sorted": end_idx})

    # Map sorted indices back to original indices for completeness
    for seg in segments:
        seg["start_idx"] = int(order[seg["start_sorted"]])
        seg["end_idx"] = int(order[seg["end_sorted"]])
    return segments


def extract_cut_indices(pos: torch.Tensor, x_cut: float, eps: float, fallback: bool = True) -> torch.Tensor:
    """Pick nodes whose x is within eps of x_cut; optionally fall back to nearest x when empty.

    Fallback avoids dropping samples when x_cut is slightly off-grid; it takes all nodes that share
    the minimum |x - x_cut| within a small tolerance.
    """
    x = pos[:, 0]
    mask = torch.isfinite(x) & (x.sub(x_cut).abs() <= eps)
    idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if idx.numel() == 0 and fallback:
        dist = x.sub(x_cut).abs()
        min_dist = torch.min(dist)
        tol = min_dist * 0.05  # allow slight slack so ties are included
        idx = torch.nonzero(dist <= min_dist + tol, as_tuple=False).squeeze(-1)
    return idx


def process_sample(
    model: MeshGraphNet,
    normalizer: Normalizer,
    data,
    x_cut: float,
    eps: float,
    use_fallback: bool,
    device: torch.device,
    h5_dir: Path,
    sample_tag: str,
) -> Tuple[Dict[str, float], List[str]]:
    """Run forward, backward, export H5 for one sample. Returns region sensitivities and region order."""
    data = data.to(device)
    data.x.requires_grad_(True)
    model.zero_grad(set_to_none=True)

    pred_norm = model(data)[config.OUTPUT_FIELD].squeeze(-1)
    cut_idx = extract_cut_indices(data.pos, x_cut, eps, fallback=use_fallback)
    if cut_idx.numel() < 2:
        if use_fallback:
            x_min, x_max = float(torch.min(data.pos[:, 0]).cpu()), float(torch.max(data.pos[:, 0]).cpu())
            logger.log(
                f"Sample {sample_tag}: fallback picked nearest x (x_min={x_min:.4f}, x_max={x_max:.4f}, x_cut={x_cut}, eps={eps})"
            )
        else:
            logger.log(f"Sample {sample_tag}: no nodes within epsilon and fallback disabled; skipping.")
            return {}, []
    if cut_idx.numel() < 2:
        logger.log(f"Sample {sample_tag}: still too few nodes after fallback; skipping.")
        return {}, []

    # Sort along y ascending
    y_coords = data.pos[cut_idx, 1]
    sort_order = torch.argsort(y_coords)
    cut_idx = cut_idx[sort_order]
    y_coords = y_coords[sort_order]

    cut_idx_cpu = cut_idx.detach().cpu().numpy()

    pred_cut_norm = pred_norm[cut_idx]
    target_cut_norm = data.y.squeeze(-1)[cut_idx]

    vds_val = float(data.vds.cpu().item())
    pred_cut_phys = invert_field_torch(pred_cut_norm, normalizer, vds_val)
    target_cut_phys = invert_field_torch(target_cut_norm, normalizer, vds_val)

    # Loss: field variance (uniformity metric)
    loss = torch.var(pred_cut_phys)
    loss.backward()

    doping_grad_abs = data.x.grad[cut_idx, 2].detach().abs().cpu().numpy()

    # Denormalize to numpy for export/geometry parsing
    y_np = y_coords.detach().cpu().numpy()
    pred_np = pred_cut_phys.detach().cpu().numpy()
    target_np = target_cut_phys.detach().cpu().numpy()
    doping_norm_np = data.x.detach().cpu().numpy()[cut_idx_cpu, 2]
    doping_phys_np = denorm_doping(normalizer, doping_norm_np)

    segments = parse_geometry(y_np, doping_phys_np)

    # Aggregate sensitivity by segment
    region_sens: Dict[str, float] = {}
    region_order: List[str] = []
    for seg in segments:
        label = seg["label"]
        region_order.append(label)
        start_s = seg["start_sorted"]
        end_s = seg["end_sorted"]
        idx_slice = np.arange(start_s, end_s + 1)
        if idx_slice.size == 0:
            continue
        region_sens[label] = float(np.mean(doping_grad_abs[idx_slice]))

    # Save H5 per sample
    h5_dir.mkdir(parents=True, exist_ok=True)
    h5_path = h5_dir / f"{sample_tag}.h5"
    with h5py.File(h5_path, "w") as f:
        grp = f.create_group(sample_tag)
        grp.create_dataset("y_coords", data=y_np)
        grp.create_dataset("doping", data=doping_phys_np)
        grp.create_dataset("E_pred", data=pred_np)
        grp.create_dataset("E_true", data=target_np)
        grp.create_dataset("grad_abs", data=doping_grad_abs)
        grp.attrs["x_cut"] = x_cut
        grp.attrs["epsilon"] = eps
        grp.attrs["loss_var"] = float(loss.detach().cpu().item())
    logger.log(f"Saved cutline H5 to {h5_path}")

    return region_sens, region_order


def plot_heatmap(
    sens_matrix: np.ndarray,
    region_labels: List[str],
    sample_labels: List[str],
    save_path: Path,
) -> None:
    plt.figure(figsize=(max(6, len(sample_labels) * 0.5), max(4, len(region_labels) * 0.5)))
    im = plt.imshow(sens_matrix, aspect="auto", cmap="inferno")
    plt.colorbar(im, label="Mean |dL/dDoping|")
    plt.xticks(ticks=np.arange(len(sample_labels)), labels=sample_labels, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(region_labels)), labels=region_labels)
    plt.xlabel("Sample")
    plt.ylabel("Region")
    plt.title("Doping Sensitivity Heatmap (variance objective)")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.log(f"Saved sensitivity heatmap to {save_path}")


def main() -> None:
    args = parse_args()
    config.ensure_output_dirs()
    output_dir = Path(args.output_dir)
    h5_dir = output_dir / "linecuts"
    heatmap_path = output_dir / "sensitivity_heatmap.png"

    device = config.DEVICE
    normalizer = load_normalizer()
    ckpt_path = Path(args.checkpoint) if args.checkpoint else latest_checkpoint()
    logger.log(f"Using checkpoint: {ckpt_path}")

    model = build_model(normalizer, device, ckpt_path)

    all_samples = enumerate_samples(str(config.HDF5_PATH))
    if len(all_samples) == 0:
        logger.log("No samples found; aborting analysis.")
        return
    if args.limit is not None:
        all_samples = all_samples[: args.limit]
    if len(all_samples) == 0:
        logger.log("Zero samples after applying limit; aborting analysis.")
        return
    dataset = H5MeshGraphDataset(
        h5_path=str(config.HDF5_PATH),
        samples=all_samples,
        output_field=config.OUTPUT_FIELD,
        normalizer=normalizer,
        boundary_percentile=config.BOUNDARY_PERCENTILE,
    )

    sample_labels: List[str] = []
    region_names: List[str] = []
    sens_records: Dict[str, Dict[str, float]] = {}

    for idx in range(len(dataset)):
        data = dataset[idx]
        spec = all_samples[idx]
        sample_tag = f"{spec.group}_s{spec.sheet}"
        sample_labels.append(sample_tag)

        region_sens, region_order = process_sample(
            model=model,
            normalizer=normalizer,
            data=data,
            x_cut=args.x_cut,
            eps=args.epsilon,
            use_fallback=not args.no_fallback,
            device=device,
            h5_dir=h5_dir,
            sample_tag=sample_tag,
        )
        sens_records[sample_tag] = region_sens
        for name in region_order:
            if name not in region_names:
                region_names.append(name)

    if args.no_heatmap or len(sens_records) == 0 or len(region_names) == 0:
        logger.log("Heatmap skipped or no sensitivity data available.")
        return

    # Build matrix [regions x samples]
    sens_matrix = np.full((len(region_names), len(sample_labels)), np.nan, dtype=np.float32)
    for j, sample_tag in enumerate(sample_labels):
        for i, region_name in enumerate(region_names):
            if sample_tag in sens_records and region_name in sens_records[sample_tag]:
                sens_matrix[i, j] = sens_records[sample_tag][region_name]

    plot_heatmap(sens_matrix, region_names, sample_labels, heatmap_path)


if __name__ == "__main__":
    main()
