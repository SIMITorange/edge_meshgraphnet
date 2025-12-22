"""
Invert a trained MeshGraphNet to recover node doping from a target field by
optimizing node-wise normalized doping inputs with a frozen forward model.

Usage:
    python inverse_infer.py --group n43 --sheet 0 --checkpoint outputs/checkpoints/meshgraphnet_epoch_200.pt
    python inverse_infer.py --group n105 --sheet 0 --checkpoint outputs/checkpoints/meshgraphnet_epoch_400.pt --lr 1e-2 --iters 20 --smooth 1e-2 --l2 1e-4 --init mean
Notes:
    - This script treats x,y,pos and vds as known and optimizes the normalized
      doping coordinate used as input to the surrogate.
    - Regularization options (smoothness, L2) are provided to reduce ill-posedness.
    - The method relies on the forward model being differentiable and the
      dataset normalizer for consistent normalization logic.
"""

import argparse
from pathlib import Path
from typing import Sequence, Optional

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data

import config
from data import H5MeshGraphDataset, SampleSpec, enumerate_samples
from model import MeshGraphNet
from normalization import Normalizer
from utils import logger, visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inverse inference for MeshGraphNet")
    parser.add_argument("--group", type=str, default=None, help="Group name, e.g., n1")
    parser.add_argument("--sheet", type=int, default=None, help="Sheet index s")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for optimizer")
    parser.add_argument("--iters", type=int, default=2000, help="Number of gradient steps")
    parser.add_argument("--smooth", type=float, default=1e-2, help="Smoothness weight")
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 regularization weight")
    parser.add_argument("--init", type=str, choices=["mean", "truth", "random"], default="mean")
    return parser.parse_args()


def torch_normalize_constants(normalizer: Normalizer, x_coord: np.ndarray, y_coord: np.ndarray, vds: float, device):
    # x,y are normalized in python numpy in Normalizer, but we recreate them as tensors
    x_norm = (torch.from_numpy(x_coord).float().to(device) - normalizer.x_mean) / normalizer.x_std
    y_norm = (torch.from_numpy(y_coord).float().to(device) - normalizer.x_mean) / normalizer.x_std
    vds_norm = (vds - normalizer.vds_mean) / normalizer.vds_std
    vds_norm_t = torch.full_like(x_norm, float(vds_norm)).to(device)
    return x_norm, y_norm, vds_norm_t


def physical_from_u(u: torch.Tensor, normalizer: Normalizer) -> torch.Tensor:
    # u is normalized arcsinh(doping/doping_scale) variable used as input feature
    # doping_phys = sinh( u * doping_log_std + doping_log_mean ) * doping_scale
    arg = u * normalizer.doping_log_std + normalizer.doping_log_mean
    return torch.sinh(arg) * normalizer.doping_scale


def run_inversion(
    model: MeshGraphNet,
    normalizer: Normalizer,
    sample: SampleSpec,
    device: torch.device,
    lr: float,
    iters: int,
    smooth_w: float,
    l2_w: float,
    init_strategy: str,
):
    # Load mesh data and target
    dataset = H5MeshGraphDataset(
        h5_path=str(config.HDF5_PATH),
        samples=[sample],
        output_field=config.OUTPUT_FIELD,
        normalizer=None,  # load raw for ground-truth values
        boundary_percentile=config.BOUNDARY_PERCENTILE,
    )
    data = dataset[0]
    # Extract raw arrays
    pos = data.pos.cpu().numpy()
    edge_index = data.edge_index.to(device)
    x_coord, y_coord = pos[:, 0].astype(np.float32), pos[:, 1].astype(np.float32)
    vds = float(data.vds.item())
    # Target (physical) values from data.y are normalized when using normalizer; our dataset returns raw 'y' because normalizer=None
    # But in dataset we set normalizer=None, so y is in raw units.
    target_phys = data.y.squeeze(-1).cpu().numpy() if data.y is not None else None

    # Build normalized constants in torch
    x_norm_t, y_norm_t, vds_norm_t = torch_normalize_constants(normalizer, x_coord, y_coord, vds, device)

    # Initial u value (normalized doping variable used in network input)
    # We can initialize u using three strategies: mean (0), truth (from dataset if available), random
    if init_strategy == "mean":
        u_init = torch.zeros_like(x_norm_t).to(device)
    elif init_strategy == "random":
        u_init = 0.01 * torch.randn_like(x_norm_t).to(device)
    else:  # truth
        # Use normalization chosen by the normalizer: u = (asinh(doping/doping_scale) - mean)/std
        # We need raw doping from HDF5 so get it by reading from the file directly
        dataset_raw = H5MeshGraphDataset(
            h5_path=str(config.HDF5_PATH),
            samples=[sample],
            output_field=config.OUTPUT_FIELD,
            normalizer=None,
        )
        raw_data = dataset_raw[0]
        # raw_doping is the 3rd column in x (feature order x,y,doping,vds)
        raw_doping = raw_data.x.squeeze(-1)[:, 2].cpu().numpy() if raw_data.x is not None else None
        if raw_doping is None:
            u_init = torch.zeros_like(x_norm_t).to(device)
        else:
            # compute u_init via normalizer formula
            doping_asinh = np.arcsinh(raw_doping / normalizer.doping_scale)
            u_init_np = (doping_asinh - normalizer.doping_log_mean) / normalizer.doping_log_std
            u_init = torch.from_numpy(u_init_np.astype(np.float32)).to(device)

    # Parameter to optimize
    u = nn.Parameter(u_init)
    optimizer = torch.optim.Adam([u], lr=lr)
    loss_fn = torch.nn.MSELoss()

    # Prepare target normalized values using the normalizer
    if target_phys is None:
        raise ValueError("Target array required. Make sure HDF5 dataset contains target values.")
    target_norm_np = normalizer.transform_y(target=target_phys, vds=vds, output_field=config.OUTPUT_FIELD)
    target_norm_t = torch.from_numpy(target_norm_np.squeeze(-1)).float().to(device)

    # Precompute constants
    x_coord_norm_t = x_norm_t.detach()
    y_coord_norm_t = y_norm_t.detach()
    vds_norm_t = vds_norm_t.detach()

    # For smoothness, edges
    src, dst = edge_index

    best_u = None
    best_loss = float("inf")
    for step in range(iters):
        optimizer.zero_grad()
        # Build input features [x_norm, y_norm, u, vds_norm]
        # u has shape [N]; need to stack
        x_in = torch.stack([x_coord_norm_t, y_coord_norm_t, u, vds_norm_t], dim=1)
        graph_data = Data(x=x_in, edge_index=edge_index, pos=torch.from_numpy(pos).float().to(device), vds=torch.tensor([vds], device=device))

        pred_dict = model(graph_data)
        pred = pred_dict[config.OUTPUT_FIELD].squeeze(-1)
        data_loss = loss_fn(pred, target_norm_t)

        # Smoothness: squared differences across edges using u
        smooth_loss = torch.mean((u[src] - u[dst]) ** 2)
        l2_loss = torch.mean(u ** 2)
        loss = data_loss + smooth_w * smooth_loss + l2_w * l2_loss

        loss.backward()
        optimizer.step()

        # Optional: clamp u to sensible range corresponding to physical bounds
        # Here we clamp to +/- 5 standard deviations in normalized space
        with torch.no_grad():
            clip_val = 5.0
            u.clamp_(min=-clip_val, max=clip_val)

        if step % 100 == 0 or step == iters - 1:
            loss_val = float(loss.detach().cpu().item())
            logger.log(f"Step {step}/{iters} loss={loss_val:.6f} data_loss={float(data_loss):.6f}")
        if float(data_loss.detach().cpu().item()) < best_loss:
            best_loss = float(data_loss.detach().cpu().item())
            best_u = u.detach().clone()

    # After optimization, compute final physical doping
    u_final = best_u if best_u is not None else u.detach()
    doping_phys_t = physical_from_u(u_final, normalizer).cpu().numpy()

    # produce predicted field based on final u
    with torch.no_grad():
        x_in = torch.stack([x_coord_norm_t, y_coord_norm_t, u_final, vds_norm_t], dim=1)
        graph_data = Data(x=x_in, edge_index=edge_index, pos=torch.from_numpy(pos).float().to(device), vds=torch.tensor([vds], device=device))
        pred_dict = model(graph_data)
        pred_norm = pred_dict[config.OUTPUT_FIELD].squeeze(-1).cpu().numpy()
        pred_phys = normalizer.inverse_transform_y(pred_norm, vds=vds)

    # Save results
    out_dir = config.OUTPUT_DIR / "inverse"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{sample.group}_s{sample.sheet}_inverse"
    np.savez(base.with_suffix(".npz"), pred=pred_phys, target=target_phys, doping=doping_phys_t, vds=vds)
    logger.log(f"Saved inversion result to {base}.npz")

    # Visualization: overlay predicted and target fields
    visualization.scatter_field_comparison(
        pos=torch.from_numpy(pos).float().to(device),
        pred=torch.from_numpy(pred_phys),
        target=torch.from_numpy(target_phys),
        save_prefix=base,
        title_prefix=f"Inverse {sample.group} s{sample.sheet}",
        edge_index=edge_index,
        use_mesh=True,
        field_name=config.OUTPUT_FIELD,
    )
    logger.log("Saved inverse field visualizations")

    # -----------------------
    # Plot predicted vs ground-truth Doping input
    # -----------------------
    try:
        raw_doping_np = data.x[:, 2].cpu().numpy()
    except Exception:
        raw_doping_np = None
    if raw_doping_np is not None:
        # Create a new prefix for doping plots
        base_doping = base.with_name(f"{base.name}_doping")
        visualization.scatter_field_comparison(
            pos=torch.from_numpy(pos).float(),
            pred=torch.from_numpy(doping_phys_t),
            target=torch.from_numpy(raw_doping_np),
            save_prefix=base_doping,
            title_prefix=f"Inverse {sample.group} s{sample.sheet} (Doping)",
            edge_index=data.edge_index,
            use_mesh=True,
            field_name="DopingConcentration",
        )
        logger.log("Saved inverted doping visualizations")
    else:
        logger.log("No raw doping found in dataset; skipping doping visualization.")


def main() -> None:
    args = parse_args()
    device = config.DEVICE
    config.ensure_output_dirs()

    norm_path = config.NORM_DIR / f"{config.OUTPUT_FIELD}_normalizer.npz"
    if not norm_path.exists():
        raise FileNotFoundError(f"Normalizer not found at {norm_path}. Please run training first.")
    normalizer = Normalizer.load(norm_path)

    if args.checkpoint is None:
        ckpts = sorted(config.CHECKPOINT_DIR.glob("meshgraphnet_epoch_*.pt"))
        if not ckpts:
            raise FileNotFoundError("No checkpoints found. Train the model first.")
        checkpoint_path = ckpts[-1]
    else:
        checkpoint_path = Path(args.checkpoint)
    logger.log(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    if args.group is None or args.sheet is None:
        raise ValueError("--group and --sheet are required.")
    sample = SampleSpec(group=args.group, sheet=args.sheet)

    model = MeshGraphNet(
        input_dim=4,
        hidden_dim=config.HIDDEN_DIM,
        num_message_passing_steps=config.NUM_MESSAGE_PASSING_STEPS,
        activation=config.ACTIVATION,
        dropout=config.DROPOUT,
        target_field=config.OUTPUT_FIELD,
        use_grad_checkpoint=config.USE_GRAD_CHECKPOINT,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    run_inversion(
        model=model,
        normalizer=normalizer,
        sample=sample,
        device=device,
        lr=args.lr,
        iters=args.iters,
        smooth_w=args.smooth,
        l2_w=args.l2,
        init_strategy=args.init,
    )


if __name__ == "__main__":
    main()
