"""
Advanced visualization script for space charge field analysis.

Generates high-quality mesh visualizations with automatic depletion region detection
and boundary line tracing.

Usage:
    python visualize.py --group n43 --sheet 0 --checkpoint outputs/checkpoints/meshgraphnet_epoch_300.pt [--mode mesh|scatter]
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from torch_geometric.loader import DataLoader

import config
from data import H5MeshGraphDataset, SampleSpec
from model import MeshGraphNet
from normalization import Normalizer
from utils import logger, visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate advanced visualizations of field predictions.")
    parser.add_argument("--group", type=str, required=True, help="Group name, e.g., n1")
    parser.add_argument("--sheet", type=int, required=True, help="Sheet index s")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (.pt). If not provided, uses latest in checkpoint dir.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["mesh", "scatter"],
        default="mesh",
        help="Visualization mode: 'mesh' for continuous surface, 'scatter' for point cloud.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Custom output filename prefix. If not provided, uses default naming.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved figures.",
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    return ckpt


def main() -> None:
    args = parse_args()
    config.ensure_output_dirs()
    device = config.DEVICE
    use_amp = config.USE_MIXED_PRECISION and device.type == "cuda"

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
    ckpt = load_checkpoint(checkpoint_path)

    sample = SampleSpec(group=args.group, sheet=args.sheet)
    dataset = H5MeshGraphDataset(
        h5_path=str(config.HDF5_PATH),
        samples=[sample],
        output_field=config.OUTPUT_FIELD,
        normalizer=normalizer,
        boundary_percentile=config.BOUNDARY_PERCENTILE,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = MeshGraphNet(
        input_dim=dataset[0].x.shape[1],
        hidden_dim=config.HIDDEN_DIM,
        num_message_passing_steps=config.NUM_MESSAGE_PASSING_STEPS,
        activation=config.ACTIVATION,
        dropout=config.DROPOUT,
        target_field=config.OUTPUT_FIELD,
        use_grad_checkpoint=config.USE_GRAD_CHECKPOINT,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    batch = next(iter(loader)).to(device)
    with torch.no_grad():
        from torch import amp
        with amp.autocast(
            device_type=device.type if device.type != "meta" else "cuda",
            dtype=config.AMP_DTYPE if device.type == "cuda" else None,
            enabled=use_amp,
        ):
            pred = model(batch)[config.OUTPUT_FIELD]

    # Convert to float32 for numpy compatibility
    pred = pred.float() if pred.dtype != torch.float32 else pred
    batch.y = batch.y.float() if batch.y.dtype != torch.float32 else batch.y
    
    pred_np = pred.squeeze(-1).detach().cpu().numpy()
    target_np = batch.y.squeeze(-1).detach().cpu().numpy()
    vds_val = float(batch.vds.cpu().item())

    pred_phys = normalizer.inverse_transform_y(pred_np, vds=vds_val)
    target_phys = normalizer.inverse_transform_y(target_np, vds=vds_val)

    # Determine output name
    if args.output_name is None:
        output_name = f"viz_{args.group}_s{args.sheet}_{config.OUTPUT_FIELD}"
    else:
        output_name = args.output_name

    save_prefix = config.FIG_DIR / output_name
    
    logger.log(f"Mode: {args.mode}")
    logger.log(f"Generating visualizations with {len(batch.pos)} mesh nodes...")
    
    visualization.scatter_field_comparison(
        pos=batch.pos,
        pred=torch.from_numpy(pred_phys),
        target=torch.from_numpy(target_phys),
        save_prefix=save_prefix,
        title_prefix=f"{args.group} Sheet {args.sheet}",
        edge_index=batch.edge_index,
        use_mesh=(args.mode == "mesh"),
    )
    
    logger.log(f"Visualizations saved to {save_prefix.parent}/")
    logger.log("Files generated:")
    logger.log(f"  - {output_name}_pred.png (Prediction)")
    logger.log(f"  - {output_name}_true.png (Ground Truth)")
    logger.log(f"  - {output_name}_error.png (Error)")


if __name__ == "__main__":
    main()
