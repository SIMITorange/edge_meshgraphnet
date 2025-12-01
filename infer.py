"""
Module purpose:
    Inference script for running the trained surrogate on a specified sample.
Inputs:
    Command-line args:
        --group: HDF5 group name (e.g., n1)
        --sheet: Integer sheet index s
        --checkpoint: Path to model checkpoint (defaults to latest in checkpoints dir)
Outputs:
    Saves numpy arrays of prediction/target and optional figures to configured folders.

use:
    python infer.py --group n43 --sheet 0 --checkpoint outputs/checkpoints/meshgraphnet_epoch_X.pt
"""

import argparse
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

import config
from data import H5MeshGraphDataset, SampleSpec
from model import MeshGraphNet
from normalization import Normalizer
from utils import logger, visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single sample.")
    parser.add_argument("--group", type=str, required=True, help="Group name, e.g., n1")
    parser.add_argument("--sheet", type=int, required=True, help="Sheet index s")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (.pt). If not provided, uses latest in checkpoint dir.",
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    return ckpt


def main() -> None:
    args = parse_args()
    config.ensure_output_dirs()
    device = config.DEVICE

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
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    batch = next(iter(loader)).to(device)
    with torch.no_grad():
        pred = model(batch)[config.OUTPUT_FIELD]

    pred_np = pred.squeeze(-1).detach().cpu().numpy()
    target_np = batch.y.squeeze(-1).detach().cpu().numpy()
    vds_val = float(batch.vds.cpu().item())

    pred_phys = normalizer.inverse_transform_y(pred_np, vds=vds_val)
    target_phys = normalizer.inverse_transform_y(target_np, vds=vds_val)

    out_prefix = config.OUTPUT_DIR / f"infer_{args.group}_s{args.sheet}_{config.OUTPUT_FIELD}"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = out_prefix.with_suffix(".npz")
    import numpy as np  # Local import to avoid dependency if not used elsewhere

    np.savez(
        npz_path,
        pred=pred_phys,
        target=target_phys,
        vds=vds_val,
        group=args.group,
        sheet=args.sheet,
    )
    logger.log(f"Saved predictions to {npz_path}")

    visualization.scatter_field_comparison(
        pos=batch.pos,
        pred=torch.from_numpy(pred_phys),
        target=torch.from_numpy(target_phys),
        save_prefix=config.FIG_DIR / f"infer_{args.group}_s{args.sheet}_{config.OUTPUT_FIELD}",
        title_prefix=f"Infer {args.group} s{args.sheet}",
    )


if __name__ == "__main__":
    main()

