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
    python infer.py --all --checkpoint outputs/checkpoints/meshgraphnet_epoch_2000.pt
"""

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import amp
from torch_geometric.loader import DataLoader

import config
from data import H5MeshGraphDataset, SampleSpec, enumerate_samples
from model import MeshGraphNet
from normalization import Normalizer
from utils import logger, visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single sample.")
    parser.add_argument("--group", type=str, default=None, help="Group name, e.g., n1")
    parser.add_argument("--sheet", type=int, default=None, help="Sheet index s")
    parser.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="Run inference for all groups and sheets in the HDF5 file; ignores --group/--sheet.",
    )
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


def run_inference_for_samples(
    model: MeshGraphNet,
    normalizer: Normalizer,
    samples: Sequence[SampleSpec],
    device: torch.device,
    use_amp: bool,
) -> None:
    """
    Run inference/visualization for a list of SampleSpec items.
    Saves figures into per-group subfolders under config.FIG_DIR.
    """

    dataset = H5MeshGraphDataset(
        h5_path=str(config.HDF5_PATH),
        samples=samples,
        output_field=config.OUTPUT_FIELD,
        normalizer=normalizer,
        boundary_percentile=config.BOUNDARY_PERCENTILE,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for data, spec in zip(loader, samples):
        data = data.to(device)
        with torch.no_grad():
            with amp.autocast(
                device_type=device.type if device.type != "meta" else "cuda",
                dtype=config.AMP_DTYPE if device.type == "cuda" else None,
                enabled=use_amp,
            ):
                pred = model(data)[config.OUTPUT_FIELD]

        pred = pred.float() if pred.dtype != torch.float32 else pred
        data.y = data.y.float() if data.y.dtype != torch.float32 else data.y

        pred_np = pred.squeeze(-1).detach().cpu().numpy()
        target_np = data.y.squeeze(-1).detach().cpu().numpy()
        vds_val = float(data.vds.cpu().item())

        pred_phys = normalizer.inverse_transform_y(pred_np, vds=vds_val)
        target_phys = normalizer.inverse_transform_y(target_np, vds=vds_val)

        # Error percentage statistics + plots
        err_percent = visualization.compute_error_percent(pred_phys, target_phys, eps=1e-12)
        error_txt_dir = config.FIG_DIR / "error" / "txt"
        error_plot_dir = config.FIG_DIR / "error" / "plot"
        error_txt_dir.mkdir(parents=True, exist_ok=True)
        error_plot_dir.mkdir(parents=True, exist_ok=True)
        err_base = f"{spec.group}_s{spec.sheet}_{config.OUTPUT_FIELD}_error_percent"
        err_txt_path = error_txt_dir / f"{err_base}.txt"
        err_plot_path = error_plot_dir / f"{err_base}.png"

        visualization.save_error_percent_bins_txt(
            err_percent,
            save_path=err_txt_path,
            title=f"Error% bins for {spec.group} s{spec.sheet} ({config.OUTPUT_FIELD})",
        )
        visualization.plot_error_percent_hist(
            err_percent,
            save_path=err_plot_path,
            title=f"{spec.group} s{spec.sheet} {config.OUTPUT_FIELD} error% histogram",
        )
        logger.log(f"Saved error% bin stats to {err_txt_path}")
        logger.log(f"Saved error% histogram to {err_plot_path}")

        group_dir = config.FIG_DIR / spec.group
        group_dir.mkdir(parents=True, exist_ok=True)
        save_prefix = group_dir / f"{spec.group}_s{spec.sheet}_{config.OUTPUT_FIELD}"

        # Save numeric outputs for reference
        npz_path = save_prefix.with_suffix(".npz")
        np.savez(
            npz_path,
            pred=pred_phys,
            target=target_phys,
            vds=vds_val,
            group=spec.group,
            sheet=spec.sheet,
        )

        logger.log(f"Saved predictions to {npz_path}")
        logger.log(f"Generating visualizations for {spec.group} s{spec.sheet}...")
        visualization.scatter_field_comparison(
            pos=data.pos,
            pred=torch.from_numpy(pred_phys),
            target=torch.from_numpy(target_phys),
            save_prefix=save_prefix,
            title_prefix=f"Infer {spec.group} s{spec.sheet}",
            edge_index=data.edge_index,
            use_mesh=True,
            field_name=config.OUTPUT_FIELD,
        )
        logger.log("Visualizations saved successfully!")


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

    if args.run_all:
        samples = enumerate_samples(str(config.HDF5_PATH))
        logger.log(f"Running inference for all groups/sheets: {len(samples)} samples")
    else:
        if args.group is None or args.sheet is None:
            raise ValueError("--group and --sheet are required unless --all is set")
        samples = [SampleSpec(group=args.group, sheet=args.sheet)]

    model = MeshGraphNet(
        input_dim=H5MeshGraphDataset(
            h5_path=str(config.HDF5_PATH),
            samples=[samples[0]],
            output_field=config.OUTPUT_FIELD,
            normalizer=normalizer,
            boundary_percentile=config.BOUNDARY_PERCENTILE,
        )[0].x.shape[1],
        hidden_dim=config.HIDDEN_DIM,
        num_message_passing_steps=config.NUM_MESSAGE_PASSING_STEPS,
        activation=config.ACTIVATION,
        dropout=config.DROPOUT,
        target_field=config.OUTPUT_FIELD,
        use_grad_checkpoint=config.USE_GRAD_CHECKPOINT,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    run_inference_for_samples(model, normalizer, samples, device, use_amp)


if __name__ == "__main__":
    main()
