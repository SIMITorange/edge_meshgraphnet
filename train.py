"""
Module purpose:
    Training script for MeshGraphNet surrogate on TCAD HDF5 data.
Inputs:
    Configuration from config.py; reads HDF5 dataset specified there.
Outputs:
    Trained model checkpoints, normalization parameters, training logs, and example figures.
"""

import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import amp
from torch_geometric.loader import DataLoader

import config
from data import H5MeshGraphDataset, enumerate_samples, split_samples
from losses import compute_loss
from model import MeshGraphNet
from normalization import Normalizer
from utils import logger, visualization


def set_seed(seed: int) -> None:
    """
    Set seeds for reproducibility.
    Inputs:
        seed: Integer seed.
    Outputs:
        None; random state mutated.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: MeshGraphNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: amp.GradScaler,
    use_amp: bool,
) -> Dict[str, float]:
    """
    Single training epoch.
    Inputs:
        model: MeshGraphNet instance.
        loader: Training DataLoader.
        optimizer: Optimizer.
        device: torch.device.
        scaler: Gradient scaler for mixed precision.
        use_amp: Whether to enable autocast.
    Outputs:
        Dictionary of averaged losses.
    """
    model.train()
    total_loss = 0.0
    total_node = 0.0
    total_grad = 0.0
    total_boundary = 0.0
    count = 0
    for batch in loader:
        batch = batch.to(device)
        # Skip batches with invalid values to prevent NaNs early
        if not torch.isfinite(batch.x).all() or not torch.isfinite(batch.y).all():
            logger.log("Skipping batch with non-finite inputs/targets.")
            continue
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(
            device_type=device.type if device.type != "meta" else "cuda",
            dtype=config.AMP_DTYPE if device.type == "cuda" else None,
            enabled=use_amp,
        ):
            outputs = model(batch)
            pred = outputs[config.OUTPUT_FIELD]
            loss, comps = compute_loss(
                pred=pred,
                target=batch.y,
                edge_index=batch.edge_index,
                boundary_mask=getattr(batch, "boundary_mask", None),
            )
        if not torch.isfinite(loss):
            logger.log("Non-finite loss encountered; skipping step.")
            continue
        if use_amp:
            scaler.scale(loss).backward()
            if config.GRAD_CLIP is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.GRAD_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()

        total_loss += comps["total"]
        total_node += comps["node_loss"]
        total_grad += comps["grad_loss"]
        total_boundary += comps["boundary_loss"]
        count += 1
    return {
        "total": total_loss / max(count, 1),
        "node": total_node / max(count, 1),
        "grad": total_grad / max(count, 1),
        "boundary": total_boundary / max(count, 1),
    }


def evaluate(
    model: MeshGraphNet,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    """
    Run evaluation without gradient updates.
    Inputs:
        model: MeshGraphNet instance.
        loader: Validation DataLoader.
        device: torch.device.
        use_amp: Whether to enable autocast.
    Outputs:
        Averaged loss components.
    """
    model.eval()
    total_loss = 0.0
    total_node = 0.0
    total_grad = 0.0
    total_boundary = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if not torch.isfinite(batch.x).all() or not torch.isfinite(batch.y).all():
                logger.log("Skipping val batch with non-finite inputs/targets.")
                continue
            with amp.autocast(
                device_type=device.type if device.type != "meta" else "cuda",
                dtype=config.AMP_DTYPE if device.type == "cuda" else None,
                enabled=use_amp,
            ):
                outputs = model(batch)
                pred = outputs[config.OUTPUT_FIELD]
                loss, comps = compute_loss(
                    pred=pred,
                    target=batch.y,
                    edge_index=batch.edge_index,
                    boundary_mask=getattr(batch, "boundary_mask", None),
                )
            total_loss += comps["total"]
            total_node += comps["node_loss"]
            total_grad += comps["grad_loss"]
            total_boundary += comps["boundary_loss"]
            count += 1
    return {
        "total": total_loss / max(count, 1),
        "node": total_node / max(count, 1),
        "grad": total_grad / max(count, 1),
        "boundary": total_boundary / max(count, 1),
    }


def maybe_load_or_fit_normalizer(
    normalizer_path: Path, h5_path: str, train_samples
) -> Normalizer:
    """
    Load an existing normalizer or fit a new one on training samples.
    Inputs:
        normalizer_path: Path to npz file.
        h5_path: HDF5 data path.
        train_samples: List of SampleSpec for training split.
    Outputs:
        Fitted Normalizer.
    """
    if normalizer_path.exists():
        logger.log(f"Loading normalizer from {normalizer_path}")
        return Normalizer.load(normalizer_path)
    logger.log("Fitting normalizer on training split...")
    normalizer = Normalizer(output_field=config.OUTPUT_FIELD)
    normalizer.fit_from_samples(h5_path, train_samples)
    normalizer.save(normalizer_path)
    logger.log(f"Saved normalizer to {normalizer_path}")
    return normalizer


def main() -> None:
    config.ensure_output_dirs()
    set_seed(config.RANDOM_SEED)
    device = config.DEVICE
    use_amp = config.USE_MIXED_PRECISION and device.type == "cuda"
    logger.log(f"Using device: {device}")

    all_samples = enumerate_samples(str(config.HDF5_PATH))
    train_samples, val_samples = split_samples(
        all_samples, train_fraction=config.TRAIN_VAL_SPLIT, seed=config.RANDOM_SEED
    )

    norm_path = config.NORM_DIR / f"{config.OUTPUT_FIELD}_normalizer.npz"
    normalizer = maybe_load_or_fit_normalizer(norm_path, str(config.HDF5_PATH), train_samples)

    train_dataset = H5MeshGraphDataset(
        h5_path=str(config.HDF5_PATH),
        samples=train_samples,
        output_field=config.OUTPUT_FIELD,
        normalizer=normalizer,
        boundary_percentile=config.BOUNDARY_PERCENTILE,
    )
    val_dataset = H5MeshGraphDataset(
        h5_path=str(config.HDF5_PATH),
        samples=val_samples,
        output_field=config.OUTPUT_FIELD,
        normalizer=normalizer,
        boundary_percentile=config.BOUNDARY_PERCENTILE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    model = MeshGraphNet(
        input_dim=train_dataset[0].x.shape[1],
        hidden_dim=config.HIDDEN_DIM,
        num_message_passing_steps=config.NUM_MESSAGE_PASSING_STEPS,
        activation=config.ACTIVATION,
        dropout=config.DROPOUT,
        target_field=config.OUTPUT_FIELD,
        use_grad_checkpoint=config.USE_GRAD_CHECKPOINT,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scaler = amp.GradScaler(enabled=use_amp)

    history = {"train_total": [], "val_total": [], "train_node": [], "val_node": [], "train_grad": [], "val_grad": []}

    for epoch in range(1, config.EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, scaler, use_amp)
        history["train_total"].append(train_metrics["total"])
        history["train_node"].append(train_metrics["node"])
        history["train_grad"].append(train_metrics["grad"])

        val_metrics = None
        if epoch % config.VALIDATE_EVERY == 0:
            val_metrics = evaluate(model, val_loader, device, use_amp)
            history["val_total"].append(val_metrics["total"])
            history["val_node"].append(val_metrics["node"])
            history["val_grad"].append(val_metrics["grad"])
            logger.log(
                f"Epoch {epoch}: train_total={train_metrics['total']:.4f} "
                f"val_total={val_metrics['total']:.4f}"
            )
        else:
            logger.log(f"Epoch {epoch}: train_total={train_metrics['total']:.4f}")
            history["val_total"].append(float("nan"))
            history["val_node"].append(float("nan"))
            history["val_grad"].append(float("nan"))

        if epoch % config.CHECKPOINT_EVERY == 0 or epoch == config.EPOCHS:
            ckpt_path = config.CHECKPOINT_DIR / f"meshgraphnet_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "output_field": config.OUTPUT_FIELD,
                },
                ckpt_path,
            )
            logger.log(f"Saved checkpoint to {ckpt_path}")

    # Save training curves
    visualization.plot_training_curves(
        history=history,
        save_path=config.FIG_DIR / f"training_curves_{config.OUTPUT_FIELD}.png",
        title=f"Loss Curves ({config.OUTPUT_FIELD})",
    )

    # Quick qualitative check on the first validation sample
    if len(val_dataset) > 0:
        sample = val_dataset[0]
        sample = sample.to(device)
        model.eval()
        with torch.no_grad():
            pred = model(sample)[config.OUTPUT_FIELD]
        # Convert BFloat16 or other non-standard types to float32 for numpy compatibility
        pred = pred.float() if pred.dtype != torch.float32 else pred
        sample.y = sample.y.float() if sample.y.dtype != torch.float32 else sample.y
        pred_np = pred.squeeze(-1).detach().cpu().numpy()
        target_np = sample.y.squeeze(-1).detach().cpu().numpy()
        pred_phys = normalizer.inverse_transform_y(pred_np, vds=float(sample.vds.cpu().item()))
        target_phys = normalizer.inverse_transform_y(target_np, vds=float(sample.vds.cpu().item()))
        visualization.scatter_field_comparison(
            pos=sample.pos,
            pred=torch.from_numpy(pred_phys),
            target=torch.from_numpy(target_phys),
            save_prefix=config.FIG_DIR / f"val_sample_{sample.uid}_s{sample.sheet_idx}_{config.OUTPUT_FIELD}",
            title_prefix=f"Val n{sample.uid} s{sample.sheet_idx}",
            field_name=config.OUTPUT_FIELD,
        )


if __name__ == "__main__":
    main()
