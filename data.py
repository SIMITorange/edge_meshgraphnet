"""
Module purpose:
    Data loading utilities for HDF5-based TCAD graph data and PyTorch Geometric datasets.
Inputs:
    HDF5 file with groups n{uid} containing pos, edge_index, and fields datasets.
Outputs:
    PyG Data objects with normalized node features, edge_index, targets, and auxiliary metadata.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import config
from normalization import Normalizer


# Mapping of field name to index is sourced from config.FIELD_TO_INDEX.


@dataclass
class SampleSpec:
    """
    Simple container describing one training sample identified by group and sheet.
    Attributes:
        group: Group name in the HDF5 file, e.g., "n1".
        sheet: Integer index s selecting the field slice.
    """

    group: str
    sheet: int


def enumerate_samples(h5_path: str) -> List[SampleSpec]:
    """
    Enumerate all (group, sheet) combinations present in the HDF5 file.
    Inputs:
        h5_path: Path to meshgraph_data.h5.
    Outputs:
        List of SampleSpec entries for every sheet of every group.
    """
    samples: List[SampleSpec] = []
    with h5py.File(h5_path, "r") as f:
        for group in f.keys():
            num_sheets = f[group]["fields"].shape[0]
            for s in range(num_sheets):
                samples.append(SampleSpec(group=group, sheet=s))
    return samples


class H5MeshGraphDataset(Dataset):
    """
    PyTorch Dataset wrapping the HDF5 TCAD data.
    Inputs:
        h5_path: Path to HDF5.
        samples: Sequence of SampleSpec defining which samples belong to this split.
        output_field: Which physical field to predict (matches config.AVAILABLE_OUTPUT_FIELDS).
        normalizer: Normalizer instance. If None, raw (unnormalized) data is returned.
        boundary_percentile: Percentile for detecting boundary nodes via doping gradient.
    Outputs:
        torch_geometric.data.Data objects with attributes:
            x: [N, F] node features (normalized if normalizer provided)
            y: [N, 1] target values (normalized if normalizer provided)
            edge_index: [2, E] edge indices
            pos: [N, 2] raw coordinates (useful for visualization)
            vds: scalar tensor holding raw Vds for this sample
            boundary_mask: [N] float mask (1 for boundary nodes, 0 otherwise)
            uid: integer UID parsed from group name if possible
            sheet_idx: integer sheet index
    """

    def __init__(
        self,
        h5_path: str,
        samples: Sequence[SampleSpec],
        output_field: str,
        normalizer: Optional[Normalizer],
        boundary_percentile: float = 90.0,
    ) -> None:
        super().__init__()
        self.h5_path = h5_path
        self.samples = list(samples)
        self.output_field = output_field
        self.normalizer = normalizer
        self.boundary_percentile = boundary_percentile
        self.target_idx = config.FIELD_TO_INDEX[output_field]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        spec = self.samples[idx]
        with h5py.File(self.h5_path, "r") as f:
            grp = f[spec.group]
            pos = grp["pos"][:].astype(np.float32)  # (N, 2)
            edge_index = grp["edge_index"][:].astype(np.int64)  # (2, E)
            fields = grp["fields"][spec.sheet].astype(np.float32)  # (N, 7)

        doping = fields[:, config.FIELD_TO_INDEX["DopingConcentration"]]
        target = fields[:, self.target_idx]
        space_charge = fields[:, config.FIELD_TO_INDEX["SpaceCharge"]]
        vds = float(fields[:, config.FIELD_TO_INDEX["ElectrostaticPotential"]].max())

        # Build raw node feature components
        x_coord = pos[:, 0]
        y_coord = pos[:, 1]

        if self.normalizer is not None:
            x_feat = self.normalizer.transform_x(
                x_coord=x_coord,
                y_coord=y_coord,
                doping=doping,
                vds=vds,
            )
            y_target = self.normalizer.transform_y(
                target=target,
                vds=vds,
                output_field=self.output_field,
            )
        else:
            # Raw fallback: stack features without normalization.
            vds_feat = np.full_like(x_coord, vds, dtype=np.float32)
            x_feat = np.stack([x_coord, y_coord, doping, vds_feat], axis=1)
            y_target = target[:, None]

        x_tensor = torch.from_numpy(x_feat).float()
        y_tensor = torch.from_numpy(y_target).float()
        edge_index_tensor = torch.from_numpy(edge_index).long()
        pos_tensor = torch.from_numpy(pos).float()

        boundary_mask = self._compute_boundary_mask(
            boundary_field=torch.from_numpy(space_charge).float(),
            edge_index=edge_index_tensor,
            percentile=self.boundary_percentile,
        )

        data = Data(
            x=x_tensor,
            y=y_tensor,
            edge_index=edge_index_tensor,
            pos=pos_tensor,
            vds=torch.tensor([vds], dtype=torch.float32),
            boundary_mask=boundary_mask,
            uid=int(spec.group[1:]) if spec.group.startswith("n") else -1,
            sheet_idx=spec.sheet,
        )
        return data

    @staticmethod
    def _compute_boundary_mask(
        boundary_field: torch.Tensor, edge_index: torch.Tensor, percentile: float
    ) -> torch.Tensor:
        """
        Identify boundary nodes by large gradients of the true SpaceCharge field.
        Inputs:
            boundary_field: [N] tensor of SpaceCharge ground truth values.
            edge_index: [2, E] edge indices.
            percentile: Percentile threshold.
        Outputs:
            boundary_mask: [N] float tensor with 1 at boundary nodes, 0 elsewhere.
        """
        src, dst = edge_index
        diff = torch.abs(boundary_field[src] - boundary_field[dst])
        num_nodes = boundary_field.shape[0]
        # Aggregate gradient magnitude to nodes using scatter_add to avoid torch_scatter dependency
        grad_src_sum = torch.zeros(num_nodes, device=boundary_field.device).scatter_add_(0, src, diff)
        grad_src_cnt = torch.zeros(num_nodes, device=boundary_field.device).scatter_add_(
            0, src, torch.ones_like(diff)
        )
        grad_src = grad_src_sum / grad_src_cnt.clamp(min=1)
        grad_dst_sum = torch.zeros(num_nodes, device=boundary_field.device).scatter_add_(0, dst, diff)
        grad_dst_cnt = torch.zeros(num_nodes, device=boundary_field.device).scatter_add_(
            0, dst, torch.ones_like(diff)
        )
        grad_dst = grad_dst_sum / grad_dst_cnt.clamp(min=1)
        grad = 0.5 * (grad_src + grad_dst)
        threshold = torch.quantile(grad, percentile / 100.0)
        mask = (grad >= threshold).float()
        return mask


def split_samples(
    samples: Sequence[SampleSpec], train_fraction: float, seed: int
) -> Tuple[List[SampleSpec], List[SampleSpec]]:
    """
    Shuffle and split samples into train and validation subsets.
    Inputs:
        samples: List of SampleSpec.
        train_fraction: Fraction assigned to training.
        seed: Random seed for reproducibility.
    Outputs:
        (train_samples, val_samples)
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    split = int(len(indices) * train_fraction)
    train_idx = indices[:split]
    val_idx = indices[split:]
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    return train_samples, val_samples


def build_dataloaders(
    h5_path: str,
    output_field: str,
    normalizer: Normalizer,
    train_fraction: float,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    boundary_percentile: float,
) -> Tuple[DataLoader, DataLoader, List[SampleSpec], List[SampleSpec]]:
    """
    Convenience function to create train/val datasets and dataloaders.
    Inputs:
        h5_path: Path to HDF5.
        output_field: Target field name.
        normalizer: Fitted Normalizer instance.
        train_fraction: Fraction of samples for training.
        batch_size, num_workers, pin_memory: DataLoader options.
        seed: Random seed.
        boundary_percentile: Percentile used for boundary detection.
    Outputs:
        train_loader, val_loader, train_samples, val_samples
    """
    samples = enumerate_samples(h5_path)
    train_samples, val_samples = split_samples(samples, train_fraction, seed)

    train_dataset = H5MeshGraphDataset(
        h5_path=h5_path,
        samples=train_samples,
        output_field=output_field,
        normalizer=normalizer,
        boundary_percentile=boundary_percentile,
    )
    val_dataset = H5MeshGraphDataset(
        h5_path=h5_path,
        samples=val_samples,
        output_field=output_field,
        normalizer=normalizer,
        boundary_percentile=boundary_percentile,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, train_samples, val_samples
