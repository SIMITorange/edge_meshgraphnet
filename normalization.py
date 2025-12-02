"""
Module purpose:
    Normalization utilities for node inputs and target fields with persistence.
Inputs:
    Raw arrays from HDF5 datasets (coordinates, doping, Vds, target field).
Outputs:
    Normalized numpy arrays and ability to invert predictions back to physical units.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, TYPE_CHECKING

import h5py
import numpy as np

import config

if TYPE_CHECKING:
    from data import SampleSpec  # pragma: no cover


def _ensure_path(path_like) -> Path:
    return Path(path_like)


@dataclass
class RunningMeanStd:
    """Online mean and variance estimator using Welford's algorithm."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: np.ndarray) -> None:
        flat = np.asarray(value).reshape(-1)
        for v in flat:
            self.count += 1
            delta = v - self.mean
            self.mean += delta / self.count
            delta2 = v - self.mean
            self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.m2 / (self.count - 1) if self.count > 1 else 0.0

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance + 1e-12))


class Normalizer:
    """
    Handles normalization/inverse-normalization for inputs and outputs.
    Inputs:
        output_field: Target field name.
    Outputs:
        transform_x / transform_y methods producing normalized arrays.
        inverse_transform_y for restoring physical units.
    """

    def __init__(self, output_field: str) -> None:
        self.output_field = output_field
        # Input stats
        self.x_mean = 0.0
        self.x_std = 1.0
        self.y_mean = 0.0
        self.y_std = 1.0
        self.doping_log_mean = 0.0
        self.doping_log_std = 1.0
        self.vds_mean = 0.0
        self.vds_std = 1.0
        # Output scaling extra parameter (e.g., q0 for space charge)
        self.q0 = 1.0
        self.fitted = False
        self.eps = config.NORMALIZATION_EPS

    # ------------------------------
    # Fitting utilities
    # ------------------------------
    def fit_from_samples(self, h5_path: str, samples: Sequence["SampleSpec"]) -> None:
        """
        Compute normalization statistics from a list of samples.
        Inputs:
            h5_path: Path to HDF5 file.
            samples: Sequence of SampleSpec to iterate over (typically training set).
        Outputs:
            Populates mean/std attributes and marks normalizer as fitted.
        """
        x_stat = RunningMeanStd()
        y_stat = RunningMeanStd()
        doping_stat = RunningMeanStd()
        vds_stat = RunningMeanStd()

        # For SpaceCharge signed-log scaling
        q_abs_samples: List[float] = []
        max_q_samples = 100000

        # First pass: gather input stats and, when possible, output stats.
        with h5py.File(h5_path, "r") as f:
            for spec in samples:
                grp = f[spec.group]
                pos = grp["pos"][:]  # (N, 2)
                fields = grp["fields"][spec.sheet]
                x_coord = pos[:, 0]
                y_coord = pos[:, 1]
                doping = fields[:, config.FIELD_TO_INDEX["DopingConcentration"]]
                target = fields[:, config.FIELD_TO_INDEX[self.output_field]]
                vds = float(
                    fields[:, config.FIELD_TO_INDEX["ElectrostaticPotential"]].max()
                )

                doping_log = self._signed_log10(doping)
                x_stat.update(x_coord)
                x_stat.update(y_coord)
                doping_stat.update(doping_log)
                vds_stat.update(np.array([vds], dtype=np.float64))

                if self.output_field == "SpaceCharge":
                    abs_vals = np.abs(target.reshape(-1))
                    for val in abs_vals:
                        if len(q_abs_samples) < max_q_samples:
                            q_abs_samples.append(float(val))
                        else:
                            replace_idx = np.random.randint(0, len(q_abs_samples))
                            q_abs_samples[replace_idx] = float(val)
                else:
                    prepared_target = self._prepare_target_for_stats(
                        target=target, vds=vds
                    )
                    y_stat.update(prepared_target)

        self.x_mean = x_stat.mean
        self.x_std = max(x_stat.std, self.eps)
        self.doping_log_mean = doping_stat.mean
        self.doping_log_std = max(doping_stat.std, self.eps)
        self.vds_mean = vds_stat.mean
        self.vds_std = max(vds_stat.std, self.eps)

        if self.output_field == "SpaceCharge":
            if len(q_abs_samples) == 0:
                self.q0 = 1.0
            else:
                median_abs = float(np.median(np.asarray(q_abs_samples)))
                self.q0 = max(median_abs * config.SPACECHARGE_Q0_SCALE, self.eps)
            # Second pass for y statistics using finalized q0
            with h5py.File(h5_path, "r") as f:
                for spec in samples:
                    grp = f[spec.group]
                    fields = grp["fields"][spec.sheet]
                    target = fields[:, config.FIELD_TO_INDEX[self.output_field]]
                    vds = float(
                        fields[:, config.FIELD_TO_INDEX["ElectrostaticPotential"]].max()
                    )
                    prepared_target = self._prepare_target_for_stats(
                        target=target, vds=vds
                    )
                    y_stat.update(prepared_target)

        self.y_mean = y_stat.mean
        self.y_std = max(y_stat.std, self.eps)
        self.fitted = True

    # ------------------------------
    # Transformations
    # ------------------------------
    def transform_x(
        self,
        x_coord: np.ndarray,
        y_coord: np.ndarray,
        doping: np.ndarray,
        vds: float,
    ) -> np.ndarray:
        """
        Normalize node inputs.
        Inputs:
            x_coord, y_coord: Arrays of coordinates.
            doping: Array of doping concentrations.
            vds: Scalar drain-source bias.
        Outputs:
            Stacked normalized features of shape [N, 4].
        """
        assert self.fitted, "Normalizer must be fitted before calling transform_x."
        x_norm = (x_coord - self.x_mean) / self.x_std
        y_norm = (y_coord - self.x_mean) / self.x_std  # share stats for simplicity
        doping_log = self._signed_log10(doping)
        doping_norm = (doping_log - self.doping_log_mean) / self.doping_log_std
        vds_norm = (vds - self.vds_mean) / self.vds_std
        vds_norm_vec = np.full_like(x_norm, vds_norm, dtype=np.float32)
        features = np.stack([x_norm, y_norm, doping_norm, vds_norm_vec], axis=1)
        return features.astype(np.float32)

    def transform_y(self, target: np.ndarray, vds: float, output_field: str) -> np.ndarray:
        """
        Normalize target values according to the configured output_field.
        Inputs:
            target: Array [N] of raw target values.
            vds: Scalar Vds for this sample.
            output_field: Name of the output field (must match fitted field).
        Outputs:
            Normalized target array of shape [N, 1].
        """
        assert self.fitted, "Normalizer must be fitted before calling transform_y."
        assert (
            output_field == self.output_field
        ), "Output field mismatch between config and normalizer."
        prepared = self._prepare_target_for_stats(target=target, vds=vds)
        normalized = (prepared - self.y_mean) / self.y_std
        return normalized[:, None].astype(np.float32)

    def inverse_transform_y(self, pred: np.ndarray, vds: float) -> np.ndarray:
        """
        Inverse normalization for predictions.
        Inputs:
            pred: Array [N] of normalized predictions.
            vds: Scalar Vds corresponding to the sample.
        Outputs:
            Predictions in physical units, shape [N].
        """
        assert self.fitted, "Normalizer must be fitted before calling inverse_transform_y."
        pred = pred.reshape(-1)
        unnorm = pred * self.y_std + self.y_mean
        if self.output_field == "ElectrostaticPotential":
            return unnorm * vds
        if self.output_field in ("ElectricField_x", "ElectricField_y"):
            return unnorm
        if self.output_field == "SpaceCharge":
            signed = np.sign(unnorm)
            return signed * (np.expm1(np.abs(unnorm)) * (self.q0 + self.eps))
        if self.output_field in ("eDensity", "hDensity"):
            return np.power(10.0, unnorm)
        return unnorm

    # ------------------------------
    # Persistence
    # ------------------------------
    def save(self, path: Path) -> None:
        """
        Save normalization parameters to npz.
        Inputs:
            path: Destination file path.
        Outputs:
            File written to disk.
        """
        path = _ensure_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            output_field=self.output_field,
            x_mean=self.x_mean,
            x_std=self.x_std,
            y_mean=self.y_mean,
            y_std=self.y_std,
            doping_log_mean=self.doping_log_mean,
            doping_log_std=self.doping_log_std,
            vds_mean=self.vds_mean,
            vds_std=self.vds_std,
            q0=self.q0,
            fitted=self.fitted,
            eps=self.eps,
        )

    @classmethod
    def load(cls, path: Path) -> "Normalizer":
        """
        Load normalization parameters from disk.
        Inputs:
            path: Path to npz file created by save().
        Outputs:
            Normalizer instance.
        """
        data = np.load(path, allow_pickle=True)
        output_field = str(data["output_field"])
        norm = cls(output_field=output_field)
        norm.x_mean = float(data["x_mean"])
        norm.x_std = float(data["x_std"])
        norm.y_mean = float(data["y_mean"])
        norm.y_std = float(data["y_std"])
        norm.doping_log_mean = float(data["doping_log_mean"])
        norm.doping_log_std = float(data["doping_log_std"])
        norm.vds_mean = float(data["vds_mean"])
        norm.vds_std = float(data["vds_std"])
        norm.q0 = float(data["q0"])
        norm.fitted = bool(data["fitted"])
        norm.eps = float(data["eps"])
        return norm

    # ------------------------------
    # Helpers
    # ------------------------------
    def _signed_log10(self, arr: np.ndarray) -> np.ndarray:
        """
        Symmetric log transform that supports signed inputs.
        Preserves sign and compresses magnitude to keep large dynamic ranges stable.
        """
        return np.sign(arr) * np.log10(np.abs(arr) + self.eps)

    def _prepare_target_for_stats(self, target: np.ndarray, vds: float) -> np.ndarray:
        """
        Apply field-specific pre-processing before computing mean/std.
        Inputs:
            target: Raw target array [N].
            vds: Scalar Vds.
        Outputs:
            Processed target array [N].
        """
        if self.output_field == "ElectrostaticPotential":
            scaled = target / max(vds, self.eps)
            return scaled
        if self.output_field in ("ElectricField_x", "ElectricField_y"):
            # Clip extremes per-sample before stats to reduce outlier influence
            clip_val = np.percentile(np.abs(target), 99.0)
            clipped = np.clip(target, -clip_val, clip_val)
            return clipped
        if self.output_field == "SpaceCharge":
            return np.sign(target) * np.log1p(np.abs(target) / (self.q0 + self.eps))
        if self.output_field in ("eDensity", "hDensity"):
            return np.log10(target + self.eps)
        return target
