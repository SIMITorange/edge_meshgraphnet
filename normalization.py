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
        self.doping_scale = 1.0
        self.y_asinh_scale = 1.0
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
        vds_stat = RunningMeanStd()

        # Robust sampling pools for scale estimates
        doping_abs_samples: List[float] = []
        target_abs_samples: List[float] = []
        q_abs_samples: List[float] = []
        max_q_samples = 100000
        max_pool_samples = 100000
        rng = np.random.default_rng(config.RANDOM_SEED)

        def _sample_pool(values: np.ndarray, pool: List[float], max_size: int) -> None:
            flat = np.asarray(values).reshape(-1)
            if flat.size == 0:
                return
            take = min(flat.size, 1024)
            subset = rng.choice(flat, size=take, replace=False)
            pool.extend(subset.tolist())
            if len(pool) > max_size:
                # Keep the most recent samples; order does not matter for median
                del pool[: len(pool) - max_size]

        # First pass: gather coordinate/vds stats and scale estimates.
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

                x_stat.update(x_coord)
                x_stat.update(y_coord)
                _sample_pool(np.abs(doping), doping_abs_samples, max_pool_samples)
                vds_stat.update(np.array([vds], dtype=np.float64))

                if self.output_field == "SpaceCharge":
                    _sample_pool(np.abs(target), q_abs_samples, max_q_samples)
                else:
                    # Collect target magnitudes for asinh scaling (after simple per-field preprocessing)
                    if self.output_field == "ElectrostaticPotential":
                        base = target / max(vds, self.eps)
                    else:
                        base = target
                    _sample_pool(np.abs(base), target_abs_samples, max_pool_samples)

        self.doping_scale = (
            max(float(np.median(np.asarray(doping_abs_samples))), self.eps)
            if len(doping_abs_samples) > 0
            else 1.0
        )

        if self.output_field == "SpaceCharge":
            if len(q_abs_samples) == 0:
                self.q0 = 1.0
            else:
                median_abs = float(np.median(np.asarray(q_abs_samples)))
                self.q0 = max(median_abs * config.SPACECHARGE_Q0_SCALE, self.eps)
            self.y_asinh_scale = self.q0 + self.eps
        else:
            if len(target_abs_samples) == 0:
                self.y_asinh_scale = 1.0
            else:
                self.y_asinh_scale = max(float(np.median(np.asarray(target_abs_samples))), self.eps)

        # Second pass: compute stats using finalized scales
        doping_stat = RunningMeanStd()
        y_stat = RunningMeanStd()
        with h5py.File(h5_path, "r") as f:
            for spec in samples:
                grp = f[spec.group]
                pos = grp["pos"][:]  # (N, 2)
                fields = grp["fields"][spec.sheet]
                doping = fields[:, config.FIELD_TO_INDEX["DopingConcentration"]]
                target = fields[:, config.FIELD_TO_INDEX[self.output_field]]
                vds = float(
                    fields[:, config.FIELD_TO_INDEX["ElectrostaticPotential"]].max()
                )

                doping_asinh = self._asinh_scaled(doping, self.doping_scale)
                doping_stat.update(doping_asinh)

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
        doping_asinh = self._asinh_scaled(doping, self.doping_scale)
        doping_norm = (doping_asinh - self.doping_log_mean) / self.doping_log_std
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
            scaled = np.sinh(unnorm) * self.y_asinh_scale
            return scaled * vds
        if self.output_field in ("ElectricField_x", "ElectricField_y"):
            return np.sinh(unnorm) * self.y_asinh_scale
        if self.output_field == "SpaceCharge":
            return np.sinh(unnorm) * (self.q0 + self.eps)
        if self.output_field in ("eDensity", "hDensity"):
            return np.sinh(unnorm) * self.y_asinh_scale
        return np.sinh(unnorm) * self.y_asinh_scale

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
            doping_scale=self.doping_scale,
            y_asinh_scale=self.y_asinh_scale,
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
        norm.doping_scale = float(data.get("doping_scale", 1.0))
        norm.y_asinh_scale = float(data.get("y_asinh_scale", 1.0))
        norm.q0 = float(data["q0"])
        norm.fitted = bool(data["fitted"])
        norm.eps = float(data["eps"])
        return norm

    # ------------------------------
    # Helpers
    # ------------------------------
    def _asinh_scaled(self, arr: np.ndarray, scale: float) -> np.ndarray:
        """Symmetric asinh transform that handles signed inputs with a robust scale."""
        return np.arcsinh(arr / max(scale, self.eps))

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
            return self._asinh_scaled(scaled, self.y_asinh_scale)
        if self.output_field in ("ElectricField_x", "ElectricField_y"):
            # Clip extremes per-sample before stats to reduce outlier influence
            clip_val = np.percentile(np.abs(target), 99.0)
            clipped = np.clip(target, -clip_val, clip_val)
            return self._asinh_scaled(clipped, self.y_asinh_scale)
        if self.output_field == "SpaceCharge":
            return self._asinh_scaled(target, self.q0 + self.eps)
        if self.output_field in ("eDensity", "hDensity"):
            return self._asinh_scaled(target, self.y_asinh_scale)
        return self._asinh_scaled(target, self.y_asinh_scale)
