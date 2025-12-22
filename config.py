"""
Module purpose:
    Central configuration for data paths, model hyperparameters, and training/inference defaults.
Inputs:
    Imported by training and inference scripts; no direct CLI inputs.
Outputs:
    Provides constants and simple helper maps consumed by other modules.
"""

from pathlib import Path
import torch

# ------------------------------
# Filesystem configuration
# ------------------------------
# HDF5 dataset path (update to your actual location before running).
HDF5_PATH = Path(
    r"D:\paper_GNN_2025\train_data\dat\edge\dfise_results\GaN_FinJFET_halfcell_termin_BV_FLR_14n_1.5int\train_data\meshgraph_data.h5"
)

# Root working directory (project root).
PROJECT_ROOT = Path(__file__).parent

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
FIG_DIR = OUTPUT_DIR / "figures"
NORM_DIR = OUTPUT_DIR / "normalizers"

# Ensure directories are created lazily by calling code (train.py will create them).

# ------------------------------
# Training configuration
# ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42

OUTPUT_FIELD = "ElectrostaticPotential"  # Default target field
TRAIN_VAL_SPLIT = 0.85  # Fraction of samples used for training
BATCH_SIZE = 1  # Large graphs, keep batch size at 1
NUM_WORKERS = 0  # Increase if your HDF5 reads benefit from multiprocessing
PIN_MEMORY = True

EPOCHS = 3000
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0

NODE_LOSS_WEIGHT = 1.0
GRAD_LOSS_WEIGHT = 0.5
BOUNDARY_LOSS_WEIGHT = 0.25  # Set to 0.0 to disable boundary weighting
BOUNDARY_PERCENTILE = 90.0  # Percentile of doping gradient used to detect boundaries

# ------------------------------
# Model hyperparameters
# ------------------------------
HIDDEN_DIM = 128  # reduce from 256 to shrink activations/parameters
NUM_MESSAGE_PASSING_STEPS = 4  # reduce message passing depth to save memory
DROPOUT = 0.05
ACTIVATION = "gelu"  # Options: "relu", "gelu"

# ------------------------------
# Precision / memory optimization
# ------------------------------
USE_MIXED_PRECISION = True  # use torch.cuda.amp when running on CUDA
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
USE_GRAD_CHECKPOINT = True  # checkpoint GNN blocks during training to cut activation memory

# ------------------------------
# Normalization options
# ------------------------------
NORMALIZATION_EPS = 1e-8
SPACECHARGE_Q0_SCALE = 1.0  # Multiplicative factor applied to median |q| for signed-log scaling

# ------------------------------
# Checkpointing / logging
# ------------------------------
CHECKPOINT_EVERY = 100  # epochs
VALIDATE_EVERY = 5
PRINT_EVERY = 10

# Path to the checkpoint file for resuming training (set to None to disable)
CHECKPOINT_PATH = CHECKPOINT_DIR / "meshgraphnet_epoch_4000.pt"

# Continue-from-checkpoint settings
CONTINUE_FROM_CHECKPOINT = False
CONTINUE_EPOCHS = 1000

# ------------------------------
# Field mapping
# ------------------------------
FIELD_TO_INDEX = {
    "ElectrostaticPotential": 0,
    "eDensity": 1,
    "hDensity": 2,
    "SpaceCharge": 3,
    "ElectricField_x": 4,
    "ElectricField_y": 5,
    "DopingConcentration": 6,
}

# Multi-head support: list of heads available.
AVAILABLE_OUTPUT_FIELDS = [
    "ElectrostaticPotential",
    "ElectricField_x",
    "ElectricField_y",
    "SpaceCharge",
    "eDensity",
    "hDensity",
]


def ensure_output_dirs() -> None:
    """
    Create output directories if they do not exist.
    Inputs: None
    Outputs: None (directories created on disk)
    """
    for path in [OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, FIG_DIR, NORM_DIR]:
        path.mkdir(parents=True, exist_ok=True)
