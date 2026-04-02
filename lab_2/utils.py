"""Utility functions for autoencoder anomaly detection pipeline."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.manual_seed(seed)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_label(filename: str) -> int:
    """Extract class label from filename (e.g., image_00001_1.png -> 1)."""
    stem = os.path.splitext(filename)[0]
    return int(stem.split("_")[-1])


def get_file_list(data_dir: str) -> list[tuple[str, int]]:
    """Get list of (filepath, label) tuples from a directory."""
    files = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.lower().endswith(".png"):
            label = parse_label(fname)
            files.append((os.path.join(data_dir, fname), label))
    return files


def ensure_dirs(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


class EarlyStoppingLoss:
    """Early stopping based on a monitored loss (lower is better)."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, loss: float, model: torch.nn.Module) -> bool:
        """Returns True if training should stop."""
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            self.best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            return False
        self.counter += 1
        return self.counter >= self.patience

    def load_best(self, model: torch.nn.Module):
        """Restore model to best state."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
