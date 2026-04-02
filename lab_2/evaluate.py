"""Anomaly scoring, threshold selection, and evaluation metrics."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from tqdm import tqdm

from config import THRESHOLD_PERCENTILE, THRESHOLD_NUM_POINTS


def compute_reconstruction_errors(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Compute per-image reconstruction errors and collect results.

    Returns dict with keys:
        errors: np.ndarray (N,) -- mean reconstruction error per image
        error_maps: list of np.ndarray (H, W) -- per-pixel error heatmap
        labels: np.ndarray (N,)
        filepaths: list of str
        originals: list of np.ndarray (H, W, 3) -- original images [0,1]
        reconstructions: list of np.ndarray (H, W, 3) -- AE outputs [0,1]
    """
    model.eval()
    all_errors = []
    all_error_maps = []
    all_labels = []
    all_filepaths = []
    all_originals = []
    all_reconstructions = []

    with torch.no_grad():
        for images, labels_or_targets, filepaths in tqdm(dataloader, desc="Computing errors"):
            images = images.to(device)
            outputs = model(images)

            # Per-pixel MSE, averaged over channels -> (B, H, W)
            error_maps = ((images - outputs) ** 2).mean(dim=1).cpu().numpy()
            # Per-image mean error
            errors = error_maps.mean(axis=(1, 2))

            all_errors.extend(errors)
            all_error_maps.extend(error_maps)
            all_filepaths.extend(filepaths)
            all_originals.extend(images.cpu().permute(0, 2, 3, 1).numpy())
            all_reconstructions.extend(outputs.cpu().permute(0, 2, 3, 1).numpy())

            # labels_or_targets could be int labels or tensor images
            if isinstance(labels_or_targets, torch.Tensor) and labels_or_targets.dim() == 1:
                all_labels.extend(labels_or_targets.numpy())
            elif isinstance(labels_or_targets, (int, np.integer)):
                all_labels.append(int(labels_or_targets))
            else:
                # In reconstruct mode, targets are images -- no labels
                all_labels.extend([-1] * len(images))

    return {
        "errors": np.array(all_errors),
        "error_maps": all_error_maps,
        "labels": np.array(all_labels),
        "filepaths": all_filepaths,
        "originals": all_originals,
        "reconstructions": all_reconstructions,
    }


def set_threshold_from_clean(
    clean_errors: np.ndarray,
    percentile: float = THRESHOLD_PERCENTILE,
) -> float:
    """Set anomaly threshold as a percentile of clean image errors."""
    return float(np.percentile(clean_errors, percentile))


def optimize_threshold_on_test(
    errors: np.ndarray,
    labels: np.ndarray,
    num_points: int = THRESHOLD_NUM_POINTS,
) -> tuple[float, float]:
    """Grid search for threshold maximizing macro F1 on test set (oracle).

    Returns (best_threshold, best_f1).
    """
    thresholds = np.linspace(errors.min(), errors.max(), num_points)
    best_f1 = 0.0
    best_threshold = thresholds[0]

    for t in thresholds:
        preds = np.where(errors > t, 0, 1)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return float(best_threshold), float(best_f1)


def evaluate_anomaly_detection(
    errors: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict:
    """Evaluate anomaly detection at a given threshold.

    Label mapping: error > threshold -> predict 0 (artifact), else 1 (clean).

    Returns dict with macro_f1, roc_auc, confusion_matrix, report, predictions.
    """
    predictions = np.where(errors > threshold, 0, 1)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    report = classification_report(
        labels, predictions, target_names=["artifact", "clean"], zero_division=0
    )

    try:
        # For ROC-AUC: higher error = more likely artifact (class 0)
        # roc_auc_score expects higher score for positive class
        # If we treat clean (1) as positive, score = -error
        roc_auc = roc_auc_score(labels, -errors)
    except ValueError:
        roc_auc = float("nan")

    return {
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "report": report,
        "predictions": predictions,
        "threshold": threshold,
    }
