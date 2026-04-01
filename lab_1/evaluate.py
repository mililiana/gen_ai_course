"""Evaluation metrics, threshold optimization, and visualization."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
)
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    CLASS_NAMES, THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP, OUTPUT_DIR,
)
from utils import ensure_dirs


def optimize_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold_min: float = THRESHOLD_MIN,
    threshold_max: float = THRESHOLD_MAX,
    threshold_step: float = THRESHOLD_STEP,
) -> tuple[float, float]:
    """Find threshold that maximizes macro F1 on given probabilities.

    Args:
        labels: ground truth labels
        probs: predicted probabilities (N, 2) — class 1 probability used for thresholding

    Returns:
        (best_threshold, best_f1)
    """
    prob_class1 = probs[:, 1]
    thresholds = np.arange(threshold_min, threshold_max + threshold_step, threshold_step)
    best_f1, best_thresh = 0.0, 0.5

    for t in thresholds:
        preds = (prob_class1 >= t).astype(int)
        f1 = f1_score(labels, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh, best_f1


@torch.no_grad()
def predict_probs(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Get predictions from a model. Returns (labels, probs, filepaths)."""
    model.eval()
    all_labels, all_probs, all_paths = [], [], []

    for images, labels, paths in tqdm(dataloader, desc="Predicting", leave=False):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
        all_paths.extend(paths)

    return np.concatenate(all_labels), np.concatenate(all_probs), all_paths


def full_evaluation(
    labels: np.ndarray,
    probs: np.ndarray,
    model_name: str = "Model",
    threshold: float | None = None,
) -> dict:
    """Run full evaluation: threshold optimization, metrics, report.

    Returns dict with metrics and optimal threshold.
    """
    # Optimize threshold if not provided
    if threshold is None:
        opt_thresh, opt_f1 = optimize_threshold(labels, probs)
    else:
        opt_thresh = threshold
        preds = (probs[:, 1] >= opt_thresh).astype(int)
        opt_f1 = f1_score(labels, preds, average="macro")

    preds = (probs[:, 1] >= opt_thresh).astype(int)

    macro_f1 = f1_score(labels, preds, average="macro")
    report = classification_report(
        labels, preds, target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
    )
    cm = confusion_matrix(labels, preds)

    try:
        roc_auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        roc_auc = float("nan")

    print(f"\n{'='*50}")
    print(f"Evaluation: {model_name}")
    print(f"{'='*50}")
    print(f"Optimal threshold: {opt_thresh:.2f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"\n{report}")
    print(f"Confusion Matrix:\n{cm}")

    return {
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "threshold": opt_thresh,
        "confusion_matrix": cm,
        "report": report,
        "predictions": preds,
    }


def plot_threshold_curve(labels: np.ndarray, probs: np.ndarray, title: str = ""):
    """Plot macro F1 vs threshold."""
    prob_class1 = probs[:, 1]
    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP)
    f1_scores = []
    for t in thresholds:
        preds = (prob_class1 >= t).astype(int)
        f1_scores.append(f1_score(labels, preds, average="macro"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1_scores, "b-o", markersize=3)
    best_idx = np.argmax(f1_scores)
    ax.axvline(thresholds[best_idx], color="r", linestyle="--",
               label=f"Best: {thresholds[best_idx]:.2f} (F1={f1_scores[best_idx]:.4f})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"Threshold vs Macro F1 — {title}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, title: str = ""):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[CLASS_NAMES[0], CLASS_NAMES[1]],
        yticklabels=[CLASS_NAMES[0], CLASS_NAMES[1]],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {title}")
    plt.tight_layout()
    return fig


def plot_roc_pr_curves(labels: np.ndarray, probs: np.ndarray, title: str = ""):
    """Plot ROC and Precision-Recall curves side by side."""
    prob_class1 = probs[:, 1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ROC
    fpr, tpr, _ = roc_curve(labels, prob_class1)
    auc = roc_auc_score(labels, prob_class1)
    ax1.plot(fpr, tpr, "b-", label=f"AUC = {auc:.4f}")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(f"ROC Curve — {title}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(labels, prob_class1)
    ap = average_precision_score(labels, prob_class1)
    ax2.plot(recall, precision, "r-", label=f"AP = {ap:.4f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"Precision-Recall Curve — {title}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_training_history(history: dict, title: str = ""):
    """Plot training loss, val loss, and val F1 over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=3)
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss — {title}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_f1"], "g-o", label="Val Macro F1", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro F1")
    ax2.set_title(f"Validation F1 — {title}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
