"""Visualization functions for autoencoder anomaly detection."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from config import CLASS_NAMES


def plot_training_history(history: dict, title: str = "Training History"):
    """Plot training and validation reconstruction loss."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss")
    ax.plot(epochs, history["val_loss"], "r-o", markersize=3, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_error_distribution(
    errors: np.ndarray,
    labels: np.ndarray,
    threshold: float | None = None,
    title: str = "Reconstruction Error Distribution",
):
    """Histogram of reconstruction errors for clean vs artifact images."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    clean_errors = errors[labels == 1]
    artifact_errors = errors[labels == 0]

    ax.hist(clean_errors, bins=30, alpha=0.6, label="Clean", color="green", density=True)
    ax.hist(artifact_errors, bins=30, alpha=0.6, label="Artifact", color="red", density=True)

    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold={threshold:.6f}")

    ax.set_xlabel("Mean Reconstruction Error (MSE)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_error_maps(
    originals: list[np.ndarray],
    reconstructions: list[np.ndarray],
    error_maps: list[np.ndarray],
    labels: np.ndarray,
    n_samples: int = 5,
    title: str = "Reconstruction Error Maps",
):
    """Grid visualization: original | reconstruction | error heatmap.

    Shows n_samples clean images on top and n_samples artifact images on bottom.
    """
    clean_idx = np.where(labels == 1)[0]
    artifact_idx = np.where(labels == 0)[0]

    # Select samples
    n_clean = min(n_samples, len(clean_idx))
    n_artifact = min(n_samples, len(artifact_idx))
    selected_clean = clean_idx[:n_clean]
    selected_artifact = artifact_idx[:n_artifact]
    selected = np.concatenate([selected_clean, selected_artifact])

    n_rows = n_clean + n_artifact
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(title, fontsize=14, y=1.02)

    for i, idx in enumerate(selected):
        orig = np.clip(originals[idx], 0, 1)
        recon = np.clip(reconstructions[idx], 0, 1)
        emap = error_maps[idx]
        label = "Clean" if labels[idx] == 1 else "Artifact"

        axes[i, 0].imshow(orig)
        axes[i, 0].set_title(f"{label} - Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(recon)
        axes[i, 1].set_title("Reconstruction")
        axes[i, 1].axis("off")

        im = axes[i, 2].imshow(emap, cmap="hot", vmin=0)
        axes[i, 2].set_title(f"Error Map (mean={emap.mean():.5f})")
        axes[i, 2].axis("off")
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def plot_artifact_cleanup(
    originals: list[np.ndarray],
    reconstructions: list[np.ndarray],
    error_maps: list[np.ndarray],
    labels: np.ndarray,
    n_samples: int = 5,
):
    """Side-by-side: artifact original | AE 'cleaned' output | error map."""
    artifact_idx = np.where(labels == 0)[0]
    n_show = min(n_samples, len(artifact_idx))
    selected = artifact_idx[:n_show]

    fig, axes = plt.subplots(n_show, 3, figsize=(12, 3 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle("Artifact Cleanup via Autoencoder", fontsize=14, y=1.02)

    for i, idx in enumerate(selected):
        orig = np.clip(originals[idx], 0, 1)
        recon = np.clip(reconstructions[idx], 0, 1)
        emap = error_maps[idx]

        axes[i, 0].imshow(orig)
        axes[i, 0].set_title("Original (with artifact)")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(recon)
        axes[i, 1].set_title("AE Output (cleaned)")
        axes[i, 1].axis("off")

        im = axes[i, 2].imshow(emap, cmap="hot", vmin=0)
        axes[i, 2].set_title("Error Map (artifact regions)")
        axes[i, 2].axis("off")
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix"):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["artifact", "clean"],
        yticklabels=["artifact", "clean"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_roc_pr_curves(errors: np.ndarray, labels: np.ndarray):
    """Plot ROC and Precision-Recall curves."""
    # Use negative error as score (lower error = more likely clean = class 1)
    scores = -errors

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc_val = auc(fpr, tpr)
    ax1.plot(fpr, tpr, "b-", linewidth=2, label=f"AUC = {roc_auc_val:.4f}")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)
    ax2.plot(recall, precision, "r-", linewidth=2, label=f"AP = {ap:.4f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_threshold_curve(
    errors: np.ndarray,
    labels: np.ndarray,
    best_threshold: float | None = None,
    num_points: int = 200,
):
    """Macro F1 vs threshold curve."""
    from sklearn.metrics import f1_score as _f1_score

    thresholds = np.linspace(errors.min(), errors.max(), num_points)
    f1_scores = []
    for t in thresholds:
        preds = np.where(errors > t, 0, 1)
        f1_scores.append(_f1_score(labels, preds, average="macro", zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1_scores, "b-", linewidth=2)
    if best_threshold is not None:
        ax.axvline(best_threshold, color="red", linestyle="--", label=f"Threshold={best_threshold:.6f}")
        ax.legend()
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Macro F1")
    ax.set_title("Macro F1 vs Threshold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_bottleneck_comparison(results: dict):
    """Compare bottleneck sizes: F1 bar chart and error gap chart.

    Args:
        results: dict mapping bottleneck_channels -> {
            'macro_f1': float,
            'clean_mean_error': float,
            'artifact_mean_error': float,
        }
    """
    sizes = sorted(results.keys())
    f1_scores = [results[s]["macro_f1"] for s in sizes]
    clean_errors = [results[s]["clean_mean_error"] for s in sizes]
    artifact_errors = [results[s]["artifact_mean_error"] for s in sizes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # F1 bar chart
    x = range(len(sizes))
    bars = ax1.bar(x, f1_scores, color="steelblue", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in sizes])
    ax1.set_xlabel("Bottleneck Channels")
    ax1.set_ylabel("Macro F1")
    ax1.set_title("Anomaly Detection F1 vs Bottleneck Size")
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, score in zip(bars, f1_scores):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{score:.3f}", ha="center", fontsize=10)

    # Error gap chart
    ax2.plot(sizes, clean_errors, "go-", linewidth=2, markersize=8, label="Clean (mean)")
    ax2.plot(sizes, artifact_errors, "ro-", linewidth=2, markersize=8, label="Artifact (mean)")
    ax2.fill_between(sizes, clean_errors, artifact_errors, alpha=0.15, color="gray")
    ax2.set_xlabel("Bottleneck Channels")
    ax2.set_ylabel("Mean Reconstruction Error")
    ax2.set_title("Error Gap: Clean vs Artifact")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_bottleneck_reconstructions(
    all_results: dict,
    original_clean: np.ndarray,
    original_artifact: np.ndarray,
    recon_clean: dict,
    recon_artifact: dict,
):
    """Show reconstructions of one clean and one artifact image across bottleneck sizes.

    Args:
        all_results: not used (kept for interface consistency)
        original_clean: (H, W, 3) original clean image
        original_artifact: (H, W, 3) original artifact image
        recon_clean: dict mapping bottleneck_channels -> (H, W, 3) reconstruction
        recon_artifact: dict mapping bottleneck_channels -> (H, W, 3) reconstruction
    """
    sizes = sorted(recon_clean.keys())
    n_cols = len(sizes) + 1  # +1 for original

    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 7))

    # Clean row
    axes[0, 0].imshow(np.clip(original_clean, 0, 1))
    axes[0, 0].set_title("Original\n(Clean)")
    axes[0, 0].axis("off")

    for j, s in enumerate(sizes):
        axes[0, j + 1].imshow(np.clip(recon_clean[s], 0, 1))
        axes[0, j + 1].set_title(f"B={s}")
        axes[0, j + 1].axis("off")

    # Artifact row
    axes[1, 0].imshow(np.clip(original_artifact, 0, 1))
    axes[1, 0].set_title("Original\n(Artifact)")
    axes[1, 0].axis("off")

    for j, s in enumerate(sizes):
        axes[1, j + 1].imshow(np.clip(recon_artifact[s], 0, 1))
        axes[1, j + 1].set_title(f"B={s}")
        axes[1, j + 1].axis("off")

    fig.suptitle("Reconstructions Across Bottleneck Sizes", fontsize=14, y=1.02)
    plt.tight_layout()
    return fig
