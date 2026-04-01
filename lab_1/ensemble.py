"""Ensemble methods combining multiple model predictions."""

import os
import numpy as np
import pandas as pd
import torch

from config import (
    OUTPUT_DIR, IMG_SIZE_EFFICIENTNET, IMG_SIZE_VIT, SEED, TEST_DIR,
)
from utils import set_seed, get_device, ensure_dirs, get_file_list
from inference import load_model, get_test_dataloader
from evaluate import predict_probs, full_evaluation, optimize_threshold
from dataset import get_val_transforms, ArtifactDataset, split_train_val
from torch.utils.data import DataLoader
from config import BATCH_SIZE, NUM_WORKERS


def get_val_predictions(device: torch.device) -> dict:
    """Get validation set predictions from both models for threshold tuning."""
    train_files, val_files = split_train_val()

    results = {}
    for model_type, img_size in [("efficientnet", IMG_SIZE_EFFICIENTNET), ("vit", IMG_SIZE_VIT)]:
        model = load_model(model_type, device)
        val_ds = ArtifactDataset(val_files, transform=get_val_transforms(img_size))
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=False,
        )
        labels, probs, paths = predict_probs(model, val_loader, device)
        results[model_type] = {"labels": labels, "probs": probs, "paths": paths}

    return results


def weighted_average_ensemble(
    probs_list: list[np.ndarray],
    weights: list[float] | None = None,
) -> np.ndarray:
    """Combine predictions via weighted averaging.

    Args:
        probs_list: list of (N, 2) probability arrays from each model
        weights: model weights (e.g., validation F1 scores). If None, equal weights.

    Returns:
        Combined (N, 2) probability array.
    """
    if weights is None:
        weights = [1.0] * len(probs_list)

    weights = np.array(weights) / sum(weights)
    combined = sum(w * p for w, p in zip(weights, probs_list))
    return combined


def run_ensemble(
    val_f1_efficientnet: float,
    val_f1_vit: float,
) -> dict:
    """Run ensemble inference on test set.

    Args:
        val_f1_efficientnet: validation F1 of EfficientNet (used as weight)
        val_f1_vit: validation F1 of ViT (used as weight)

    Returns:
        Dictionary with ensemble predictions and metrics.
    """
    set_seed(SEED)
    device = get_device()
    ensure_dirs(OUTPUT_DIR)

    # Get validation predictions for threshold optimization
    print("Getting validation predictions for threshold tuning...")
    val_preds = get_val_predictions(device)

    val_probs_combined = weighted_average_ensemble(
        [val_preds["efficientnet"]["probs"], val_preds["vit"]["probs"]],
        weights=[val_f1_efficientnet, val_f1_vit],
    )
    val_labels = val_preds["efficientnet"]["labels"]
    opt_thresh, opt_f1 = optimize_threshold(val_labels, val_probs_combined)
    print(f"Ensemble optimal threshold on val: {opt_thresh:.2f} (F1={opt_f1:.4f})")

    # Test set predictions
    print("\nRunning ensemble on test set...")
    test_probs = {}
    test_labels = None
    test_paths = None

    for model_type, img_size in [("efficientnet", IMG_SIZE_EFFICIENTNET), ("vit", IMG_SIZE_VIT)]:
        model = load_model(model_type, device)
        test_loader = get_test_dataloader(img_size)
        labels, probs, paths = predict_probs(model, test_loader, device)
        test_probs[model_type] = probs
        test_labels = labels
        test_paths = paths

    # Combine
    combined_probs = weighted_average_ensemble(
        [test_probs["efficientnet"], test_probs["vit"]],
        weights=[val_f1_efficientnet, val_f1_vit],
    )

    results = full_evaluation(
        test_labels, combined_probs,
        model_name="Ensemble (test)", threshold=opt_thresh,
    )

    # Save
    filenames = [os.path.basename(fp) for fp in test_paths]
    df = pd.DataFrame({
        "filename": filenames,
        "true_label": test_labels,
        "predicted_label": results["predictions"],
        "prob_artifact": combined_probs[:, 0],
        "prob_clean": combined_probs[:, 1],
        "prob_eff_artifact": test_probs["efficientnet"][:, 0],
        "prob_vit_artifact": test_probs["vit"][:, 0],
    })
    csv_path = os.path.join(OUTPUT_DIR, "predictions_ensemble.csv")
    df.to_csv(csv_path, index=False)
    print(f"Ensemble predictions saved to {csv_path}")

    results["dataframe"] = df
    results["probs"] = combined_probs
    results["labels"] = test_labels
    results["threshold"] = opt_thresh
    return results
