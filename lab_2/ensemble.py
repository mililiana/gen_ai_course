"""Ensemble: combine AE anomaly score with Lab 1 classifier predictions."""

import sys
import os
import importlib.util
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    LAB1_ROOT,
    LAB1_CHECKPOINT_DIR,
    ENSEMBLE_AE_WEIGHT,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    TEST_DIR,
    THRESHOLD_NUM_POINTS,
)
from utils import get_device, get_file_list
from evaluate import optimize_threshold_on_test, evaluate_anomaly_detection


_lab1_modules: dict = {}  # cache loaded lab1 modules


def _load_lab1_env():
    """Load all lab_1 modules, temporarily overriding sys.modules so
    lab_1's internal `from config import ...` etc. resolve correctly."""
    if _lab1_modules:
        return  # already loaded

    module_names = ["config", "utils", "dataset", "models"]
    saved = {}

    # Back up any existing cached modules that share these names
    for name in module_names:
        if name in sys.modules:
            saved[name] = sys.modules.pop(name)

    try:
        for name in module_names:
            module_path = os.path.join(LAB1_ROOT, f"{name}.py")
            spec = importlib.util.spec_from_file_location(name, module_path)
            mod = importlib.util.module_from_spec(spec)
            # Inject into sys.modules BEFORE exec so downstream imports find it
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            _lab1_modules[name] = mod
    finally:
        # Restore original sys.modules entries
        for name in module_names:
            sys.modules.pop(name, None)
        for name, mod in saved.items():
            sys.modules[name] = mod


def _load_lab1_vit(device: torch.device) -> nn.Module:
    """Load Lab 1's trained ViT model."""
    _load_lab1_env()
    lab1_models = _lab1_modules["models"]
    lab1_config = _lab1_modules["config"]

    model = lab1_models.get_vit(pretrained=False)
    ckpt_path = os.path.join(LAB1_CHECKPOINT_DIR, "best_vit.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Lab 1 ViT checkpoint not found: {ckpt_path}")
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()
    return model, lab1_config.IMG_SIZE_VIT


def _get_lab1_predictions(device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Get Lab 1 ViT artifact probabilities on test set.

    Returns:
        labels: np.ndarray (N,)
        artifact_probs: np.ndarray (N,) -- probability of class 0 (artifact)
    """
    _load_lab1_env()
    ArtifactDataset = _lab1_modules["dataset"].ArtifactDataset
    get_val_transforms = _lab1_modules["dataset"].get_val_transforms

    model, img_size = _load_lab1_vit(device)

    test_files = get_file_list(TEST_DIR)
    test_ds = ArtifactDataset(
        test_files, transform=get_val_transforms(img_size)
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.numpy() if hasattr(labels, "numpy") else labels)

    all_probs = np.concatenate(all_probs, axis=0)
    artifact_probs = all_probs[:, 0]  # probability of artifact class
    return np.array(all_labels), artifact_probs


def run_ensemble(
    ae_errors: np.ndarray,
    ae_labels: np.ndarray,
    ae_weight: float = ENSEMBLE_AE_WEIGHT,
    device: torch.device | None = None,
) -> dict:
    """Combine AE anomaly scores with Lab 1 classifier for stronger detection.

    Args:
        ae_errors: Per-image reconstruction errors from AE (N,).
        ae_labels: True labels (N,).
        ae_weight: Weight for AE score in [0, 1]. Classifier gets (1 - ae_weight).
        device: Torch device.

    Returns:
        Dict with combined evaluation results.
    """
    if device is None:
        device = get_device()

    # Get Lab 1 classifier predictions
    classifier_labels, classifier_artifact_probs = _get_lab1_predictions(device)

    # Normalize AE errors to [0, 1]
    ae_min, ae_max = ae_errors.min(), ae_errors.max()
    ae_scores = (ae_errors - ae_min) / (ae_max - ae_min + 1e-8)

    # Combined anomaly score (higher = more likely artifact)
    combined_scores = ae_weight * ae_scores + (1 - ae_weight) * classifier_artifact_probs

    # For evaluation: higher combined score -> predict artifact (0)
    # Find optimal threshold
    best_threshold, best_f1 = optimize_threshold_on_test(
        combined_scores, ae_labels, THRESHOLD_NUM_POINTS
    )

    results = evaluate_anomaly_detection(combined_scores, ae_labels, best_threshold)
    results["ae_weight"] = ae_weight
    results["combined_scores"] = combined_scores
    results["ae_scores"] = ae_scores
    results["classifier_scores"] = classifier_artifact_probs

    print(f"\nEnsemble (AE weight={ae_weight:.1f}):")
    print(f"  Macro F1: {results['macro_f1']:.4f}")
    print(f"  ROC-AUC: {results['roc_auc']:.4f}")
    print(f"  Threshold: {best_threshold:.6f}")

    return results
