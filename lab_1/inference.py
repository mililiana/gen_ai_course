"""Inference on the test set."""

import os
import pandas as pd
import torch

from config import (
    CHECKPOINT_DIR, OUTPUT_DIR, IMG_SIZE_EFFICIENTNET, IMG_SIZE_VIT,
    BATCH_SIZE, NUM_WORKERS, TEST_DIR, SEED,
)
from utils import set_seed, get_device, ensure_dirs, get_file_list
from dataset import ArtifactDataset, get_val_transforms
from torch.utils.data import DataLoader
from models import get_efficientnet, get_vit
from evaluate import predict_probs, full_evaluation, optimize_threshold


def load_model(model_type: str, device: torch.device) -> torch.nn.Module:
    """Load a trained model from checkpoint."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_{model_type}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if model_type == "efficientnet":
        model = get_efficientnet(pretrained=False)
    elif model_type == "vit":
        model = get_vit(pretrained=False)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model


def get_test_dataloader(img_size: int) -> DataLoader:
    """Create test DataLoader."""
    test_files = get_file_list(TEST_DIR)
    test_ds = ArtifactDataset(test_files, transform=get_val_transforms(img_size))
    return DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
    )


def run_inference(
    model_type: str,
    threshold: float | None = None,
) -> dict:
    """Run inference on test set with a single model.

    Args:
        model_type: "efficientnet" or "vit"
        threshold: classification threshold. If None, uses default 0.5.

    Returns:
        Dictionary with predictions, probabilities, metrics.
    """
    set_seed(SEED)
    device = get_device()
    ensure_dirs(OUTPUT_DIR)

    model = load_model(model_type, device)
    img_size = IMG_SIZE_EFFICIENTNET if model_type == "efficientnet" else IMG_SIZE_VIT
    test_loader = get_test_dataloader(img_size)

    labels, probs, filepaths = predict_probs(model, test_loader, device)

    results = full_evaluation(labels, probs, model_name=f"{model_type} (test)", threshold=threshold)

    # Save predictions CSV
    filenames = [os.path.basename(fp) for fp in filepaths]
    df = pd.DataFrame({
        "filename": filenames,
        "true_label": labels,
        "predicted_label": results["predictions"],
        "prob_artifact": probs[:, 0],
        "prob_clean": probs[:, 1],
    })
    csv_path = os.path.join(OUTPUT_DIR, f"predictions_{model_type}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Predictions saved to {csv_path}")

    results["dataframe"] = df
    results["probs"] = probs
    results["labels"] = labels
    results["filepaths"] = filepaths
    return results


if __name__ == "__main__":
    for mt in ["efficientnet", "vit"]:
        try:
            run_inference(mt)
        except FileNotFoundError as e:
            print(f"Skipping {mt}: {e}")
