"""Training loop for artifact detection models."""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

from config import (
    NUM_EPOCHS, BACKBONE_LR, HEAD_LR, WEIGHT_DECAY, CLASS_WEIGHTS,
    WARMUP_EPOCHS, FREEZE_EPOCHS, EARLY_STOPPING_PATIENCE,
    CHECKPOINT_DIR, IMG_SIZE_EFFICIENTNET, IMG_SIZE_VIT, BATCH_SIZE, SEED,
)
from utils import set_seed, get_device, ensure_dirs, EarlyStopping
from dataset import get_dataloaders
from models import (
    get_efficientnet, get_vit, freeze_backbone, unfreeze_all, get_param_groups,
)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    running_loss = 0.0
    num_batches = 0

    for images, labels, _ in tqdm(dataloader, desc="  Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Validate model. Returns (loss, macro_f1, all_labels, all_probs)."""
    model.eval()
    running_loss = 0.0
    num_batches = 0
    all_labels = []
    all_probs = []

    for images, labels, _ in tqdm(dataloader, desc="  Val", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

        running_loss += loss.item()
        num_batches += 1

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    preds = all_probs.argmax(axis=1)
    macro_f1 = f1_score(all_labels, preds, average="macro")

    return running_loss / max(num_batches, 1), macro_f1, all_labels, all_probs


def train_model(
    model_type: str = "efficientnet",
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Full training pipeline for a given model type.

    Args:
        model_type: "efficientnet" or "vit"
        num_epochs: number of training epochs
        batch_size: batch size for DataLoader

    Returns:
        Dictionary with training history and best metrics.
    """
    set_seed(SEED)
    device = get_device()
    ensure_dirs(CHECKPOINT_DIR)

    # Setup model and image size
    if model_type == "efficientnet":
        model = get_efficientnet()
        img_size = IMG_SIZE_EFFICIENTNET
    elif model_type == "vit":
        model = get_vit()
        img_size = IMG_SIZE_VIT
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model.to(device)

    # Data
    dataloaders = get_dataloaders(img_size=img_size, batch_size=batch_size)

    # Loss with class weights
    class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Phase 1: Frozen backbone — train only head
    freeze_backbone(model, model_type)
    head_lr = HEAD_LR
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=head_lr, weight_decay=WEIGHT_DECAY,
    )

    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "lr": []}

    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} | device={device} | img_size={img_size}")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        # Unfreeze backbone after FREEZE_EPOCHS
        if epoch == FREEZE_EPOCHS + 1:
            print(f"\n>> Unfreezing backbone at epoch {epoch}")
            unfreeze_all(model)
            param_groups = get_param_groups(model, model_type, BACKBONE_LR, HEAD_LR)
            optimizer = AdamW(param_groups, weight_decay=WEIGHT_DECAY)
            scheduler = SequentialLR(
                optimizer,
                schedulers=[
                    LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS),
                    CosineAnnealingLR(optimizer, T_max=num_epochs - FREEZE_EPOCHS - WARMUP_EPOCHS),
                ],
                milestones=[WARMUP_EPOCHS],
            )

        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
        val_loss, val_f1, _, _ = validate(model, dataloaders["val"], criterion, device)

        # Step scheduler only after unfreezing
        if epoch > FREEZE_EPOCHS:
            scheduler.step()

        current_lr = optimizer.param_groups[-1]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_f1={val_f1:.4f}  lr={current_lr:.2e}")

        # Early stopping
        if early_stopping.step(val_f1, model):
            print(f"\nEarly stopping at epoch {epoch}. Best val F1: {early_stopping.best_score:.4f}")
            break

    # Restore best model
    early_stopping.load_best(model)

    # Save checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_{model_type}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nBest model saved to {ckpt_path}")
    print(f"Best val macro F1: {early_stopping.best_score:.4f}")

    return {
        "model": model,
        "history": history,
        "best_val_f1": early_stopping.best_score,
        "checkpoint_path": ckpt_path,
        "model_type": model_type,
        "img_size": img_size,
    }


if __name__ == "__main__":
    # Train both models sequentially
    results_eff = train_model("efficientnet")
    results_vit = train_model("vit")
    print(f"\nEfficientNet best F1: {results_eff['best_val_f1']:.4f}")
    print(f"ViT best F1: {results_vit['best_val_f1']:.4f}")
