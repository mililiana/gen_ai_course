"""Training pipeline for the convolutional autoencoder."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    SCHEDULER_PATIENCE,
    SCHEDULER_FACTOR,
    CHECKPOINT_DIR,
    DEFAULT_BOTTLENECK,
    IMG_SIZE,
)
from utils import get_device, set_seed, ensure_dirs, EarlyStoppingLoss
from dataset import get_ae_dataloaders
from models import ConvAutoencoder, count_parameters


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average reconstruction loss."""
    model.train()
    total_loss = 0.0

    for images, targets, _ in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(dataloader.dataset)


def validate_ae(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Validate autoencoder. Returns average reconstruction loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, targets, _ in tqdm(dataloader, desc="Val", leave=False):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)

    return total_loss / len(dataloader.dataset)


def train_autoencoder(
    bottleneck_channels: int = DEFAULT_BOTTLENECK,
    criterion: nn.Module | None = None,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = EARLY_STOPPING_PATIENCE,
    img_size: int = IMG_SIZE,
    device: torch.device | None = None,
    dataloaders: dict[str, DataLoader] | None = None,
) -> tuple[ConvAutoencoder, dict]:
    """Full training pipeline for one autoencoder configuration.

    Returns:
        model: Trained ConvAutoencoder (best checkpoint).
        history: Dict with 'train_loss', 'val_loss', 'lr' lists.
    """
    if device is None:
        device = get_device()

    if dataloaders is None:
        dataloaders = get_ae_dataloaders(img_size=img_size)

    model = ConvAutoencoder(bottleneck_channels).to(device)
    print(
        f"Bottleneck={bottleneck_channels}, "
        f"Parameters={count_parameters(model):,}"
    )

    if criterion is None:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )
    early_stopping = EarlyStoppingLoss(patience=patience)

    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
        val_loss = validate_ae(model, dataloaders["val"], criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        scheduler.step(val_loss)

        if early_stopping.step(val_loss, model):
            print(f"Early stopping at epoch {epoch}")
            break

    # Restore best model
    early_stopping.load_best(model)

    # Save checkpoint
    ensure_dirs(CHECKPOINT_DIR)
    ckpt_path = f"{CHECKPOINT_DIR}/ae_bottleneck_{bottleneck_channels}.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    return model, history
