"""Dataset classes, transforms, and data loading for autoencoder pipeline."""

import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import ShuffleSplit
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import TRAIN_DIR, TEST_DIR, VAL_RATIO, SEED, BATCH_SIZE, NUM_WORKERS, IMG_SIZE
from utils import get_file_list


class AEDataset(Dataset):
    """Dataset for autoencoder training and evaluation.

    In 'reconstruct' mode: returns (image, image, filepath) for AE training.
    In 'evaluate' mode: returns (image, label, filepath) for anomaly detection.
    """

    def __init__(
        self,
        file_list: list[tuple[str, int]],
        transform=None,
        mode: str = "reconstruct",
    ):
        self.file_list = file_list
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath, label = self.file_list[idx]
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.mode == "reconstruct":
            return image, image.clone(), filepath
        return image, label, filepath


def get_ae_train_transforms(img_size: int = IMG_SIZE) -> A.Compose:
    """Training transforms for autoencoder (minimal augmentation)."""
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, p=0.3
            ),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ]
    )


def get_ae_val_transforms(img_size: int = IMG_SIZE) -> A.Compose:
    """Validation/test transforms (no augmentation, [0,1] range)."""
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ]
    )


def get_clean_file_list(data_dir: str) -> list[tuple[str, int]]:
    """Get only clean (label=1) images from a directory."""
    return [(fp, label) for fp, label in get_file_list(data_dir) if label == 1]


def split_clean_train_val(
    train_dir: str = TRAIN_DIR,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Split clean training images into train/val sets."""
    clean_files = get_clean_file_list(train_dir)

    splitter = ShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    indices = list(range(len(clean_files)))
    train_idx, val_idx = next(splitter.split(indices))

    train_files = [clean_files[i] for i in train_idx]
    val_files = [clean_files[i] for i in val_idx]
    return train_files, val_files


def get_ae_dataloaders(
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> dict[str, DataLoader]:
    """Create train (clean-only), val (clean-only), and test (all) DataLoaders."""
    train_files, val_files = split_clean_train_val()
    test_files = get_file_list(TEST_DIR)

    train_ds = AEDataset(
        train_files, transform=get_ae_train_transforms(img_size), mode="reconstruct"
    )
    val_ds = AEDataset(
        val_files, transform=get_ae_val_transforms(img_size), mode="reconstruct"
    )
    test_ds = AEDataset(
        test_files, transform=get_ae_val_transforms(img_size), mode="evaluate"
    )

    return {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
    }
