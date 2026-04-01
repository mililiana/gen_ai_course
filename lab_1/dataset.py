"""Dataset classes, transforms, and data loading utilities."""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedShuffleSplit

from config import (
    TRAIN_DIR,
    TEST_DIR,
    VAL_RATIO,
    SEED,
    BATCH_SIZE,
    NUM_WORKERS,
    IMG_SIZE_EFFICIENTNET,
    IMG_SIZE_VIT,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from utils import get_file_list


class ArtifactDataset(Dataset):
    """Dataset for artifact detection in AI-generated images."""

    def __init__(self, file_list: list[tuple[str, int]], transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath, label = self.file_list[idx]
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label, filepath


def get_train_transforms(img_size: int) -> A.Compose:
    """Training augmentation pipeline."""
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3, border_mode=cv2.BORDER_REFLECT_101),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=15,
                p=0.5,
                border_mode=cv2.BORDER_REFLECT_101,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=20, p=0.3
            ),
            A.GaussNoise(p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                p=0.3,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(img_size: int) -> A.Compose:
    """Validation/test transform pipeline (no augmentation)."""
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def split_train_val(
    train_dir: str = TRAIN_DIR,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Stratified train/val split."""
    all_files = get_file_list(train_dir)
    filepaths = [f[0] for f in all_files]
    labels = np.array([f[1] for f in all_files])

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio, random_state=seed
    )
    train_idx, val_idx = next(splitter.split(filepaths, labels))

    train_files = [all_files[i] for i in train_idx]
    val_files = [all_files[i] for i in val_idx]
    return train_files, val_files


def make_weighted_sampler(file_list: list[tuple[str, int]]) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler to balance classes during training."""
    labels = [f[1] for f in file_list]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_dataloaders(
    img_size: int,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> dict[str, DataLoader]:
    """Create train, val, and test DataLoaders."""
    train_files, val_files = split_train_val()
    test_files = get_file_list(TEST_DIR)

    train_ds = ArtifactDataset(train_files, transform=get_train_transforms(img_size))
    val_ds = ArtifactDataset(val_files, transform=get_val_transforms(img_size))
    test_ds = ArtifactDataset(test_files, transform=get_val_transforms(img_size))

    sampler = make_weighted_sampler(train_files)

    return {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
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
