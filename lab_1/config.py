"""Central configuration for artifact detection pipeline."""

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "trainee_dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# Reproducibility
SEED = 42

# Data
VAL_RATIO = 0.15
IMG_SIZE_EFFICIENTNET = 384
IMG_SIZE_VIT = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Class labels
CLASS_NAMES = {0: "artifact", 1: "clean"}
NUM_CLASSES = 2
# Inverse frequency weights for CrossEntropyLoss (artifact is 10x rarer)
CLASS_WEIGHTS = [9.0, 1.0]

# Training
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
BACKBONE_LR = 1e-5
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 3
FREEZE_EPOCHS = 3  # epochs to keep backbone frozen
EARLY_STOPPING_PATIENCE = 7
NUM_WORKERS = 0  # MPS on macOS requires 0

# Model names (timm)
EFFICIENTNET_MODEL = "efficientnet_b0"
VIT_MODEL = "vit_base_patch16_224"

# Threshold optimization
THRESHOLD_MIN = 0.1
THRESHOLD_MAX = 0.9
THRESHOLD_STEP = 0.05
DEFAULT_THRESHOLD = 0.5
