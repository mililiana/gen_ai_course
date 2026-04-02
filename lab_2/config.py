"""Central configuration for autoencoder anomaly detection pipeline."""

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "trainee_dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Lab 1 paths (for ensemble)
LAB1_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "lab_1")
LAB1_CHECKPOINT_DIR = os.path.join(LAB1_ROOT, "outputs", "checkpoints")

# Reproducibility
SEED = 42

# Data
VAL_RATIO = 0.15
IMG_SIZE = 256

# Class labels (same convention as Lab 1)
CLASS_NAMES = {0: "artifact", 1: "clean"}

# Training
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5
EARLY_STOPPING_PATIENCE = 10
NUM_WORKERS = 0  # MPS on macOS requires 0

# Autoencoder architecture
DEFAULT_BOTTLENECK = 128
BOTTLENECK_CHANNELS_LIST = [32, 64, 128, 256, 512]

# Anomaly detection
THRESHOLD_PERCENTILE = 95
THRESHOLD_NUM_POINTS = 200

# Perceptual loss (advanced)
PERCEPTUAL_WEIGHT = 0.5
VGG_LAYERS = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]

# Ensemble (advanced)
ENSEMBLE_AE_WEIGHT = 0.5
