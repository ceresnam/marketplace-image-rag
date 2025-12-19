from pathlib import Path

# Dataset paths
DATA_ROOT = Path("./data")  # symlink to /mnt/data

# DINOv2 settings
# DINOv2 uses patch size 14, so input must be divisible by 14
# Common sizes: 224 (16x16 patches), 518 (37x37 patches)
IMAGE_SIZE = 224

# ImageNet normalization (used by DINOv2)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training settings
BATCH_SIZE = 32
NUM_WORKERS = 6
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
