import os
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

# https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md
EMBEDDING_DIM = (
    # 384  # DINOv2 ViT-S model
    768  # DINOv2 ViT-B model
    # 1024  # DINOv2 ViT-L model
    # 1536  # DINOv2 ViT-g model
)

# see scripts/devenv.sh
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")
