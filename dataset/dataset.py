import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder, VisionDataset

from .const import (
    BATCH_SIZE,
    DATA_ROOT,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_WORKERS,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        # Perform a random affine transformation: shifting and scaling the image.
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

VAL_TRANSFORM = transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


class MarketplaceDataModule(pl.LightningDataModule):
    full_dataset: VisionDataset | None = None
    train_dataset: VisionDataset | None = None
    val_dataset: VisionDataset | None = None
    test_dataset: VisionDataset | None = None
    predict_dataset: VisionDataset | None = None

    def __init__(
        self,
        data_dir: str = DATA_ROOT,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = TRAIN_TRANSFORM
        self.val_transform = VAL_TRANSFORM

    def setup(self, stage):
        self.full_dataset = ImageFolder(root=self.data_dir)
        full_dataset_with_val_transform = ImageFolder(
            root=self.data_dir, transform=self.val_transform
        )
        full_dataset_with_train_transform = ImageFolder(
            root=self.data_dir, transform=self.val_transform
        )

        # Stratified split into train/val/test sets keeping class distribution
        train_indices, val_indices, test_indices = _stratified_split(
            self.full_dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
        )
        self.train_dataset = Subset(full_dataset_with_train_transform, train_indices)
        self.val_dataset = Subset(full_dataset_with_val_transform, val_indices)
        self.test_dataset = Subset(full_dataset_with_val_transform, test_indices)
        self.predict_dataset = full_dataset_with_val_transform

    def train_dataloader(self):
        """Returns the DataLoader for the training set."""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        """Returns the DataLoader for the validation set."""
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return val_loader

    def test_dataloader(self):
        """Returns the DataLoader for the test set."""
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_loader

    def predict_dataloader(self):
        """Returns the DataLoader for the prediction set."""
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def _stratified_split(
    dataset: VisionDataset,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
):
    """
    Create stratified train/val/test split preserving class distribution.

    Args:
        dataset: ImageFolder dataset
        indices: valid sample indices (after filtering)
        train_ratio, val_ratio, test_ratio: split ratios
        seed: random seed
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    np.random.seed(seed)
    targets = np.array(dataset.targets)

    all_indices = np.arange(len(dataset))
    train_indices = []
    val_indices = []
    test_indices = []

    # Get unique classes in the filtered indices
    filtered_targets = targets[all_indices]
    unique_classes = np.unique(filtered_targets)

    # Split each class separately to maintain distribution
    for class_idx in unique_classes:
        class_mask = targets[all_indices] == class_idx
        class_indices = all_indices[class_mask]
        np.random.shuffle(class_indices)

        n = len(class_indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train : n_train + n_val])
        test_indices.extend(class_indices[n_train + n_val :])

    assert len(train_indices) + len(val_indices) + len(test_indices) == len(all_indices)

    return np.array(train_indices), np.array(val_indices), np.array(test_indices)
