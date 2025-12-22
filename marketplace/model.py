from typing import Self, TypedDict

import lightning.pytorch as pl
import torch
from torch import nn, optim
from torchmetrics import Accuracy

from .const import EMBEDDING_DIM
from .loss import WeightedASLSingleLabel


class DinoV2Base(torch.nn.Module):
    """
    DINOv2 outputs unnormalized embeddings by default

    We normalize them to L2, this is standard choice
    for embeddings because:
      - Embeddings live in continuous semantic space
      - Better geometric properties for nearest neighbor search
      - HNSW performs well with L2 distance
    """

    def __init__(self):
        super().__init__()
        model_variant = {
            384: "dinov2_vits14_reg",
            768: "dinov2_vitb14_reg",
            1024: "dinov2_vitl14_reg",
            1536: "dinov2_vitg14_reg",
        }[EMBEDDING_DIM]
        self.model = torch.hub.load("facebookresearch/dinov2", model_variant)

    def forward(self, x):
        return self.model(x)


class DinoV2WithNormalize(DinoV2Base):
    def forward(self, samples):
        return nn.functional.normalize(self.model(samples), dim=-1, p=2)


class DinoV2ClassificationHParams(TypedDict):
    learning_rate: float
    weight_decay: float
    embed_dim: int
    dropout: float
    num_classes: int
    mlp: bool


class DinoV2Classification(pl.LightningModule):
    hparams: DinoV2ClassificationHParams  # type: ignore[assignment]

    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        embed_dim: int = EMBEDDING_DIM,
        dropout: float = 0.2,
        num_classes: int = 10,
        mlp: bool = False,
        class_weights: list[float] | None = None,
    ):
        """
        Initializes the LightningModule with configurable layer parameters.
        """
        super().__init__()
        # Save the hyperparameters passed to the constructor.
        self.save_hyperparameters()

        # create backbone and freeze its parameters
        backbone = DinoV2Base()
        for param in backbone.parameters():
            param.requires_grad = False

        if self.hparams.mlp:
            head = nn.Sequential(
                nn.Linear(self.hparams.embed_dim, self.hparams.embed_dim),
                # help training stability
                nn.BatchNorm1d(self.hparams.embed_dim),
                nn.ReLU(),
                # prevent overfitting on frozen features
                nn.Dropout(self.hparams.dropout),
                nn.Linear(self.hparams.embed_dim, self.hparams.num_classes),
            )
        else:
            head = nn.Linear(self.hparams.embed_dim, self.hparams.num_classes)

        self.model = nn.Sequential(backbone, head)

        # Initialize the loss function.
        # self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = WeightedASLSingleLabel(class_weights=class_weights, gamma_neg=2)

        # Initialize metrics to track accuracy for training and validation.
        self.val_accuracy = Accuracy(
            task="multiclass", num_classes=self.hparams.num_classes
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self: Self, batch, batch_idx=None):
        """
        Performs a single training step.
        """
        # perform a forward pass to get the model's predictions (logits).
        inputs, labels = batch
        outputs = self(inputs)

        # calculate the loss
        loss = self.loss_fn(outputs, labels)
        self.log("train_loss", loss)

        return loss

    def validation_step(self: Self, batch, batch_idx=None):
        """
        Performs a single validation step.
        """
        # perform a forward pass to get the model's predictions (logits).
        inputs, labels = batch
        outputs = self(inputs)

        # calculate the loss.
        loss = self.loss_fn(outputs, labels)
        self.log("val_loss", loss, prog_bar=True)

        # update the validation accuracy metric with the current batch's results.
        self.val_accuracy(outputs, labels)
        self.log("val_accuracy", self.val_accuracy, prog_bar=True)

    def configure_optimizers(self: Self) -> optim.Optimizer:
        """
        Configures and returns the model's optimizer.
        """
        # Create and return the AdamW optimizer.
        return optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
