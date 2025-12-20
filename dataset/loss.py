import torch
from torch import nn


class ASLSingleLabel(nn.Module):
    """
    This loss is intended for single-label classification problems
    """

    def __init__(self, gamma_pos=0, gamma_neg=2, eps: float = 0.1, reduction="mean"):
        super().__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        """
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        """
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(
            1, target.long().unsqueeze(1), 1
        )

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(
            1 - xs_pos - xs_neg,
            self.gamma_pos * targets + self.gamma_neg * anti_targets,
        )
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(
                self.eps / num_classes
            )

        # loss calculation
        loss = -self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == "mean":
            loss = loss.mean()

        return loss


class WeightedASLSingleLabel(nn.Module):
    """
    Asymmetric Loss with optional class weights for handling class imbalance.
    Combines ASL focal weighting with per-class reweighting.

    If class_weights is None, uses uniform weights (equivalent to standard ASL).
    If class_weights is provided, applies per-sample weighting based on class.
    """

    def __init__(self, class_weights=None, gamma_pos=0, gamma_neg=2, eps=0.1):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        # Register class_weights as buffer (moves to GPU with model)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) - logits from model
            targets: (batch_size,) - class indices

        Returns:
            Weighted ASL loss scalar
        """
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)

        # Create one-hot encoded targets
        targets_classes = torch.zeros_like(inputs).scatter_(
            1, targets.long().unsqueeze(1), 1
        )

        # ASL asymmetric weights
        anti_targets = 1 - targets_classes
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets_classes
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(
            1 - xs_pos - xs_neg,
            self.gamma_pos * targets_classes + self.gamma_neg * anti_targets,
        )
        log_preds = log_preds * asymmetric_w

        # Label smoothing
        if self.eps > 0:
            targets_classes = targets_classes.mul(1 - self.eps).add(
                self.eps / num_classes
            )

        # Compute base loss
        loss = -targets_classes.mul(log_preds)
        loss = loss.sum(dim=-1)  # (batch_size,)

        # Apply class weights if provided
        if self.class_weights is not None:
            sample_weights = self.class_weights[targets]
            loss = (loss * sample_weights).mean()
        else:
            loss = loss.mean()

        return loss
