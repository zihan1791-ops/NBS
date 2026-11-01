from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DICE_EPSILON


class DiceLoss(nn.Module):
    """与交叉熵损失配合使用的多类别 Soft Dice 损失。"""

    def __init__(self, eps: float = DICE_EPSILON) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        num_classes: Optional[int] = None,
    ) -> torch.Tensor:
        classes = num_classes or logits.shape[1]
        probabilities = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets.clamp(min=0), num_classes=classes)
        one_hot = one_hot.permute(0, 3, 1, 2).to(dtype=probabilities.dtype)
        dims = (0, 2, 3)
        intersection = torch.sum(probabilities * one_hot, dim=dims)
        cardinality = torch.sum(probabilities + one_hot, dim=dims)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return (1.0 - dice).mean()


@torch.no_grad()
def update_confusion_matrix(
    confusion: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> None:
    predictions = logits.argmax(dim=1).reshape(-1)
    targets = targets.reshape(-1)
    valid = (targets >= 0) & (targets < num_classes)
    counts = torch.bincount(
        num_classes * targets[valid] + predictions[valid],
        minlength=num_classes * num_classes,
    )
    confusion += counts.reshape(num_classes, num_classes)


def metrics_from_confusion(confusion: torch.Tensor) -> Dict[str, object]:
    confusion = confusion.to(dtype=torch.float64)
    true_positive = confusion.diag()
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    union = support + predicted - true_positive
    iou = true_positive / union.clamp(min=1)
    dice = 2 * true_positive / (support + predicted).clamp(min=1)
    accuracy = true_positive.sum() / confusion.sum().clamp(min=1)
    return {
        "mIoU": iou.mean().item(),
        "mDice": dice.mean().item(),
        "accuracy": accuracy.item(),
        "per_class_iou": [value.item() for value in iou],
        "per_class_dice": [value.item() for value in dice],
        "confusion_matrix": confusion.to(dtype=torch.int64).tolist(),
    }
