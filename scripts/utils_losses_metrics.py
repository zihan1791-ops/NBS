from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """多类别 Dice Loss（与 CrossEntropy 可加权相加）。"""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, num_classes: Optional[int] = None) -> torch.Tensor:
        # logits: (N, C, H, W), targets: (N, H, W) with values in [0, C-1]
        n, c, h, w = logits.shape
        if num_classes is None:
            num_classes = c

        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets.clamp(min=0), num_classes=num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_oh, dims)
        cardinality = torch.sum(probs + targets_oh, dims)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        loss = 1.0 - dice
        return loss.mean()


@torch.no_grad()
def compute_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    # pred, target: (N, H, W) int64
    pred = pred.view(-1)
    target = target.view(-1)
    mask = (target >= 0) & (target < num_classes)
    indices = num_classes * target[mask] + pred[mask]
    cm = torch.bincount(indices, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return cm


@torch.no_grad()
def metrics_from_confmat(cm: torch.Tensor) -> Dict[str, torch.Tensor]:
    # cm: (C, C) where rows are gt, cols are pred
    tp = cm.diag()
    support = cm.sum(dim=1)
    pred_sum = cm.sum(dim=0)
    union = support + pred_sum - tp
    iou = tp.float() / union.clamp(min=1)
    miou = iou.mean()
    acc = tp.sum().float() / cm.sum().clamp(min=1)

    # Dice = 2TP / (2TP + FP + FN) = 2TP / (support + pred_sum)
    dice_c = (2 * tp.float()) / (support + pred_sum).clamp(min=1)
    mdice = dice_c.mean()
    
    # 计算每个类别的准确率 (每个类别的TP / 该类别在ground truth中的总数)
    class_acc = tp.float() / support.clamp(min=1)
    
    return {
        "mIoU": miou.item() if isinstance(miou, torch.Tensor) else miou,
        "mDice": mdice.item() if isinstance(mdice, torch.Tensor) else mdice,
        "Acc": acc.item() if isinstance(acc, torch.Tensor) else acc,
        "per_class_iou": [x.item() for x in iou],
        "per_class_dice": [x.item() for x in dice_c],
        "per_class_acc": [x.item() for x in class_acc],
    }


@torch.no_grad()
def evaluate_metrics(targets, preds, num_classes):
    """将 numpy 数组转为混淆矩阵并计算指标，返回标量 dict。

    targets, preds: np.ndarray, shape (N, H, W), dtype int
    """
    import numpy as np

    pred_t = torch.from_numpy(preds.astype(np.int64))
    tgt_t = torch.from_numpy(targets.astype(np.int64))
    cm = compute_confusion_matrix(pred_t, tgt_t, num_classes)
    return metrics_from_confmat(cm)





