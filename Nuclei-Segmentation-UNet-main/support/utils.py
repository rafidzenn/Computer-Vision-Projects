"""Reproducibility, metrics, and loss for the nuclei-segmentation PoC."""
import random

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed=42):
    """Seed python / numpy / torch so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def dice_coef(logits, targets, eps=1e-6):
    """Mean Dice over the batch. logits: (N,1,H,W), targets: (N,1,H,W) in {0,1}."""
    preds = (torch.sigmoid(logits) > 0.5).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (union + eps)).mean().item()


@torch.no_grad()
def iou_score(logits, targets, eps=1e-6):
    """Mean IoU (Jaccard) over the batch."""
    preds = (torch.sigmoid(logits) > 0.5).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) >= 1).float().sum(dim=(1, 2, 3))
    return ((inter + eps) / (union + eps)).mean().item()


class DiceBCELoss(nn.Module):
    """BCE + soft-Dice. Dice term counters the heavy background/foreground
    imbalance that plain BCE handles poorly on sparse nuclei masks."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = 1 - ((2 * inter + 1) / (union + 1)).mean()
        return bce + dice
