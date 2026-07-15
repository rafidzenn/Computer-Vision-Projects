"""Evaluate a trained model on a split and return Dice / IoU."""
import torch
from torch.utils.data import DataLoader

from support.dataset import NucleiDataset
from support.utils import dice_coef, iou_score


@torch.no_grad()
def evaluate_model(model, data_root, device, split="val", batch_size=8, size=128):
    ds = NucleiDataset(f"{data_root}/{split}", size=size, augment=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.eval().to(device)
    dices, ious = [], []
    for img, mask in dl:
        img, mask = img.to(device), mask.to(device)
        logits = model(img)
        dices.append(dice_coef(logits, mask))
        ious.append(iou_score(logits, mask))

    return {"dice": sum(dices) / len(dices), "iou": sum(ious) / len(ious)}
