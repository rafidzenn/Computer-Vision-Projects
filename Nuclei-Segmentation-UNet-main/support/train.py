"""Training loop for the nuclei-segmentation PoC."""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from support.dataset import NucleiDataset
from support.utils import DiceBCELoss, dice_coef, iou_score


def train_model(model, data_root, device, epochs=5, batch_size=8, lr=1e-3, size=128):
    train_ds = NucleiDataset(f"{data_root}/train", size=size, augment=True)
    val_ds = NucleiDataset(f"{data_root}/val", size=size, augment=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = DiceBCELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for img, mask in tqdm(train_dl, desc=f"epoch {epoch}/{epochs}", leave=False):
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(img)
            loss = loss_fn(logits, mask)
            loss.backward()
            optimizer.step()
            running += loss.item() * img.size(0)
        train_loss = running / len(train_ds)

        model.eval()
        dices, ious = [], []
        with torch.no_grad():
            for img, mask in val_dl:
                img, mask = img.to(device), mask.to(device)
                logits = model(img)
                dices.append(dice_coef(logits, mask))
                ious.append(iou_score(logits, mask))
        val_dice = sum(dices) / len(dices)
        val_iou = sum(ious) / len(ious)
        print(f"epoch {epoch:02d} | train_loss {train_loss:.4f} "
              f"| val_dice {val_dice:.4f} | val_iou {val_iou:.4f}")

    return model
