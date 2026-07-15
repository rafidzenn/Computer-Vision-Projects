"""PyTorch Dataset for paired nuclei images and binary masks.

Expects a folder with:
    <root>/images/*.png
    <root>/masks/*.png     (same filenames; mask is 0/255, foreground = nucleus)
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class NucleiDataset(Dataset):
    def __init__(self, root, size=128, augment=False):
        self.img_dir = os.path.join(root, "images")
        self.msk_dir = os.path.join(root, "masks")
        self.files = sorted(os.listdir(self.img_dir))
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def _augment(self, img, mask):
        # cheap, label-preserving flips/rotations (safe for segmentation)
        if np.random.rand() < 0.5:
            img, mask = img[:, ::-1, :].copy(), mask[:, ::-1].copy()
        if np.random.rand() < 0.5:
            img, mask = img[::-1, :, :].copy(), mask[::-1, :].copy()
        k = np.random.randint(0, 4)
        if k:
            img = np.rot90(img, k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k).copy()
        return img, mask

    def __getitem__(self, idx):
        fn = self.files[idx]
        img = (Image.open(os.path.join(self.img_dir, fn))
               .convert("RGB").resize((self.size, self.size)))
        mask = (Image.open(os.path.join(self.msk_dir, fn))
                .convert("L").resize((self.size, self.size)))

        img = np.asarray(img, dtype=np.float32) / 255.0
        mask = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)

        if self.augment:
            img, mask = self._augment(img, mask)

        img = torch.from_numpy(img.transpose(2, 0, 1))   # (3, H, W)
        mask = torch.from_numpy(mask)[None, ...]          # (1, H, W)
        return img, mask
