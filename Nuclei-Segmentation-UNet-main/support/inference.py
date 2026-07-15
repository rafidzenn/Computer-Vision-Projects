"""Run the trained model on a single image and save a 3-panel visualization
(input | predicted probability | mask overlay)."""
import os

import matplotlib
matplotlib.use("Agg")  # headless backend; safe on servers with no display
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


@torch.no_grad()
def predict_image(model, image_path, device, size=128):
    model.eval().to(device)
    img = Image.open(image_path).convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr.transpose(2, 0, 1))[None].to(device)
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    mask = (prob > 0.5).astype(np.uint8)
    return arr, prob, mask


def save_overlay(arr, prob, mask, out_path):
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    ax[0].imshow(arr); ax[0].set_title("input"); ax[0].axis("off")
    ax[1].imshow(prob, cmap="magma"); ax[1].set_title("probability"); ax[1].axis("off")
    ax[2].imshow(arr); ax[2].imshow(mask, cmap="Reds", alpha=0.4)
    ax[2].set_title("overlay"); ax[2].axis("off")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
