"""Generate a tiny SYNTHETIC nuclei dataset so the whole pipeline is
verifiable end-to-end without any download.

It paints dark elliptical "nuclei" on an H&E-ish pink/purple background and
writes a matching binary mask. This is ONLY a plumbing test (a tracer bullet).
Replace it with a real dataset (e.g. MoNuSeg) for meaningful results — see
data/README.md.

Layout produced:
    <root>/train/images/*.png , <root>/train/masks/*.png
    <root>/val/images/*.png   , <root>/val/masks/*.png
"""
import os

import numpy as np
from PIL import Image, ImageDraw


def _make_one(size, seed):
    rng = np.random.default_rng(seed)
    bg = np.array([225, 190, 215], dtype=np.int16)               # pale H&E pink
    img = np.ones((size, size, 3), dtype=np.int16) * bg
    img += rng.integers(-10, 10, img.shape)                       # light texture
    img = np.clip(img, 0, 255).astype(np.uint8)

    im = Image.fromarray(img)
    mask = Image.new("L", (size, size), 0)
    d_im, d_mask = ImageDraw.Draw(im), ImageDraw.Draw(mask)

    n = int(rng.integers(15, 40))
    for _ in range(n):
        cx, cy = rng.integers(6, size - 6, size=2)
        rx, ry = int(rng.integers(4, 9)), int(rng.integers(4, 9))
        bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
        col = (int(rng.integers(80, 120)), int(rng.integers(40, 80)),
               int(rng.integers(110, 150)))                        # dark purple nucleus
        d_im.ellipse(bbox, fill=col)
        d_mask.ellipse(bbox, fill=255)

    return np.asarray(im), np.asarray(mask)


def generate_dataset(root, n_train=40, n_val=10, size=128, seed=42):
    for split, n, offset in [("train", n_train, 0), ("val", n_val, 10_000)]:
        img_dir = os.path.join(root, split, "images")
        msk_dir = os.path.join(root, split, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(msk_dir, exist_ok=True)
        for i in range(n):
            img, mask = _make_one(size=size, seed=seed + offset + i)
            Image.fromarray(img).save(os.path.join(img_dir, f"{i:03d}.png"))
            Image.fromarray(mask).save(os.path.join(msk_dir, f"{i:03d}.png"))
    print(f"[data] synthetic set written to {root} "
          f"({n_train} train / {n_val} val, {size}x{size})")
