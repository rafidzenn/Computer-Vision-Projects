"""
Nuclei Segmentation - vanilla U-Net proof-of-concept (tracer bullet).

Runs the whole pipeline end-to-end out of the box:
  1. If no dataset is found under data/, a small SYNTHETIC nuclei set is
     generated so the pipeline is verifiable without any download.
  2. Trains a small U-Net (binary: nucleus vs. background).
  3. Evaluates with Dice + IoU on the validation split.
  4. Runs inference on one sample and saves a 3-panel overlay to outputs/.

To use a REAL dataset (e.g. MoNuSeg), place it under data/nuclei/ in this
layout, then run again (the synthetic step is skipped automatically):
    data/nuclei/train/images/*.png   data/nuclei/train/masks/*.png
    data/nuclei/val/images/*.png     data/nuclei/val/masks/*.png

Usage:
    python main.py                          # defaults (5 epochs, 128px)
    python main.py --epochs 20 --size 256   # larger run
"""
import argparse
import os

import torch

from support.evaluate import evaluate_model
from support.inference import predict_image, save_overlay
from support.model import UNet
from support.synthetic_data import generate_dataset
from support.train import train_model
from support.utils import seed_everything

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "data", "nuclei")
OUT_DIR = os.path.join(HERE, "outputs")


def ensure_data(size):
    train_imgs = os.path.join(DATA_ROOT, "train", "images")
    if os.path.isdir(train_imgs) and len(os.listdir(train_imgs)) > 0:
        print(f"[data] using existing dataset at {DATA_ROOT}")
        return
    print("[data] no dataset found -> generating synthetic tracer-bullet data")
    generate_dataset(DATA_ROOT, n_train=40, n_val=10, size=size)


def parse_args():
    p = argparse.ArgumentParser(description="Nuclei segmentation U-Net PoC")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--size", type=int, default=128, help="must be divisible by 16")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    assert args.size % 16 == 0, "--size must be divisible by 16 for this U-Net"

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    ensure_data(args.size)

    model = UNet(in_ch=3, out_ch=1, base=32)
    model = train_model(model, DATA_ROOT, device, epochs=args.epochs,
                        batch_size=args.batch_size, lr=args.lr, size=args.size)

    metrics = evaluate_model(model, DATA_ROOT, device, split="val", size=args.size)
    print(f"[eval] val Dice {metrics['dice']:.4f} | val IoU {metrics['iou']:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "unet_poc.pt"))

    val_img_dir = os.path.join(DATA_ROOT, "val", "images")
    sample = os.path.join(val_img_dir, sorted(os.listdir(val_img_dir))[0])
    arr, prob, mask = predict_image(model, sample, device, size=args.size)
    save_overlay(arr, prob, mask, os.path.join(OUT_DIR, "sample_overlay.png"))
    print(f"[done] weights + overlay saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
