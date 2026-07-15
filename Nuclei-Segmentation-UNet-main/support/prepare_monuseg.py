"""
Convert a raw MoNuSeg download into the layout this repo trains on.

MoNuSeg gives, per slide:
    <name>.tif   H&E image (~1000x1000 RGB)
    <name>.xml   Aperio-style polygon annotations of every nucleus

This script:
  1. rasterizes each XML into a binary nucleus mask,
  2. splits SLIDES (not tiles) into train / val  -> prevents data leakage,
  3. tiles each slide into fixed-size patches (edge-clamped for full coverage),
  4. writes  <out>/train|val/images/*.png  and  <out>/train|val/masks/*.png .

After running this, `python main.py` trains on real data with no other changes.

Get the data (CC BY-NC-SA 4.0) from the official challenge page:
    https://monuseg.grand-challenge.org/Data/
Unzip it anywhere; this script finds the .tif/.xml pairs recursively.

Usage:
    python support/prepare_monuseg.py --src /path/to/MoNuSeg/raw --out data/nuclei
    python support/prepare_monuseg.py --src ./MoNuSeg --tile-size 256 --stride 128 --min-fg 0.01
"""
import argparse
import os
import random
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def _local(tag):
    """Strip any XML namespace, e.g. '{ns}Region' -> 'Region'."""
    return tag.rsplit("}", 1)[-1]


def find_pairs(src):
    """Match <stem>.tif with <stem>.xml anywhere under src."""
    images, xmls = {}, {}
    for root, _, files in os.walk(src):
        for f in files:
            stem, ext = os.path.splitext(f)
            ext = ext.lower()
            if ext in IMAGE_EXTS:
                images[stem] = os.path.join(root, f)
            elif ext == ".xml":
                xmls[stem] = os.path.join(root, f)
    return sorted((images[s], xmls[s]) for s in images if s in xmls)


def rasterize_xml(xml_path, size):
    """Fill every annotated nucleus polygon onto a binary mask of (W, H)."""
    width, height = size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    root = ET.parse(xml_path).getroot()
    for region in (el for el in root.iter() if _local(el.tag) == "Region"):
        pts = [(float(v.get("X")), float(v.get("Y")))
               for v in region.iter() if _local(v.tag) == "Vertex"]
        if len(pts) >= 3:
            draw.polygon(pts, fill=255)
    return np.array(mask)


def tile_positions(length, tile, stride):
    """Start coordinates covering `length`, with the last tile clamped to the
    edge so the whole image is covered (small overlap at the border is fine)."""
    if length <= tile:
        return [0]
    pos = list(range(0, length - tile + 1, stride))
    if pos[-1] != length - tile:
        pos.append(length - tile)
    return pos


def parse_args():
    ap = argparse.ArgumentParser(description="Prepare MoNuSeg for this repo")
    ap.add_argument("--src", required=True, help="folder containing MoNuSeg .tif + .xml")
    ap.add_argument("--out", default="data/nuclei", help="output dataset root")
    ap.add_argument("--tile-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256, help="use < tile-size for overlap")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--min-fg", type=float, default=0.0,
                    help="drop tiles with less than this foreground fraction (0 = keep all)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    pairs = find_pairs(args.src)
    if not pairs:
        raise SystemExit(f"No .tif/.xml pairs found under {args.src!r}")

    random.seed(args.seed)
    random.shuffle(pairs)
    n_val = max(1, round(len(pairs) * args.val_frac))
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}
    print(f"[prep] {len(pairs)} slides -> {len(splits['train'])} train / "
          f"{len(splits['val'])} val (split by slide, no leakage)")

    for split, split_pairs in splits.items():
        img_dir = os.path.join(args.out, split, "images")
        msk_dir = os.path.join(args.out, split, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(msk_dir, exist_ok=True)

        n_tiles = 0
        for img_path, xml_path in split_pairs:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            img = np.array(Image.open(img_path).convert("RGB"))
            h, w = img.shape[:2]
            mask = rasterize_xml(xml_path, (w, h))

            for y in tile_positions(h, args.tile_size, args.stride):
                for x in tile_positions(w, args.tile_size, args.stride):
                    m = mask[y:y + args.tile_size, x:x + args.tile_size]
                    if args.min_fg > 0 and (m > 0).mean() < args.min_fg:
                        continue
                    p = img[y:y + args.tile_size, x:x + args.tile_size]
                    name = f"{stem}_{y:04d}_{x:04d}.png"
                    Image.fromarray(p).save(os.path.join(img_dir, name))
                    Image.fromarray(m).save(os.path.join(msk_dir, name))
                    n_tiles += 1
        print(f"[prep] {split}: {n_tiles} tiles")

    print(f"[prep] done -> {args.out}\n"
          f"[prep] now train on real data:  python main.py --epochs 40 --size {args.tile_size}")


if __name__ == "__main__":
    main()
