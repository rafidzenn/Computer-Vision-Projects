# data/

Datasets live here. Large image files should **not** be committed to Git (add
them to `.gitignore`); keep only small samples or a download script in the repo.

## Expected layout

```
data/nuclei/
├── train/
│   ├── images/   # RGB patches (.png)
│   └── masks/    # binary masks (.png), same filenames, foreground = nucleus
└── val/
    ├── images/
    └── masks/
```

## Proof-of-concept (default)

If this folder is empty, `main.py` auto-generates a small **synthetic** set here
so the pipeline runs end-to-end with zero setup. Synthetic data only tests the
plumbing — it is not meaningful for results.

## Using a real dataset (recommended for real results)

**MoNuSeg** (H&E stained, multiple organs → the "heterogeneous backgrounds" the
project brief asks for) is a good first real dataset. Its annotations come as
per-image XML polygons, so there is one extra step: rasterize each XML into a
binary PNG mask, then drop images/masks into the layout above.

Other common options: **Data Science Bowl 2018** (fastest; masks already
provided) and **PanNuke** / **CoNSeP** (multi-class, harder).

> Split by **slide**, not by patch — patches from the same slide must not appear
> in both train and val, or the reported metrics will be optimistically biased.
