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
project brief asks for) is a good first real dataset:
[monuseg.grand-challenge.org/Data](https://monuseg.grand-challenge.org/Data/).
It ships `.tif` images with `.xml` polygon annotations.

Don't build the masks by hand — run the included converter, which rasterizes the
XML polygons, splits by slide, and tiles the images into the layout above:

```bash
python support/prepare_monuseg.py --src /path/to/MoNuSeg --out data/nuclei
```

Other common options: **Data Science Bowl 2018** (fastest; masks already
provided) and **PanNuke** / **CoNSeP** (multi-class, harder).

> The converter splits by **slide**, not by patch — patches from the same slide
> never appear in both train and val, so reported metrics aren't inflated.
