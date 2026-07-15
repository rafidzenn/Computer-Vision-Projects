# Nuclei Segmentation (U-Net) — Proof of Concept

Binary segmentation of cell nuclei in H&E histopathology image patches, built as
a **vanilla U-Net proof-of-concept** first. The goal of this PoC is a thin,
working end-to-end slice (data → model → training → metrics → visualization)
that can be verified in minutes and then scaled up in clear, deliberate steps.

Part of the [Computer-Vision-Projects](../) portfolio.

---

## What it does

- Trains a small U-Net to label each pixel as **nucleus** or **background**.
- Reports **Dice** and **IoU** on a held-out validation split.
- Saves a 3-panel visualization (input · predicted probability · mask overlay).
- Runs **out of the box**: if no dataset is present, it generates a tiny
  synthetic nuclei set so the pipeline is provably functional before any real
  data is downloaded.

## Repository structure

```
.
├── main.py            # entry point: data → train → evaluate → infer
├── requirements.txt   # dependencies
├── README.md          # this file
├── data/              # datasets (auto-generated synthetic set, or real data)
├── support/           # all supporting code
│   ├── model.py         # U-Net
│   ├── dataset.py       # Dataset + augmentation
│   ├── train.py         # training loop
│   ├── evaluate.py      # Dice / IoU
│   ├── inference.py     # prediction + overlay
│   ├── synthetic_data.py# tracer-bullet data generator
│   └── utils.py         # seeding, metrics, loss
└── others/            # presentation, reports, demo video (course deliverables)
```

## Quick start

```bash
pip install -r requirements.txt
python main.py                        # 5 epochs on 128px synthetic data (CPU-friendly)
```

Outputs (`unet_poc.pt` weights and `sample_overlay.png`) are written to `outputs/`.

### Run on real data

Place a real dataset under `data/nuclei/` in the layout described in
[`data/README.md`](data/README.md), then:

```bash
python main.py --epochs 30 --size 256
```

## Method (PoC)

| Component      | Choice                                  |
|----------------|-----------------------------------------|
| Task           | Binary semantic segmentation            |
| Model          | U-Net (4 down / 4 up, BatchNorm + ReLU) |
| Loss           | BCE + soft-Dice (handles class imbalance)|
| Metrics        | Dice, IoU                               |
| Augmentation   | Flips + 90° rotations                   |

## Known limitations (by design, for the PoC)

- **Semantic, not instance** — touching nuclei merge into one blob. Separating
  them needs instance-aware methods (see roadmap).
- **No stain normalization** — real H&E colour varies across labs/scanners.
- **Synthetic default data** proves plumbing only, not accuracy.

## Roadmap: scaling from PoC → functional system

1. **Real data + honest splits** — MoNuSeg/PanNuke, split by slide, report on
   unseen slides.
2. **Instance segmentation** — watershed post-processing, or switch to
   **StarDist / HoVer-Net / Cellpose**; evaluate with **AJI** and **Panoptic
   Quality**, not just Dice.
3. **Stain normalization** (Macenko/Vahadane) and colour augmentation for
   cross-scanner generalization.
4. **Stronger backbones** — pretrained encoders (ResNet/EfficientNet/ViT), or
   pathology foundation models.
5. **Whole-slide inference** — tiled reading via OpenSlide, overlap-tile
   stitching, memory-aware batching.
6. **Engineering** — config files, experiment tracking, checkpointing, mixed
   precision, reproducible seeds.

## References / tools

U-Net (Ronneberger et al., 2015); MoNuSeg and Data Science Bowl 2018 datasets;
StarDist, HoVer-Net, Cellpose for instance segmentation. This scaffold was
prepared with the help of an AI assistant and is meant to be understood,
modified, and extended by the team.
