# Dataset Setup Guide

## Recommended Kaggle Datasets

### Option 1 (Best for beginners) — Bone X-Ray Tumor Dataset
- URL: https://www.kaggle.com/datasets/tushardhumal/bone-tumor-dataset-for-classification
- Classes: tumor / no_tumor  
- Size: ~500 MB  
- Images: ~2000 total

### Option 2 — Bone Lesion X-ray
- URL: https://www.kaggle.com/datasets/andrewmvd/bone-marrow-cell-classification
- Larger dataset, more variety

### Option 3 — Musculoskeletal Radiographs (MURA)
- URL: https://stanfordmlgroup.github.io/competitions/mura/
- Stanford dataset — high quality, well-cited for papers

---

## Download Steps

1. Create a Kaggle account at https://www.kaggle.com
2. Go to your profile → Settings → API → Create New Token
3. Download `kaggle.json` to `~/.kaggle/kaggle.json`
4. In terminal:

```bash
pip install kaggle
kaggle datasets download -d tushardhumal/bone-tumor-dataset-for-classification
unzip bone-tumor-dataset*.zip -d dataset/
```

5. Verify your folder looks like:

```
dataset/
├── tumor/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── no_tumor/
    ├── image001.jpg
    ├── image002.jpg
    └── ...
```

---

## Important Notes

- Aim for **at least 200 images per class** (400 total minimum)
- Images **do not** need to be the same size — the model resizes them
- Accepted formats: `.jpg`, `.jpeg`, `.png`
- Balanced classes (equal tumor/no_tumor images) → better accuracy
- Remove any corrupted or empty files

---

## Quick Dataset Sanity Check

Run this Python snippet to verify:

```python
import os

for folder in ["dataset/tumor", "dataset/no_tumor"]:
    count = len(os.listdir(folder))
    print(f"{folder}: {count} images")
```

You should see something like:
```
dataset/tumor: 450 images
dataset/no_tumor: 480 images
```
