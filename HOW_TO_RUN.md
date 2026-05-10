# 🦴 Bone Tumor Detection — Complete Setup & Run Guide

> **Everything you need to go from zero to working demo.**  
> Follow these steps in order. Don't skip any.

---

## Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Python | 3.9 – 3.11 | `python3 --version` |
| pip | Latest | `pip --version` |
| Git (optional) | Any | `git --version` |
| 4 GB free disk | — | For model + dataset |

---

## STEP 1 — Create Project Folder

Open Terminal (on Mac: `Cmd + Space` → type "Terminal"):

```bash
mkdir bone-tumor-detection
cd bone-tumor-detection
```

---

## STEP 2 — Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

✅ You'll see `(venv)` at the start of your terminal line.

> **Windows users:** Use `venv\Scripts\activate` instead.

---

## STEP 3 — Copy Project Files

Place all project files inside your `bone-tumor-detection/` folder:

```
bone-tumor-detection/
│
├── dataset/
│   ├── tumor/          ← put X-ray images WITH tumors here
│   └── no_tumor/       ← put X-ray images WITHOUT tumors here
│
├── model/              ← auto-created when you train
│
├── app.py              ← Streamlit web app
├── train.py            ← model training script
├── requirements.txt    ← Python dependencies
└── HOW_TO_RUN.md       ← this file
```

---

## STEP 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: TensorFlow, Streamlit, OpenCV, Matplotlib, NumPy, Pandas, Pillow, Scikit-learn.

> ⏳ First install takes 5–10 minutes (TensorFlow is large). Be patient.

If you get an error about TensorFlow on Mac M1/M2/M3:
```bash
pip install tensorflow-macos tensorflow-metal
```

---

## STEP 5 — Download Dataset

### Option A: Kaggle (Recommended)

1. Go to [kaggle.com](https://www.kaggle.com) → create a free account
2. Profile → Settings → API → **Create New Token** → downloads `kaggle.json`
3. Move it:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```
4. Download dataset:
```bash
pip install kaggle
kaggle datasets download -d tushardhumal/bone-tumor-dataset-for-classification
unzip bone-tumor-dataset*.zip
```
5. Move images:
```bash
# Adjust paths based on how the ZIP unpacks
mv extracted_folder/tumor/* dataset/tumor/
mv extracted_folder/no_tumor/* dataset/no_tumor/
```

### Option B: Manual Download

1. Go to: https://www.kaggle.com/datasets/tushardhumal/bone-tumor-dataset-for-classification
2. Click **Download** (ZIP file)
3. Extract and copy images into:
   - `dataset/tumor/` — images that show tumors
   - `dataset/no_tumor/` — images that don't show tumors

### Verify Dataset

```bash
python3 -c "
import os
for f in ['dataset/tumor', 'dataset/no_tumor']:
    n = len([x for x in os.listdir(f) if x.endswith(('.jpg','.jpeg','.png'))])
    print(f'{f}: {n} images')
"
```

Minimum: **200 images per class** (400 total). Ideal: 500+ per class.

---

## STEP 6 — Train the Model

```bash
python train.py
```

### What Happens:
1. Loads and augments your dataset
2. Loads pretrained MobileNetV2 (downloads ~14 MB from Google)
3. Trains the top layers (Phase 1) — ~5–15 minutes
4. Fine-tunes deeper layers (Phase 2) — ~5–10 minutes
5. Saves model to `model/bone_tumor_model.h5`
6. Saves training curves to `model/training_curves.png`

### Expected Output:
```
[1/5] Loading and augmenting dataset...
  Train samples : 720
  Val samples   : 180
  Classes       : {'no_tumor': 0, 'tumor': 1}

[2/5] Building MobileNetV2 model...
[3/5] Training model...
Epoch 1/20 - loss: 0.6234 - accuracy: 0.6812 - val_accuracy: 0.7500
...
Epoch 15/20 - loss: 0.1823 - accuracy: 0.9312 - val_accuracy: 0.9100

✅  Training complete!
```

### Target Accuracy:
- **80–85%** = Good for a mini project ✅
- **85–92%** = Excellent ✅✅
- Don't worry about getting 99% — that's unrealistic for small datasets.

---

## STEP 7 — Launch the Web App

```bash
streamlit run app.py
```

The app opens automatically in your browser at:  
**http://localhost:8501**

### Using the App:
1. Click **Browse files** or drag & drop an X-ray image (JPG/PNG)
2. Preview the uploaded image
3. Click **🔍 Analyze X-Ray**
4. View:
   - Prediction (Tumor / No Tumor)
   - Confidence percentage
   - Grad-CAM heatmap showing where the model focused

---

## STEP 8 — Stop the App

Press `Ctrl + C` in Terminal to stop Streamlit.

To deactivate the virtual environment:
```bash
deactivate
```

---

## Common Errors & Fixes

### "ModuleNotFoundError: No module named 'tensorflow'"
```bash
# Make sure venv is active
source venv/bin/activate
pip install tensorflow
```

### "Model not found"
- You need to run `python train.py` before running `streamlit run app.py`
- Check that `model/bone_tumor_model.h5` exists after training

### "No images found in dataset"
- Make sure images are inside `dataset/tumor/` and `dataset/no_tumor/`
- Check that files end in `.jpg`, `.jpeg`, or `.png`
- Avoid subfolders inside tumor/no_tumor

### TensorFlow slow on Mac
```bash
pip uninstall tensorflow
pip install tensorflow-macos tensorflow-metal
```

### Streamlit port already in use
```bash
streamlit run app.py --server.port 8502
```

### Low accuracy (below 70%)
- Add more training images (aim for 500+ per class)
- Make sure your images are actual X-rays, not random photos
- Increase `EPOCHS` in `train.py` from 20 to 30

---

## Full File Reference

| File | Purpose |
|------|---------|
| `train.py` | Trains MobileNetV2 on your dataset, saves model |
| `app.py` | Streamlit web UI for uploading images and getting predictions |
| `requirements.txt` | All Python packages needed |
| `dataset/tumor/` | X-ray images WITH bone tumors |
| `dataset/no_tumor/` | X-ray images WITHOUT bone tumors |
| `model/bone_tumor_model.h5` | Saved trained model (auto-created) |
| `model/class_indices.json` | Label mapping (auto-created) |
| `model/training_curves.png` | Accuracy/loss graphs (auto-created) |

---

## Recommended Day-by-Day Plan

| Day | Task |
|-----|------|
| 1 | Setup project + download dataset |
| 2 | Run `train.py`, understand output |
| 3 | Improve training (tweak epochs/augmentation) |
| 4 | Run `app.py`, test with real X-rays |
| 5 | Polish UI, test edge cases |
| 6 | Prepare PPT and report |
| 7 | Final demo rehearsal |

---

## Viva/Presentation Talking Points

**Q: Why MobileNetV2?**  
A: It's a lightweight, efficient CNN pretrained on 1.2M ImageNet images. Transfer learning lets us leverage those learned features for medical imaging without needing millions of X-ray images.

**Q: What is Grad-CAM?**  
A: Gradient-weighted Class Activation Mapping. It generates a heatmap showing which pixels the model found most influential for its decision — making the AI explainable.

**Q: Why binary classification?**  
A: We start with tumor vs. no-tumor as a proof of concept. Future work can extend to multi-class (osteosarcoma, chondrosarcoma, etc.).

**Q: What's the accuracy?**  
A: ~85–90% on validation set. Medical AI typically requires clinical trials and much larger datasets for deployment — this is a research prototype.

---

*Good luck with your project! 🎓*
