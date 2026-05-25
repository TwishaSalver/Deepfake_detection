# Model Training Progress & Setup

**Last Updated:** May 22, 2026
**Status:** Setting up Python 3.11 environment (alternative approach)

---

## Current Installation Status

### ✅ Successfully Installed (18+ packages)
- NumPy 2.4.1, Pandas 2.3.3, Pillow 12.0.0
- OpenCV (opencv-python-headless) 4.13.0.92
- Scikit-learn 1.8.0, SciPy 1.17.1, Matplotlib 3.10.9
- Streamlit 1.56.0, Seaborn 0.13.2
- MTCNN 1.0.0, MediaPipe 0.10.35, TQDM 4.67.3
- PyTorch 2.10.0 (CUDA 12.6), TorchVision 0.25.0, TorchAudio 2.11.0
- Timm 1.0.27, Keras 3.14.1, Protobuf 5.29.6

### ❌ Issue: Python Version Mismatch
- **Current:** Python 3.14 (incompatible with TensorFlow)
- **Required:** Python 3.11
- **Missing:** TensorFlow (needed for 4 CNN region models)

---

## Overview
This guide helps you train the deepfake detection models with real data.

## Option 1: Use Kaggle Dataset (Recommended - Easiest)

The training script automatically downloads from Kaggle if credentials are configured.

### Step 1: Get Kaggle API Credentials
1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token" → saves `kaggle.json`
3. Save the file to: `C:\Users\YOUR_USERNAME\.kaggle\kaggle.json`
4. Set permissions: Right-click → Properties → Security → Edit → Full Control

### Step 2: Run Training
```bash
python train.py
```

The script will:
- Download ~140k real and fake faces from Kaggle
- Extract 200 real + 200 fake images
- Preprocess all images (MTCNN face detection, MediaPipe keypoints)
- Train 5 models (4 CNNs for regions + CViT for full face)
- Save trained models to `saved_models/`
- Generate metrics report

**Training time:** 30-60 minutes depending on GPU availability

---

## Option 2: Use Your Own Dataset

### Directory Structure
Create this structure before running training:

```
data/
├── raw/
│   ├── real/       ← Put your REAL face images here
│   └── fake/       ← Put your AI-generated faces here
```

### Upload Your Data
1. Create the directories (if they don't exist):
   ```bash
   mkdir -p data\raw\real
   mkdir -p data\raw\fake
   ```

2. Copy your images:
   - Real images → `data/raw/real/` (must contain "real" in path)
   - Fake images → `data/raw/fake/` (must contain "fake" in path)

   Or provide additional dataset folders directly at training time:
   ```bash
   python train.py --external-real path/to/real_dataset --external-fake path/to/fake_dataset
   ```
   Or pass a full dataset root with real/fake subfolders:
   ```bash
   python train.py --external-dataset path/to/FaceForensics
   ```

3. Run training:
   ```bash
   python train.py
   ```

### Supported Formats
- `.jpg`, `.jpeg`, `.png`
- Images will be auto-resized to 224×224 during preprocessing
- Minimum 50 images per class recommended

---

## Option 3: Test with Dummy Data (Quick Demo)

If you just want to test the pipeline quickly:

```bash
python train.py
```

Without Kaggle credentials or custom data, it generates 200 synthetic images.

---

## After Training

✅ Models saved to: `saved_models/`
- `cnn_eyes.h5`
- `cnn_nose.h5`
- `cnn_chin.h5`
- `cnn_ears.h5`
- `cvit_face.pth`
- `posture_mlp.h5`

✅ Metrics saved to: `saved_models/metrics.json`

✅ Test the app:
```bash
streamlit run app.py
```

---

---

## 🚀 What I Need From You - Choose ONE Action:

### Action 1: Install Python 3.11 (Recommended - Full Model)
```bash
# Download: https://www.python.org/downloads/release/python-3111/
# After installation, I'll set up everything automatically
```
✅ Full 5-model ensemble will work
✅ Highest accuracy
⏱️ ~45-60 minutes training time

### Action 2: Provide Kaggle Credentials (Option 1A)
```
1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token" → saves kaggle.json
3. Save to: C:\Users\YOUR_USERNAME\.kaggle\kaggle.json
4. Reply: "Kaggle ready"
```

### Action 3: Upload Your Own Face Images (Option 2)
```
Create:
- data/raw/real/     ← Put real face images here
- data/raw/fake/     ← Put AI-generated faces here

Reply: "Data uploaded"
```

### Action 4: Quick Test with Dummy Data (Option 3)
```
Reply: "Test with dummy data"
```
⚠️ Quick demo only (synthetic images)

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "ModuleNotFoundError: tensorflow" | Install Python 3.11 (TensorFlow not available for Python 3.14) |
| "Kaggle API not found" | Configure credentials or provide custom data |
| Training is slow | Normal on CPU. CUDA available if GPU installed |
| Out of memory error | Reduce `NUM_PER_CLASS` in `train.py` from 200 to 100 |

---

## Next Steps

👉 **Reply with ONE of these:**
- "Install Python 3.11"
- "Kaggle ready"
- "Data uploaded"
- "Test with dummy data"
