# Deepfake Detection System

Multi-model ensemble for detecting AI-generated (fake) vs authentic (real) face images. Combines **CViT** (full face), **CNNs** (eyes, nose, chin, ears), and a **posture MLP** (MediaPipe keypoints) with majority-vote fusion.

## Project structure

```
deepfake_detection/
├── app.py              # Streamlit dashboard
├── preprocessing.py    # MTCNN + region crops + MediaPipe
├── models.py           # CNN, CViT, MLP builders & loaders
├── ensemble.py         # Majority-vote inference
├── train.py            # Download → preprocess → train → metrics
├── requirements.txt
├── README.md
├── saved_models/       # Trained weights (created by train.py)
└── data/               # Raw & processed numpy arrays
```

## Setup

```bash
cd deepfake_detection
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Recommended: staged install (handles slow downloads / timeouts)
chmod +x install.sh
./install.sh
```

Or manual install with a longer timeout:

```bash
python -m pip install --upgrade pip
pip install --default-timeout=1000 --retries 10 -r requirements.txt
```

### If `pip` times out (e.g. on `opencv-python`)

Large wheels (~40MB+) often fail on slow Wi‑Fi. Fix:

```bash
source venv/bin/activate
python -m pip install --upgrade pip
pip install --default-timeout=1000 --retries 10 opencv-python-headless
pip install --default-timeout=1000 --retries 10 -r requirements.txt
```

We use **opencv-python-headless** (same `cv2` API, no GUI) — sufficient for this project.

### If `mediapipe` install fails (`SyntaxError: invalid character '∂'`)

Recent MediaPipe wheels ship a corrupted test file; pip’s bytecode compile step crashes. Fix:

```bash
pip uninstall mediapipe -y 2>/dev/null || true
pip install --no-compile --no-deps "mediapipe==0.10.11"
pip install "absl-py" "attrs>=19.1.0" "flatbuffers>=2.0" "protobuf>=3.11,<4" \
  "sounddevice>=0.4.4" "opencv-contrib-python>=4.8,<4.11"
pip install "numpy>=1.23,<2.0" "opencv-python-headless>=4.8,<4.11"
```

Or run `./install.sh` (handles this automatically).

### If `mtcnn` fails (`No module named 'pkg_resources'`)

```bash
pip install "setuptools>=65,<82"
python -c "from mtcnn import MTCNN; print('mtcnn OK')"
```

### Verify everything installed

```bash
python verify_install.py
```

If Streamlit crashes with `cannot import name 'TypeAliasType' from 'typing_extensions'`:

```bash
pip install "typing-extensions>=4.10,<5"
streamlit run app.py
```

(TensorFlow installs an older `typing-extensions`; upgrading it is safe for this project.)

### Kaggle (optional)

To download the real dataset instead of dummy images:

1. Create API token at https://www.kaggle.com/settings
2. Place `kaggle.json` in `~/.kaggle/`
3. Run training — it will try several dataset slugs for `140k-real-and-fake-faces`

## Train all models

```bash
python train.py
```

This will:

1. Download **200 real + 200 fake** images (or generate dummy PNGs if Kaggle fails)
2. Split **80% train / 20% test**
3. Preprocess with MTCNN + region crops + MediaPipe posture
4. Train CNNs (20 epochs), CViT (10 epochs), posture MLP (20 epochs)
5. Save weights under `saved_models/` and metrics to `saved_models/metrics.json`

## Run the dashboard

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

### Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (includes `.python-version` with **3.11** — required).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app**.
3. Repo: `varadasaiakhila/Deepfake_detection`, branch `main`, file `app.py`.
4. In **Advanced settings**, set **Python version** to **3.11** if the deploy still uses 3.14.
5. **Reboot** the app after pushing dependency fixes.

**If install fails** (TensorFlow / typing-extensions / memory):

1. Streamlit app → **Settings** → **Advanced**
2. Set **Python version** to **3.11**
3. Set **Requirements file** to `requirements-cloud.txt` (demo + UI only, fast)
4. **Reboot** app

Full ML stack uses `requirements.txt` (no `typing-extensions` pin — avoids TF conflict).

Trained models are gitignored — use **Demo Mode** on Cloud unless you upload weights.

Local training:

```bash
pip install -r requirements-full.txt
```

## Standalone modules

```bash
python preprocessing.py path/to/image.jpg
python models.py
python ensemble.py path/to/image.jpg
```

## Labels

- `0` = REAL  
- `1` = FAKE  

Ensemble returns `"REAL"` or `"FAKE"` with confidence and per-model votes.
