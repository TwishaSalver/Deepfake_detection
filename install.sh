#!/usr/bin/env bash
# Reliable install on slow/unstable networks (macOS/Linux)
set -euo pipefail

cd "$(dirname "$0")"
VENV="${VENV:-venv}"

if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

export PIP_DEFAULT_TIMEOUT=1000
export PIP_RETRIES=10

echo "==> Upgrading pip..."
python -m pip install --upgrade pip wheel
pip install "setuptools>=65,<82"

echo "==> Pinning numpy (<2 for TensorFlow 2.13)..."
pip install --default-timeout=1000 --retries 10 "numpy>=1.23,<2.0"

echo "==> Installing lightweight packages..."
pip install --default-timeout=1000 --retries 10 \
  Pillow tqdm pandas scikit-learn matplotlib seaborn streamlit kaggle

echo "==> Installing OpenCV headless..."
pip install --default-timeout=1000 --retries 10 "opencv-python-headless>=4.8,<4.11"

echo "==> Installing mtcnn..."
pip install --default-timeout=1000 --retries 10 mtcnn

echo "==> Installing PyTorch (before mediapipe to avoid version override)..."
pip install --default-timeout=1000 --retries 10 torch==2.0.1 torchvision==0.15.2 timm

echo "==> Installing TensorFlow..."
pip install --default-timeout=1000 --retries 10 tensorflow==2.13.0

echo "==> Installing mediapipe (--no-compile avoids broken test file in wheel)..."
pip uninstall mediapipe -y 2>/dev/null || true
pip install --default-timeout=1000 --retries 10 --no-compile --no-deps "mediapipe==0.10.11"
pip install --default-timeout=1000 --retries 10 \
  "absl-py" "attrs>=19.1.0" "flatbuffers>=2.0" "matplotlib" \
  "protobuf>=3.11,<4" "sounddevice>=0.4.4" \
  "opencv-contrib-python>=4.8,<4.11"

echo "==> Re-pin numpy + opencv after mediapipe deps..."
pip install --default-timeout=1000 --retries 10 "numpy>=1.23,<2.0" "opencv-python-headless>=4.8,<4.11"

echo "==> Fix typing_extensions for Streamlit (TensorFlow pins an old version)..."
pip install "typing-extensions>=4.10,<5"

echo "==> Done. Verify:"
python verify_install.py
