#!/usr/bin/env python3
"""Check that all deepfake-detection dependencies import correctly."""

import sys

CHECKS = [
    ("numpy", "import numpy; print(numpy.__version__)"),
    ("cv2", "import cv2; print(cv2.__version__)"),
    ("PIL", "from PIL import Image"),
    ("mtcnn", "from mtcnn import MTCNN"),
    ("mediapipe", "import mediapipe as mp; print(mp.__version__)"),
    ("torch", "import torch; print(torch.__version__)"),
    ("timm", "import timm"),
    ("tensorflow", "import tensorflow as tf; print(tf.__version__)"),
    ("streamlit", "import streamlit"),
    ("sklearn", "import sklearn"),
]


def main():
    failed = []
    print("Deepfake Detection — dependency check\n")
    for name, code in CHECKS:
        try:
            exec(code, {})
            print(f"  OK  {name}")
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            failed.append(name)

    print()
    if failed:
        print(f"Failed: {', '.join(failed)}")
        sys.exit(1)
    print("All checks passed. Run: python train.py")


if __name__ == "__main__":
    main()
