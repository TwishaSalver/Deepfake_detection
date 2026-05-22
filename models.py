"""
Model definitions and loaders for deepfake detection.
Works standalone: python models.py
"""

import os
from pathlib import Path

import numpy as np

SAVED_DIR = Path(__file__).resolve().parent / "saved_models"
LABEL_REAL = 0
LABEL_FAKE = 1


def build_cnn(input_shape):
    """TensorFlow CNN for 50x50 region classifiers."""
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (
        BatchNormalization,
        Conv2D,
        Dense,
        Dropout,
        Flatten,
        MaxPooling2D,
    )

    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.3),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.3),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.3),
            Dense(2, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_posture_mlp(input_shape=(132,)):
    """TensorFlow MLP for posture keypoints."""
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout

    model = Sequential(
        [
            Dense(128, activation="relu", input_shape=input_shape),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(2, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cvit(num_classes=2, pretrained=True):
    """PyTorch ViT via timm for full-face classification."""
    import torch
    import timm

    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def load_keras_model(path):
    import tensorflow as tf

    return tf.keras.models.load_model(path)


def load_cvit(path, device=None):
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cvit(pretrained=False)
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    elif isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def predict_keras(model, batch, class_names=("REAL", "FAKE")):
    """Return label string and confidence % for one batch item."""
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    probs = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx] * 100)
    return class_names[idx], conf, probs


def predict_cvit(model, device, face_rgb):
    """face_rgb: uint8 (224,224,3)."""
    import torch
    import torchvision.transforms as T

    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    tensor = transform(face_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    names = ("REAL", "FAKE")
    return names[idx], float(probs[idx] * 100), probs


def prepare_cnn_batch(region_uint8):
    """Single 50x50x3 -> batch (1,50,50,3) float32 /255."""
    x = region_uint8.astype(np.float32) / 255.0
    return np.expand_dims(x, axis=0)


def prepare_posture_batch(posture_vec):
    return np.expand_dims(posture_vec.astype(np.float32), axis=0)


def cnn_paths():
    return {
        "eyes": SAVED_DIR / "cnn_eyes.h5",
        "nose": SAVED_DIR / "cnn_nose.h5",
        "chin": SAVED_DIR / "cnn_chin.h5",
        "ears": SAVED_DIR / "cnn_ears.h5",
    }


def cvit_path():
    return SAVED_DIR / "cvit_face.pth"


def posture_path():
    return SAVED_DIR / "posture_model.h5"


def models_available():
    """Which trained model files exist."""
    status = {}
    for name, path in cnn_paths().items():
        status[name] = path.exists()
    status["face"] = cvit_path().exists()
    status["posture"] = posture_path().exists()
    return status


if __name__ == "__main__":
    print("Building CNN (dry run)...")
    m = build_cnn((50, 50, 3))
    print(m.summary())
    print("CViT available:", cvit_path())
    print("Models on disk:", models_available())
