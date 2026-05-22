"""
Download data, preprocess, train all models, save metrics.
Works standalone: python train.py
"""

import json
import os
import random
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from models import (
    SAVED_DIR,
    build_cnn,
    build_cvit,
    build_posture_mlp,
    cnn_paths,
    cvit_path,
    posture_path,
)
from preprocessing import preprocess_batch, save_processed_arrays

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
METRICS_PATH = SAVED_DIR / "metrics.json"

REAL_DIR = DATA_RAW / "real"
FAKE_DIR = DATA_RAW / "fake"
NUM_PER_CLASS = 200
TRAIN_RATIO = 0.8
KAGGLE_DATASETS = [
    "xhlulu/140k-real-and-fake-faces",
    "trainingdatapro/140k-real-and-fake-faces",
    "soumikrakshit/140k-real-and-fake-faces",
]


def download_dataset():
    """Download 200 real + 200 fake from Kaggle or generate dummy images."""
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    FAKE_DIR.mkdir(parents=True, exist_ok=True)

    existing = list(REAL_DIR.glob("*")) + list(FAKE_DIR.glob("*"))
    if len(existing) >= NUM_PER_CLASS * 2:
        print(f"Using existing raw data ({len(existing)} files).")
        return

    downloaded = False
    for slug in KAGGLE_DATASETS:
        try:
            print(f"Trying Kaggle dataset: {slug}")
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", slug, "-p", str(DATA_RAW), "--unzip"],
                check=True,
                capture_output=True,
                timeout=600,
            )
            downloaded = True
            break
        except Exception as e:
            print(f"  Kaggle failed: {e}")

    if downloaded:
        _collect_kaggle_images()
    else:
        print("Kaggle unavailable — generating dummy face images.")
        _generate_dummy_data()


def _collect_kaggle_images():
    """Walk extracted Kaggle tree and copy up to 200 real / 200 fake."""
    real_paths, fake_paths = [], []

    for path in DATA_RAW.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        parts = [p.lower() for p in path.parts]
        if "fake" in parts:
            fake_paths.append(path)
            continue
        if "real" in parts:
            real_paths.append(path)
            continue
        lower = str(path).lower()
        if "/fake/" in lower.replace("\\", "/"):
            fake_paths.append(path)
        elif "/real/" in lower.replace("\\", "/"):
            real_paths.append(path)

    random.shuffle(real_paths)
    random.shuffle(fake_paths)

    for i, src in enumerate(real_paths[:NUM_PER_CLASS]):
        dst = REAL_DIR / f"real_{i:04d}{src.suffix}"
        shutil.copy2(src, dst)
    for i, src in enumerate(fake_paths[:NUM_PER_CLASS]):
        dst = FAKE_DIR / f"fake_{i:04d}{src.suffix}"
        shutil.copy2(src, dst)

    if len(list(REAL_DIR.glob("*"))) < 50:
        print("Kaggle layout not recognized — falling back to dummy data.")
        shutil.rmtree(REAL_DIR, ignore_errors=True)
        shutil.rmtree(FAKE_DIR, ignore_errors=True)
        REAL_DIR.mkdir(parents=True, exist_ok=True)
        FAKE_DIR.mkdir(parents=True, exist_ok=True)
        _generate_dummy_data()


def _generate_dummy_data():
    """Random (224,224,3) arrays saved as PNG for testing."""
    np.random.seed(42)
    for i in range(NUM_PER_CLASS):
        real_img = (np.random.rand(224, 224, 3) * 180 + 40).astype(np.uint8)
        fake_img = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        # slight pattern difference for learnability
        fake_img[:, :112, :] = 255 - fake_img[:, :112, :]
        Image.fromarray(real_img).save(REAL_DIR / f"real_{i:04d}.png")
        Image.fromarray(fake_img).save(FAKE_DIR / f"fake_{i:04d}.png")
    print(f"Generated {NUM_PER_CLASS} real + {NUM_PER_CLASS} fake dummy images.")


def build_file_lists():
    real_files = sorted(REAL_DIR.glob("*"))
    fake_files = sorted(FAKE_DIR.glob("*"))
    paths = [str(p) for p in real_files + fake_files]
    labels = [0] * len(real_files) + [1] * len(fake_files)
    return paths, np.array(labels, dtype=np.int32)


def run_preprocessing(paths, labels):
    train_paths, test_paths, train_y, test_y = train_test_split(
        paths, labels, test_size=1 - TRAIN_RATIO, random_state=42, stratify=labels
    )

    print(f"Train: {len(train_paths)}, Test: {len(test_paths)}")
    train_data = preprocess_batch(train_paths, train_y.tolist())
    test_data = preprocess_batch(test_paths, test_y.tolist())

    save_processed_arrays(train_data, DATA_PROC / "train")
    save_processed_arrays(test_data, DATA_PROC / "test")
    return train_data, test_data


def train_cnn_models(train_data, test_data):
    regions = ["eyes", "nose", "chin", "ears"]
    key_map = {"eyes": "X_eyes", "nose": "X_nose", "chin": "X_chin", "ears": "X_ears"}
    metrics = {}

    for region in regions:
        print(f"\n=== Training CNN: {region} ===")
        X_key = key_map[region]
        X_train = train_data[X_key].astype(np.float32) / 255.0
        X_test = test_data[X_key].astype(np.float32) / 255.0
        y_train = train_data["y_labels"]
        y_test = test_data["y_labels"]

        model = build_cnn((50, 50, 3))
        model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1, verbose=1)

        out_path = SAVED_DIR / f"cnn_{region}.h5"
        SAVED_DIR.mkdir(parents=True, exist_ok=True)
        model.save(out_path)
        print(f"Saved {out_path}")

        probs = model.predict(X_test, verbose=0)
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(y_test, preds)
        metrics[region] = _eval_binary(y_test, probs[:, 1], preds, acc)

    return metrics


def train_cvit(train_data, test_data):
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, Dataset

    print("\n=== Training CViT (face) ===")

    class FaceDataset(Dataset):
        def __init__(self, images, labels, transform):
            self.images = images
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            img = self.images[idx]
            return self.transform(img), self.labels[idx]

    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    X_train = train_data["X_face"]
    X_test = test_data["X_face"]
    y_train = train_data["y_labels"]
    y_test = test_data["y_labels"]

    train_ds = FaceDataset(X_train, y_train, transform)
    test_ds = FaceDataset(X_test, y_test, transform)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cvit().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/10 loss={total_loss / max(len(train_loader), 1):.4f}")

    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cvit_path())
    print(f"Saved {cvit_path()}")

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch_y.numpy())

    probs = np.vstack(all_probs)
    y_true = np.concatenate(all_labels)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_true, preds)
    return {"face": _eval_binary(y_true, probs[:, 1], preds, acc)}


def train_posture_mlp(train_data, test_data):
    print("\n=== Training Posture MLP ===")
    X_train = train_data["X_posture"]
    X_test = test_data["X_posture"]
    y_train = train_data["y_labels"]
    y_test = test_data["y_labels"]

    model = build_posture_mlp((132,))
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1, verbose=1)

    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    model.save(posture_path())
    print(f"Saved {posture_path()}")

    probs = model.predict(X_test, verbose=0)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_test, preds)
    return {"posture": _eval_binary(y_test, probs[:, 1], preds, acc)}


def _eval_binary(y_true, prob_fake, preds, accuracy):
    cm = confusion_matrix(y_true, preds).tolist()
    out = {
        "accuracy": float(accuracy),
        "confusion_matrix": cm,
        "roc_fpr": [],
        "roc_tpr": [],
        "auc": None,
    }
    try:
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, prob_fake)
            out["roc_fpr"] = fpr.tolist()
            out["roc_tpr"] = tpr.tolist()
            out["auc"] = float(roc_auc_score(y_true, prob_fake))
    except Exception:
        pass
    return out


def main():
    print("=" * 60)
    print("Deepfake Detection — Training Pipeline")
    print("=" * 60)

    download_dataset()
    paths, labels = build_file_lists()
    print(f"Total images: {len(paths)} (real=0, fake=1)")

    train_data, test_data = run_preprocessing(paths, labels)

    all_metrics = {}
    all_metrics.update(train_cnn_models(train_data, test_data))
    all_metrics.update(train_cvit(train_data, test_data))
    all_metrics.update(train_posture_mlp(train_data, test_data))

    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete. Model accuracies (test set):")
    print("=" * 60)
    for name, m in all_metrics.items():
        auc_str = f", AUC={m['auc']:.3f}" if m.get("auc") else ""
        print(f"  {name:10s} accuracy={m['accuracy']:.3f}{auc_str}")
    print(f"\nMetrics saved to {METRICS_PATH}")
    print("Saved models:")
    for p in sorted(SAVED_DIR.glob("*")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
