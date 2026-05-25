"""
Download data, preprocess, train all models, save metrics.
Works standalone: python train.py
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
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
NUM_PER_CLASS = 300
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATASET_LABEL_PATTERNS = {
    "real": {"real", "original", "original_sequences", "real_v_fake", "real-vs-fake", "real_vs_fake"},
    "fake": {"fake", "manipulated", "deepfake", "c23", "c40", "manipulated_sequences"},
}
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
EPOCHS_CNN = 60
EPOCHS_CVIT = 20
EPOCHS_POSTURE = 80
BATCH_SIZE_CNN = 12
BATCH_SIZE_CVIT = 8
BATCH_SIZE_POSTURE = 16
KAGGLE_DATASETS = [
    "xhlulu/140k-real-and-fake-faces",
    "trainingdatapro/140k-real-and-fake-faces",
    "soumikrakshit/140k-real-and-fake-faces",
]
KAGGLE_DOWNLOAD_TIMEOUT = 3600


def download_dataset(
    allow_synthetic=False,
    external_real_dirs=None,
    external_fake_dirs=None,
    external_dataset_dirs=None,
    num_per_class=NUM_PER_CLASS,
):
    """Download real/fake images from Kaggle or validate existing local raw data.
    Synthetic dummy data is only created when allow_synthetic=True."""
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    FAKE_DIR.mkdir(parents=True, exist_ok=True)

    real_files = list(REAL_DIR.glob("*"))
    fake_files = list(FAKE_DIR.glob("*"))
    external_real = _collect_external_images(external_real_dirs)
    external_fake = _collect_external_images(external_fake_dirs)
    dataset_real, dataset_fake = _discover_external_dataset_images(external_dataset_dirs)

    if (real_files or external_real or dataset_real) and (fake_files or external_fake or dataset_fake):
        total_real = len(real_files) + len(external_real) + len(dataset_real)
        total_fake = len(fake_files) + len(external_fake) + len(dataset_fake)
        print(f"Using raw/external data: real={total_real}, fake={total_fake}")
        if not allow_synthetic and not (external_real or external_fake or dataset_real or dataset_fake):
            if not _validate_raw_images(real_files) or not _validate_raw_images(fake_files):
                raise RuntimeError(
                    "Existing raw data does not appear to contain real face images. "
                    "Replace data/raw/real and data/raw/fake with true face datasets, "
                    "or rerun with --allow-synthetic to generate synthetic test images."
                )
        return

    _remove_partial_kaggle_downloads()

    downloaded = False
    for slug in KAGGLE_DATASETS:
        try:
            print(f"Trying Kaggle dataset: {slug}")
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", slug, "-p", str(DATA_RAW), "--unzip", "--force"],
                check=True,
                timeout=KAGGLE_DOWNLOAD_TIMEOUT,
            )
            downloaded = True
            break
        except subprocess.TimeoutExpired:
            print(f"  Kaggle failed: download timed out after {KAGGLE_DOWNLOAD_TIMEOUT} seconds.")
        except subprocess.CalledProcessError as e:
            print(f"  Kaggle failed: {e}")
        except FileNotFoundError as e:
            print(f"  Kaggle CLI not found: {e}")
            break

    if downloaded:
        _collect_kaggle_images(allow_synthetic=allow_synthetic)
        real_files = list(REAL_DIR.glob("*"))
        fake_files = list(FAKE_DIR.glob("*"))
        if real_files and fake_files:
            return

    if allow_synthetic:
        print("Kaggle unavailable or raw data missing — generating dummy face images.")
        print("Warning: running on synthetic dummy data only. Models trained on this data may not generalize to real faces.")
        _generate_dummy_data()
        return

    raise RuntimeError(
        "No raw real/fake data found and Kaggle download failed. "
        "Place real images in data/raw/real and fake images in data/raw/fake, "
        "or rerun with --allow-synthetic to create dummy data for testing."
    )

def _remove_partial_kaggle_downloads():
    """Remove stale raw dataset zip files before retrying a Kaggle download."""
    for path in DATA_RAW.glob("*.zip"):
        try:
            path.unlink()
        except OSError:
            pass
    for path in DATA_RAW.glob("*.zip.*"):
        try:
            path.unlink()
        except OSError:
            pass


def _collect_external_images(directories):
    """Recursively collect supported images from external dataset directories."""
    image_paths = []
    for root in directories or []:
        path = Path(root)
        if not path.exists():
            continue
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(path)
            continue
        for item in path.rglob("*"):
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(item)
    return sorted({p.resolve() for p in image_paths})


def _discover_external_dataset_images(directories):
    """Collect images from labeled external dataset directories with real/fake subfolders."""
    real_images, fake_images = [], []
    for root in directories or []:
        path = Path(root)
        if not path.exists() or not path.is_dir():
            continue

        for child in path.rglob("*"):
            if not child.is_dir():
                continue
            name = child.name.lower()
            if name in DATASET_LABEL_PATTERNS["real"]:
                real_images.extend(_collect_external_images([child]))
            elif name in DATASET_LABEL_PATTERNS["fake"]:
                fake_images.extend(_collect_external_images([child]))

        # Fall back to the root itself if it looks labeled
        root_name = path.name.lower()
        if root_name in DATASET_LABEL_PATTERNS["real"]:
            real_images.extend(_collect_external_images([path]))
        elif root_name in DATASET_LABEL_PATTERNS["fake"]:
            fake_images.extend(_collect_external_images([path]))

    return sorted({p.resolve() for p in real_images}), sorted({p.resolve() for p in fake_images})


def _make_class_weights(y):
    if len(np.unique(y)) < 2:
        return None
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def _balance_dataset(real_files, fake_files):
    if not real_files or not fake_files:
        return real_files, fake_files
    real_files = list(real_files)
    fake_files = list(fake_files)
    if len(real_files) == len(fake_files):
        return real_files, fake_files
    if len(real_files) < len(fake_files):
        real_files.extend(np.random.choice(real_files, size=len(fake_files) - len(real_files), replace=True).tolist())
    else:
        fake_files.extend(np.random.choice(fake_files, size=len(real_files) - len(fake_files), replace=True).tolist())
    random.shuffle(real_files)
    random.shuffle(fake_files)
    return real_files, fake_files


def _collect_kaggle_images(allow_synthetic=False, num_per_class=NUM_PER_CLASS):
    """Walk extracted Kaggle tree and copy up to num_per_class real / num_per_class fake."""
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

    for i, src in enumerate(real_paths[:num_per_class]):
        dst = REAL_DIR / f"real_{i:04d}{src.suffix}"
        shutil.copy2(src, dst)
    for i, src in enumerate(fake_paths[:num_per_class]):
        dst = FAKE_DIR / f"fake_{i:04d}{src.suffix}"
        shutil.copy2(src, dst)

    if len(list(REAL_DIR.glob("*"))) < 50 or len(list(FAKE_DIR.glob("*"))) < 50:
        print("Kaggle layout not recognized or raw data incomplete.")
        shutil.rmtree(REAL_DIR, ignore_errors=True)
        shutil.rmtree(FAKE_DIR, ignore_errors=True)
        REAL_DIR.mkdir(parents=True, exist_ok=True)
        FAKE_DIR.mkdir(parents=True, exist_ok=True)
        if allow_synthetic:
            _generate_dummy_data()
        else:
            raise RuntimeError(
                "Kaggle download did not produce usable raw real/fake images. "
                "Place data under data/raw/real and data/raw/fake, or use --allow-synthetic."
            )


def _generate_dummy_data():
    """Random (224,224,3) arrays saved as PNG for testing."""
    np.random.seed(RANDOM_SEED)
    for i in range(NUM_PER_CLASS):
        base_img = (np.random.rand(224, 224, 3) * 100 + 80).astype(np.float32)
        real_img = np.clip(base_img, 0, 255).astype(np.uint8)
        fake_img = base_img.copy()
        # add a stronger structured difference for learnability while preserving base noise
        fake_img[:, :112, 0] = np.clip(fake_img[:, :112, 0] + 120, 0, 255)
        fake_img[:, 112:, 1] = np.clip(fake_img[:, 112:, 1] - 120, 0, 255)
        fake_img = fake_img.astype(np.uint8)
        Image.fromarray(real_img).save(REAL_DIR / f"real_{i:04d}.png")
        Image.fromarray(fake_img).save(FAKE_DIR / f"fake_{i:04d}.png")
    print(f"Generated {NUM_PER_CLASS} real + {NUM_PER_CLASS} fake dummy images.")


def _validate_raw_images(image_paths, min_face_ratio=0.5, sample_size=20):
    from preprocessing import detect_face_box

    if not image_paths:
        return False

    sample = list(image_paths)
    if len(sample) > sample_size:
        sample = random.sample(sample, sample_size)

    detected = 0
    for path in sample:
        try:
            img = Image.open(path).convert("RGB")
            rgb = np.array(img)
            if detect_face_box(rgb) is not None:
                detected += 1
        except Exception:
            continue

    return detected / len(sample) >= min_face_ratio


def _build_video_from_images(image_paths, output_path, fps=2):
    if not image_paths:
        return False
    output_path = Path(output_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (224, 224))
    if not writer.isOpened():
        return False

    for path in image_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        writer.write(frame)

    writer.release()
    return True


def evaluate_video_ensemble(test_paths, test_labels, videos_per_class=4, frames_per_video=8):
    try:
        from ensemble import predict_deepfake_video
    except Exception:
        return {}

    video_records = []
    for label in [0, 1]:
        label_paths = [p for p, y in zip(test_paths, test_labels) if y == label]
        if not label_paths:
            continue
        for video_idx in range(videos_per_class):
            if len(label_paths) >= frames_per_video:
                selected = random.sample(label_paths, frames_per_video)
            else:
                selected = random.choices(label_paths, k=frames_per_video)

            with tempfile.TemporaryDirectory() as tmpdir:
                video_file = Path(tmpdir) / f"video_{label}_{video_idx}.mp4"
                if not _build_video_from_images(selected, video_file):
                    continue
                result = predict_deepfake_video(video_file, demo_mode=False, max_frames=frames_per_video)
                verdict = result.get("verdict", "UNKNOWN")
                pred_label = 1 if verdict == "FAKE" else 0
                video_records.append((label, pred_label))

    if not video_records:
        return {}

    true_labels = [r[0] for r in video_records]
    pred_labels = [r[1] for r in video_records]
    accuracy = float(accuracy_score(true_labels, pred_labels))
    cm = confusion_matrix(true_labels, pred_labels).tolist()

    return {
        "video": {
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "roc_fpr": [],
            "roc_tpr": [],
            "auc": None,
        }
    }


def build_file_lists(external_real_dirs=None, external_fake_dirs=None, external_dataset_dirs=None):
    raw_real_files = list(REAL_DIR.glob("*"))
    raw_fake_files = list(FAKE_DIR.glob("*"))
    external_real_files = _collect_external_images(external_real_dirs)
    external_fake_files = _collect_external_images(external_fake_dirs)
    dataset_real_files, dataset_fake_files = _discover_external_dataset_images(external_dataset_dirs)

    real_files = sorted({*raw_real_files, *external_real_files, *dataset_real_files})
    fake_files = sorted({*raw_fake_files, *external_fake_files, *dataset_fake_files})
    if not real_files or not fake_files:
        raise RuntimeError(
            "No real or fake images found. Populate data/raw/real and data/raw/fake, "
            "or provide additional directories with --external-real and --external-fake."
        )

    real_files, fake_files = _balance_dataset(real_files, fake_files)
    paths = [str(p) for p in real_files + fake_files]
    labels = [0] * len(real_files) + [1] * len(fake_files)
    return paths, np.array(labels, dtype=np.int32)


def run_preprocessing(train_paths, test_paths, train_y, test_y):
    print(f"Train: {len(train_paths)}, Test: {len(test_paths)}")
    train_data = preprocess_batch(train_paths, train_y.tolist())
    test_data = preprocess_batch(test_paths, test_y.tolist())

    save_processed_arrays(train_data, DATA_PROC / "train")
    save_processed_arrays(test_data, DATA_PROC / "test")
    return train_data, test_data


def train_cnn_models(train_data, test_data):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    regions = ["eyes", "nose", "chin", "ears"]
    key_map = {"eyes": "X_eyes", "nose": "X_nose", "chin": "X_chin", "ears": "X_ears"}
    metrics = {}

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.12,
        height_shift_range=0.12,
        zoom_range=0.2,
        shear_range=0.1,
        brightness_range=(0.7, 1.3),
        channel_shift_range=30.0,
        horizontal_flip=True,
        fill_mode="reflect",
    )

    for region in regions:
        print(f"\n=== Training CNN: {region} ===")
        X_key = key_map[region]
        X_train = train_data[X_key].astype(np.float32) / 255.0
        X_test = test_data[X_key].astype(np.float32) / 255.0
        y_train = train_data["y_labels"]
        y_test = test_data["y_labels"]

        class_weights = _make_class_weights(y_train)
        model = build_cnn((128, 128, 3))
        callbacks = [
            EarlyStopping(
                monitor="val_accuracy",
                patience=12,
                min_delta=1e-4,
                restore_best_weights=True,
                verbose=1,
                mode="max",
            ),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
        ]
        model.fit(
            datagen.flow(X_train, y_train, batch_size=BATCH_SIZE_CNN, seed=RANDOM_SEED),
            steps_per_epoch=max(1, len(X_train) // BATCH_SIZE_CNN),
            epochs=EPOCHS_CNN,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1,
            shuffle=True,
            class_weight=class_weights,
        )

        out_path = SAVED_DIR / f"cnn_{region}.keras"
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

    transform_train = T.Compose([
        T.ToPILImage(),
        T.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=8),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    transform_test = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    X_train = train_data["X_face"]
    X_test = test_data["X_face"]
    y_train = train_data["y_labels"]
    y_test = test_data["y_labels"]

    train_ds = FaceDataset(X_train, y_train, transform_train)
    test_ds = FaceDataset(X_test, y_test, transform_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_CVIT, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE_CVIT, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cvit(pretrained=True).to(device)
    class_weights = _make_class_weights(y_train)
    if class_weights is not None:
        weight = torch.tensor([class_weights[0], class_weights[1]], device=device)
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()

    # Train the new classification head first to preserve pretrained features.
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False

    head_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-5,
    )
    head_scheduler = torch.optim.lr_scheduler.StepLR(head_optimizer, step_size=4, gamma=0.5)
    head_epochs = min(8, EPOCHS_CVIT)

    print(f"Training CViT classification head for {head_epochs} epochs.")
    for epoch in range(head_epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()
            head_optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            head_optimizer.step()
            total_loss += loss.item()
        head_scheduler.step()
        print(f"Head epoch {epoch + 1}/{head_epochs} loss={total_loss / max(len(train_loader), 1):.4f}")

    if EPOCHS_CVIT > head_epochs:
        print("Fine-tuning full CViT model.")
        for param in model.parameters():
            param.requires_grad = True

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

        for epoch in range(head_epochs, EPOCHS_CVIT):
            model.train()
            total_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            print(f"Fine-tune epoch {epoch + 1}/{EPOCHS_CVIT} loss={total_loss / max(len(train_loader), 1):.4f}")

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
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    print("\n=== Training Posture MLP ===")
    X_train = train_data["X_posture"]
    X_test = test_data["X_posture"]
    y_train = train_data["y_labels"]
    y_test = test_data["y_labels"]

    if X_train.size == 0 or not np.any(np.abs(X_train)):
        print("Posture input is not informative; skipping posture model training.")
        return {}

    class_weights = _make_class_weights(y_train)
    model = build_posture_mlp((132,))
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=12,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
            mode="max",
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
    ]
    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS_POSTURE,
        batch_size=BATCH_SIZE_POSTURE,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weights,
    )

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


def parse_args():
    parser = argparse.ArgumentParser(description="Train deepfake detection models.")
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow fallback to synthetic dummy data if no real/fake raw images are available.",
    )
    parser.add_argument(
        "--external-real",
        nargs="+",
        default=[],
        help="One or more directories containing additional real face images.",
    )
    parser.add_argument(
        "--external-fake",
        nargs="+",
        default=[],
        help="One or more directories containing additional fake face images.",
    )
    parser.add_argument(
        "--external-dataset",
        nargs="+",
        default=[],
        help="One or more dataset roots containing both real/fake subfolders for automatic discovery.",
    )
    parser.add_argument(
        "--num-samples-per-class",
        type=int,
        default=NUM_PER_CLASS,
        help="Maximum number of Kaggle real/fake images to copy when downloading datasets.",
    )
    parser.add_argument(
        "--save-region-debug",
        action="store_true",
        help="Save sample aligned region crops for inspection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Deepfake Detection — Training Pipeline")
    print("=" * 60)

    download_dataset(
        allow_synthetic=args.allow_synthetic,
        external_real_dirs=args.external_real,
        external_fake_dirs=args.external_fake,
        external_dataset_dirs=args.external_dataset,
        num_per_class=args.num_samples_per_class,
    )
    paths, labels = build_file_lists(
        external_real_dirs=args.external_real,
        external_fake_dirs=args.external_fake,
        external_dataset_dirs=args.external_dataset,
    )
    real_count = int((labels == 0).sum())
    fake_count = int((labels == 1).sum())
    print(f"Total images: {len(paths)} (real={real_count}, fake={fake_count})")

    if args.save_region_debug:
        debug_dir = SAVED_DIR / "region_debug"
        print(f"Saving debug region crops to {debug_dir}")
        preprocess_batch(paths, labels.tolist(), verbose=False, save_debug_dir=debug_dir, max_debug=50)

    train_paths, test_paths, train_y, test_y = train_test_split(
        paths, labels, test_size=1 - TRAIN_RATIO, random_state=RANDOM_SEED, stratify=labels
    )
    train_data, test_data = run_preprocessing(train_paths, test_paths, train_y, test_y)

    all_metrics = {}
    all_metrics.update(train_cnn_models(train_data, test_data))
    all_metrics.update(train_cvit(train_data, test_data))
    all_metrics.update(train_posture_mlp(train_data, test_data))
    all_metrics.update(evaluate_video_ensemble(test_paths, test_y))

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
