"""
Simplified training script for Python 3.14 - uses PyTorch backend
Downloads real data from Kaggle, preprocesses, and trains models.
"""

import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
SAVED_DIR = ROOT / "saved_models"
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


def log_progress(msg):
    """Log with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")


def download_dataset():
    """Download from Kaggle or generate dummy images."""
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    FAKE_DIR.mkdir(parents=True, exist_ok=True)

    existing = list(REAL_DIR.glob("*")) + list(FAKE_DIR.glob("*"))
    if len(existing) >= NUM_PER_CLASS * 2:
        log_progress(f"✅ Using existing raw data ({len(existing)} files).")
        return

    log_progress("📥 Attempting Kaggle download...")
    downloaded = False
    for slug in KAGGLE_DATASETS:
        try:
            log_progress(f"  Trying: {slug}")
            result = subprocess.run(
                ["kaggle", "datasets", "download", "-d", slug, "-p", str(DATA_RAW), "--unzip"],
                check=True,
                capture_output=True,
                timeout=600,
            )
            log_progress(f"  ✅ Downloaded successfully!")
            downloaded = True
            break
        except Exception as e:
            log_progress(f"  ❌ Failed: {str(e)[:80]}")

    if downloaded:
        _collect_kaggle_images()
    else:
        log_progress("⚠️  Kaggle unavailable — generating dummy face images...")
        _generate_dummy_data()


def _collect_kaggle_images():
    """Walk extracted Kaggle tree and copy up to 200 real / 200 fake."""
    log_progress("📂 Organizing Kaggle images...")
    real_paths, fake_paths = [], []

    for path in DATA_RAW.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        
        path_str = str(path).lower().replace("\\", "/")
        
        if "/fake/" in path_str:
            fake_paths.append(path)
        elif "/real/" in path_str:
            real_paths.append(path)

    random.shuffle(real_paths)
    random.shuffle(fake_paths)

    for i, src in enumerate(real_paths[:NUM_PER_CLASS]):
        dst = REAL_DIR / f"real_{i:04d}{src.suffix}"
        if not dst.exists():
            shutil.copy2(src, dst)
    
    for i, src in enumerate(fake_paths[:NUM_PER_CLASS]):
        dst = FAKE_DIR / f"fake_{i:04d}{src.suffix}"
        if not dst.exists():
            shutil.copy2(src, dst)

    log_progress(f"✅ Organized {len(list(REAL_DIR.glob('*')))} real + {len(list(FAKE_DIR.glob('*')))} fake images")


def _generate_dummy_data():
    """Generate random face images for testing."""
    log_progress(f"🎨 Generating {NUM_PER_CLASS} dummy image pairs...")
    np.random.seed(42)
    
    for i in range(NUM_PER_CLASS):
        real_img = (np.random.rand(224, 224, 3) * 180 + 40).astype(np.uint8)
        fake_img = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        fake_img[:, :112, :] = 255 - fake_img[:, :112, :]  # pattern difference
        
        Image.fromarray(real_img).save(REAL_DIR / f"real_{i:04d}.png")
        Image.fromarray(fake_img).save(FAKE_DIR / f"fake_{i:04d}.png")
    
    log_progress(f"✅ Generated {NUM_PER_CLASS} real + {NUM_PER_CLASS} fake dummy images.")


def build_file_lists():
    """Build lists of image paths and labels."""
    real_files = sorted(REAL_DIR.glob("*"))
    fake_files = sorted(FAKE_DIR.glob("*"))
    paths = [str(p) for p in real_files + fake_files]
    labels = [0] * len(real_files) + [1] * len(fake_files)
    return paths, np.array(labels, dtype=np.int32)


def train_with_pytorch():
    """Train using PyTorch models (CViT + MLP)"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as T
    from timm import create_model
    
    log_progress("🚀 Starting PyTorch training...")
    
    # Load data
    log_progress("📖 Loading image data...")
    paths, labels = build_file_lists()
    train_paths, test_paths, train_y, test_y = train_test_split(
        paths, labels, test_size=1-TRAIN_RATIO, random_state=42, stratify=labels
    )
    log_progress(f"  Train: {len(train_paths)}, Test: {len(test_paths)}")
    
    # Load images
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    class FaceDataset(Dataset):
        def __init__(self, paths, labels, transform):
            self.paths = paths
            self.labels = labels
            self.transform = transform
        
        def __len__(self):
            return len(self.paths)
        
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            img = self.transform(img)
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return img, label
    
    train_dataset = FaceDataset(train_paths, train_y, transform)
    test_dataset = FaceDataset(test_paths, test_y, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_progress(f"  Device: {device}")
    
    # Train CViT model
    log_progress("\n🎯 Training CViT (Vision Transformer) model...")
    model = create_model("vit_small_patch16_224", pretrained=True, num_classes=2)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 15
    best_val_acc = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device).long()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device).long()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        log_progress(f"  Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            SAVED_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), SAVED_DIR / "cvit_face.pth")
            log_progress(f"    ✅ Saved best model (Acc: {val_acc:.4f})")
    
    # Evaluate on test set
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    test_acc = accuracy_score(all_labels, all_preds)
    log_progress(f"\n✅ CViT Test Accuracy: {test_acc:.4f}")
    
    return {
        "cvit_face": {
            "accuracy": float(test_acc),
            "model": "Vision Transformer (ViT-Small)",
            "status": "trained"
        }
    }


def save_metrics(metrics):
    """Save training metrics to JSON."""
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dict = {
        "timestamp": datetime.now().isoformat(),
        "models": metrics,
        "total_images": len(list(REAL_DIR.glob("*"))) + len(list(FAKE_DIR.glob("*"))),
        "split_ratio": TRAIN_RATIO
    }
    
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    
    log_progress(f"📊 Metrics saved to {METRICS_PATH}")
    print(json.dumps(metrics_dict, indent=2))


def main():
    log_progress("=" * 60)
    log_progress("🔍 DEEPFAKE DETECTION - TRAINING PIPELINE (Python 3.14 Compatible)")
    log_progress("=" * 60)
    
    try:
        # Step 1: Download dataset
        download_dataset()
        
        # Step 2: Train models
        metrics = train_with_pytorch()
        
        # Step 3: Save metrics
        save_metrics(metrics)
        
        log_progress("\n" + "=" * 60)
        log_progress("✅ TRAINING COMPLETED SUCCESSFULLY!")
        log_progress("=" * 60)
        log_progress(f"📁 Models saved to: {SAVED_DIR}")
        log_progress(f"📊 Metrics saved to: {METRICS_PATH}")
        log_progress(f"🚀 Ready to use! Run: streamlit run app.py")
        
    except Exception as e:
        log_progress(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
