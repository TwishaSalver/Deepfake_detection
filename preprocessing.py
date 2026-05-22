"""
Face and region preprocessing for deepfake detection.
Works standalone: python preprocessing.py
"""

import os
from pathlib import Path

import cv2
import numpy as np
from mtcnn import MTCNN
from PIL import Image

try:
    import mediapipe as mp

    _MEDIAPIPE_OK = True
except Exception:
    mp = None
    _MEDIAPIPE_OK = False

FACE_SIZE = (224, 224)
REGION_SIZE = (50, 50)
POSTURE_DIM = 132

_mtcnn = None
_mp_pose = None


def _get_mtcnn():
    global _mtcnn
    if _mtcnn is None:
        _mtcnn = MTCNN()
    return _mtcnn


def _get_pose():
    global _mp_pose
    if not _MEDIAPIPE_OK:
        return None
    if _mp_pose is None:
        _mp_pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
        )
    return _mp_pose


def _to_rgb_array(image_input):
    """Load path, PIL Image, or ndarray as RGB uint8 array."""
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
        return np.array(img)
    if isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"))
    arr = np.asarray(image_input)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    return arr


def detect_face_box(rgb):
    """Return (x, y, w, h) face box or None if MTCNN fails."""
    try:
        detector = _get_mtcnn()
        results = detector.detect_faces(rgb)
        if not results:
            return None
        best = max(results, key=lambda r: r.get("confidence", 0))
        if best.get("confidence", 0) < 0.9:
            # still use best box if any detection
            if best.get("confidence", 0) < 0.5:
                return None
        box = best["box"]
        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x, y = max(0, x), max(0, y)
        return x, y, w, h
    except Exception:
        return None


def crop_face(rgb, box=None):
    """Crop face region; fallback to full image."""
    h, w = rgb.shape[:2]
    if box is not None:
        x, y, bw, bh = box
        x2, y2 = min(w, x + bw), min(h, y + bh)
        face = rgb[y:y2, x:x2]
        if face.size > 0:
            return face
    return rgb


def _resize_region(region, size=REGION_SIZE):
    if region.size == 0:
        return np.zeros((*size, 3), dtype=np.uint8)
    return cv2.resize(region, size, interpolation=cv2.INTER_AREA)


def _region_from_face(face, y0, y1, x0, x1):
    fh, fw = face.shape[:2]
    y0p, y1p = int(fh * y0), int(fh * y1)
    x0p, x1p = int(fw * x0), int(fw * x1)
    y0p, y1p = max(0, y0p), min(fh, y1p)
    x0p, x1p = max(0, x0p), min(fw, x1p)
    if y1p <= y0p or x1p <= x0p:
        return np.zeros((*REGION_SIZE, 3), dtype=np.uint8)
    return _resize_region(face[y0p:y1p, x0p:x1p])


def extract_facial_regions(face_rgb):
    """
    Heuristic crops from aligned face (proportions on frontal face).
    Returns dict with face, eyes, nose, chin, ears arrays.
    """
    face_full = _resize_region(face_rgb, FACE_SIZE)
    eyes = _region_from_face(face_rgb, 0.22, 0.48, 0.18, 0.82)
    nose = _region_from_face(face_rgb, 0.40, 0.62, 0.35, 0.65)
    chin = _region_from_face(face_rgb, 0.62, 0.95, 0.25, 0.75)
    # ears: combine left + right strips
    left_ear = _region_from_face(face_rgb, 0.25, 0.55, 0.0, 0.22)
    right_ear = _region_from_face(face_rgb, 0.25, 0.55, 0.78, 1.0)
    ears = cv2.resize(
        np.hstack([left_ear, right_ear]) if left_ear.size and right_ear.size else left_ear,
        REGION_SIZE,
        interpolation=cv2.INTER_AREA,
    )
    return {
        "face": face_full,
        "eyes": eyes,
        "nose": nose,
        "chin": chin,
        "ears": ears,
    }


def extract_posture_keypoints(rgb):
    """MediaPipe Pose → 33 landmarks × 4 = 132 floats."""
    zeros = np.zeros(POSTURE_DIM, dtype=np.float32)
    try:
        pose = _get_pose()
        if pose is None:
            return zeros
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        results = pose.process(bgr)
        if not results.pose_landmarks:
            return zeros
        vals = []
        for lm in results.pose_landmarks.landmark:
            vals.extend([lm.x, lm.y, lm.z, lm.visibility])
        arr = np.array(vals[:POSTURE_DIM], dtype=np.float32)
        if arr.shape[0] < POSTURE_DIM:
            arr = np.pad(arr, (0, POSTURE_DIM - arr.shape[0]))
        return arr
    except Exception:
        return zeros


def preprocess_image(image_input):
    """
    Full pipeline for one image.
    Returns dict: face, eyes, nose, chin, ears (uint8 arrays), posture (float32 132,).
    """
    rgb = _to_rgb_array(image_input)
    box = detect_face_box(rgb)
    face_crop = crop_face(rgb, box)
    regions = extract_facial_regions(face_crop)
    regions["posture"] = extract_posture_keypoints(rgb)
    return regions


def preprocess_batch(image_paths, labels=None, verbose=True):
    """Process list of paths into numpy arrays."""
    from tqdm import tqdm

    paths = list(image_paths)
    iterator = tqdm(paths, desc="Preprocessing") if verbose else paths

    faces, eyes, noses, chins, ears_list, postures = [], [], [], [], [], []
    y = []

    for i, path in enumerate(iterator):
        try:
            regions = preprocess_image(path)
            faces.append(regions["face"])
            eyes.append(regions["eyes"])
            noses.append(regions["nose"])
            chins.append(regions["chin"])
            ears_list.append(regions["ears"])
            postures.append(regions["posture"])
            if labels is not None:
                y.append(labels[i])
        except Exception:
            continue

    out = {
        "X_face": np.array(faces, dtype=np.uint8),
        "X_eyes": np.array(eyes, dtype=np.uint8),
        "X_nose": np.array(noses, dtype=np.uint8),
        "X_chin": np.array(chins, dtype=np.uint8),
        "X_ears": np.array(ears_list, dtype=np.uint8),
        "X_posture": np.array(postures, dtype=np.float32),
    }
    if labels is not None and y:
        out["y_labels"] = np.array(y, dtype=np.int32)
    return out


def save_processed_arrays(data, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, arr in data.items():
        np.save(output_dir / f"{key}.npy", arr)


def load_processed_arrays(input_dir):
    input_dir = Path(input_dir)
    keys = ["X_face", "X_eyes", "X_nose", "X_chin", "X_ears", "X_posture", "y_labels"]
    data = {}
    for key in keys:
        path = input_dir / f"{key}.npy"
        if path.exists():
            data[key] = np.load(path)
    return data


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <image_path>")
        sys.exit(1)
    result = preprocess_image(sys.argv[1])
    print("Keys:", list(result.keys()))
    print("Face shape:", result["face"].shape)
    print("Posture shape:", result["posture"].shape)
