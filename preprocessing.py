"""
Face and region preprocessing for deepfake detection.
Works standalone: python preprocessing.py
"""

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    import mediapipe as mp

    _MEDIAPIPE_OK = True
except Exception:
    mp = None
    _MEDIAPIPE_OK = False

FACE_SIZE = (256, 256)
REGION_SIZE = (128, 128)
POSTURE_DIM = 132

EYE_LANDMARKS = [33, 133, 362, 263, 159, 145, 386, 374]
NOSE_LANDMARKS = [1, 2, 5, 98, 327, 168]
CHIN_LANDMARKS = [152, 148, 176, 149, 150, 136, 172]

_mtcnn = None
_mp_pose = None
_mp_face_mesh = None


def _get_mtcnn():
    global _mtcnn
    if _mtcnn is None:
        from mtcnn import MTCNN

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


def _get_face_mesh():
    global _mp_face_mesh
    if not _MEDIAPIPE_OK:
        return None
    if _mp_face_mesh is None:
        try:
            _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
            )
        except Exception:
            return None
    return _mp_face_mesh


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


def _transform_landmarks(landmarks, M):
    if landmarks is None:
        return None
    pts = np.array([[lm[0], lm[1], 1.0] for lm in landmarks], dtype=np.float32)
    transformed = pts.dot(M.T)
    return [(float(x), float(y), lm[2]) for (x, y), lm in zip(transformed.tolist(), landmarks)]


def _extract_face_mesh_landmarks(rgb):
    face_mesh = _get_face_mesh()
    if face_mesh is None:
        return None
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0].landmark
    return [(lm.x * rgb.shape[1], lm.y * rgb.shape[0], lm.z) for lm in landmarks]


def _align_face_by_eyes(face, landmarks):
    if not landmarks:
        return face, landmarks

    left = np.mean([np.array(landmarks[i][:2], dtype=np.float32) for i in [33, 133, 159, 145]], axis=0)
    right = np.mean([np.array(landmarks[i][:2], dtype=np.float32) for i in [263, 362, 386, 374]], axis=0)
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    if np.hypot(dx, dy) < 1.0:
        return face, landmarks

    angle = np.degrees(np.arctan2(dy, dx))
    center = tuple(((left + right) / 2).tolist())
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(face, M, (face.shape[1], face.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    aligned_landmarks = _transform_landmarks(landmarks, M)
    return aligned, aligned_landmarks


def _crop_from_landmarks(face, landmarks, indices, scale=2.0):
    if landmarks is None:
        return None
    pts = np.array([landmarks[i][:2] for i in indices], dtype=np.float32)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    if x1 <= x0 or y1 <= y0:
        return None
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    size = max(x1 - x0, y1 - y0) * scale
    x0 = int(max(0, cx - size / 2.0))
    y0 = int(max(0, cy - size / 2.0))
    x1 = int(min(face.shape[1], cx + size / 2.0))
    y1 = int(min(face.shape[0], cy + size / 2.0))
    if x1 <= x0 or y1 <= y0:
        return None
    return _resize_region(face[y0:y1, x0:x1])


def _crop_ears(face):
    fh, fw = face.shape[:2]
    y0 = int(fh * 0.18)
    y1 = int(fh * 0.55)
    left = face[y0:y1, : int(fw * 0.22)]
    right = face[y0:y1, int(fw * 0.78) :]
    if left.size == 0 and right.size == 0:
        return np.zeros((*REGION_SIZE, 3), dtype=np.uint8)
    if left.size and right.size:
        combined = np.hstack([_resize_region(left), _resize_region(right)])
        return _resize_region(combined)
    return _resize_region(left if left.size else right)


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
    Landmark-guided crops from the detected face.
    Returns dict with face, eyes, nose, chin, ears arrays.
    """
    face_resized = _resize_region(face_rgb, FACE_SIZE)
    landmarks = _extract_face_mesh_landmarks(face_resized)
    if landmarks is not None:
        aligned_face, aligned_landmarks = _align_face_by_eyes(face_resized, landmarks)
        face_full = _resize_region(aligned_face, FACE_SIZE)
        eyes = _crop_from_landmarks(face_full, aligned_landmarks, EYE_LANDMARKS, scale=2.0)
        nose = _crop_from_landmarks(face_full, aligned_landmarks, NOSE_LANDMARKS, scale=2.2)
        chin = _crop_from_landmarks(face_full, aligned_landmarks, CHIN_LANDMARKS, scale=2.4)
        ears = _crop_ears(face_full)
        if eyes is None:
            eyes = _region_from_face(face_full, 0.22, 0.48, 0.18, 0.82)
        if nose is None:
            nose = _region_from_face(face_full, 0.40, 0.62, 0.35, 0.65)
        if chin is None:
            chin = _region_from_face(face_full, 0.62, 0.95, 0.25, 0.75)
    else:
        face_full = _resize_region(face_resized, FACE_SIZE)
        eyes = _region_from_face(face_full, 0.22, 0.48, 0.18, 0.82)
        nose = _region_from_face(face_full, 0.40, 0.62, 0.35, 0.65)
        chin = _region_from_face(face_full, 0.62, 0.95, 0.25, 0.75)
        ears = _crop_ears(face_full)
    return {
        "face": face_full,
        "eyes": eyes,
        "nose": nose,
        "chin": chin,
        "ears": ears,
    }


def extract_posture_keypoints(rgb):
    """MediaPipe FaceMesh → 33 landmarks × 4 = 132 floats."""
    zeros = np.zeros(POSTURE_DIM, dtype=np.float32)
    try:
        face_mesh = _get_face_mesh()
        if face_mesh is None:
            return zeros
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return zeros
        landmarks = results.multi_face_landmarks[0].landmark
        selected_indices = np.linspace(0, len(landmarks) - 1, num=33, dtype=int)
        vals = []
        for idx in selected_indices:
            lm = landmarks[idx]
            vals.extend([lm.x, lm.y, lm.z, 1.0])
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


def preprocess_batch(image_paths, labels=None, verbose=True, save_debug_dir=None, max_debug=50):
    """Process list of paths into numpy arrays."""
    from tqdm import tqdm

    paths = list(image_paths)
    iterator = tqdm(paths, desc="Preprocessing") if verbose else paths

    faces, eyes, noses, chins, ears_list, postures = [], [], [], [], [], []
    y = []

    debug_dir = Path(save_debug_dir) if save_debug_dir else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

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

            if debug_dir is not None and i < max_debug:
                for name in ["face", "eyes", "nose", "chin", "ears"]:
                    out_path = debug_dir / f"{i:03d}_{name}.png"
                    Image.fromarray(regions[name]).save(out_path)
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
