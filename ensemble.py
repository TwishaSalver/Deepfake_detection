"""
Ensemble inference: majority vote across 6 models.
Works standalone: python ensemble.py <image_path>
"""

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from models import (
    LABEL_FAKE,
    LABEL_REAL,
    cnn_paths,
    cvit_path,
    load_cvit,
    load_keras_model,
    models_available,
    posture_path,
    predict_cvit,
    predict_keras,
    prepare_cnn_batch,
    prepare_posture_batch,
)
SAVED_DIR = Path(__file__).resolve().parent / "saved_models"
METRICS_PATH = SAVED_DIR / "metrics.json"
MIN_MODEL_ACCURACY = 0.52  # Ignore region models that are effectively chance-level
MIN_MODEL_AUC = 0.52
FACE_WEIGHT_MULTIPLIER = 1.5
POSTURE_WEIGHT_MIN = 0.55


def _load_metrics():
    if not METRICS_PATH.exists():
        return {}
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

_METRICS = _load_metrics()


def _model_vote_weight(name):
    model_meta = _METRICS.get(name, {})
    acc = model_meta.get("accuracy")
    auc = model_meta.get("auc")

    if acc is None:
        return 0.0

    acc = float(acc)
    score = acc
    if auc is not None:
        try:
            auc = float(auc)
            score = max(score, auc) * 0.9 + min(score, auc) * 0.1
        except Exception:
            pass

    if name == "face":
        return min(1.0, max(0.75, score * FACE_WEIGHT_MULTIPLIER))

    if name == "posture":
        if score < POSTURE_WEIGHT_MIN:
            return 0.0
        return min(1.0, score + 0.05)

    if score < MIN_MODEL_ACCURACY and (auc is None or float(auc) < MIN_MODEL_AUC):
        return 0.0

    return score


def _verdict_from_label(idx):
    return "FAKE" if idx == LABEL_FAKE else "REAL"


def _demo_prediction():
    """Deterministic demo output when no models are trained."""
    return {
        "verdict": "FAKE",
        "confidence": 66.7,
        "model_predictions": {
            "eyes": "FAKE",
            "nose": "REAL",
            "chin": "FAKE",
            "ears": "FAKE",
            "face": "FAKE",
            "posture": "FAKE",
        },
        "demo_mode": True,
        "message": "Demo mode — train models with python train.py for real inference.",
    }


def _posture_metrics_good():
    metrics_file = SAVED_DIR / "metrics.json"
    if not metrics_file.exists():
        return True
    try:
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        acc = metrics.get("posture", {}).get("accuracy")
        return acc is None or float(acc) > 0.55
    except Exception:
        return True


def _load_all_models():
    """Load available models; failed loads are skipped."""
    loaded = {}

    for region, path in cnn_paths().items():
        if not path.exists():
            continue
        try:
            loaded[region] = load_keras_model(str(path))
        except Exception:
            pass

    if cvit_path().exists():
        try:
            model, device = load_cvit(str(cvit_path()))
            loaded["face"] = (model, device)
        except Exception:
            pass

    if posture_path().exists():
        try:
            if _posture_metrics_good():
                loaded["posture"] = load_keras_model(str(posture_path()))
        except Exception:
            pass

    return loaded


def _predict_single(model_entry, region_name, regions):
    """Return (label_str, confidence, raw_class_idx) or None."""
    try:
        if region_name in ("eyes", "nose", "chin", "ears"):
            batch = prepare_cnn_batch(regions[region_name])
            label, conf, probs = predict_keras(model_entry, batch)
            return label, conf, int(np.argmax(probs))

        if region_name == "face":
            model, device = model_entry
            label, conf, probs = predict_cvit(model, device, regions["face"])
            return label, conf, int(np.argmax(probs))

        if region_name == "posture":
            if not np.any(regions["posture"]):
                return None
            batch = prepare_posture_batch(regions["posture"])
            label, conf, probs = predict_keras(model_entry, batch)
            return label, conf, int(np.argmax(probs))
    except Exception:
        return None
    return None


def _aggregate_model_predictions(frame_results):
    votes = defaultdict(lambda: {"REAL": 0, "FAKE": 0})
    for frame in frame_results:
        for name, label in frame.get("model_predictions", {}).items():
            votes[name][label] += 1

    aggregated = {}
    for name, counts in votes.items():
        aggregated[name] = "FAKE" if counts["FAKE"] >= counts["REAL"] else "REAL"
    return aggregated


def _predict_regions(models, regions):
    model_predictions = {}
    weighted_score_fake = 0.0
    weighted_score_real = 0.0
    confidences = []

    order = ["face", "eyes", "nose", "chin", "ears", "posture"]
    for name in order:
        if name not in models:
            continue
        weight = _model_vote_weight(name)
        if weight <= 0:
            continue

        result = _predict_single(models[name], name, regions)
        if result is None:
            continue

        label, conf, _ = result
        model_predictions[name] = label
        confidences.append(conf)

        score = weight * (0.75 * (conf / 100.0) + 0.25)
        if label == "FAKE":
            weighted_score_fake += score
        else:
            weighted_score_real += score

    if not model_predictions:
        return None

    if weighted_score_fake >= weighted_score_real:
        verdict = "FAKE"
    else:
        verdict = "REAL"

    total_score = weighted_score_fake + weighted_score_real
    confidence = round((max(weighted_score_fake, weighted_score_real) / max(total_score, 1e-6)) * 100, 1)
    if confidences:
        confidence = round((confidence + np.mean(confidences)) / 2, 1)

    flagged = [k for k, v in model_predictions.items() if v == "FAKE"]
    return {
        "verdict": verdict,
        "confidence": confidence,
        "model_predictions": model_predictions,
        "flagged_regions": flagged,
        "demo_mode": False,
    }


def predict_deepfake(image_path, demo_mode=False):
    """
    Run full ensemble on an image path or array.

    Returns dict with verdict, confidence, model_predictions.
    """
    if demo_mode:
        return _demo_prediction()

    if not any(models_available().values()):
        return _demo_prediction()

    try:
        from preprocessing import preprocess_image

        regions = preprocess_image(image_path)
    except Exception as e:
        return {
            "verdict": "UNKNOWN",
            "confidence": 0.0,
            "model_predictions": {},
            "error": f"Preprocessing failed: {e}",
        }

    models = _load_all_models()
    if not models:
        return _demo_prediction()

    prediction = _predict_regions(models, regions)
    if prediction is None:
        return _demo_prediction()
    return prediction


def predict_deepfake_video(video_path, demo_mode=False, max_frames=8):
    """
    Run ensemble on a video by sampling frames and aggregating per-frame predictions.
    """
    if demo_mode:
        return _demo_prediction()

    if not any(models_available().values()):
        return _demo_prediction()

    models = _load_all_models()
    if not models:
        return _demo_prediction()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "verdict": "UNKNOWN",
            "confidence": 0.0,
            "model_predictions": {},
            "error": "Unable to open video file.",
        }

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, total_frames // max_frames) if total_frames else 1

        frame_results = []
        frame_index = 0
        sampled = 0
        while sampled < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    from preprocessing import preprocess_image

                    regions = preprocess_image(rgb)
                    prediction = _predict_regions(models, regions)
                    if prediction is not None:
                        prediction["frame_index"] = frame_index
                        frame_results.append(prediction)
                except Exception:
                    pass
                sampled += 1
            frame_index += 1
    finally:
        cap.release()

    if not frame_results:
        return {
            "verdict": "UNKNOWN",
            "confidence": 0.0,
            "model_predictions": {},
            "error": "No valid frames could be processed from the video.",
        }

    aggregated = _aggregate_model_predictions(frame_results)
    fake_votes = sum(1 for v in aggregated.values() if v == "FAKE")
    real_votes = sum(1 for v in aggregated.values() if v == "REAL")
    verdict = "FAKE" if fake_votes >= real_votes else "REAL"
    confidence = round(float(np.mean([f["confidence"] for f in frame_results])), 1)
    flagged = sorted({region for f in frame_results for region in f.get("flagged_regions", [])})

    return {
        "verdict": verdict,
        "confidence": confidence,
        "model_predictions": aggregated,
        "flagged_regions": flagged,
        "processed_frames": len(frame_results),
        "frame_results": frame_results,
        "demo_mode": False,
    }


def predict_deepfake_demo():
    return _demo_prediction()


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        out = predict_deepfake(path)
    else:
        out = predict_deepfake_demo()
    print(json.dumps(out, indent=2))
