"""
Ensemble inference: majority vote across 6 models.
Works standalone: python ensemble.py <image_path>
"""

import json
from pathlib import Path

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
            batch = prepare_posture_batch(regions["posture"])
            label, conf, probs = predict_keras(model_entry, batch)
            return label, conf, int(np.argmax(probs))
    except Exception:
        return None
    return None


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

    model_predictions = {}
    votes_fake = 0
    votes_real = 0
    confidences = []

    order = ["eyes", "nose", "chin", "ears", "face", "posture"]
    for name in order:
        if name not in models:
            continue
        result = _predict_single(models[name], name, regions)
        if result is None:
            continue
        label, conf, _ = result
        model_predictions[name] = label
        confidences.append(conf)
        if label == "FAKE":
            votes_fake += 1
        else:
            votes_real += 1

    if not model_predictions:
        return _demo_prediction()

    if votes_fake >= votes_real:
        verdict = "FAKE"
        winning_votes = votes_fake
    else:
        verdict = "REAL"
        winning_votes = votes_real

    total = votes_fake + votes_real
    confidence = round((winning_votes / total) * 100, 1)
    if confidences:
        # blend vote share with mean model confidence for display
        confidence = round((confidence + np.mean(confidences)) / 2, 1)

    flagged = [k for k, v in model_predictions.items() if v == "FAKE"]

    return {
        "verdict": verdict,
        "confidence": confidence,
        "model_predictions": model_predictions,
        "flagged_regions": flagged,
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
