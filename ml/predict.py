"""ML prediction helpers for Priority Care triage model."""

import os
import pickle
from typing import Any

import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "triage_model.pkl")

FEATURE_COLUMNS = ["age", "pain_level", "resp_rate", "heart_rate", "oxygen_sat"]

LABEL_MAP = {0: "GREEN", 1: "YELLOW", 2: "RED"}

ACTION_MAP = {
    "RED": "Immediate assessment — notify doctor now",
    "YELLOW": "Urgent — assess within 30 minutes",
    "GREEN": "Non-urgent — standard queue",
}

_model: Any = None


def load_model() -> Any | None:
    """Load trained scikit-learn model from disk."""
    global _model

    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        return None

    try:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        return _model
    except Exception as exc:
        print(f"[predict] Warning: could not load model — {exc}")
        return None


def _safe_float(value, default: float) -> float:
    """Convert input to float safely."""
    try:
        return float(value) if value not in (None, "", "null") else default
    except (ValueError, TypeError):
        return default


def preprocess(form_data: dict) -> np.ndarray:
    """Convert form data to numeric array in model feature order."""
    defaults = {
        "age": 35,
        "pain_level": 0,
        "resp_rate": 16,
        "heart_rate": 72,
        "oxygen_sat": 98.0,
    }

    row = []
    for col in FEATURE_COLUMNS:
        row.append(_safe_float(form_data.get(col), defaults[col]))

    return np.array([row])


def _extract_vitals(form_data: dict) -> dict:
    """Extract patient vitals safely from form data."""
    return {
        "age": _safe_float(form_data.get("age"), 35),
        "pain_level": _safe_float(form_data.get("pain_level"), 0),
        "resp_rate": _safe_float(form_data.get("resp_rate"), 16),
        "heart_rate": _safe_float(form_data.get("heart_rate"), 72),
        "oxygen_sat": _safe_float(form_data.get("oxygen_sat"), 98.0),
    }


def _unique_top_factors(factors: list[str]) -> list[str]:
    """Return max 3 unique top factors."""
    ordered = []

    for factor in factors + FEATURE_COLUMNS:
        if factor not in ordered:
            ordered.append(factor)

    return ordered[:3]


def _clinical_safety_override(form_data: dict) -> dict | None:
    """
    Hybrid safety layer.

    Critical cases are handled before ML prediction so unsafe outputs
    such as GREEN for low oxygen or severe pain are prevented.
    """
    vitals = _extract_vitals(form_data)

    age = vitals["age"]
    pain = vitals["pain_level"]
    resp = vitals["resp_rate"]
    heart = vitals["heart_rate"]
    spo2 = vitals["oxygen_sat"]

    red_factors = []

    if spo2 <= 90:
        red_factors.append("oxygen_sat")
    if pain >= 9:
        red_factors.append("pain_level")
    if resp >= 25:
        red_factors.append("resp_rate")
    if heart >= 130 or heart <= 45:
        red_factors.append("heart_rate")
    if age >= 75 and spo2 <= 94:
        red_factors.append("age")

    if red_factors:
        return {
            "label": "RED",
            "confidence": 0.95,
            "suggested_action": ACTION_MAP["RED"],
            "top_factors": _unique_top_factors(red_factors),
            "method": "safety_override",
        }

    yellow_factors = []

    if spo2 <= 94:
        yellow_factors.append("oxygen_sat")
    if pain >= 5:
        yellow_factors.append("pain_level")
    if resp >= 20:
        yellow_factors.append("resp_rate")
    if heart >= 110 or heart <= 55:
        yellow_factors.append("heart_rate")
    if age >= 75 and (pain >= 4 or spo2 <= 95):
        yellow_factors.append("age")

    if yellow_factors:
        return {
            "label": "YELLOW",
            "confidence": 0.88,
            "suggested_action": ACTION_MAP["YELLOW"],
            "top_factors": _unique_top_factors(yellow_factors),
            "method": "clinical_override",
        }

    return None


def _rule_based_predict(form_data: dict) -> dict:
    """Fallback rule-based triage when no ML model is available."""
    override = _clinical_safety_override(form_data)

    if override:
        override["method"] = "rule_based"
        return override

    return {
        "label": "GREEN",
        "confidence": 0.88,
        "suggested_action": ACTION_MAP["GREEN"],
        "top_factors": ["pain_level", "age", "heart_rate"],
        "method": "rule_based",
    }


def predict(form_data: dict) -> dict:
    """
    Run hybrid triage prediction.

    Flow:
    1. Apply clinical safety override.
    2. If no critical/urgent warning signs, use ML model.
    3. If model is missing or fails, use rule-based fallback.
    """
    override = _clinical_safety_override(form_data)

    if override:
        return override

    model = load_model()

    if model is None:
        return _rule_based_predict(form_data)

    try:
        features = preprocess(form_data)
        raw_label = model.predict(features)[0]

        if isinstance(raw_label, (int, np.integer)):
            label = LABEL_MAP.get(int(raw_label), "GREEN")
        else:
            label = str(raw_label).upper()
            if label not in ("RED", "YELLOW", "GREEN"):
                label = "GREEN"

        confidence = 0.80

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            confidence = round(float(max(proba)), 2)

        top_factors = FEATURE_COLUMNS[:3]

        return {
            "label": label,
            "confidence": confidence,
            "suggested_action": ACTION_MAP.get(label, "Assess patient"),
            "top_factors": top_factors,
            "method": "ml_model",
        }

    except Exception as exc:
        print(f"[predict] ML prediction error ({exc}); using rule-based fallback")
        return _rule_based_predict(form_data)