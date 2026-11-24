"""Robustness and geometric feature invariance tests for the upgraded model."""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from landmarks import extract_features, extract_features_batch
from model import SignLanguageModel

MODEL_PKL = os.path.join(ROOT, "models", "sign_language_model.pkl")
DATASET_CSV = os.path.join(ROOT, "data", "processed", "asl_alphabet_landmarks.csv")

needs_artifacts = pytest.mark.skipif(
    not (os.path.exists(MODEL_PKL) and os.path.exists(DATASET_CSV)),
    reason="trained model or processed CSV not present",
)


def _rot2d(X, deg):
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    Xr = X.reshape(-1, 21, 3).copy()
    x, y = Xr[:, :, 0].copy(), Xr[:, :, 1].copy()
    Xr[:, :, 0] = x * c - y * s
    Xr[:, :, 1] = x * s + y * c
    return Xr.reshape(-1, 63)


def _pitch3d(X, deg):
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    Xr = X.reshape(-1, 21, 3).copy()
    y, z = Xr[:, :, 1].copy(), Xr[:, :, 2].copy()
    Xr[:, :, 1] = y * c - z * s
    Xr[:, :, 2] = y * s + z * c
    sc = np.linalg.norm(Xr[:, 9, :], axis=1, keepdims=True)
    return (Xr / np.maximum(sc[:, None, :], 1e-8)).reshape(-1, 63)


def _mirror_x(X):
    Xm = X.reshape(-1, 21, 3).copy()
    Xm[:, :, 0] *= -1
    return Xm.reshape(-1, 63)


def test_extract_features_handles_degenerate_input():
    zeros = np.zeros(63, dtype=float)
    feats = extract_features(zeros)
    assert feats.ndim == 1
    assert feats.shape[0] == 210
    assert np.all(np.isfinite(feats))


@needs_artifacts
def test_extract_features_is_mirror_invariant():
    df = pd.read_csv(DATASET_CSV)
    X = df[[f"feature_{i}" for i in range(63)]].values[:30]
    f_orig = extract_features_batch(X)
    f_mir = extract_features_batch(_mirror_x(X))
    assert np.allclose(f_orig, f_mir, atol=1e-12)


@needs_artifacts
def test_model_accuracy_under_domain_shifts():
    model = SignLanguageModel()
    model.load_model()
    model.load_metadata()

    df = pd.read_csv(DATASET_CSV)
    X = df[[f"feature_{i}" for i in range(63)]].values
    y = np.array([model.label_encoder[lbl] for lbl in df["label"].values])
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    def _acc(X_in):
        feats = extract_features_batch(X_in)
        scaled = model.scaler.transform(feats)
        preds = model.model.predict(scaled)
        return accuracy_score(y_test, preds)

    assert _acc(X_test) >= 0.98
    assert _acc(_mirror_x(X_test)) >= 0.98
    assert _acc(_rot2d(X_test, 18)) >= 0.96
    assert _acc(_rot2d(X_test, -18)) >= 0.96
    assert _acc(_pitch3d(X_test, 20)) >= 0.96


@needs_artifacts
def test_synthetic_line_stays_low_confidence():
    model = SignLanguageModel()
    model.load_model()
    model.load_metadata()
    pts = np.array([[0.5 + i * 0.01, 0.5, 0.0] for i in range(21)])
    pts = (pts - pts[0:1]) / np.linalg.norm(pts[9] - pts[0])
    _, conf = model.predict(pts.reshape(63))
    assert conf < 0.6
