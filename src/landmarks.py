"""
Landmark geometry: the normalization that turns raw hand landmarks into the
63 features the classifier was trained on.

This module deliberately imports nothing beyond numpy. MediaPipe is a very
heavy dependency (it pulls in jax, jaxlib, OpenCV and matplotlib, around 1 GB
unzipped on Linux), and the deployed web API does not need it because the
browser now performs landmark detection. Keeping the normalization here lets
the server stay the single source of truth for feature construction without
dragging MediaPipe into the serverless bundle.

`HandTracker.normalize_landmarks` delegates to `normalize_landmarks` below, so
the desktop and web paths run the exact same code.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config

NUM_LANDMARKS = config.NUM_LANDMARKS
LANDMARK_DIMENSIONS = config.LANDMARK_DIMENSIONS
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9

TOTAL_FEATURES = NUM_LANDMARKS * LANDMARK_DIMENSIONS

_JOINT_ANGLE_TRIPLES = (
    (0, 1, 2), (1, 2, 3), (2, 3, 4), (1, 2, 4),       # Thumb
    (0, 5, 6), (5, 6, 7), (6, 7, 8), (0, 5, 8),       # Index
    (0, 9, 10), (9, 10, 11), (10, 11, 12), (0, 9, 12),  # Middle
    (0, 13, 14), (13, 14, 15), (14, 15, 16), (0, 13, 16),  # Ring
    (0, 17, 18), (17, 18, 19), (18, 19, 20), (0, 17, 20),  # Pinky
)

_SPREAD_PAIRS = (
    ((2, 4), (5, 8)), ((5, 8), (9, 12)), ((9, 12), (13, 16)), ((13, 16), (17, 20)),
    ((5, 6), (9, 10)), ((9, 10), (13, 14)), ((13, 14), (17, 18)),
)

_DISTANCE_PAIRS = (
    # Fingertips to wrist (extension)
    (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),
    # Fingertips to own MCP (curl tightness)
    (2, 4), (5, 8), (9, 12), (13, 16), (17, 20),
    # All pairwise fingertip distances
    (4, 8), (4, 12), (4, 16), (4, 20),
    (8, 12), (8, 16), (8, 20), (12, 16), (12, 20), (16, 20),
    # Thumb tip (4) and IP (3) to finger MCP/PIP/DIP joints (M/N/T/S/A/E/K)
    (4, 5), (4, 6), (4, 7), (4, 9), (4, 10), (4, 11),
    (4, 13), (4, 14), (4, 15), (4, 17), (4, 18), (4, 19),
    (3, 5), (3, 6), (3, 9), (3, 10), (3, 13), (3, 14), (3, 17), (3, 18),
    # Inter-finger PIP/DIP spacing (R vs U vs V)
    (6, 10), (10, 14), (14, 18), (7, 11),
)


def _unit_vectors(vecs, eps=1e-8):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, eps)


def extract_features_batch(normalized_landmarks):
    """
    Build geometric, palm-frame, and coarse-orientation features from a batch
    of wrist-centered 63-value landmark vectors.

    Features are invariant to translation, hand scale, and left/right mirroring
    (x -> -x), and robust to moderate 2D wrist tilt and 3D pitch/yaw, while
    preserving up/sideways/down orientation needed by letters like D/G/Q, U/H, K/P.

    Args:
        normalized_landmarks: array-like of shape (N, 63)

    Returns:
        np.ndarray of shape (N, num_features)
    """
    arr = np.asarray(normalized_landmarks, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    P = arr.reshape(-1, NUM_LANDMARKS, LANDMARK_DIMENSIONS)
    eps = 1e-8

    # Re-normalize by mean wrist-to-MCP distance (4x less sensitive to single-joint
    # depth noise or foreshortening than dividing solely by wrist->middle_MCP).
    mcps = P[:, [5, 9, 13, 17], :]
    palm_scale = np.linalg.norm(mcps - P[:, 0:1, :], axis=2).mean(axis=1, keepdims=True)
    Pn = (P - P[:, 0:1, :]) / np.maximum(palm_scale[:, :, None], eps)

    # 3D canonical palm basis: u_y along palm, u_x from pinky->index, u_z palm normal.
    # Orient u_z by camera-z sign so horizontal mirroring preserves chirality.
    uy = _unit_vectors(Pn[:, 9, :], eps)
    across = Pn[:, 5, :] - Pn[:, 17, :]
    across = across - np.sum(across * uy, axis=1, keepdims=True) * uy
    ux = _unit_vectors(across, eps)
    uz = _unit_vectors(np.cross(ux, uy), eps)
    uz = uz * np.where(uz[:, 2:3] < 0, -1.0, 1.0)

    P_nonwrist = Pn[:, 1:, :]
    lx = np.einsum("nkd,nd->nk", P_nonwrist, ux)
    ly = np.einsum("nkd,nd->nk", P_nonwrist, uy)
    lz = np.einsum("nkd,nd->nk", P_nonwrist, uz)
    local_coords = np.stack([lx, ly, lz], axis=2).reshape(-1, 60)

    # 2D canonical palm coordinates (immune to monocular z estimation drift)
    uy2 = _unit_vectors(Pn[:, 9, :2], eps)
    across2 = Pn[:, 5, :2] - Pn[:, 17, :2]
    across2 = across2 - np.sum(across2 * uy2, axis=1, keepdims=True) * uy2
    ux2 = _unit_vectors(across2, eps)
    lx2 = np.einsum("nkd,nd->nk", P_nonwrist[:, :, :2], ux2)
    ly2 = np.einsum("nkd,nd->nk", P_nonwrist[:, :, :2], uy2)
    local_2d = np.stack([lx2, ly2], axis=2).reshape(-1, 40)

    angles = np.stack([
        np.sum(_unit_vectors(Pn[:, b, :] - Pn[:, a, :], eps) *
               _unit_vectors(Pn[:, c, :] - Pn[:, b, :], eps), axis=1)
        for a, b, c in _JOINT_ANGLE_TRIPLES
    ], axis=1)

    spreads = np.stack([
        np.sum(_unit_vectors(Pn[:, b1, :] - Pn[:, a1, :], eps) *
               _unit_vectors(Pn[:, b2, :] - Pn[:, a2, :], eps), axis=1)
        for (a1, b1), (a2, b2) in _SPREAD_PAIRS
    ], axis=1)

    dists = np.stack([
        np.linalg.norm(Pn[:, a, :] - Pn[:, b, :], axis=1)
        for a, b in _DISTANCE_PAIRS
    ], axis=1)

    # Signed local offsets separating R vs U and fist-thumb letters (A/E/M/N/S/T)
    signed_diffs = np.stack([
        lx[:, 8 - 1] - lx[:, 12 - 1],
        lx[:, 7 - 1] - lx[:, 11 - 1],
        lx[:, 4 - 1] - lx[:, 5 - 1],
        lx[:, 4 - 1] - lx[:, 9 - 1],
        lx[:, 4 - 1] - lx[:, 13 - 1],
        lx[:, 4 - 1] - lx[:, 17 - 1],
        lx[:, 3 - 1] - lx[:, 9 - 1],
        lx[:, 3 - 1] - lx[:, 13 - 1],
        ly[:, 4 - 1] - ly[:, 6 - 1],
        ly[:, 4 - 1] - ly[:, 10 - 1],
        ly[:, 4 - 1] - ly[:, 14 - 1],
        ly[:, 4 - 1] - ly[:, 18 - 1],
        lz[:, 4 - 1] - lz[:, 6 - 1],
        lz[:, 4 - 1] - lz[:, 10 - 1],
        lz[:, 4 - 1] - lz[:, 14 - 1],
        lz[:, 14 - 1] - lz[:, 10 - 1],
        lz[:, 15 - 1] - lz[:, 11 - 1],
        lz[:, 8 - 1] - lz[:, 12 - 1],
    ], axis=1)

    # Mirror-symmetric coarse global orientation: (|dx|, dy, dz)
    orient_vecs = (
        Pn[:, 9, :],
        Pn[:, 5, :],
        Pn[:, 5, :] - Pn[:, 17, :],
        Pn[:, 4, :] - Pn[:, 2, :],
        Pn[:, 8, :] - Pn[:, 5, :],
        Pn[:, 12, :] - Pn[:, 9, :],
        Pn[:, 20, :] - Pn[:, 17, :],
    )
    orient_feats = []
    for vec in orient_vecs:
        vn = _unit_vectors(vec, eps)
        orient_feats.extend([np.abs(vn[:, 0]), vn[:, 1], vn[:, 2]])
    orient_feats = np.stack(orient_feats, axis=1)

    return np.hstack([
        local_coords, local_2d, angles, spreads, dists, signed_diffs, orient_feats
    ])


def extract_features(normalized_landmarks):
    """Extract the 1D engineered feature vector from a single 63-value vector."""
    return extract_features_batch(np.asarray(normalized_landmarks, dtype=np.float64).reshape(1, -1))[0]


def normalize_landmarks(landmarks):
    """
    Normalize landmarks for model input.

    Args:
        landmarks: list of raw landmark vectors, each 63 values laid out as
            21 consecutive (x, y, z) triples, or None

    Returns:
        List of normalized (63,) arrays, or None when there is nothing to
        normalize.
    """
    if landmarks is None:
        return None

    normalized = []
    for hand_landmarks in landmarks:
        # Convert to numpy array if needed
        if not isinstance(hand_landmarks, np.ndarray):
            hand_landmarks = np.array(hand_landmarks)

        # Reshape flat 63 values into 21 (x, y, z) points so the wrist
        # offset and scale apply per-landmark rather than broadcasting.
        points = hand_landmarks.reshape(NUM_LANDMARKS, LANDMARK_DIMENSIONS)

        # Normalize: use the wrist (landmark 0) as reference point
        normalized_points = points - points[WRIST_IDX]

        # Scale by hand size (distance from wrist to middle finger MCP)
        scale = np.linalg.norm(points[MIDDLE_MCP_IDX] - points[WRIST_IDX])
        if scale > 0:
            normalized_points = normalized_points / scale

        normalized.append(normalized_points.reshape(-1))

    return normalized if len(normalized) > 0 else None


def landmarks_to_points(landmarks):
    """
    Convert raw landmark vectors into (x, y) pairs in [0,1] within the square
    crop, for drawing in a browser.

    Returns a list of hands, each a list of 21 (x, y) pairs.
    """
    if not landmarks:
        return []
    return [[(float(h[i]), float(h[i + 1])) for i in range(0, len(h), 3)]
            for h in landmarks]


def flatten_landmark_dicts(items):
    """
    Turn a browser payload of 21 {x, y, z} objects into the flat 63-value
    vector the rest of the pipeline expects.

    The browser runs MediaPipe on the same centred square region the Python
    pipeline cropped to, so these coordinates carry the same meaning as the
    ones `HandTracker.extract_landmarks` produces.
    """
    flat = []
    for p in items:
        flat.extend([float(p["x"]), float(p["y"]), float(p["z"])])
    return np.array(flat, dtype=float)
