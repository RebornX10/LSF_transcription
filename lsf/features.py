"""Turn a `LandmarkFrame` into a fixed-length, normalised feature vector.

Design goals
------------
* **Fixed length** regardless of which body parts are visible (missing groups
  are zero-filled and flagged), so downstream buffers/matchers stay simple.
* **Body-relative** coordinates: we re-express every landmark in a frame whose
  origin is the shoulder centre and whose unit is the shoulder width. This makes
  the features invariant to where the signer stands and how far they are from
  the camera, while preserving the three things LSF phonology cares about:
  hand *location* relative to the body, hand *shape* (all 21 finger points),
  and *movement* (because successive frames share the same reference).
* Only the face points that carry linguistic meaning in LSF are kept: eyes /
  gaze, eyebrows, and mouth (mouthing + non-manual markers).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .landmarks import LandmarkFrame

# Pose points relevant to arms + chest (MediaPipe Pose indices).
POSE_IDX = [
    0,    # nose
    11, 12,   # shoulders (chest line)
    13, 14,   # elbows
    15, 16,   # wrists
    23, 24,   # hips (torso anchor)
]

# Curated face mesh points: brows, eyes (+ iris when refined), nose, lips.
FACE_IDX = [
    # eyebrows
    70, 105, 107, 336, 334, 300,
    # left eye ring
    33, 159, 145, 133,
    # right eye ring
    362, 386, 374, 263,
    # irises (present only with refine_face_landmarks=True)
    468, 473,
    # nose tip + bridge
    1, 4,
    # outer lips
    61, 291, 0, 17,
    # inner lips (mouth opening)
    13, 14,
]

N_POSE_SEL = len(POSE_IDX)
N_FACE_SEL = len(FACE_IDX)
N_HAND = 21

# 3 coords (x, y, z) per point, for pose-subset + face-subset + two hands,
# plus 4 presence flags (pose, face, left hand, right hand).
FEATURE_DIM = (N_POSE_SEL + N_FACE_SEL + N_HAND + N_HAND) * 3 + 4


def _reference(frame: LandmarkFrame) -> tuple[np.ndarray, float]:
    """Pick an (origin_xy z), scale to normalise a frame into a body frame.

    Prefers shoulders; falls back to face span, then to hand wrists, so the
    pipeline still produces stable features when only part of the body shows.
    """
    if frame.pose is not None:
        l_sh, r_sh = frame.pose[11, :3], frame.pose[12, :3]
        origin = (l_sh + r_sh) / 2.0
        scale = float(np.linalg.norm(l_sh[:2] - r_sh[:2]))
    elif frame.face is not None:
        # Use inter-ocular-ish span (eye corners) as scale.
        origin = frame.face[1, :3]  # nose tip
        scale = float(np.linalg.norm(frame.face[33, :2] - frame.face[263, :2]))
    else:
        wrists = [h[0, :3] for h in (frame.left_hand, frame.right_hand) if h is not None]
        origin = np.mean(wrists, axis=0) if wrists else np.zeros(3, np.float32)
        scale = 0.2  # arbitrary but non-zero; single-hand close-up
    return origin.astype(np.float32), (scale or 1e-3)


def _normalise(points: Optional[np.ndarray], idx, origin, scale) -> np.ndarray:
    """Return (len(idx), 3) body-relative coords, or zeros when absent."""
    n = len(idx)
    if points is None:
        return np.zeros((n, 3), dtype=np.float32)
    # Guard against meshes without iris points (refine disabled).
    safe_idx = [i if i < points.shape[0] else 0 for i in idx]
    sel = points[safe_idx, :3]
    return (sel - origin) / scale


def _normalise_all(points: Optional[np.ndarray], origin, scale) -> np.ndarray:
    n = N_HAND
    if points is None:
        return np.zeros((n, 3), dtype=np.float32)
    return (points[:, :3] - origin) / scale


def extract_features(frame: LandmarkFrame) -> np.ndarray:
    """Flatten a LandmarkFrame into a (FEATURE_DIM,) float32 vector."""
    origin, scale = _reference(frame)

    pose = _normalise(frame.pose, POSE_IDX, origin, scale)
    face = _normalise(frame.face, FACE_IDX, origin, scale)
    lh = _normalise_all(frame.left_hand, origin, scale)
    rh = _normalise_all(frame.right_hand, origin, scale)

    flags = np.array(
        [
            frame.pose is not None,
            frame.face is not None,
            frame.left_hand is not None,
            frame.right_hand is not None,
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        [pose.ravel(), face.ravel(), lh.ravel(), rh.ravel(), flags]
    ).astype(np.float32)


def mouth_openness(frame: LandmarkFrame) -> float:
    """Vertical lip gap / mouth width, a cheap non-manual marker. 0 when no face."""
    if frame.face is None:
        return 0.0
    top, bottom = frame.face[13, :2], frame.face[14, :2]
    left, right = frame.face[61, :2], frame.face[291, :2]
    width = float(np.linalg.norm(left - right)) or 1e-3
    return float(np.linalg.norm(top - bottom)) / width
