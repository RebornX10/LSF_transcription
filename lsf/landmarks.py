"""MediaPipe Holistic wrapper.

`Holistic` gives us, in a single pass, every body part the brief asks for:

    * fingers / hand shape   -> left_hand / right_hand  (21 landmarks each)
    * eyes / mouth / face    -> face_landmarks           (468 landmarks)
    * chest / arms / shoulders -> pose_landmarks         (33 landmarks)

Each landmark is (x, y, z) with x/y normalised to the image size and z a
relative depth (smaller = closer to camera). We keep the raw MediaPipe result
plus convenient numpy views so the rest of the pipeline never touches the
MediaPipe types directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - surfaced at runtime
    raise ImportError(
        "mediapipe is required. Install it with `pip install -r requirements.txt`."
    ) from exc


# Number of landmarks per group (fixed by the MediaPipe models).
N_POSE = 33
N_HAND = 21
# Face mesh is 468 points, or 478 when refine_face_landmarks adds the 10 iris
# points. We size face arrays from the actual result rather than this constant.
N_FACE = 478


def _to_array(landmark_list) -> Optional[np.ndarray]:
    """Convert a MediaPipe landmark list to an (N, 4) float32 array.

    N is taken from the result itself (the face mesh is 468 or 478 points
    depending on iris refinement, so we never hardcode it). Columns are
    (x, y, z, visibility); returns None when the group was not detected.
    """
    if landmark_list is None:
        return None
    points = landmark_list.landmark
    out = np.zeros((len(points), 4), dtype=np.float32)
    for i, lm in enumerate(points):
        out[i, 0] = lm.x
        out[i, 1] = lm.y
        out[i, 2] = lm.z
        # Face / hand landmarks have no visibility field; default to 1.0.
        out[i, 3] = getattr(lm, "visibility", 1.0)
    return out


@dataclass
class LandmarkFrame:
    """All landmarks detected in a single frame (or None per group)."""

    pose: Optional[np.ndarray]        # (33, 4)
    face: Optional[np.ndarray]        # (468, 4)
    left_hand: Optional[np.ndarray]   # (21, 4)
    right_hand: Optional[np.ndarray]  # (21, 4)
    raw: object = None                # original MediaPipe result (for drawing)

    @property
    def has_hands(self) -> bool:
        return self.left_hand is not None or self.right_hand is not None

    @property
    def has_body(self) -> bool:
        return self.pose is not None


def points_payload(frame: "LandmarkFrame") -> dict:
    """Compact 2D points for the browser to draw the skeleton.

    Only pose + both hands are sent (face mesh is omitted to keep the payload
    tiny and the overlay clean); x/y are the normalised image coords [0, 1].
    Rounded to 4 decimals to shrink the JSON.
    """

    def xy(arr):
        if arr is None:
            return None
        return [[round(float(x), 4), round(float(y), 4)] for x, y in arr[:, :2]]

    return {
        "pose": xy(frame.pose),
        "left_hand": xy(frame.left_hand),
        "right_hand": xy(frame.right_hand),
    }


class HolisticTracker:
    """Stateful wrapper around `mediapipe.solutions.holistic.Holistic`.

    The underlying graph is stateful (it tracks across frames), so create one
    tracker per video stream and call `process` for each frame in order.
    """

    def __init__(
        self,
        *,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        refine_face_landmarks: bool = True,
    ) -> None:
        self._mp_holistic = mp.solutions.holistic
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        # refine_face_landmarks gives the iris/eye detail we use for gaze.
        self._holistic = self._mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            refine_face_landmarks=refine_face_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, rgb_frame: np.ndarray) -> LandmarkFrame:
        """Run holistic detection on an RGB (not BGR) frame."""
        rgb_frame.flags.writeable = False
        result = self._holistic.process(rgb_frame)
        rgb_frame.flags.writeable = True
        return LandmarkFrame(
            pose=_to_array(result.pose_landmarks),
            face=_to_array(result.face_landmarks),
            left_hand=_to_array(result.left_hand_landmarks),
            right_hand=_to_array(result.right_hand_landmarks),
            raw=result,
        )

    # Expose the MediaPipe handles so skeleton.py can draw without re-importing.
    @property
    def mp_holistic(self):
        return self._mp_holistic

    @property
    def mp_drawing(self):
        return self._mp_drawing

    @property
    def mp_styles(self):
        return self._mp_styles

    def close(self) -> None:
        self._holistic.close()

    def __enter__(self) -> "HolisticTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
