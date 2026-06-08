"""Draw the holistic skeleton (face mesh, pose, both hands) onto a frame."""

from __future__ import annotations

import numpy as np

from .landmarks import HolisticTracker, LandmarkFrame


def draw_skeleton(
    bgr_frame: np.ndarray,
    frame: LandmarkFrame,
    tracker: HolisticTracker,
    *,
    draw_face: bool = True,
    draw_pose: bool = True,
    draw_hands: bool = True,
) -> np.ndarray:
    """Overlay the detected skeleton on a BGR frame, in place, and return it."""
    if frame.raw is None:
        return bgr_frame

    mp_h = tracker.mp_holistic
    mp_d = tracker.mp_drawing
    mp_s = tracker.mp_styles
    result = frame.raw

    if draw_face and result.face_landmarks is not None:
        # Tesselation gives the full mesh; contours highlight eyes + lips.
        mp_d.draw_landmarks(
            bgr_frame,
            result.face_landmarks,
            mp_h.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_s.get_default_face_mesh_tesselation_style(),
        )
        mp_d.draw_landmarks(
            bgr_frame,
            result.face_landmarks,
            mp_h.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_s.get_default_face_mesh_contours_style(),
        )

    if draw_pose and result.pose_landmarks is not None:
        mp_d.draw_landmarks(
            bgr_frame,
            result.pose_landmarks,
            mp_h.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_s.get_default_pose_landmarks_style(),
        )

    if draw_hands:
        if result.left_hand_landmarks is not None:
            mp_d.draw_landmarks(
                bgr_frame,
                result.left_hand_landmarks,
                mp_h.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_s.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_s.get_default_hand_connections_style(),
            )
        if result.right_hand_landmarks is not None:
            mp_d.draw_landmarks(
                bgr_frame,
                result.right_hand_landmarks,
                mp_h.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_s.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_s.get_default_hand_connections_style(),
            )

    return bgr_frame
