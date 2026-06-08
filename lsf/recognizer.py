"""Sign recogniser: turn a stream of landmark frames into LSF glosses.

There is no off-the-shelf, production-grade LSF translation model, and training
one needs a large annotated corpus. So this recogniser is built to be *useful
out of the box and extensible*:

1. **Motion segmentation** — watches hand presence + motion energy to find where
   one sign begins and ends (so we classify gestures, not every frame).

2. **Learn-by-example (DTW)** — the primary path. You record a few samples of
   each sign through the UI; we store the normalised landmark sequence as a
   template and match new segments with Dynamic Time Warping. This works on a
   laptop CPU, with no GPU and no training run, for a small/medium vocabulary.

3. **Deep-model hook** — if a trained Keras model (`model.h5`) and its label map
   are present, we use it instead of/along DTW. Plug your own sequence model in.

4. **Heuristic fallbacks** — a couple of signs detectable by simple geometry so
   the demo produces output before you've recorded anything.

Everything operates on the body-relative feature vectors from `features.py`.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .features import (
    FEATURE_DIM,
    N_FACE_SEL,
    N_POSE_SEL,
    extract_features,
    mouth_openness,
)
from .landmarks import LandmarkFrame

# Offsets into the feature vector (see features.extract_features layout).
_POSE_LEN = N_POSE_SEL * 3
_FACE_LEN = N_FACE_SEL * 3
_HANDS_LEN = 21 * 3 * 2
_HANDS_START = _POSE_LEN + _FACE_LEN

# Offsets *within the matching vector* ([pose(27), left hand(63), right hand(63)]).
# POSE_IDX order is [nose, L/R shoulder, L/R elbow, L/R wrist, L/R hip]; the arm
# block (elbows+wrists, selection indices 3..6) is coords 9..21.
_ARM_SLICE = slice(9, 21)
_LEFT_HAND_SLICE = slice(_POSE_LEN, _POSE_LEN + 63)
_RIGHT_HAND_SLICE = slice(_POSE_LEN + 63, _POSE_LEN + 126)

# Tunables
TEMPLATE_LEN = 32        # frames each segment is resampled to before DTW
MIN_SEGMENT_FRAMES = 6   # ignore micro-twitches
MOTION_THRESHOLD = 0.025  # per-frame RMS motion (of moving parts) to be "active"
SETTLE_FRAMES = 8        # consecutive calm frames that close a segment
DTW_BAND = 8             # Sakoe-Chiba band radius
REJECT_DISTANCE = 3.5    # avg aligned distance above which we emit nothing


@dataclass
class Recognition:
    """A recognised sign."""

    gloss: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    source: str = "dtw"  # "dtw" | "model" | "heuristic"


def _matching_vector(feat: np.ndarray) -> np.ndarray:
    """Pose + hands slice of a feature vector (face/flags dropped for matching).

    Facial expression varies a lot between repetitions of the same sign, so we
    align on the manual channel (arms + both hands) which carries the gesture.
    """
    pose = feat[:_POSE_LEN]
    hands = feat[_HANDS_START:_HANDS_START + _HANDS_LEN]
    return np.concatenate([pose, hands])


def _resample(seq: np.ndarray, n: int) -> np.ndarray:
    """Linearly resample a (T, D) sequence to (n, D) along the time axis."""
    t = seq.shape[0]
    if t == n:
        return seq
    if t == 1:
        return np.repeat(seq, n, axis=0)
    src = np.linspace(0.0, 1.0, t)
    dst = np.linspace(0.0, 1.0, n)
    out = np.empty((n, seq.shape[1]), dtype=np.float32)
    for d in range(seq.shape[1]):
        out[:, d] = np.interp(dst, src, seq[:, d])
    return out


def _dtw_distance(a: np.ndarray, b: np.ndarray, band: int = DTW_BAND) -> float:
    """Banded DTW distance between two (N, D) sequences, normalised by path len."""
    n, m = a.shape[0], b.shape[0]
    inf = np.inf
    cost = np.full((n + 1, m + 1), inf, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        j_lo = max(1, i - band)
        j_hi = min(m, i + band)
        ai = a[i - 1]
        for j in range(j_lo, j_hi + 1):
            d = float(np.linalg.norm(ai - b[j - 1]))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    # Normalise by the diagonal path length so different lengths compare fairly.
    return cost[n, m] / (n + m)


class SignRecognizer:
    """Stateful streaming recogniser.

    Call `update(frame, t)` once per video frame. It returns a `Recognition`
    on the frame where a completed sign is identified, otherwise None.
    """

    def __init__(
        self,
        templates_path: str = "signs/lsf_signs.json",
        model_path: Optional[str] = None,
        *,
        use_heuristics: bool = True,
    ) -> None:
        self.templates_path = templates_path
        self.use_heuristics = use_heuristics
        # templates: gloss -> list of (TEMPLATE_LEN, D) matching sequences
        self.templates: dict[str, List[np.ndarray]] = {}
        self._load_templates()

        # streaming state
        self._buffer: List[np.ndarray] = []       # matching vectors in segment
        self._prev_match: Optional[np.ndarray] = None
        self._calm_run = 0
        self._active = False
        self._last_emit = 0.0

        # recording state (learn-by-example)
        self._recording = False
        self._record_buffer: List[np.ndarray] = []
        self._record_label: Optional[str] = None

        # optional deep model
        self._model = None
        self._labels: List[str] = []
        if model_path and os.path.exists(model_path):
            self._try_load_model(model_path)

    # -- persistence ----------------------------------------------------------
    def _load_templates(self) -> None:
        if not os.path.exists(self.templates_path):
            return
        with open(self.templates_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        for gloss, samples in raw.items():
            self.templates[gloss] = [np.asarray(s, dtype=np.float32) for s in samples]

    def _save_templates(self) -> None:
        serialisable = {
            gloss: [s.tolist() for s in samples]
            for gloss, samples in self.templates.items()
        }
        try:
            os.makedirs(os.path.dirname(self.templates_path) or ".", exist_ok=True)
            with open(self.templates_path, "w", encoding="utf-8") as fh:
                json.dump(serialisable, fh)
        except OSError:
            # Read-only/ephemeral filesystem (e.g. a hosted Space): keep the
            # templates in memory so the session still works, just not persisted.
            pass

    @property
    def vocabulary(self) -> List[str]:
        return sorted(self.templates.keys())

    def template_counts(self) -> dict[str, int]:
        """How many recorded samples back each trained gloss."""
        return {gloss: len(samples) for gloss, samples in self.templates.items()}

    def delete_gloss(self, gloss: str) -> bool:
        """Forget a trained sign entirely. Returns True if it existed."""
        if gloss in self.templates:
            del self.templates[gloss]
            self._save_templates()
            return True
        return False

    # -- deep model hook ------------------------------------------------------
    def _try_load_model(self, model_path: str) -> None:
        try:
            import tensorflow as tf  # noqa: F401
            from tensorflow import keras

            self._model = keras.models.load_model(model_path)
            label_path = os.path.splitext(model_path)[0] + ".labels.json"
            if os.path.exists(label_path):
                with open(label_path, encoding="utf-8") as fh:
                    self._labels = json.load(fh)
        except Exception:
            self._model = None  # stay on DTW/heuristics

    # -- recording API (learn-by-example) ------------------------------------
    def start_recording(self, label: str) -> None:
        self._recording = True
        self._record_label = label
        self._record_buffer = []

    def stop_recording(self) -> Optional[str]:
        """Finalise a recording into a template. Returns the label or None."""
        self._recording = False
        if not self._record_label or len(self._record_buffer) < MIN_SEGMENT_FRAMES:
            self._record_buffer = []
            return None
        seq = _resample(np.asarray(self._record_buffer, dtype=np.float32), TEMPLATE_LEN)
        self.templates.setdefault(self._record_label, []).append(seq)
        self._save_templates()
        label = self._record_label
        self._record_label = None
        self._record_buffer = []
        return label

    def cancel_recording(self) -> None:
        """Abort the current recording without saving anything."""
        self._recording = False
        self._record_label = None
        self._record_buffer = []

    @property
    def is_recording(self) -> bool:
        return self._recording

    # -- motion gating --------------------------------------------------------
    def _motion_energy(self, frame: LandmarkFrame, match: np.ndarray) -> float:
        """RMS per-coordinate displacement over the present moving body parts.

        Uses the hands when visible (the primary signing channel), else falls
        back to the arms (elbows/wrists), so segmentation works for one hand,
        two hands, or arm-only gestures without static channels washing it out.
        """
        if self._prev_match is None:
            return 0.0
        diff = match - self._prev_match
        idx = []
        if frame.left_hand is not None:
            idx.append(diff[_LEFT_HAND_SLICE])
        if frame.right_hand is not None:
            idx.append(diff[_RIGHT_HAND_SLICE])
        if not idx:  # no hands -> use arm motion
            idx.append(diff[_ARM_SLICE])
        d = np.concatenate(idx)
        return float(np.linalg.norm(d)) / np.sqrt(d.size)

    # -- main streaming entry point ------------------------------------------
    def update(self, frame: LandmarkFrame, t: Optional[float] = None) -> Optional[Recognition]:
        t = time.time() if t is None else t
        feat = extract_features(frame)
        match = _matching_vector(feat)

        if self._recording:
            self._record_buffer.append(match)

        # Motion energy between consecutive frames, measured only over the body
        # parts that are actually present (so static/zero channels don't dilute it).
        motion = self._motion_energy(frame, match)
        self._prev_match = match

        active_now = motion > MOTION_THRESHOLD

        if active_now:
            if not self._active:
                self._active = True
                self._buffer = []
            self._buffer.append(match)
            self._calm_run = 0
            return None

        # Not active this frame.
        if self._active:
            self._calm_run += 1
            if self._calm_run >= SETTLE_FRAMES:
                # Segment finished -> classify.
                segment = self._buffer
                self._active = False
                self._buffer = []
                self._calm_run = 0
                if len(segment) >= MIN_SEGMENT_FRAMES and (t - self._last_emit) > 0.3:
                    rec = self._classify(np.asarray(segment, dtype=np.float32), frame, t)
                    if rec is not None:
                        self._last_emit = t
                    return rec
        return None

    # -- classification -------------------------------------------------------
    def _classify(
        self, segment: np.ndarray, frame: LandmarkFrame, t: float
    ) -> Optional[Recognition]:
        seq = _resample(segment, TEMPLATE_LEN)

        if self._model is not None:
            rec = self._classify_model(seq, t)
            if rec is not None:
                return rec

        rec = self._classify_dtw(seq, t)
        if rec is not None:
            return rec

        if self.use_heuristics:
            return self._classify_heuristic(segment, frame, t)
        return None

    def _classify_model(self, seq: np.ndarray, t: float) -> Optional[Recognition]:
        probs = self._model.predict(seq[None, ...], verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        if conf < 0.5 or idx >= len(self._labels):
            return None
        return Recognition(self._labels[idx], conf, t, source="model")

    def _classify_dtw(self, seq: np.ndarray, t: float) -> Optional[Recognition]:
        if not self.templates:
            return None
        best_gloss, best_dist = None, np.inf
        for gloss, samples in self.templates.items():
            for tmpl in samples:
                d = _dtw_distance(seq, tmpl)
                if d < best_dist:
                    best_dist, best_gloss = d, gloss
        if best_gloss is None or best_dist > REJECT_DISTANCE:
            return None
        conf = max(0.0, 1.0 - best_dist / REJECT_DISTANCE)
        return Recognition(best_gloss, conf, t, source="dtw")

    def _classify_heuristic(
        self, segment: np.ndarray, frame: LandmarkFrame, t: float
    ) -> Optional[Recognition]:
        """A couple of geometry-only signs so the demo isn't silent.

        These are deliberately simple and approximate — real coverage comes from
        recorded templates or a trained model.
        """
        # Lateral oscillation of an open hand -> greeting "Bonjour". We track the
        # x of the first hand-wrist channel (right after the pose block).
        if segment.shape[0] >= MIN_SEGMENT_FRAMES:
            x_track = segment[:, _POSE_LEN]  # x of left-hand wrist in matching vec
            sweep = float(np.max(x_track) - np.min(x_track))
            zero_cross = np.sum(np.abs(np.diff(np.sign(np.diff(x_track)))) > 0)
            if sweep > 0.6 and zero_cross >= 2:
                return Recognition("BONJOUR", 0.4, t, source="heuristic")

        # Open mouth + raised hand held still -> tentative "OUI" placeholder.
        if mouth_openness(frame) > 0.35 and frame.has_hands:
            return Recognition("OUI", 0.3, t, source="heuristic")
        return None
