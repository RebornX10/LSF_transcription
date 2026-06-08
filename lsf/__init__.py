"""LSF_transcripter — French Sign Language (Langue des Signes Française) interpreter.

The package turns a camera frame into:
  1. A holistic skeleton (face / pose / both hands) via MediaPipe.
  2. A normalised feature vector describing fingers, eyes, mouth, chest and arms.
  3. A recognised LSF gloss using a learn-by-example (DTW) recogniser, with an
     optional deep-learning model hook and a few built-in heuristic signs.

Optional depth refinement is applied when the camera exposes a dual-pixel /
depth stream (or a monocular-depth model is installed).
"""

from .landmarks import HolisticTracker, LandmarkFrame
from .recognizer import SignRecognizer, Recognition

__all__ = [
    "HolisticTracker",
    "LandmarkFrame",
    "SignRecognizer",
    "Recognition",
]

__version__ = "0.1.0"
