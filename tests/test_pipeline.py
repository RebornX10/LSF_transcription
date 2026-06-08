"""Camera-free self-test of the core pipeline.

Validates feature extraction, motion segmentation, learn-by-example recording,
and DTW matching using synthetic landmark frames — no webcam or real video
needed. Run after installing requirements:

    python tests/test_pipeline.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lsf.features import FEATURE_DIM, extract_features  # noqa: E402
from lsf.landmarks import LandmarkFrame  # noqa: E402
from lsf.recognizer import SignRecognizer  # noqa: E402


def make_frame(progress: float, jitter: float = 0.0) -> LandmarkFrame:
    """Synthetic frame: fixed shoulders, a right hand translating left→right."""
    rng = np.random.default_rng(0)
    pose = np.zeros((33, 4), dtype=np.float32)
    pose[:, 3] = 1.0
    pose[11, :3] = (0.40, 0.50, 0.0)   # left shoulder
    pose[12, :3] = (0.60, 0.50, 0.0)   # right shoulder
    pose[16, :3] = (0.55 + 0.2 * progress, 0.55, 0.0)  # right wrist

    hand = np.zeros((21, 4), dtype=np.float32)
    hand[:, 3] = 1.0
    base_x = 0.50 + 0.25 * progress
    for i in range(21):
        hand[i, 0] = base_x + 0.01 * i + jitter * rng.standard_normal()
        hand[i, 1] = 0.55 + 0.01 * (i % 5) + jitter * rng.standard_normal()
    return LandmarkFrame(pose=pose, face=None, left_hand=None, right_hand=hand)


def gesture_frames(n: int = 20, jitter: float = 0.0):
    return [make_frame(k / (n - 1), jitter) for k in range(n)]


def resting_frames(n: int = 12):
    return [make_frame(1.0) for _ in range(n)]  # hand held still


def test_feature_dim():
    feat = extract_features(make_frame(0.0))
    assert feat.shape == (FEATURE_DIM,), feat.shape
    assert np.isfinite(feat).all()
    print(f"  feature vector dim = {FEATURE_DIM}  ✓")


def test_record_and_recognise():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "signs.json")
        rec = SignRecognizer(templates_path=path, use_heuristics=False)

        # 1) Record a template for "DEMO".
        rec.start_recording("DEMO")
        for f in gesture_frames(jitter=0.0):
            rec.update(f, t=0.0)
        saved = rec.stop_recording()
        assert saved == "DEMO", saved
        assert "DEMO" in rec.vocabulary
        assert os.path.exists(path)
        print("  recorded template 'DEMO'  ✓")

        # 2) Reload from disk to prove persistence works.
        rec2 = SignRecognizer(templates_path=path, use_heuristics=False)
        assert "DEMO" in rec2.vocabulary
        print("  reloaded template from disk  ✓")

        # 3) Replay a similar gesture, then settle -> should recognise "DEMO".
        result = None
        t = 1.0
        for f in gesture_frames(jitter=0.003):
            r = rec2.update(f, t)
            result = r or result
            t += 1 / 30
        for f in resting_frames():
            r = rec2.update(f, t)
            result = r or result
            t += 1 / 30

        assert result is not None, "no sign was recognised"
        assert result.gloss == "DEMO", result.gloss
        print(f"  recognised '{result.gloss}' (conf {result.confidence:.2f}, "
              f"{result.source})  ✓")


def main() -> int:
    print("test_feature_dim")
    test_feature_dim()
    print("test_record_and_recognise")
    test_record_and_recognise()
    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
