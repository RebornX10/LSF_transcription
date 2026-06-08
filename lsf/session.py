"""Frame-push interpreter for the web app (browser-camera architecture).

Unlike `CameraInterpreter` (which owns a server-side webcam — only usable
locally), this receives frames pushed from the browser over HTTP, runs the
holistic tracker + recognizer, and returns compact landmarks + glosses for the
browser to draw. This is what makes the app deployable to Hugging Face, where
the server has no camera.

Tuned for CPU-only hosts (HF free tier): MediaPipe at `model_complexity=0`,
face refinement off, server-side drawing skipped. MediaPipe's graph is stateful
and not thread-safe, so all processing is serialised behind one lock — fine for
a demo; for many concurrent users you'd shard sessions.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, List, Optional

import numpy as np

from .camera import InterpreterPipeline
from .landmarks import points_payload
from .recognizer import Recognition


class FrameSession:
    """Process-wide singleton that turns pushed frames into landmarks + glosses."""

    _instance: Optional["FrameSession"] = None
    _init_lock = threading.Lock()

    def __init__(
        self,
        *,
        templates_path: str = "signs/lsf_signs.json",
        model_path: Optional[str] = None,
        model_complexity: int = 0,
        refine_face_landmarks: bool = False,
        max_transcript: int = 200,
    ) -> None:
        self._pipeline = InterpreterPipeline(
            templates_path=templates_path,
            model_path=model_path,
            model_complexity=model_complexity,
            refine_face_landmarks=refine_face_landmarks,
        )
        self._lock = threading.Lock()
        self._transcript: Deque[dict] = deque(maxlen=max_transcript)
        self._seq = 0
        self._fps = 0.0
        self._last_t = 0.0
        # Depth capability reported by the client device (dual-pixel detection).
        self._depth_source = "mediapipe"
        self._depth_reason = "No client depth reported yet."

    @classmethod
    def instance(cls, **kwargs) -> "FrameSession":
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls(**kwargs)
            return cls._instance

    # -- depth capability (set by the browser) -------------------------------
    def set_depth_capability(self, *, dual_pixel: bool, reason: str = "", label: str = "") -> None:
        if dual_pixel:
            self._depth_source = "dual_pixel"
            self._depth_reason = f"Dual-pixel/depth-capable device: {label or 'detected'}."
        else:
            self._depth_source = "mediapipe"
            self._depth_reason = reason or "No dual-pixel depth on this device; using MediaPipe z."

    # -- main entry point -----------------------------------------------------
    def process_jpeg(self, jpeg_bytes: bytes) -> dict:
        """Decode a pushed JPEG, run detection + recognition, return a payload."""
        import cv2  # local import keeps module import cheap

        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"ok": False, "error": "Could not decode frame."}

        with self._lock:
            result = self._pipeline.process(frame, draw=False)
            reco = result.recognition
            if reco is not None:
                self._append(reco)
            now = time.time()
            if self._last_t:
                self._fps = 0.85 * self._fps + 0.15 * (1.0 / max(now - self._last_t, 1e-3))
            self._last_t = now

        return {
            "ok": True,
            "landmarks": points_payload(result.landmarks),
            "reco": self._reco_dict(reco) if reco else None,
            "status": self.status,
        }

    # -- transcript -----------------------------------------------------------
    def _append(self, reco: Recognition) -> None:
        self._seq += 1
        self._transcript.append(self._reco_dict(reco, seq=self._seq))

    @staticmethod
    def _reco_dict(reco: Recognition, seq: int = 0) -> dict:
        return {
            "seq": seq,
            "gloss": reco.gloss,
            "confidence": round(reco.confidence, 3),
            "source": reco.source,
            "t": reco.timestamp,
        }

    def transcript(self, since: int = 0) -> List[dict]:
        with self._lock:
            return [e for e in self._transcript if e["seq"] > since]

    def clear_transcript(self) -> None:
        with self._lock:
            self._transcript.clear()

    # -- recording / vocabulary proxies --------------------------------------
    def start_recording(self, label: str) -> None:
        self._pipeline.recognizer.start_recording(label)

    def stop_recording(self) -> Optional[str]:
        return self._pipeline.recognizer.stop_recording()

    def cancel_recording(self) -> None:
        self._pipeline.recognizer.cancel_recording()

    def template_counts(self) -> dict:
        return self._pipeline.recognizer.template_counts()

    def delete_gloss(self, gloss: str) -> bool:
        return self._pipeline.recognizer.delete_gloss(gloss)

    @property
    def status(self) -> dict:
        rec = self._pipeline.recognizer
        return {
            "depth_source": self._depth_source,
            "depth_reason": self._depth_reason,
            "fps": round(self._fps, 1),
            "recording": rec.is_recording,
            "trained": len(rec.vocabulary),
        }
