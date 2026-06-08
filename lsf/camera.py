"""Frame pipeline + threaded camera manager.

`InterpreterPipeline` is the stateless-per-instance glue: frame in, annotated
frame + optional recognition out. It is shared by the CLI runner and the web app.

`CameraInterpreter` owns a camera, runs the pipeline on a background thread, and
keeps the latest annotated JPEG + transcript so a web server can stream them
without blocking. It is a process-wide singleton (one camera, many viewers).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np

from .depth import DepthEstimator
from .landmarks import HolisticTracker, LandmarkFrame
from .recognizer import Recognition, SignRecognizer
from .skeleton import draw_skeleton

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("opencv-python is required (pip install -r requirements.txt)") from exc


@dataclass
class FrameResult:
    annotated: np.ndarray
    landmarks: LandmarkFrame
    recognition: Optional[Recognition]
    depth_source: str


class InterpreterPipeline:
    """Holistic detection + depth refinement + skeleton draw + recognition."""

    def __init__(
        self,
        *,
        templates_path: str = "signs/lsf_signs.json",
        model_path: Optional[str] = None,
        enable_monocular_depth: bool = False,
        model_complexity: int = 1,
        refine_face_landmarks: bool = True,
    ) -> None:
        self.tracker = HolisticTracker(
            model_complexity=model_complexity,
            refine_face_landmarks=refine_face_landmarks,
        )
        self.depth = DepthEstimator(enable_monocular=enable_monocular_depth)
        self.recognizer = SignRecognizer(
            templates_path=templates_path, model_path=model_path
        )

    def probe_depth(self, capture) -> None:
        self.depth.probe(capture)

    def process(
        self,
        bgr_frame: np.ndarray,
        external_depth: Optional[np.ndarray] = None,
        *,
        draw: bool = True,
    ) -> FrameResult:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        lf = self.tracker.process(rgb)

        depth_info = self.depth.depth_for_frame(rgb, external_depth)
        lf = self.depth.refine_landmarks(lf, depth_info)

        # Server-side drawing is only used by the CLI; the web app draws the
        # skeleton in the browser from returned landmarks, so it skips this.
        annotated = draw_skeleton(bgr_frame, lf, self.tracker) if draw else bgr_frame
        recognition = self.recognizer.update(lf)
        return FrameResult(annotated, lf, recognition, depth_info.source)

    def close(self) -> None:
        self.tracker.close()


class CameraInterpreter:
    """Threaded singleton that captures, processes, and buffers results."""

    _instance: Optional["CameraInterpreter"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        camera_index: int = 0,
        *,
        templates_path: str = "signs/lsf_signs.json",
        model_path: Optional[str] = None,
        enable_monocular_depth: bool = False,
        max_transcript: int = 200,
    ) -> None:
        self.camera_index = camera_index
        self._pipeline = InterpreterPipeline(
            templates_path=templates_path,
            model_path=model_path,
            enable_monocular_depth=enable_monocular_depth,
        )
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._transcript: Deque[dict] = deque(maxlen=max_transcript)
        self._transcript_seq = 0
        self._depth_source = "mediapipe"
        self._fps = 0.0

    # -- singleton access -----------------------------------------------------
    @classmethod
    def instance(cls, **kwargs) -> "CameraInterpreter":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(**kwargs)
                cls._instance.start()
            return cls._instance

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.camera_index}.")
        self._pipeline.probe_depth(self._cap)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        self._pipeline.close()

    # -- worker loop ----------------------------------------------------------
    def _loop(self) -> None:
        last = time.time()
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)  # mirror, like a selfie view
            result = self._pipeline.process(frame)
            self._depth_source = result.depth_source

            if result.recognition is not None:
                self._append_transcript(result.recognition)

            self._overlay_hud(result.annotated)

            ok, buf = cv2.imencode(".jpg", result.annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                with self._frame_lock:
                    self._latest_jpeg = buf.tobytes()

            now = time.time()
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(now - last, 1e-3))
            last = now

    def _overlay_hud(self, frame: np.ndarray) -> None:
        rec = self._pipeline.recognizer
        status = "REC " + (rec._record_label or "") if rec.is_recording else ""
        text = f"depth:{self._depth_source}  fps:{self._fps:0.0f}  {status}"
        cv2.putText(
            frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 0, 0), 4, cv2.LINE_AA,
        )
        cv2.putText(
            frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 128), 1, cv2.LINE_AA,
        )

    # -- shared state accessors ----------------------------------------------
    def _append_transcript(self, rec: Recognition) -> None:
        self._transcript_seq += 1
        with self._frame_lock:
            self._transcript.append(
                {
                    "seq": self._transcript_seq,
                    "gloss": rec.gloss,
                    "confidence": round(rec.confidence, 3),
                    "source": rec.source,
                    "t": rec.timestamp,
                }
            )

    def latest_jpeg(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._latest_jpeg

    def transcript(self, since: int = 0) -> List[dict]:
        with self._frame_lock:
            return [e for e in self._transcript if e["seq"] > since]

    def clear_transcript(self) -> None:
        with self._frame_lock:
            self._transcript.clear()

    # -- recording controls (proxied to recognizer) --------------------------
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
            "depth_reason": self._pipeline.depth.reason,
            "fps": round(self._fps, 1),
            "recording": rec.is_recording,
            "vocabulary": rec.vocabulary,
        }
