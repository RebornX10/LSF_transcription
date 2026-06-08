"""Standalone CLI runner for the LSF interpreter (no web server).

Opens a window showing the live camera feed with the holistic skeleton drawn on
top and the running transcription printed in the console and on-screen.

Keys
----
  q / Esc : quit
  r       : start recording a sign sample; you'll be asked for its gloss in the
            console, then perform the sign. Press  s  to save it as a template.
  s       : stop + save the current recording
  c       : clear the on-screen transcript

For the Django web UI instead, run:  python web/manage.py runserver
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque

import cv2

from lsf.camera import InterpreterPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="French Sign Language interpreter (CLI).")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default 0).")
    p.add_argument(
        "--templates",
        default="signs/lsf_signs.json",
        help="Path to the learned-sign templates file.",
    )
    p.add_argument("--model", default=None, help="Optional Keras model.h5 path.")
    p.add_argument(
        "--monocular-depth",
        action="store_true",
        help="Enable MiDaS monocular depth fallback (needs torch installed).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: could not open camera index {args.camera}.", file=sys.stderr)
        return 1

    pipeline = InterpreterPipeline(
        templates_path=args.templates,
        model_path=args.model,
        enable_monocular_depth=args.monocular_depth,
    )
    pipeline.probe_depth(cap)
    print(f"Depth source: {pipeline.depth.status} ({pipeline.depth.reason})")
    print("Press 'r' to record a sign, 's' to save, 'c' to clear, 'q' to quit.")

    transcript: deque[str] = deque(maxlen=8)
    recording = False
    fps = 0.0
    last = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            result = pipeline.process(frame)

            if result.recognition is not None:
                line = f"{result.recognition.gloss} ({result.recognition.confidence:.2f})"
                transcript.append(line)
                print("→", line)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-3))
            last = now

            _draw_overlay(result.annotated, transcript, result.depth_source, fps, recording)
            cv2.imshow("LSF interpreter", result.annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("r") and not recording:
                label = input("Gloss for this sign (then perform it): ").strip()
                if label:
                    pipeline.recognizer.start_recording(label)
                    recording = True
                    print(f"Recording '{label}'… press 's' to save.")
            elif key == ord("s") and recording:
                saved = pipeline.recognizer.stop_recording()
                recording = False
                print(f"Saved template: {saved}" if saved else "Recording too short, discarded.")
            elif key == ord("c"):
                transcript.clear()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pipeline.close()
    return 0


def _draw_overlay(frame, transcript, depth_source, fps, recording) -> None:
    h = frame.shape[0]
    status = "REC" if recording else ""
    hud = f"depth:{depth_source}  fps:{fps:0.0f}  {status}"
    cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 1)
    for i, line in enumerate(list(transcript)[-6:]):
        y = h - 20 - (5 - i) * 26
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)


if __name__ == "__main__":
    raise SystemExit(main())
