"""Batch-import sign clips into recognisable ("pre-trained") words.

This is how you get pre-trained words without recording each sign live in the
UI: point it at reference clips and it runs them through the *same* pipeline the
live recorder uses, writing templates into `signs/lsf_signs.json`. After that
the words are recognised immediately — by the CLI and the web UI alike.

Accepted layouts (gloss is taken from the file/folder name, uppercased):

    clips/BONJOUR.mp4                      -> one sample  for BONJOUR
    clips/MERCI.mov
    clips/AU REVOIR/01.mp4                 -> several samples for AU REVOIR
    clips/AU REVOIR/02.mp4

Record several clips per sign (different speed / slightly different framing) for
much better accuracy — DTW matches against all of them.

Where to get clips:
  * Film them yourself once (phone/webcam): one short clip per sign.
  * Any LSF video you have the right to use (check each source's terms).

Usage:
    python tools/import_videos.py clips/
    python tools/import_videos.py one_clip.mp4 --gloss BONJOUR
    python tools/import_videos.py clips/ --templates signs/lsf_signs.json
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lsf.landmarks import HolisticTracker  # noqa: E402
from lsf.recognizer import SignRecognizer  # noqa: E402

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}


def find_clips(path: str) -> list[tuple[str, str]]:
    """Return (gloss, video_path) pairs from a file or directory tree."""
    if os.path.isfile(path):
        gloss = os.path.splitext(os.path.basename(path))[0].upper()
        return [(gloss, path)]

    pairs: list[tuple[str, str]] = []
    for root, _dirs, files in os.walk(path):
        for name in sorted(files):
            ext = os.path.splitext(name)[1].lower()
            if ext not in VIDEO_EXTS:
                continue
            full = os.path.join(root, name)
            # gloss = subfolder name if nested, else the file stem.
            if os.path.abspath(root) != os.path.abspath(path):
                gloss = os.path.basename(root).upper()
            else:
                gloss = os.path.splitext(name)[0].upper()
            pairs.append((gloss, full))
    return pairs


def import_clip(
    tracker: HolisticTracker, recognizer: SignRecognizer, gloss: str, video_path: str
) -> bool:
    """Run one clip through the pipeline and store it as a template for `gloss`."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ! could not open {video_path}")
        return False

    recognizer.start_recording(gloss)
    frames = 0
    detected = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lf = tracker.process(rgb)
            if lf.has_hands:
                detected += 1
            recognizer.update(lf)
    finally:
        cap.release()

    if detected == 0:
        recognizer.cancel_recording()
        print(f"  ! no hands detected in {os.path.basename(video_path)} — skipped")
        return False

    saved = recognizer.stop_recording()
    if saved is None:
        print(f"  ! clip too short ({frames} frames) — skipped")
        return False
    print(f"  ✓ {gloss:18s} <- {os.path.basename(video_path)} "
          f"({frames} frames, hands in {detected})")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="Import sign clips into templates.")
    p.add_argument("path", help="A video file or a directory of clips.")
    p.add_argument("--templates", default="signs/lsf_signs.json",
                   help="Templates file to write (default: signs/lsf_signs.json).")
    p.add_argument("--gloss", default=None,
                   help="Force the gloss (only when importing a single file).")
    args = p.parse_args()

    clips = find_clips(args.path)
    if args.gloss and len(clips) == 1:
        clips = [(args.gloss.upper(), clips[0][1])]
    if not clips:
        print(f"No video clips found under {args.path!r}.")
        return 1

    print(f"Importing {len(clips)} clip(s) into {args.templates} …")
    recognizer = SignRecognizer(templates_path=args.templates, use_heuristics=False)
    tracker = HolisticTracker()
    ok = 0
    try:
        for gloss, video_path in clips:
            if import_clip(tracker, recognizer, gloss, video_path):
                ok += 1
    finally:
        tracker.close()

    print(f"\nDone: {ok}/{len(clips)} clip(s) imported.")
    print(f"Vocabulary now trained: {', '.join(recognizer.vocabulary) or '(none)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
