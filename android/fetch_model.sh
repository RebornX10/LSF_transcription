#!/usr/bin/env bash
# Download the MediaPipe HolisticLandmarker model into the app's assets.
# Required before building — the .task model is ~13 MB and not committed.
set -euo pipefail
cd "$(dirname "$0")"

DEST="app/src/main/assets"
URL="https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"

mkdir -p "$DEST"
echo "Downloading holistic_landmarker.task (~13 MB)…"
curl -L --fail -o "$DEST/holistic_landmarker.task" "$URL"
echo "Saved to $DEST/holistic_landmarker.task"
