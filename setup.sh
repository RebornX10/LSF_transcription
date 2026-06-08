#!/usr/bin/env bash
# One-shot environment setup for the LSF interpreter.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

echo "==> Creating virtual environment in .venv"
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing requirements (this pulls mediapipe/opencv and can take a few minutes)"
pip install -r requirements.txt

echo
echo "Done. Activate the env with:  source .venv/bin/activate"
echo "Run the web UI with:          python web/manage.py runserver"
echo "Or the standalone window:     python main.py"
