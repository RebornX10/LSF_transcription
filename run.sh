#!/usr/bin/env bash
# One-command launcher for the LSF interpreter web UI.
# Ships pre-trained (72 LSF signs). Creates the venv on first run, then serves.
#
#   ./run.sh                 # http://127.0.0.1:8000
#   ./run.sh 0.0.0.0:8000    # bind all interfaces (use localhost for the camera)
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "First run — creating the virtual environment (a few minutes)…"
  ./setup.sh
fi

# shellcheck disable=SC1091
source .venv/bin/activate

ADDR="${1:-127.0.0.1:8000}"
echo "LSF interpreter → http://${ADDR}   (allow the camera; Ctrl+C to stop)"
exec python web/manage.py runserver "$ADDR"
