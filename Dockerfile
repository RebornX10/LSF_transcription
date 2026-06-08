# LSF interpreter — browser-camera Django app, ready for Hugging Face Spaces.
FROM python:3.11-slim

# System libs needed by OpenCV / MediaPipe at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LSF_DEBUG=0 \
    # Max-speed defaults for the free CPU tier; override at deploy time.
    LSF_MODEL_COMPLEXITY=0 \
    LSF_REFINE_FACE=0

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Collect static assets so whitenoise can serve them under gunicorn.
RUN python web/manage.py collectstatic --noinput

EXPOSE 7860

# One worker (MediaPipe session + transcript are in-memory shared state),
# multiple threads for concurrent static/API requests.
CMD gunicorn --chdir web web.wsgi:application \
    --bind 0.0.0.0:${PORT:-7860} \
    --workers 1 --threads 8 --timeout 120
