"""Django settings for the LSF interpreter UI.

Browser-camera architecture: the server processes pushed frames, so this runs
unchanged locally and on a hosted Space. Config is env-overridable for
deployment. No database is used.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent          # the `web/` dir
PROJECT_ROOT = BASE_DIR.parent                             # repo root


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


# Learned templates + vocabulary (writable; falls back to in-memory if read-only).
TEMPLATES_PATH = os.environ.get("LSF_TEMPLATES_PATH", str(PROJECT_ROOT / "signs" / "lsf_signs.json"))
LEXICON_PATH = os.environ.get("LSF_LEXICON_PATH", str(PROJECT_ROOT / "signs" / "lexicon.json"))
MODEL_PATH = os.environ.get("LSF_MODEL_PATH", str(PROJECT_ROOT / "models" / "model.h5"))

# MediaPipe speed/accuracy knobs. Defaults are "max speed" for CPU hosts.
MODEL_COMPLEXITY = int(os.environ.get("LSF_MODEL_COMPLEXITY", "0"))     # 0 fastest, 2 best
REFINE_FACE_LANDMARKS = _flag("LSF_REFINE_FACE", "0")                   # iris detail (slower)

SECRET_KEY = os.environ.get("LSF_SECRET_KEY", "dev-only-not-secret-change-me")
DEBUG = _flag("LSF_DEBUG", "1")
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "interpreter",
]

MIDDLEWARE = [
    "django.middleware.common.CommonMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
]

ROOT_URLCONF = "web.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": []},
    },
]

WSGI_APPLICATION = "web.wsgi.application"

DATABASES = {}

STATIC_URL = "static/"
STATIC_ROOT = str(BASE_DIR / "staticfiles")
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "whitenoise.storage.CompressedStaticFilesStorage"},
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
