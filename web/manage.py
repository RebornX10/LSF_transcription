#!/usr/bin/env python
"""Django management entry point for the LSF interpreter web UI."""

import os
import sys
from pathlib import Path

# Make the project root importable so `import lsf` works from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Django is not installed. Run `pip install -r requirements.txt`."
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
