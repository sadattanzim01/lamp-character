"""Loads the project's settings, mainly your Anthropic API key.

What this file does, in plain words:
    Secrets like the API key must never be written into the code or pushed
    to GitHub. Instead they live in a file called `.env` in the project folder
    (which .gitignore keeps off GitHub). It looks like:

        ANTHROPIC_API_KEY=sk-ant-...
        LAMP_MODEL=claude-haiku-5-5        # optional overrides, one per line

    When the brain starts, this file reads `.env` and copies each line into
    the program's environment variables, where the rest of the code can find
    them. Optional settings it can hold: LAMP_MODEL, LAMP_VISION_MODEL,
    LAMP_STT_MODEL and LAMP_VOICE (see the README's Configuration table).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # the lamp-character folder


def _load_dotenv(path: Path) -> None:
    """Read KEY=value lines from a .env file into the environment.

    Blank lines, comments (#) and lines without "=" are skipped, and quotes
    around values are removed. A value that's already set in the real
    environment wins over the file ("setdefault"), so you can override the
    file temporarily from the terminal.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)            # split on the first "=" only
        os.environ.setdefault(key.strip(), val.strip().strip("'\""))


# Runs once, the first time any part of the brain imports this file.
_load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Read-only settings for the rest of the brain (frozen = can't be changed by accident)."""
    anthropic_key: str = os.environ.get("ANTHROPIC_API_KEY", "")   # "" if missing


# Other files use it like:  from brain.config import settings;  settings.anthropic_key
settings = Settings()
