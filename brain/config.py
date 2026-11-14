"""Settings, loaded from environment variables and lamp-character/.env.

The .env file is git-ignored; it holds ANTHROPIC_API_KEY and optional
overrides (LAMP_MODEL, LAMP_STT_MODEL, LAMP_VOICE).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip("'\""))


_load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    anthropic_key: str = os.environ.get("ANTHROPIC_API_KEY", "")


settings = Settings()
