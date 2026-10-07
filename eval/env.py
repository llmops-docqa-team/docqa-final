"""Load `KEY=VALUE` lines from the project's `.env` into the process environment (eval tools only).

The API reads real environment variables (docker compose passes `.env` in). The eval runner and the judge
script are started by hand, so they read `.env` themselves. Existing environment variables win. A line that
is blank, a comment, or has no `=` is ignored; surrounding quotes are stripped; values may contain spaces.
Values are never printed.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env_file(path: str | Path | None = None) -> list[str]:
    """Set variables from the file that are not already set. Returns the names that were set."""
    path = Path(path) if path else ROOT / ".env"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    names = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip().removeprefix("export ").strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and value and not os.environ.get(key):
            os.environ[key] = value
            names.append(key)
    return names
