"""Which code produced a request: `DOCQA_GIT_SHA` (set in Docker, where there is no .git), else the short
commit hash of the checkout, else "unknown". Looked up once per process."""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache

from app.config import ROOT


@lru_cache(maxsize=1)
def get_app_version() -> str:
    env = os.environ.get("DOCQA_GIT_SHA", "").strip()
    if env:
        return env[:40]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=3
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown" if out.returncode == 0 else "unknown"
