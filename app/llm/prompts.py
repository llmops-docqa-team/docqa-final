"""Load versioned prompts from prompts/*.yaml (each file has `name`, `version`, `system`, `user`)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import yaml

from app.config import ROOT

PROMPT_DIR = ROOT / "prompts"
_PLACEHOLDER = re.compile(r"\{(\w+)\}")


@dataclass(frozen=True)
class Prompt:
    id: str  # file stem, e.g. "answer_doc_v1": what config.yaml selects
    name: str
    version: str
    system: str
    user: str

    def render_user(self, **values: str) -> str:
        """Fill `{name}` placeholders in a single pass, so a value that itself contains `{other}` is
        never substituted again (document text and questions are untrusted)."""
        return _PLACEHOLDER.sub(lambda m: str(values.get(m.group(1), m.group(0))), self.user)


@lru_cache(maxsize=None)
def load_prompt(prompt_id: str) -> Prompt:
    path = PROMPT_DIR / f"{prompt_id}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Prompt(
        id=prompt_id,
        name=data["name"],
        version=str(data["version"]),
        system=data["system"].strip(),
        user=data["user"].strip(),
    )
