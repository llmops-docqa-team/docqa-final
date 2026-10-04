"""Pull one JSON object out of an LLM reply (shared by the router and the document answerer)."""

from __future__ import annotations

import json
import re


def load_json_object(content: str) -> dict:
    """Parse the model's reply as a JSON object. Tolerates a ```json fence or prose around one object.

    Raises ValueError (json.JSONDecodeError is one) when there is no object to be found."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (content or "").strip(), flags=re.IGNORECASE)
    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object in the reply") from None
        data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("the reply is not a JSON object")
    return data
