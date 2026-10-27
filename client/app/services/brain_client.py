"""HTTP client for the JARVIS Brain API."""

import json
import re
from typing import Any

import requests

from app.config import settings


def _parse_structured_response(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raw_text = value if isinstance(value, str) else str(value)
    cleaned = raw_text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {
        "thought": "Failed to parse JSON",
        "speak": raw_text,
        "action": "none",
        "payload": {},
    }


class BrainClient:
    def __init__(self, timeout: float = 60.0) -> None:
        self.timeout = timeout

    def query(self, prompt: str, context: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
        response = requests.post(
            f"{settings.brain_url.rstrip('/')}/api/v1/query",
            json={"prompt": prompt, "context": context or {}},
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        structured = _parse_structured_response(result.get("response", {}))
        speak_text = structured.get("speak", "")
        action = {
            "action": structured.get("action", "none"),
            "payload": structured.get("payload", {}),
        }
        if not isinstance(speak_text, str):
            speak_text = str(speak_text)
        if not isinstance(action["payload"], dict):
            action["payload"] = {}
        return speak_text, action
