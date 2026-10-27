"""HTTP client for the JARVIS Brain API."""

import json
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

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
        self.last_stream_result: dict[str, Any] = {}
        self._websocket = None

    async def connect(self) -> None:
        """Open the persistent Brain WebSocket connection if needed."""
        if self._websocket is not None and self._websocket.close_code is None:
            return
        import websockets

        parsed = urlsplit(settings.brain_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        uri = urlunsplit((scheme, parsed.netloc, "/ws/chat", "", ""))
        self._websocket = await websockets.connect(
            uri,
            open_timeout=10,
            ping_interval=20,
            ping_timeout=20,
            max_size=None,
        )

    async def close(self) -> None:
        if self._websocket is not None:
            await self._websocket.close()
            self._websocket = None

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

    async def query_brain_stream(self, prompt: str):
        """Send a prompt and yield decoded speak text from WebSocket token frames."""
        raw_response = ""
        speak_started = False
        speak_finished = False
        scan_position = 0
        escaped = False
        unicode_mode = False
        unicode_digits = ""
        await self.connect()
        await self._websocket.send(prompt)
        async for message in self._websocket:
            token = message
            try:
                event = json.loads(message)
            except json.JSONDecodeError:
                event = None
            if isinstance(event, dict):
                if event.get("event") == "done":
                    break
                if event.get("event") == "error":
                    raise RuntimeError(f"Brain streaming error: {event.get('message', 'unknown error')}")
                token = event.get("text", event.get("token", ""))
            elif isinstance(event, str):
                token = event

            if not isinstance(token, str) or not token:
                continue
            raw_response += token
            if not speak_started:
                match = re.search(r'(?<!\\)"speak"\s*:\s*"', raw_response)
                if match:
                    speak_started = True
                    scan_position = match.end()
            if not speak_started or speak_finished:
                continue

            decoded: list[str] = []
            while scan_position < len(raw_response):
                char = raw_response[scan_position]
                scan_position += 1
                if unicode_mode:
                    unicode_digits += char
                    if len(unicode_digits) == 4:
                        try:
                            decoded.append(chr(int(unicode_digits, 16)))
                        except ValueError:
                            decoded.append("�")
                        unicode_mode = False
                        unicode_digits = ""
                    continue
                if escaped:
                    escaped = False
                    if char == "u":
                        unicode_mode = True
                        unicode_digits = ""
                    else:
                        decoded.append({
                            '"': '"', "\\": "\\", "/": "/",
                            "n": "\n", "r": "\r", "t": "\t",
                            "b": "\b", "f": "\f",
                        }.get(char, char))
                    continue
                if char == "\\":
                    escaped = True
                elif char == '"':
                    speak_finished = True
                    break
                else:
                    decoded.append(char)
            if decoded:
                yield "".join(decoded)
        self.last_stream_result = _parse_structured_response(raw_response)
