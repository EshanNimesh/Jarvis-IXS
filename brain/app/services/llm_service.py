"""Client for OpenAI-compatible chat completion APIs."""

import json
import logging
import re
from collections.abc import AsyncIterator

import httpx
import requests

from app.config import settings

DEFAULT_SYSTEM_PROMPT = """
You are JARVIS, an advanced, autonomous AI developer assistant.
Your output MUST be valid JSON with keys in this order: {"speak": "...", "thought": "...", "action": "none", "payload": {}}.

CRITICAL FOR VOICE LATENCY:
- CRITICAL: Place the 'speak' key as the FIRST field in your JSON response so streaming begins immediately.
- Keep the "speak" value extremely concise (1 to 2 short sentences, maximum 20 words).
- Save detailed code, logs, or lengthy explanations for the "payload" or "thought" fields.
- CRITICAL: The 'action' field MUST be strictly set to 'none' unless explicitly performing terminal or file edits. Never output invented actions like 'clarify' or 'answer_question'.
""".strip()

logger = logging.getLogger(__name__)


def _strip_markdown_fence(text: str) -> str:
    """Remove an optional Markdown code fence around model JSON."""
    cleaned = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else cleaned


class LLMService:
    """Generate assistant responses using the configured LLM provider."""

    def generate_response(self, prompt: str, system_prompt: str = "") -> dict[str, object]:
        """Return assistant content, or a structured error when unavailable."""
        if not settings.LLM_API_KEY:
            return {"error": "LLM API key is not configured."}

        effective_system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": effective_system_prompt},
            {"role": "user", "content": prompt},
        ]

        url = f"{settings.LLM_BASE_URL.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        body = {"model": settings.LLM_MODEL, "messages": messages}

        try:
            response = requests.post(url, headers=headers, json=body, timeout=60)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                return {"error": "LLM provider returned an invalid response."}
            raw_text = content
            parsed_content = json.loads(_strip_markdown_fence(raw_text))
            if not isinstance(parsed_content, dict):
                return {
                    "thought": "Failed to parse JSON",
                    "speak": raw_text,
                    "action": "none",
                    "payload": {},
                }
            return parsed_content
        except requests.RequestException:
            logger.exception("LLM provider request failed")
            return {"error": "LLM provider request failed."}
        except json.JSONDecodeError:
            logger.exception("LLM response was not valid JSON")
            return {
                "thought": "Failed to parse JSON",
                "speak": content if isinstance(content, str) else "",
                "action": "none",
                "payload": {},
            }
        except (KeyError, IndexError, TypeError, ValueError):
            logger.exception("LLM provider returned an invalid response")
            return {"error": "LLM provider returned an invalid response."}

    async def query_llm_stream(self, prompt: str) -> AsyncIterator[str]:
        """Pass provider SSE lines through as soon as they arrive."""
        if not settings.LLM_API_KEY:
            raise RuntimeError("LLM API key is not configured.")

        url = f"{settings.LLM_BASE_URL.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {settings.LLM_API_KEY}"}
        body = {
            "model": settings.LLM_MODEL,
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "temperature": 0,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("POST", url, headers=headers, json=body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        yield f"{line}\n"


llm_service = LLMService()
