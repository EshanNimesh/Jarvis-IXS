"""Client for OpenAI-compatible chat completion APIs."""

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator

import httpx
import requests

from app.config import settings

DEFAULT_SYSTEM_PROMPT = """
You are JARVIS, a fast, ultra-concise voice assistant and developer assistant.
Respond in 1 to 2 short spoken sentences maximum unless explicitly asked for detail.
Never use markdown, lists, or URLs.
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

    def __init__(self) -> None:
        self._async_client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def start(self) -> None:
        """Create the shared async HTTP client during application startup."""
        async with self._client_lock:
            if self._async_client is None or self._async_client.is_closed:
                self._async_client = httpx.AsyncClient(timeout=30.0)

    async def close(self) -> None:
        """Close the shared client during application shutdown."""
        async with self._client_lock:
            client = self._async_client
            self._async_client = None
        if client is not None and not client.is_closed:
            await client.aclose()

    async def _get_async_client(self) -> httpx.AsyncClient:
        """Return the shared client, initializing it for non-lifespan use too."""
        async with self._client_lock:
            if self._async_client is None or self._async_client.is_closed:
                self._async_client = httpx.AsyncClient(timeout=30.0)
            return self._async_client

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

        client = await self._get_async_client()
        async with client.stream("POST", url, headers=headers, json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    yield f"{line}\n"

    async def query_llm_tokens(self, prompt: str) -> AsyncIterator[str]:
        """Yield just the content delta from each upstream streaming event."""
        if not settings.LLM_API_KEY:
            raise RuntimeError("LLM API key is not configured.")

        stream_started = time.perf_counter()
        first_token_seen = False
        token_count = 0

        url = f"{settings.LLM_BASE_URL.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {settings.LLM_API_KEY}"}
        payload = {
            "model": settings.LLM_MODEL,
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "temperature": 0,
        }
        client = await self._get_async_client()
        try:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                        token = event["choices"][0]["delta"].get("content")
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                        logger.debug("Skipping malformed LLM token event")
                        continue
                    if isinstance(token, str) and token:
                        token_count += 1
                        if not first_token_seen:
                            first_token_seen = True
                            first_token_ms = (time.perf_counter() - stream_started) * 1000
                            logger.info(
                                "[LATENCY] llm_first_token_ms=%.0f",
                                first_token_ms,
                            )
                            logger.info(
                                "[TIMING] Backend upstream LLM first token: +%.3fms from request",
                                first_token_ms,
                            )
                        yield token
        finally:
            logger.info(
                "[LATENCY] llm_stream_finished_ms=%.0f token_count=%d",
                (time.perf_counter() - stream_started) * 1000,
                token_count,
            )


llm_service = LLMService()
