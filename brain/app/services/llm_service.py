"""Client for OpenAI-compatible chat completion APIs."""

import logging

import requests

from app.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Generate assistant responses using the configured LLM provider."""

    def generate_response(self, prompt: str, system_prompt: str = "") -> dict[str, object]:
        """Return assistant content, or a structured error when unavailable."""
        if not settings.LLM_API_KEY:
            return {"error": "LLM API key is not configured."}

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

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
            return {"content": content}
        except requests.RequestException:
            logger.exception("LLM provider request failed")
            return {"error": "LLM provider request failed."}
        except (KeyError, IndexError, TypeError, ValueError):
            logger.exception("LLM provider returned an invalid response")
            return {"error": "LLM provider returned an invalid response."}


llm_service = LLMService()
