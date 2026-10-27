"""Client for OpenAI-compatible chat completion APIs."""

import json
import logging

import requests

from app.config import settings

DEFAULT_SYSTEM_PROMPT = """
You are JARVIS, an advanced, autonomous AI developer assistant.
Your task is to analyze developer queries and return responses in structured JSON format with the following keys:
- "thought": A short explanation of your reasoning or plan.
- "speak": A brief, natural text summary to be read aloud via Text-to-Speech to the user.
- "action": The action type (e.g., "think", "terminal_command", "file_edit", "none").
- "payload": Details or parameters needed for the action (or empty dict if none).

Always ensure your output is valid JSON without codeblock markdown wrappers unless strictly needed.
""".strip()

logger = logging.getLogger(__name__)


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
            parsed_content = json.loads(content)
            if not isinstance(parsed_content, dict):
                return {"error": "LLM response must be a JSON object."}
            return parsed_content
        except requests.RequestException:
            logger.exception("LLM provider request failed")
            return {"error": "LLM provider request failed."}
        except json.JSONDecodeError:
            logger.exception("LLM response was not valid JSON")
            return {"error": "LLM response was not valid JSON."}
        except (KeyError, IndexError, TypeError, ValueError):
            logger.exception("LLM provider returned an invalid response")
            return {"error": "LLM provider returned an invalid response."}


llm_service = LLMService()
