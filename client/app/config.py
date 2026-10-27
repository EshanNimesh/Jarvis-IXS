"""Client configuration loaded from environment variables and .env."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    brain_url: str = "http://85.211.242.86"
    tts_voice: str = "en-US-GuyNeural"
    allow_file_edits: bool = False
    allow_terminal_commands: bool = False
    workspace_dir: Path = Path.cwd()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
