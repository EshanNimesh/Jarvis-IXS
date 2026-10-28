"""Client configuration loaded from environment variables and .env."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    brain_url: str = "http://85.211.242.86"
    tts_voice: str = "am_michael"
    kokoro_model_path: Path = Path(__file__).resolve().parents[1] / "models" / "kokoro-v1.0.onnx"
    kokoro_voices_path: Path = Path(__file__).resolve().parents[1] / "models" / "voices-v1.0.bin"
    allow_file_edits: bool = False
    allow_terminal_commands: bool = False
    workspace_dir: Path = Path.cwd()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
