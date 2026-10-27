"""Application settings loaded from the environment and ``.env``."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    port: int = 80
    env: str = "development"
    meta_api_key: str = ""
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.together.xyz/v1"
    LLM_MODEL: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
