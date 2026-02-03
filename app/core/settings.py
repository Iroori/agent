"""Application settings using pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI Configuration
    openai_api_key: str = Field(default="", description="OpenAI API Key")
    openai_model: str = Field(default="gpt-4o", description="Default OpenAI model")
    openai_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    openai_max_tokens: int = Field(default=4096, ge=1)

    # Agent Pool Configuration
    agent_pool_max_size: int = Field(default=100, ge=1)
    agent_idle_timeout_seconds: int = Field(default=3600, ge=60)

    # Server Configuration
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    debug: bool = Field(default=False)

    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )
    log_format: Literal["json", "text"] = Field(default="json")

    # Agent Configuration Directory
    agents_config_dir: str = Field(default="./agents_config")

    # Retry Configuration
    max_retries: int = Field(default=3, ge=1)
    retry_base_delay: float = Field(default=1.0, ge=0.1)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
