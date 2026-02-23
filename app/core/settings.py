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

    # Agent Configuration Loading
    agent_config_load_type: Literal["local-config", "seedai-api"] = Field(
        default="local-config",
        description="Config loading mode: local-config or seedai-api",
    )
    seedai_api_url: str = Field(default="", description="SeedAI API URL for config loading")
    seedai_api_key: str = Field(default="", description="SeedAI API Key")

    # Default Model Configuration
    default_model_type: Literal["gpt", "claude", "ollama", "gemini", "grok", "friendli", "luxia"] = Field(
        default="gpt",
        description="Default LLM provider type",
    )

    # OpenAI Configuration
    openai_api_key: str = Field(default="", description="OpenAI API Key")
    openai_model: str = Field(default="gpt-4o", description="Default OpenAI model")
    openai_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    openai_max_tokens: int = Field(default=4096, ge=1)
    openai_base_url: str | None = Field(default=None, description="OpenAI API base URL")

    # Anthropic (Claude) Configuration
    anthropic_api_key: str = Field(default="", description="Anthropic API Key")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022", description="Default Claude model")

    # Ollama Configuration
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    ollama_model: str = Field(default="llama3.2", description="Default Ollama model")

    # Google (Gemini) Configuration
    google_api_key: str = Field(default="", description="Google API Key for Gemini")
    gemini_model: str = Field(default="gemini-1.5-pro", description="Default Gemini model")

    # xAI (Grok) Configuration
    xai_api_key: str = Field(default="", description="xAI API Key for Grok")
    grok_model: str = Field(default="grok-2-latest", description="Default Grok model")

    # Friendli Configuration
    friendli_api_key: str = Field(default="", description="Friendli API Key")
    friendli_model: str = Field(default="", description="Friendli model endpoint")

    # Luxia Configuration
    luxia_api_key: str = Field(default="", description="Luxia API Key")
    luxia_base_url: str = Field(default="", description="Luxia API base URL")
    luxia_model: str = Field(default="", description="Default Luxia model")

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
    log_file_path: str = Field(
        default="./logs/antigravity.log",
        description="Path for the rotating log file",
    )
    log_rotation_backup_count: int = Field(
        default=30,
        ge=1,
        description="Number of days to retain rotated log files",
    )
    tool_latency_warning_threshold_secs: float = Field(
        default=5.0,
        ge=0.1,
        description="Latency threshold in seconds above which a SLOW_TOOL warning is emitted",
    )
    log_sensitive_masking: bool = Field(
        default=True,
        description="Mask sensitive values (API keys, tokens) in log output",
    )

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
