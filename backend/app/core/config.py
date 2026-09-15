from functools import lru_cache
from pathlib import Path
from enum import StrEnum
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderKind(StrEnum):
    OPENAI_COMPATIBLE = "openai_compatible"
    OLLAMA = "ollama"


class ProviderConfig(BaseModel):
    """Connection settings for one upstream provider.

    This model is populated from the JSON object in `NOMINAL_PROVIDER_CONFIGS`.
    `SecretStr` prevents an API key from appearing in a settings representation.
    """

    model_config = ConfigDict(extra="forbid")

    kind: ProviderKind = ProviderKind.OPENAI_COMPATIBLE
    base_url: str
    api_key: SecretStr | None = None
    timeout_seconds: PositiveFloat = 30.0

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute http(s) URL")
        return value.rstrip("/")


class Settings(BaseSettings):
    """Runtime configuration sourced from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="NOMINAL_",
        extra="ignore",
    )

    app_name: str = "NOMINAL"
    app_version: str = "0.1.0"
    app_env: str = "development"
    log_level: str = "INFO"
    database_url: str = "sqlite:///./nominal.db"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080,http://127.0.0.1:8080"
    default_model: str = "economy-gpt-4.1-mini"
    baseline_model: str = "frontier-o3"
    benchmark_frontier_model: str = "frontier-o3"
    benchmark_economy_model: str = "economy-gpt-4.1-mini"
    profiler_llm_enabled: bool = False
    quality_judge_enabled: bool = False
    max_quality_escalations: int = Field(default=1, ge=0, le=3)
    max_provider_failovers: int = Field(default=1, ge=0, le=3)
    provider_configs: dict[str, ProviderConfig] = Field(default_factory=dict)
    model_catalog_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "config" / "models.yaml"
    )
    benchmark_dataset_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "benchmarks" / "dataset.json"
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        if not value.startswith("sqlite:///"):
            raise ValueError("the MVP supports SQLite database URLs only")
        return value

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, value: str) -> str:
        origins = [origin.strip() for origin in value.split(",") if origin.strip()]
        if not origins:
            raise ValueError("at least one CORS origin is required")
        for origin in origins:
            parsed = urlparse(origin)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("CORS origins must be absolute http(s) URLs")
        return ",".join(origins)

    @field_validator("model_catalog_path", "benchmark_dataset_path")
    @classmethod
    def validate_required_file(cls, value: Path) -> Path:
        if not value.is_file():
            raise ValueError(f"required file does not exist: {value}")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
