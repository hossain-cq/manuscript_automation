from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# pydantic-settings reads .env into this Settings object only. The ported
# subgraph modules (graphs/subgraphs/*.py) each read their own per-role model
# env vars (e.g. LITERATURE_MODEL, ASSESSMENT_MODEL) via bare os.getenv(...)
# calls that predate this settings module. load_dotenv() here puts .env
# values into the actual process environment so those calls see them too -
# otherwise .env would only be a single source of truth for the fields below.
load_dotenv()


class Settings(BaseSettings):
    """Reads .env once. See .env.example for the supported keys."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str | None = None
    openai_api_base: str | None = None

    database_path: str = "./projects/managed/manuscript_system.sqlite"
    checkpoint_path: str = "./projects/managed/checkpoints.sqlite"
    artifact_store_path: str = "./projects/managed/artifacts"
    configs_path: str = "./configs"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
