"""
config.py
=========

Centralized application configuration.

Why this file exists
---------------------
Hardcoding API keys, base URLs, and default models directly inside
business logic makes a service impossible to reconfigure without a code
change and a redeploy. Pulling everything through a single Settings
object (backed by environment variables / a .env file) means:

- secrets never live in source code
- swapping providers or default models is a config change, not a code change
- tests can override settings cleanly without monkeypatching modules

We use pydantic-settings so values are validated and typed at startup
(fail fast if OPENROUTER_API_KEY is missing, rather than failing deep
inside an HTTP call later).
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Strongly-typed application settings, loaded from environment
    variables (or a local .env file during development).

    Attributes:
        openrouter_api_key: API key for OpenRouter (https://openrouter.ai).
        openrouter_base_url: OpenRouter's OpenAI-compatible API base URL.
        app_referrer: Optional "HTTP-Referer" header OpenRouter uses for
            attributing usage on their leaderboard. Not required to function.
        app_title: Optional "X-Title" header, same purpose as above.
        request_timeout_seconds: Hard timeout for a single LLM call. This
            exists so a hung provider request cannot hold a connection
            (and, indirectly, resources) forever.
        default_max_tokens: Fallback max_tokens value if a request doesn't
            specify one.
    """

    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    app_referrer: str = "http://localhost"
    app_title: str = "AI Gateway - Week 1 Project"

    request_timeout_seconds: float = 60.0
    default_max_tokens: int = 1024

    # pydantic-settings config: read from a .env file if present, ignore
    # unrelated environment variables instead of raising on them.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    Using lru_cache turns this into a cheap singleton: Settings() parses
    and validates environment variables once per process, and every
    caller (FastAPI dependency injection, module-level code, tests)
    gets the same validated object instead of re-parsing the environment
    on every call.
    """
    return Settings()
