"""Shared Anthropic client, keyed from the repo-root .env (HAIKU_API_KEY or ANTHROPIC_API_KEY)."""
from __future__ import annotations

import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

_client = None


def _load_dotenv() -> None:
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def get_client(max_retries: int = 2):
    global _client
    if _client is None:
        import anthropic

        _load_dotenv()
        api_key = os.environ.get("HAIKU_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit(f"no HAIKU_API_KEY or ANTHROPIC_API_KEY in {ENV_PATH} or the environment")
        _client = anthropic.Anthropic(api_key=api_key, max_retries=max_retries)
    return _client
