"""
Central configuration loader for NexusFlow.

Loads config.yaml + .env and exposes a single Config object so every module
reads parameters from one place (Section 73).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, *keys: str, default: Any = None) -> Any:
        """Nested get, e.g. config.get('sequential_mining', 'min_support')."""
        node = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    @property
    def database_url(self) -> str:
        url = os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError(
                "DATABASE_URL is not set. Copy .env.example to .env and fill it in."
            )
        return url

    @property
    def random_seed(self) -> int:
        return int(self.get("project", "random_seed", default=42))

    def path(self, *parts: str) -> Path:
        return PROJECT_ROOT.joinpath(*parts)


def load_config(config_path: str | Path = PROJECT_ROOT / "config.yaml") -> Config:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw)


CONFIG = load_config()
