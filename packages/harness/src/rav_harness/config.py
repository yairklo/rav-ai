"""Harness configuration: model choice and loop limits, from config/harness.toml + env."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from rav_ai.config import ROOT

CONFIG_PATH = ROOT / "config" / "harness.toml"


@dataclass(frozen=True)
class HarnessConfig:
    model: str = "anthropic/claude-sonnet-5-5"
    temperature: float = 0.0
    max_tokens: int = 4096
    max_steps: int = 12
    max_answer_retries: int = 2

    def with_model(self, model: str) -> HarnessConfig:
        return replace(self, model=model)


def load_config(path: Path | None = None) -> HarnessConfig:
    p = path or CONFIG_PATH
    cfg = HarnessConfig()
    if p.exists():
        with open(p, "rb") as f:
            raw = tomllib.load(f).get("harness", {})
        cfg = HarnessConfig(**{k: v for k, v in raw.items() if k in HarnessConfig.__annotations__})
    if env := os.environ.get("RAV_HARNESS_MODEL"):
        cfg = cfg.with_model(env)
    return cfg
