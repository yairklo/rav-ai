"""Harness configuration: model choice, loop limits and the knowledge layer, from
config/harness.toml + env."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

from rav_ai.config import ROOT

CONFIG_PATH = ROOT / "config" / "harness.toml"
KNOWLEDGE_MODES = ("off", "base", "base+lens")


@dataclass(frozen=True)
class ScopeRule:
    """kind=daf: ``work`` amudim ``from``..``to`` ("21a".."24b"); kind=number: chapters/simanim."""

    kind: str
    work: str
    lo: str | int
    hi: str | int


@dataclass(frozen=True)
class HarnessConfig:
    model: str = "anthropic/claude-sonnet-5-5"
    temperature: float = 0.0
    max_tokens: int = 4096
    max_steps: int = 12
    max_answer_retries: int = 2
    knowledge: str = "off"
    min_confidence: float = 0.5
    allow_personal_only: bool = False
    scope: tuple[ScopeRule, ...] = field(default=())

    def with_model(self, model: str) -> HarnessConfig:
        return replace(self, model=model)

    def with_knowledge(self, knowledge: str) -> HarnessConfig:
        if knowledge not in KNOWLEDGE_MODES:
            raise ValueError(f"knowledge must be one of {KNOWLEDGE_MODES}, got {knowledge!r}")
        return replace(self, knowledge=knowledge)


def _scope(raw: dict) -> tuple[ScopeRule, ...]:
    rules = []
    for r in raw.get("rules", []):
        rules.append(ScopeRule(r["kind"], r["work"], r["from"], r["to"]))
    return tuple(rules)


def load_config(path: Path | None = None) -> HarnessConfig:
    p = path or CONFIG_PATH
    cfg = HarnessConfig()
    if p.exists():
        with open(p, "rb") as f:
            data = tomllib.load(f)
        raw = data.get("harness", {})
        kw = {k: v for k, v in raw.items() if k in HarnessConfig.__annotations__ and k != "scope"}
        cfg = HarnessConfig(**kw, scope=_scope(data.get("scope", {})))
        cfg.with_knowledge(cfg.knowledge)  # validates
    if env := os.environ.get("RAV_HARNESS_MODEL"):
        cfg = cfg.with_model(env)
    return cfg
