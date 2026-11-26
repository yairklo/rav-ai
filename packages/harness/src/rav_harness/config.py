"""Harness configuration: model choice, loop limits and the knowledge layer, from
config/harness.toml + env."""

from __future__ import annotations

import os
import re
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


PHRASE_KEYS = (
    "notice",
    "sefaria_label",
    "auto_link_label",
    "candidate_msg",
    "unchecked",
    "out_of_scope",
    "partial",
    "unusual",
    "no_disputant",
    "more_readings",
)
PATTERN_KEYS = (
    "girsa_patterns",
    "girsa_forbidden_patterns",
    "holds_patterns",
    "lens_mark_patterns",
    "maran_patterns",
    "sa_psak_patterns",
    "silence_patterns",
    "disputant_patterns",
    "layer_verified_patterns",
)


@dataclass(frozen=True)
class Phrases:
    """Knowledge-layer wording and patterns from ``[knowledge.phrases]``; the prompt and the
    validators read the same object. Built only by :func:`parse_phrases` (validated)."""

    notice: str
    sefaria_label: str
    auto_link_label: str
    candidate_msg: str
    unchecked: str
    out_of_scope: str
    partial: str
    unusual: str
    no_disputant: str
    more_readings: str
    practical: re.Pattern
    girsa: re.Pattern
    girsa_forbidden: re.Pattern
    holds: re.Pattern
    lens_mark: re.Pattern
    maran: re.Pattern
    sa_psak: re.Pattern
    silence: re.Pattern
    disputant: re.Pattern
    layer_verified: re.Pattern


def _compile(key: str, pattern: str) -> re.Pattern:
    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError(f"[knowledge.phrases] {key}: empty pattern")
    try:
        return re.compile(pattern)
    except re.error as e:
        raise ValueError(f"[knowledge.phrases] {key}: bad regex {pattern!r}: {e}") from e


def parse_phrases(raw: dict) -> Phrases:
    """Validate ``[knowledge.phrases]``: every phrase non-empty, every regex compiles."""
    kw: dict = {}
    for k in PHRASE_KEYS:
        v = raw.get(k)
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"[knowledge.phrases] {k}: missing or empty")
        kw[k] = v
    kw["practical"] = _compile("practical_regex", raw.get("practical_regex", ""))
    for k in PATTERN_KEYS:
        pats = raw.get(k)
        if not isinstance(pats, list) or not pats:
            raise ValueError(f"[knowledge.phrases] {k}: missing or empty list")
        for p in pats:
            _compile(k, p)  # each alternative on its own, so the error names it
        kw[k.removesuffix("_patterns")] = _compile(k, "|".join(f"(?:{p})" for p in pats))
    return Phrases(**kw)


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
    # required (and validated) when knowledge != "off"; load_config() reads it from the toml
    phrases: Phrases | None = None

    def with_model(self, model: str) -> HarnessConfig:
        return replace(self, model=model)

    def with_knowledge(self, knowledge: str) -> HarnessConfig:
        if knowledge not in KNOWLEDGE_MODES:
            raise ValueError(f"knowledge must be one of {KNOWLEDGE_MODES}, got {knowledge!r}")
        if knowledge != "off" and self.phrases is None:
            raise ValueError("knowledge on needs [knowledge.phrases] in config/harness.toml")
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
        kw = {
            k: v
            for k, v in raw.items()
            if k in HarnessConfig.__annotations__ and k not in ("scope", "phrases")
        }
        raw_phrases = data.get("knowledge", {}).get("phrases")
        cfg = HarnessConfig(
            **kw,
            scope=_scope(data.get("scope", {})),
            phrases=parse_phrases(raw_phrases) if raw_phrases is not None else None,
        )
        cfg.with_knowledge(cfg.knowledge)  # validates
    if env := os.environ.get("RAV_HARNESS_MODEL"):
        cfg = cfg.with_model(env)
    return cfg
