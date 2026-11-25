"""Prompts live in files next to this module; version + content hash identify them in kl_jobs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

PROMPT_VERSION = "kl1.0"
_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Prompt:
    stage: str
    version: str
    text: str
    hash: str


@cache
def load_prompt(stage: str) -> Prompt:
    text = (_DIR / f"{stage}.txt").read_text(encoding="utf-8")
    common = (_DIR / "common.txt").read_text(encoding="utf-8")
    full = common.strip() + "\n\n" + text.strip()
    return Prompt(stage, PROMPT_VERSION, full, hashlib.sha256(full.encode()).hexdigest())
