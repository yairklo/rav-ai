"""Paths and license policy."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("RAV_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
DEFAULT_DB_URL = "postgresql+psycopg://rav:rav@localhost:54329/rav_ai"  # see docker-compose.yml
POLICY_PATH = ROOT / "config" / "license_policy.toml"
PRIMARY_RULES_PATH = ROOT / "config" / "primary_versions.toml"


@dataclass(frozen=True)
class Permission:
    can_index: bool
    can_display_public: bool
    can_train: bool


def _key(s: str | None) -> str:
    """Canonical license key: lowercase, whitespace/underscore/slash -> "-"."""
    return re.sub(r"[\s_/]+", "-", (s or "").strip().lower())


class LicensePolicy:
    """license string -> Permission. Exact/alias match first, then longest key-prefix match
    (so versioned or suffixed strings like "CC-BY-SA 3.0 Unported" resolve to cc-by-sa)."""

    def __init__(self, table: dict[str, Permission]):
        self.table = {_key(k): v for k, v in table.items()}
        self._by_len = sorted(self.table, key=len, reverse=True)

    @classmethod
    def load(cls, path: Path = POLICY_PATH) -> LicensePolicy:
        with open(path, "rb") as f:
            raw = tomllib.load(f)["licenses"]
        table: dict[str, Permission] = {}
        for name, v in raw.items():
            perm = Permission(v["can_index"], v["can_display_public"], v["can_train"])
            for k in (name, *v.get("aliases", [])):
                table[k] = perm
        table.setdefault("unknown", Permission(False, False, False))
        return cls(table)

    def match(self, license_str: str | None) -> str:
        """Policy key that applies to `license_str` ('unknown' if none)."""
        k = _key(license_str)
        if k in self.table:
            return k
        for cand in self._by_len:
            if (
                cand != "unknown"
                and k.startswith(cand)
                and (len(k) == len(cand) or k[len(cand)] == "-")
            ):
                return cand
        return "unknown"

    def evaluate(self, license_str: str | None) -> Permission:
        return self.table[self.match(license_str)]


def load_primary_rules(path: Path = PRIMARY_RULES_PATH) -> list[dict]:
    with open(path, "rb") as f:
        return tomllib.load(f).get("rule", [])
