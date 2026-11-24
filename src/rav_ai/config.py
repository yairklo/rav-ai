"""Paths and license policy."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DEFAULT_DB_URL = "postgresql+psycopg://rav:rav@localhost:54329/rav_ai"  # see docker-compose.yml
POLICY_PATH = ROOT / "config" / "license_policy.toml"


@dataclass(frozen=True)
class Permission:
    can_index: bool
    can_display_public: bool
    can_train: bool


class LicensePolicy:
    def __init__(self, table: dict[str, Permission]):
        self.table = table

    @classmethod
    def load(cls, path: Path = POLICY_PATH) -> LicensePolicy:
        with open(path, "rb") as f:
            raw = tomllib.load(f)["licenses"]
        table = {
            k.lower(): Permission(v["can_index"], v["can_display_public"], v["can_train"])
            for k, v in raw.items()
        }
        if "unknown" not in table:
            table["unknown"] = Permission(False, False, False)
        return cls(table)

    def evaluate(self, license_str: str | None) -> Permission:
        key = (license_str or "").strip().lower()
        return self.table.get(key, self.table["unknown"])
