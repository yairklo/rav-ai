"""config/known_gaps.toml: external source gaps (text we know exists elsewhere but cannot ingest).

Loaded and schema-validated here so the ingest report and the knowledge layer (KL-1/KL-4, for
``kl_coverage``) read one source of truth. Fail closed: a malformed file raises.
"""

from __future__ import annotations

import datetime as dt
import tomllib
from dataclasses import dataclass
from pathlib import Path

from rav_ai.config import ROOT

GAPS_PATH = ROOT / "config" / "known_gaps.toml"
STATUSES = ("partial_external", "absent_external")
_REQUIRED = ("work", "scope", "status", "source", "revid", "checked", "license", "ingested", "note")


class KnownGapsError(ValueError):
    pass


@dataclass(frozen=True)
class KnownGap:
    work: str
    scope: str
    status: str
    source: str
    revid: int
    checked: dt.date
    license: str
    ingested: bool
    note: str
    page: str | None = None


def parse(raw: dict) -> list[KnownGap]:
    gaps = raw.get("gap")
    if not isinstance(gaps, list) or not gaps:
        raise KnownGapsError("known_gaps: need at least one [[gap]] table")
    out = []
    for i, g in enumerate(gaps):
        where = f"gap #{i + 1} ({g.get('work', '?')})"
        missing = [k for k in _REQUIRED if k not in g]
        if missing:
            raise KnownGapsError(f"{where}: missing {missing}")
        unknown = set(g) - set(_REQUIRED) - {"page"}
        if unknown:
            raise KnownGapsError(f"{where}: unknown keys {sorted(unknown)}")
        if g["status"] not in STATUSES:
            raise KnownGapsError(f"{where}: status must be one of {STATUSES}")
        if not isinstance(g["revid"], int) or isinstance(g["revid"], bool) or g["revid"] < 0:
            raise KnownGapsError(f"{where}: revid must be a non-negative integer")
        if g["source"].startswith("he.wikisource") and g["revid"] <= 0:
            raise KnownGapsError(f"{where}: a wikisource gap needs a pinned revid")
        if not isinstance(g["ingested"], bool):
            raise KnownGapsError(f"{where}: ingested must be a boolean")
        try:
            checked = dt.date.fromisoformat(g["checked"])
        except (TypeError, ValueError) as e:
            raise KnownGapsError(f"{where}: checked must be an ISO date") from e
        for k in ("work", "scope", "source", "license", "note"):
            if not isinstance(g[k], str) or not g[k].strip():
                raise KnownGapsError(f"{where}: {k} must be a non-empty string")
        out.append(KnownGap(**{**g, "checked": checked}))
    return out


def load(path: Path = GAPS_PATH) -> list[KnownGap]:
    with open(path, "rb") as f:
        return parse(tomllib.load(f))
