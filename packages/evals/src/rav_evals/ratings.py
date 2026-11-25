"""Human ratings, versioned files in the repo: ``golden/kl/ratings/<rater>.<queue>.jsonl``.

Append-only events; the latest event for a (rater, key) wins. Nothing here is ever filled in
by a model: a value nobody rated stays absent and is reported as pending.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .config import RATINGS_DIR

Queue = Literal["answers", "pairs", "items"]

ERROR_TYPES = (
    "wrong_attribution",
    "unsupported_claim",
    "missed_position",
    "candidate_as_proof",
    "invented_error",
    "missing_coverage_notice",
    "sa_overweighted",
    "snif_filtered",
    "bad_paraphrase",
    "other",
)
ANSWER_GRADES = ("ok", "minor", "major")
_RATER_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def check_rater(rater_id: str) -> str:
    if not _RATER_RE.match(rater_id):
        raise ValueError(f"rater_id must match {_RATER_RE.pattern}: {rater_id!r}")
    return rater_id


class RatingStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or RATINGS_DIR)

    def path(self, rater_id: str, queue: Queue) -> Path:
        return self.root / f"{check_rater(rater_id)}.{queue}.jsonl"

    def append(self, rater_id: str, queue: Queue, record: dict[str, Any]) -> dict[str, Any]:
        rec = {"rater_id": rater_id, "ts": now_iso(), **record}
        p = self.path(rater_id, queue)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        return rec

    def events(self, queue: Queue, rater_id: str | None = None) -> Iterator[dict[str, Any]]:
        if not self.root.exists():
            return
        pattern = f"{rater_id}.{queue}.jsonl" if rater_id else f"*.{queue}.jsonl"
        for p in sorted(self.root.glob(pattern)):
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    yield json.loads(line)

    def latest(
        self, queue: Queue, key: str, rater_id: str | None = None
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """{(rater_id, record[key]): latest record}."""
        out: dict[tuple[str, str], dict[str, Any]] = {}
        for e in self.events(queue, rater_id):
            if key in e:
                out[(e["rater_id"], str(e[key]))] = e
        return out
