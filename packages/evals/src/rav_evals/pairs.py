"""Gold of ~200 pairs (research 5, nidon unification): sampling and agreement.

Labels: same_as, sub_issue_of, derives_from_yesod, depends_on, similar_to, unrelated. The three
middle ones are directional; a rating carries ``direction`` ("ab" = A -> B, "ba" = B -> A)
relative to the stored ``a_ref`` / ``b_ref`` (never the displayed order).

Unit: today a pair of segment ranges (``unit_kind="segment_range"``) - issue items do not exist
yet. After KL-4: issue vs issue, same record format with ``unit_kind="issue"``.

Balanced sample: ~1/3 explicit links (Mesorat HaShas / Ein Mishpat, as *candidates*), ~1/3
lexical (FTS) neighbours in other works, ~1/3 hard negatives (curated pairs from config + FTS
neighbours in the same work outside the slice: same wording, likely another nidon). Labels are
never filled in here: ``ratings`` starts empty.

Record: pair_id, unit_kind, a_ref, b_ref, source_bucket, swapped, dup_of, ratings[].
Rating: rater_id, label, direction, confidence 1-3, note, ts (+ recheck for self-agreement).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from .golden import is_placeholder

LABELS = ("same_as", "sub_issue_of", "derives_from_yesod", "depends_on", "similar_to", "unrelated")
DIRECTIONAL = frozenset({"sub_issue_of", "derives_from_yesod", "depends_on"})
BUCKETS = ("explicit", "neighbor", "hard_negative")
UnitKind = Literal["segment_range", "issue"]


def pair_id(a_ref: str, b_ref: str) -> str:
    lo, hi = sorted((a_ref, b_ref))
    return "p-" + hashlib.sha1(f"{lo}\x1f{hi}".encode()).hexdigest()[:10]


@dataclass(frozen=True)
class PairCandidate:
    a_ref: str
    b_ref: str
    note: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return tuple(sorted((self.a_ref, self.b_ref)))  # type: ignore[return-value]


class PairSource(Protocol):
    def explicit(self) -> list[PairCandidate]: ...
    def neighbors(self) -> list[PairCandidate]: ...
    def hard_negatives(self) -> list[PairCandidate]: ...


@dataclass
class SampleResult:
    records: list[dict[str, Any]]
    shortfall: dict[str, int] = field(default_factory=dict)

    def bucket_counts(self) -> dict[str, int]:
        c = Counter(r["source_bucket"] for r in self.records if r["dup_of"] is None)
        return {b: c.get(b, 0) for b in BUCKETS}

    def summary(self) -> dict[str, Any]:
        base = [r for r in self.records if r["dup_of"] is None]
        return {
            "pairs": len(base),
            "records": len(self.records),
            "duplicates": len(self.records) - len(base),
            "swapped": sum(r["swapped"] for r in base),
            "by_bucket": self.bucket_counts(),
            "shortfall": self.shortfall,
        }


def sample_pairs(
    source: PairSource,
    n: int = 200,
    seed: int = 0,
    *,
    duplicate_fraction: float = 0.10,
    curated: Iterable[dict[str, str]] = (),
    unit_kind: UnitKind = "segment_range",
) -> SampleResult:
    """Deterministic for a given (source content, n, seed). A/B order is randomly swapped (and
    stored), and ``round(n * duplicate_fraction)`` pairs are shown a second time in reversed
    order (``dup_of``) for the order-stability check."""
    rng = random.Random(seed)
    cur = [
        PairCandidate(c["a_ref"], c["b_ref"], c.get("note", ""))
        for c in curated
        if not (is_placeholder(c.get("a_ref", "TODO")) or is_placeholder(c.get("b_ref", "TODO")))
    ]
    pools = {
        "explicit": source.explicit(),
        "neighbor": source.neighbors(),
        "hard_negative": cur + source.hard_negatives(),
    }
    seen: set[tuple[str, str]] = set()
    clean: dict[str, list[PairCandidate]] = {}
    for b in BUCKETS:  # a pair belongs to the first bucket that offers it
        uniq = []
        for c in sorted(pools[b], key=lambda c: (c.key, c.note)):
            if c.a_ref != c.b_ref and c.key not in seen:
                seen.add(c.key)
                uniq.append(c)
        clean[b] = uniq
    # curated hard negatives come first, the rest shuffled
    curated_keys = {c.key for c in cur}
    for b in BUCKETS:
        head = [c for c in clean[b] if c.key in curated_keys]
        tail = [c for c in clean[b] if c.key not in curated_keys]
        rng.shuffle(tail)
        clean[b] = head + tail

    target = {b: n // 3 for b in BUCKETS}
    for b in BUCKETS[: n % 3]:
        target[b] += 1
    chosen: list[tuple[str, PairCandidate]] = []
    shortfall: dict[str, int] = {}
    for b in BUCKETS:
        take = clean[b][: target[b]]
        clean[b] = clean[b][target[b] :]
        chosen += [(b, c) for c in take]
        if len(take) < target[b]:
            shortfall[b] = target[b] - len(take)
    missing = sum(shortfall.values())
    for b in BUCKETS:  # fill a short bucket from the others (keeping their own label)
        while missing and clean[b]:
            chosen.append((b, clean[b].pop(0)))
            missing -= 1

    records = []
    for b, c in chosen:
        records.append(
            {
                "pair_id": pair_id(c.a_ref, c.b_ref),
                "unit_kind": unit_kind,
                "a_ref": c.a_ref,
                "b_ref": c.b_ref,
                "source_bucket": b,
                "swapped": rng.random() < 0.5,
                "dup_of": None,
                "note": c.note,
                "ratings": [],
            }
        )
    k = round(len(records) * duplicate_fraction)
    for orig in rng.sample(records, k) if k else []:
        records.append(
            {
                **orig,
                "pair_id": orig["pair_id"] + "-dup",
                "dup_of": orig["pair_id"],
                "swapped": not orig["swapped"],
                "ratings": [],
            }
        )
    rng.shuffle(records)
    return SampleResult(records, shortfall)


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---- agreement ----------------------------------------------------------------------------


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa; None when undefined (no items, or both raters used one same class)."""
    n = len(a)
    if n == 0 or n != len(b):
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def bootstrap_ci(
    a: list[str], b: list[str], reps: int = 1000, seed: int = 0, level: float = 0.95
) -> tuple[float, float] | None:
    n = len(a)
    if n == 0:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        k = cohen_kappa([a[i] for i in idx], [b[i] for i in idx])
        if k is not None:
            vals.append(k)
    if not vals:
        return None
    vals.sort()
    lo = vals[int((1 - level) / 2 * (len(vals) - 1))]
    hi = vals[int((1 + level) / 2 * (len(vals) - 1))]
    return (lo, hi)


def confusion(a: list[str], b: list[str], labels: Iterable[str] = LABELS) -> dict[str, dict]:
    m = {x: {y: 0 for y in labels} for x in labels}
    for x, y in zip(a, b, strict=True):
        m.setdefault(x, {y2: 0 for y2 in labels}).setdefault(y, 0)
        m[x][y] += 1
    return m


def per_class(a: list[str], b: list[str], labels: Iterable[str] = LABELS) -> dict[str, dict]:
    """Precision / recall of rater B against rater A as reference; None for an empty class."""
    out = {}
    for c in labels:
        tp = sum(x == c and y == c for x, y in zip(a, b, strict=True))
        pred = sum(y == c for y in b)
        ref = sum(x == c for x in a)
        out[c] = {
            "support": ref,
            "precision": tp / pred if pred else None,
            "recall": tp / ref if ref else None,
        }
    return out


def _ts(r: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(r["ts"])


def collect_ratings(
    pairs: list[dict[str, Any]], extra: Iterable[dict[str, Any]] = ()
) -> dict[str, list[dict[str, Any]]]:
    """{pair_id: [ratings]} from the records' own ``ratings`` plus ratings files, by time."""
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in pairs:
        for r in p.get("ratings", []):
            out[p["pair_id"]].append(r)
    for r in extra:
        out[r["pair_id"]].append(r)
    for v in out.values():
        v.sort(key=_ts)
    return out


def _first_by_rater(rs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each rater's first-pass rating: the latest non-recheck edit made BEFORE that rater's
    first recheck of the pair (a regular rating after a recheck never replaces it)."""
    out: dict[str, dict[str, Any]] = {}
    rechecked: set[str] = set()
    for r in rs:  # sorted by time
        rid = r["rater_id"]
        if r.get("recheck"):
            rechecked.add(rid)
        elif rid not in rechecked:
            out[rid] = r
    return out


def recheck_candidates(
    pairs: list[dict[str, Any]],
    ratings: Iterable[dict[str, Any]],
    rater_id: str,
    *,
    now: datetime,
    days: int = 14,
    fraction: float = 0.10,
    seed: int = 0,
) -> list[str]:
    """``fraction`` of the base pairs ``rater_id`` rated at least ``days`` ago and has not
    rechecked yet (seeded, deterministic): the blind self-agreement recheck set."""
    by_pair = collect_ratings(pairs, [r for r in ratings if r["rater_id"] == rater_id])
    base = [p["pair_id"] for p in pairs if p.get("dup_of") is None]
    rated = [pid for pid in base if rater_id in _first_by_rater(by_pair.get(pid, []))]
    old = [
        pid
        for pid in rated
        if (now - _ts(_first_by_rater(by_pair[pid])[rater_id])).days >= days
        and not any(r.get("recheck") for r in by_pair[pid])
    ]
    k = min(len(old), math.ceil(fraction * len(rated))) if rated else 0
    return sorted(random.Random(seed).sample(sorted(old), k))


def pairs_kappa(
    pairs: list[dict[str, Any]],
    extra_ratings: Iterable[dict[str, Any]] = (),
    *,
    raters: tuple[str, str] | None = None,
    seed: int = 0,
    reps: int = 1000,
    self_min: float = 0.85,
    recheck_days: int = 14,
    recheck_fraction: float = 0.10,
    min_n: int = 20,
) -> dict[str, Any]:
    by_pair = collect_ratings(pairs, extra_ratings)
    base_ids = [p["pair_id"] for p in pairs if p.get("dup_of") is None]
    dups = {p["pair_id"]: p["dup_of"] for p in pairs if p.get("dup_of")}
    all_raters = sorted({r["rater_id"] for rs in by_pair.values() for r in rs})
    report: dict[str, Any] = {
        "pairs": len(base_ids),
        "raters": all_raters,
        "rated_by_rater": {
            rid: sum(rid in _first_by_rater(by_pair.get(pid, [])) for pid in base_ids)
            for rid in all_raters
        },
    }

    # order stability: same rater, original vs. reversed duplicate
    stab: dict[str, dict[str, Any]] = {}
    for rid in all_raters:
        same = total = 0
        for dup_id, orig_id in dups.items():
            o = _first_by_rater(by_pair.get(orig_id, [])).get(rid)
            d = _first_by_rater(by_pair.get(dup_id, [])).get(rid)
            if o and d:
                total += 1
                same += o["label"] == d["label"] and (
                    o["label"] not in DIRECTIONAL or o.get("direction") == d.get("direction")
                )
        stab[rid] = {"n": total, "agreement": same / total if total else None}
    report["order_stability"] = stab

    if raters is None and len(all_raters) >= 2:
        raters = (all_raters[0], all_raters[1])
    if raters and all(r in all_raters for r in raters):
        ra, rb = raters
        a, b = [], []
        for pid in base_ids:
            first = _first_by_rater(by_pair.get(pid, []))
            if ra in first and rb in first:
                a.append(first[ra]["label"])
                b.append(first[rb]["label"])
        k = cohen_kappa(a, b)
        report["kappa"] = {
            "raters": [ra, rb],
            "n": len(a),
            "kappa": k,
            "ci95": bootstrap_ci(a, b, reps=reps, seed=seed),
            "observed_agreement": (sum(x == y for x, y in zip(a, b, strict=True)) / len(a))
            if a
            else None,
            "note": "kappa between two human raters is the performance ceiling",
        }
        report["per_class"] = per_class(a, b)
        report["confusion"] = confusion(a, b)
    else:
        report["kappa"] = None
        report["kappa_note"] = (
            "no kappa: fewer than two raters"
            if len(all_raters) < 2
            else f"raters {raters} not both present"
        )

    # self agreement (single rater): blind recheck >= recheck_days after the first rating
    selfag: dict[str, Any] = {}
    for rid in all_raters:
        same = total = 0
        for pid in base_ids:
            rs = [r for r in by_pair.get(pid, []) if r["rater_id"] == rid]
            first = _first_by_rater(rs).get(rid)
            rechecks = [r for r in rs if r.get("recheck")]
            if first and rechecks:
                rc = rechecks[-1]
                if (_ts(rc) - _ts(first)).days >= recheck_days:
                    total += 1
                    same += rc["label"] == first["label"]
        rate = same / total if total else None
        rated = report["rated_by_rater"].get(rid, 0)
        needed = max(min_n, math.ceil(recheck_fraction * rated))
        if total < needed or rate is None:
            verdict = "UNDECIDED"
        else:
            verdict = "PASS" if rate >= self_min else "FAIL"
        selfag[rid] = {
            "n": total,
            "needed_n": needed,
            "agreement": rate,
            "target": self_min,
            "verdict": verdict,
            "meets_target": verdict == "PASS",
        }
    report["self_agreement"] = selfag
    return report


def kappa_markdown(rep: dict[str, Any]) -> str:
    lines = ["# Pairs agreement report", "", f"- Pairs: {rep['pairs']}"]
    lines.append(f"- Raters: {', '.join(rep['raters']) or '-'} {rep['rated_by_rater']}")
    k = rep.get("kappa")
    if k:
        ci = k["ci95"]
        ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "n/a"
        kv = "undefined" if k["kappa"] is None else f"{k['kappa']:.3f}"
        lines.append(
            f"- Cohen's kappa ({k['raters'][0]} vs {k['raters'][1]}, n={k['n']}): "
            f"{kv}, bootstrap 95% CI {ci_s} (ceiling for model performance)"
        )
        lines += ["", "| class | support | precision | recall |", "|---|---|---|---|"]
        for c, v in rep["per_class"].items():
            p = "n/a" if v["precision"] is None else f"{v['precision']:.2f}"
            r = "n/a" if v["recall"] is None else f"{v['recall']:.2f}"
            lines.append(f"| {c} | {v['support']} | {p} | {r} |")
        lines += ["", "Confusion (rows: first rater, cols: second):", ""]
        lines.append("| | " + " | ".join(LABELS) + " |")
        lines.append("|---" * (len(LABELS) + 1) + "|")
        for x in LABELS:
            row = rep["confusion"].get(x, {})
            lines.append(f"| {x} | " + " | ".join(str(row.get(y, 0)) for y in LABELS) + " |")
    else:
        lines.append(f"- **{rep.get('kappa_note', 'no kappa')}** (no kappa is reported)")
    lines += ["", "Order stability (same rater, reversed duplicate):"]
    for rid, v in rep["order_stability"].items():
        a = "n/a" if v["agreement"] is None else f"{v['agreement']:.0%}"
        lines.append(f"- {rid}: {a} (n={v['n']})")
    lines += ["", "Self agreement (blind recheck):"]
    for rid, v in rep["self_agreement"].items():
        a = "n/a" if v["agreement"] is None else f"{v['agreement']:.0%}"
        lines.append(
            f"- {rid}: {a} (n={v['n']} of {v['needed_n']} needed, target {v['target']:.0%}): "
            + ("לא מוכרע" if v["verdict"] == "UNDECIDED" else v["verdict"])
        )
    return "\n".join(lines) + "\n"


# ---- Postgres source ----------------------------------------------------------------------

_SCOPE_RE = re.compile(r"^(?P<work>.+?)\s+(?P<a>[0-9]+[ab]?)(?:\s*-\s*(?P<b>[0-9]+[ab]?))?$")
_WORD = re.compile(r"[א-ת]{4,}")


def _page_sort(label: str) -> tuple[int, int]:
    m = re.match(r"(\d+)([ab]?)", label)
    return (int(m.group(1)), 0 if m.group(2) in ("", "a") else 1) if m else (0, 0)


class PostgresPairSource:
    """Pairs whose side A is in the slice ``scope`` ("Shabbat 21a-24a")."""

    def __init__(
        self,
        engine,
        scope: str,
        explicit_edge_types: Iterable[str],
        *,
        pool_limit: int = 40,
        min_shared_terms: int = 3,
    ):
        self.engine = engine
        self.scope = scope
        self.types = tuple(explicit_edge_types)
        self.pool_limit = pool_limit
        self.min_shared = min_shared_terms
        self._slice: dict[int, tuple[str, str]] | None = None
        self._lex: list[tuple[int, int, int, bool]] | None = None

    def _slice_segments(self) -> dict[int, tuple[str, str]]:
        """{segment_id: (ref, text_b)} of the slice (best indexable Hebrew text)."""
        if self._slice is not None:
            return self._slice
        from sqlalchemy import text

        m = _SCOPE_RE.match(self.scope.strip())
        if not m:
            raise ValueError(f"cannot parse scope {self.scope!r}")
        lo, hi = _page_sort(m["a"]), _page_sort(m["b"] or m["a"])
        with self.engine.connect() as c:
            rows = c.execute(
                text("""SELECT DISTINCT ON (s.id) s.id, s.ref, s.parent_ref, st.text_b
                        FROM segments s JOIN works w ON w.id = s.work_id
                        JOIN segment_text st ON st.segment_id = s.id
                        JOIN versions v ON v.id = st.version_id
                        WHERE w.title = :w AND s.parent_ref IS NOT NULL
                          AND v.can_index AND v.language = 'he'
                        ORDER BY s.id, v.is_primary DESC, v.id"""),
                {"w": m["work"]},
            ).all()
        self._slice = {
            sid: (ref, tb)
            for sid, ref, parent, tb in rows
            if lo <= _page_sort(parent.rsplit(" ", 1)[-1]) <= hi
        }
        return self._slice

    def explicit(self) -> list[PairCandidate]:
        from sqlalchemy import text

        ids = sorted(self._slice_segments())
        if not ids:
            return []
        with self.engine.connect() as c:
            rows = c.execute(
                text("""SELECT e.id, sa.ref, sb.ref, e.edge_type, e.segment_a_id
                        FROM edges e JOIN segments sa ON sa.id = e.segment_a_id
                        JOIN segments sb ON sb.id = e.segment_b_id
                        WHERE e.origin = 'sefaria' AND e.edge_type = ANY(:t)
                          AND (e.segment_a_id = ANY(:ids) OR e.segment_b_id = ANY(:ids))
                        ORDER BY e.id"""),
                {"t": list(self.types), "ids": ids},
            ).all()
        out = []
        sl = set(ids)
        for eid, ra, rb, et, a_id in rows:
            a, b = (ra, rb) if a_id in sl else (rb, ra)
            out.append(PairCandidate(a, b, f"edge {eid} ({et}; candidate)"))
        return out

    def _lexical(self) -> list[tuple[int, int, int, bool]]:
        """(slice segment, other segment, shared terms, same work) for FTS neighbours that
        have no edge between them."""
        if self._lex is not None:
            return self._lex
        from sqlalchemy import text

        sl = self._slice_segments()
        out: list[tuple[int, int, int, bool]] = []
        with self.engine.connect() as c:
            work_id = c.execute(
                text("SELECT work_id FROM segments WHERE id = :i"), {"i": next(iter(sl), 0)}
            ).scalar()
            seen: set[tuple[int, int]] = set()

            def add(rows, mine: set[str], sid: int, min_shared: int) -> None:
                for oid, otb, owork in rows:
                    shared = len(mine & set(_WORD.findall(otb or "")))
                    if shared >= min_shared and (sid, oid) not in seen:
                        seen.add((sid, oid))
                        out.append((sid, oid, shared, owork == work_id))

            for sid, (_ref, tb) in sorted(sl.items()):
                mine = set(_WORD.findall(tb or ""))
                words = sorted(mine, key=lambda w: (-len(w), w))
                if len(words) < 3:
                    continue
                # (1) the whole corpus: the 3 longest words together (lexical neighbours)
                rows = c.execute(
                    text("""SELECT st.segment_id, st.text_b, s.work_id
                            FROM segment_text st JOIN segments s ON s.id = st.segment_id
                            WHERE to_tsvector('simple', st.text_b) @@ to_tsquery('simple', :q)
                              AND st.segment_id <> ALL(:ids)
                            ORDER BY st.segment_id LIMIT :lim"""),
                    {"q": " & ".join(words[:3]), "ids": list(sl), "lim": self.pool_limit},
                ).all()
                add(rows, mine, sid, self.min_shared)
                # (2) the same work outside the slice: looser (2-word groups) -> the
                # "same wording, another nidon" hard negatives
                for g in range(0, min(len(words), 12) - 1, 2):
                    rows = c.execute(
                        text("""SELECT st.segment_id, st.text_b, s.work_id
                                FROM segment_text st JOIN segments s ON s.id = st.segment_id
                                WHERE s.work_id = :w
                                  AND to_tsvector('simple', st.text_b)
                                      @@ to_tsquery('simple', :q)
                                  AND st.segment_id <> ALL(:ids)
                                ORDER BY st.segment_id LIMIT :lim"""),
                        {
                            "w": work_id,
                            "q": " & ".join(words[g : g + 2]),
                            "ids": list(sl),
                            "lim": self.pool_limit,
                        },
                    ).all()
                    add(rows, mine, sid, max(2, self.min_shared - 1))
            others = sorted({o for _, o, _, _ in out})
            linked = set()
            if others:
                for a, b in c.execute(
                    text("""SELECT segment_a_id, segment_b_id FROM edges
                            WHERE (segment_a_id = ANY(:s) AND segment_b_id = ANY(:o))
                               OR (segment_b_id = ANY(:s) AND segment_a_id = ANY(:o))"""),
                    {"s": list(sl), "o": others},
                ):
                    linked.add(frozenset((a, b)))
            refs = dict(
                c.execute(
                    text("SELECT id, ref FROM segments WHERE id = ANY(:o)"), {"o": others}
                ).all()
            )
        self._refs = refs
        self._lex = [x for x in out if frozenset((x[0], x[1])) not in linked]
        return self._lex

    def _cands(self, same_work: bool) -> list[PairCandidate]:
        sl = self._slice_segments()
        return [
            PairCandidate(sl[s][0], self._refs[o], f"fts shared={n}")
            for s, o, n, sw in self._lexical()
            if sw == same_work
        ]

    def neighbors(self) -> list[PairCandidate]:
        return self._cands(same_work=False)

    def hard_negatives(self) -> list[PairCandidate]:
        return self._cands(same_work=True)
