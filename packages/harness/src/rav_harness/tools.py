"""Narrow, deterministic corpus tools. No LLM inside; every result is license-gated."""

from __future__ import annotations

import re
import time
from typing import Any, Protocol

import sqlalchemy as sa
from rav_ai.lookup import deployment_mode, get_texts, may_display, neighbors
from sqlalchemy.orm import Session
from textnorm import find_citations, normalize

from .config import HarnessConfig
from .knowledge_tools import KnowledgeTools

SNIPPET_CHARS = 300


class ToolBox(Protocol):
    """What the loop needs from a tool set (the tests substitute a fake)."""

    retrieved_refs: list[str]

    def specs(self) -> list[dict[str, Any]]: ...

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


def _spec(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SPECS = [
    _spec(
        "get_segment",
        "Raw text of one segment by canonical ref (e.g. 'Shabbat 21b:3'), with license metadata. "
        "Text whose license forbids display is withheld. If the ref is a whole page, returns "
        "its child segment refs instead.",
        {"ref": {"type": "string"}},
        ["ref"],
    ),
    _spec(
        "neighbors",
        "Segments linked to a ref, split into layers above (later: commentaries) and below "
        "(earlier: sources), optionally filtered by edge type (e.g. 'commentary', 'quotation').",
        {
            "ref": {"type": "string"},
            "edge_types": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": "integer", "description": "max results per layer (default 20)"},
        },
        ["ref"],
    ),
    _spec(
        "search_text",
        "Basic full-text search (any query word matches, ranked) over the normalized corpus text. "
        "Returns refs and short snippets; call get_segment to read a hit in full.",
        {"query": {"type": "string"}, "limit": {"type": "integer"}},
        ["query"],
    ),
    _spec(
        "resolve_citation",
        "Find citations in free text (e.g. 'שבת כא ע\"ב', 'שו\"ע או\"ח תרעא') and resolve them to "
        "canonical refs, saying whether each exists in the corpus.",
        {"text": {"type": "string"}},
        ["text"],
    ),
]


def _snip(s: str | None) -> tuple[str, bool]:
    s = s or ""
    return (s[:SNIPPET_CHARS], len(s) > SNIPPET_CHARS)


class CorpusTools:
    """The four tools over a SQLAlchemy session. ``retrieved_refs`` lists refs whose text the
    model has actually seen (full text, or an untruncated snippet): the verifier's V1 set."""

    def __init__(
        self, session: Session, mode: str | None = None, config: HarnessConfig | None = None
    ):
        self.session = session
        self.mode = mode or deployment_mode()
        self.retrieved_refs: list[str] = []
        # knowledge layer tools (off: not offered, v0 behaviour). They never add to
        # retrieved_refs: only get_segment (and the corpus tools) mark text as read.
        self.kl: KnowledgeTools | None = (
            KnowledgeTools(session, config, self.mode)
            if config is not None and config.knowledge != "off"
            else None
        )

    def specs(self) -> list[dict[str, Any]]:
        return [*TOOL_SPECS, *(self.kl.specs() if self.kl else [])]

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.kl is not None and self.kl.has(name):
            return self.kl.call(name, arguments)
        fn = {
            "get_segment": self.get_segment,
            "neighbors": self.neighbors,
            "search_text": self.search_text,
            "resolve_citation": self.resolve_citation,
        }.get(name)
        if fn is None:
            return {"error": f"unknown tool {name!r}"}
        try:
            return fn(**arguments)
        except TypeError as e:  # wrong/missing arguments from the model
            return {"error": f"bad arguments for {name}: {e}"}

    def _seen(self, ref: str) -> None:
        if ref not in self.retrieved_refs:
            self.retrieved_refs.append(ref)

    def get_segment(self, ref: str) -> dict[str, Any]:
        rows = get_texts(self.session, ref)
        if not rows:
            children = (
                self.session.execute(
                    sa.text(
                        "SELECT ref FROM segments WHERE parent_ref = :r ORDER BY sort_key LIMIT 60"
                    ),
                    {"r": ref},
                )
                .scalars()
                .all()
            )
            return {
                "found": False,
                "ref": ref,
                "child_refs": list(children),
                "hint": "no such segment; use child_refs, resolve_citation or search_text",
            }
        versions = []
        for r in rows:
            ok = may_display(r.can_index, r.can_display_public, self.mode)
            versions.append(
                {
                    "text": r.text_raw if ok else None,
                    "withheld": not ok,
                    "work": r.work,
                    "period": r.period,
                    "language": r.language,
                    "version_title": r.version_title,
                    "license": r.license,
                    "can_display_public": r.can_display_public,
                    "is_primary": r.is_primary,
                }
            )
        if any(not v["withheld"] for v in versions):
            self._seen(ref)
        return {"found": True, "ref": ref, "versions": versions}

    def neighbors(
        self, ref: str, edge_types: list[str] | None = None, limit: int = 20
    ) -> dict[str, Any]:
        res = neighbors(self.session, ref)
        if res is None:
            return {"found": False, "ref": ref}
        groups, my_period = res
        wanted = {("" if t == "untyped" else t) for t in edge_types} if edge_types else None
        out: dict[str, Any] = {"found": True, "ref": ref, "period": my_period}
        for side in ("above", "below", "same"):
            items = []
            for by_period in groups[side].values():
                for lst in by_period.values():
                    for n in lst:
                        if wanted is not None and n.edge_type not in wanted:
                            continue
                        withheld = n.snippet.startswith("[withheld")
                        snip, cut = _snip(n.snippet)
                        if n.snippet and not withheld and not cut:
                            self._seen(n.ref)
                        items.append(
                            {
                                "ref": n.ref,
                                "work": n.work,
                                "period": n.period,
                                "edge_type": n.edge_type or "untyped",
                                "origin": n.origin,
                                "snippet": snip,
                                "truncated": cut,
                            }
                        )
            out[side] = items[: max(1, limit)]
            out[f"{side}_total"] = len(items)
        return out

    def search_text(self, query: str, limit: int = 10) -> dict[str, Any]:
        tokens: list[str] = []
        for tok in normalize(query, "B").text.split():
            if re.fullmatch(r"\w{2,}", tok) and tok not in tokens:
                tokens.append(tok)
        tokens = tokens[:12]
        if not tokens:
            return {"query": query, "results": [], "error": "no searchable words in query"}
        gate = "v.can_display_public" if self.mode == "public" else "v.can_index"
        rows = self.session.execute(
            sa.text(f"""
            SELECT s.ref, w.title, w.period, v.language, t.text_a, r.rank FROM (
              SELECT t.id, ts_rank(to_tsvector('simple', t.text_b), q) AS rank
              FROM segment_text t
              JOIN versions v ON v.id = t.version_id AND {gate},
              to_tsquery('simple', :q) q
              WHERE to_tsvector('simple', t.text_b) @@ q
              ORDER BY rank DESC LIMIT :cand) r
            JOIN segment_text t ON t.id = r.id
            JOIN versions v ON v.id = t.version_id
            JOIN segments s ON s.id = t.segment_id JOIN works w ON w.id = s.work_id
            ORDER BY r.rank DESC, v.is_primary DESC, s.ref"""),
            {"q": " | ".join(tokens), "cand": max(1, min(limit, 50)) * 5},
        ).all()
        results, seen = [], set()
        for ref, work, period, lang, text_a, rank in rows:
            if ref in seen:
                continue
            seen.add(ref)
            snip, cut = _snip(text_a)
            if not cut:
                self._seen(ref)
            results.append(
                {
                    "ref": ref,
                    "work": work,
                    "period": period,
                    "language": lang,
                    "rank": round(float(rank), 4),
                    "snippet": snip,
                    "truncated": cut,
                }
            )
            if len(results) >= max(1, min(limit, 50)):
                break
        return {"query": query, "terms": tokens, "results": results}

    def resolve_citation(self, text: str) -> dict[str, Any]:
        found = []
        for c in find_citations(text):
            item: dict[str, Any] = {
                "text": c.text,
                "ref": c.id,
                "kind": c.kind,
                "confidence": c.confidence,
                "exists": False,
            }
            if c.id:
                hit = (
                    self.session.execute(
                        sa.text(
                            "SELECT ref FROM segments WHERE ref = :r OR parent_ref = :r "
                            "ORDER BY sort_key LIMIT 10"
                        ),
                        {"r": c.id},
                    )
                    .scalars()
                    .all()
                )
                item["exists"] = bool(hit)
                if hit and hit[0] != c.id:
                    item["segment_refs"] = list(hit)
            found.append(item)
        return {"citations": found}


def timed_call(tools: ToolBox, name: str, arguments: dict[str, Any]) -> tuple[dict, int]:
    t0 = time.perf_counter()
    try:
        result = tools.call(name, arguments)
    except Exception as e:  # noqa: BLE001 - a broken tool is reported to the model, not fatal
        result = {"error": f"{type(e).__name__}: {e}"}
    return result, int((time.perf_counter() - t0) * 1000)
