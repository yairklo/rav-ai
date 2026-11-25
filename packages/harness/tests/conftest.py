"""Fixtures. Loop tests need nothing; tool/integration tests use a throw-away Postgres DB
(docker compose up -d; skipped when unavailable) seeded with a few hand-made segments."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from typing import Any

import pytest
from rav_ai.db.models import Author, Base, Edge, Segment, SegmentText, Snapshot, Version, Work
from rav_ai.db.session import make_engine, migrate
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from textnorm import normalize

from rav_harness.llm import LLMResponse, ToolCall

ADMIN_URL = "postgresql+psycopg://rav:rav@localhost:54329/postgres"

BER_TEXT = "מאימתי קורין את שמע בערבין משעה שהכהנים נכנסים לאכול בתרומתן"
RASHI_TEXT = "משעה שהכהנים נכנסים - כשמגיע זמן אכילת תרומה"
NC_TEXT = "קטע שרישיונו אינו מתיר הצגה פומבית על שמע"
SECRET = "קטע חסוי שאסור לאנדקס"


class FakeLLM:
    """Scripted model: returns the queued responses in order and records what it was sent."""

    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages, tools, *, model, temperature, max_tokens) -> LLMResponse:
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "model": model,
                "tools": [t["function"]["name"] for t in tools],
            }
        )
        if not self.responses:
            raise AssertionError("FakeLLM script exhausted")
        return self.responses.pop(0)


def call(name: str, cid: str = "c1", **args) -> ToolCall:
    import json

    return ToolCall(cid, name, args, json.dumps(args, ensure_ascii=False))


def reply(
    *calls: ToolCall,
    content: str | None = None,
    cost: float | None = 0.01,
    tin: int = 100,
    tout: int = 20,
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=list(calls),
        input_tokens=tin,
        output_tokens=tout,
        cost_usd=cost,
        model="fake/model",
    )


def answer_args(**over) -> dict[str, Any]:
    base = {
        "claims": [
            {
                "type": "quote",
                "text": "הזמן הוא משעה שהכהנים נכנסים לאכול",
                "citations": [{"ref": "Berakhot 2a:1", "quoted_text": "משעה שהכהנים נכנסים לאכול"}],
            }
        ],
        "disclaimer": "דעת המערכת בלבד",
    }
    base.update(over)
    return base


@pytest.fixture
def engine():
    name = f"rav_harness_test_{uuid.uuid4().hex[:8]}"
    try:
        admin = create_engine(
            ADMIN_URL, isolation_level="AUTOCOMMIT", connect_args={"connect_timeout": 5}
        )
        with admin.connect() as c:
            c.execute(text(f'CREATE DATABASE "{name}"'))
    except Exception as e:  # noqa: BLE001
        if os.environ.get("REQUIRE_DB") == "1":
            pytest.fail(f"Postgres not available (REQUIRE_DB=1): {e}")
        pytest.skip(f"Postgres not available (docker compose up -d): {e}")
    url = f"postgresql+psycopg://rav:rav@localhost:54329/{name}"
    migrate(url)  # also exercises revision 0003 (run columns + FTS index)
    eng = make_engine(url)
    yield eng
    eng.dispose()
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
    admin.dispose()


@pytest.fixture
def session(engine):
    with Session(engine, expire_on_commit=False) as s:
        seed(s)
        yield s


def seed(s: Session) -> None:
    snap = Snapshot(source="sefaria", export_label="t", status="complete")
    s.add(snap)
    s.flush()
    author = Author(slug="a", name="A")
    s.add(author)
    s.flush()

    def work(title, period):
        w = Work(
            title=title,
            author_id=author.id,
            category="Talmud",
            category_path=["Talmud"],
            period=period,
            section_names=["Daf", "Line"],
            schema={},
        )
        s.add(w)
        s.flush()
        return w

    def version(w, vt, lic, idx, pub, primary=True):
        v = Version(
            work_id=w.id,
            source="sefaria",
            language="he",
            version_title=vt,
            license=lic,
            can_index=idx,
            can_display_public=pub,
            can_train=False,
            is_primary=primary,
            first_snapshot_id=snap.id,
            last_snapshot_id=snap.id,
        )
        s.add(v)
        s.flush()
        return v

    def segment(w, v, ref, body, parent):
        seg = Segment(
            ref=ref,
            work_id=w.id,
            node_path=[],
            section_path=[],
            indices=[],
            section_names=[],
            sort_key=ref,
            parent_ref=parent,
        )
        s.add(seg)
        s.flush()
        s.add(
            SegmentText(
                segment_id=seg.id,
                version_id=v.id,
                text_raw=body,
                raw_hash="h" + ref,
                text_a=normalize(body, "A").text,
                text_b=normalize(body, "B").text,
                offsets_a=b"",
                offsets_b=b"",
            )
        )
        s.flush()
        return seg

    ber, rashi, nc, sec = (
        work("Berakhot", "amoraim"),
        work("Rashi on Berakhot", "rishonim"),
        work("NC Book", "rishonim"),
        work("Secret Book", "rishonim"),
    )
    pd = version(ber, "PD", "Public Domain", True, True)
    nc_v = version(nc, "NC", "CC-BY-NC", True, False)
    sec_v = version(sec, "Secret", "Copyrighted", False, False)
    s1 = segment(ber, pd, "Berakhot 2a:1", BER_TEXT, "Berakhot 2a")
    segment(ber, pd, "Berakhot 2a:2", "ועד סוף האשמורת הראשונה", "Berakhot 2a")
    r1 = segment(
        rashi,
        version(rashi, "Rashi PD", "Public Domain", True, True),
        "Rashi on Berakhot 2a:1:1",
        RASHI_TEXT,
        "Rashi on Berakhot 2a:1",
    )
    n1 = segment(nc, nc_v, "NC Book 1:1", NC_TEXT, "NC Book 1")
    segment(sec, sec_v, "Secret Book 1:1", SECRET, "Secret Book 1")
    for other, et in ((r1, "commentary"), (n1, "quotation")):
        s.add(
            Edge(
                segment_a_id=other.id,
                segment_b_id=s1.id,
                edge_type=et,
                origin="sefaria",
                snapshot_id=snap.id,
            )
        )
    s.commit()


__all__ = ["Base", "Callable", "FakeLLM", "answer_args", "call", "reply"]
