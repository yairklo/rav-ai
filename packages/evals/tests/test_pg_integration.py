"""Integration with the real corpus schema in Postgres (see conftest for skip/REQUIRE_DB)."""

from __future__ import annotations

from rav_ai.db.models import Run, SegmentText
from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session

from conftest import RAW_SECONDARY, RAW_SHEMA
from rav_evals.golden import GoldenSet
from rav_evals.models import prompt_hash, record_report
from rav_evals.normalization import normalize
from rav_evals.pg_store import PostgresSourceStore
from rav_evals.runner import RunOutput, run_eval
from rav_evals.schema import Answer, Citation, Claim, ClaimType
from rav_evals.verifier import verify_answer


def answer_quoting(ref: str, quoted: str) -> Answer:
    claim = Claim(
        type=ClaimType.QUOTE,
        text="x",
        citations=[Citation(ref=ref, quoted_text=quoted)],
    )
    return Answer(disclaimer="דעת המערכת, לא פסק", claims=[claim])


def test_store_v1_existence(corpus):
    store = PostgresSourceStore(corpus)
    assert store.get_text("Shema 1") is not None
    assert store.get_text("Shema 2") is None


def test_store_returns_stored_text_b_which_is_normalization_stable(corpus):
    store = PostgresSourceStore(corpus)
    got = store.get_text("Shema 1")
    assert got == normalize(RAW_SHEMA)  # same policy as segment_text.text_b
    assert normalize(got) == got  # re-normalizing in the verifier changes nothing
    with Session(corpus) as s:
        assert all(normalize(t) == t for t in s.scalars(select(SegmentText.text_b)))


def test_store_respects_can_index(corpus):
    store = PostgresSourceStore(corpus)
    assert store.get_text("Noidx 1") is None  # exists, but no indexable version


def test_store_prefers_primary_version(corpus):
    # MOther has higher priority but MPrimary is_primary
    assert PostgresSourceStore(corpus).get_text("Multi 1") == normalize(RAW_SHEMA)
    assert normalize(RAW_SECONDARY) != normalize(RAW_SHEMA)


def test_verifier_end_to_end_against_postgres(corpus):
    store = PostgresSourceStore(corpus)
    quote = "שמע ישראל ה' אלהינו"  # no niqqud, no markup, different punctuation
    ok = verify_answer(answer_quoting("Shema 1", quote), ["Shema 1"], store)
    assert ok.passed, ok.failures
    bad = verify_answer(answer_quoting("Shema 1", "שמע בני"), ["Shema 1"], store)
    assert bad.failure_codes() == ["V2:quote_not_in_source"]
    missing = verify_answer(answer_quoting("Nope 1", quote), ["Nope 1"], store)
    assert missing.failure_codes() == ["V1:ref_not_found"]
    blocked = verify_answer(answer_quoting("Noidx 1", "טקסט"), ["Noidx 1"], store)
    assert blocked.failure_codes() == ["V1:ref_not_found"]


def test_runs_migration_matches_model(corpus):
    insp = inspect(corpus)
    assert "runs" in insp.get_table_names()
    cols = {c["name"] for c in insp.get_columns("runs")}
    assert cols == {c.name for c in Run.__table__.columns}
    with corpus.connect() as c:
        assert c.execute(text("SELECT version_num FROM alembic_version")).scalar() == "0002"


def test_run_logging_roundtrip(corpus):
    golden = GoldenSet(
        version="v-test",
        questions=[
            {"id": "q1", "question": "מהו?", "category": "dispute", "difficulty": "easy"},
        ],
    )
    phash = prompt_hash("system prompt + question")

    def fn(q):
        return RunOutput(
            answer=answer_quoting("Shema 1", "שמע ישראל"),
            retrieved_refs=["Shema 1"],
            model="fake-model",
            prompt_version="p1",
            prompt_hash=phash,
        )

    report = run_eval(golden, fn, PostgresSourceStore(corpus))
    with Session(corpus) as s:
        record_report(s, golden, report)
    with Session(corpus) as s:
        row = s.scalars(select(Run)).one()
        assert (row.model, row.prompt_hash, row.prompt_version) == ("fake-model", phash, "p1")
        assert row.question_id == "q1" and row.golden_version == "v-test"
        assert row.answer["claims"][0]["citations"][0]["ref"] == "Shema 1"
        assert row.verification_passed is True and row.verification["failures"] == []
        assert row.retrieved_sources == ["Shema 1"]
