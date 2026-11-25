"""V5-V10: a positive and a negative case for each (in memory and on the Postgres seed)."""

from __future__ import annotations

import pytest

from rav_evals.config import load_config
from rav_evals.kl_lookup import InMemoryKnowledgeLookup, PostgresKnowledgeLookup
from rav_evals.schema import Answer, Citation, Claim, ClaimType, RelationRef
from rav_evals.verifier import InMemorySourceStore, verify_answer

M = load_config().markers
TEXTS = {
    "Gem 1:1": "טקסט Gem שורה 1 לבדיקה",
    "Gem 1:2": "טקסט Gem שורה 2 לבדיקה",
    "Lens 1:1": "טקסט Lens שורה 1 לבדיקה",
    "Codes 1:1": "טקסט Codes שורה 1 לבדיקה",
}
STORE = InMemorySourceStore(TEXTS)
SNAP = {
    "tables": ["kl_sources", "kl_coverage", "kl_witnesses", "edges", "kl_relations", "kl_links"],
    "ref_work": {"Gem 1:1": "Gem", "Gem 1:2": "Gem", "Lens 1:1": "Lens", "Codes 1:1": "Codes"},
    "sources": {
        "Gem": {"role": "primary", "weight_tier": "core", "names": ["גמ"]},
        "Lens": {"role": "lens", "weight_tier": "extended", "names": ["עיון"]},
        "Codes": {"role": "primary", "weight_tier": "core", "names": ["פוסק"]},
    },
    "coverage": {"Gem 1": {"Gem": "covered", "Codes": "not_ingested"}},
    "edges": {
        str(i): {"origin": o, "edge_type": "", "a_ref": "Gem 1:1", "b_ref": "Codes 1:1"}
        for i, o in (
            (777, "sefaria"),
            (778, "heuristic"),
            (779, "llm"),
            (780, "weird"),
            (781, None),
            (782, "human"),
        )
    },
    "kl_relations": [4],
    "kl_links": [],
    "witnesses": [5],
}
MEM = InMemoryKnowledgeLookup(SNAP)
SCOPE = ["Gem 1"]


def cit(ref, **kw):
    return Citation(ref=ref, quoted_text=TEXTS[ref].split(" לבדיקה")[0], **kw)


def answer(*claims, books=("Codes",), scope="in_scope"):
    return Answer(
        claims=list(claims), disclaimer="דעת המערכת", scope=scope, unchecked_core_books=list(books)
    )


def codes(a, kn):
    r = verify_answer(a, list(TEXTS), STORE, knowledge=kn, scope_keys=SCOPE, markers=M)
    return {f"{f.check}:{f.code}" for f in r.failures}, r


@pytest.fixture(params=["memory", "postgres"])
def kn(request):
    if request.param == "memory":
        return MEM
    return PostgresKnowledgeLookup(request.getfixturevalue("kl_corpus"))


def test_clean_answer_passes(kn):
    a = answer(Claim(type=ClaimType.QUOTE, text="לשון", citations=[cit("Gem 1:1")]))
    c, r = codes(a, kn)
    assert c == set() and r.passed


# V5
def test_v5_lens_cited_as_primary_fails(kn):
    a = answer(Claim(type=ClaimType.INTERPRETATION, text="ביאור", citations=[cit("Lens 1:1")]))
    assert "V5:secondary_as_primary" in codes(a, kn)[0]


def test_v5_lens_marked_secondary_passes(kn):
    a = answer(
        Claim(
            type=ClaimType.INTERPRETATION,
            text="ביאור",
            citations=[cit("Lens 1:1", source_role="secondary_lens")],
        )
    )
    assert not {x for x in codes(a, kn)[0] if x.startswith("V5")}


# V6
def _rel(**kw):
    return RelationRef(
        table="edges", id=777, rel="mesorat hashas", evidence_refs=["Gem 1:1", "Codes 1:1"], **kw
    )


def test_v6_candidate_only_and_endpoints_uncited_fails(kn):
    # candidate flag is NOT set by the answer: the lookup (origin=sefaria) decides
    a = answer(
        Claim(
            type=ClaimType.INTERPRETATION,
            text="מקביל",
            citations=[cit("Gem 1:2")],
            relations=[_rel()],
        )
    )
    assert "V6:candidate_as_proof" in codes(a, kn)[0]


def test_v6_endpoint_cited_passes(kn):
    a = answer(
        Claim(
            type=ClaimType.INTERPRETATION,
            text="מקביל",
            citations=[cit("Gem 1:1"), cit("Codes 1:1")],  # BOTH endpoints
            relations=[_rel()],
        )
    )
    assert not {x for x in codes(a, kn)[0] if x.startswith("V6")}


# V7
def test_v7_error_in_system_voice_without_witness_fails(kn):
    a = answer(
        Claim(
            type=ClaimType.SYSTEM_CONCLUSION, flagged_as_system=True, text="יש כאן טעות סופר בגמרא"
        )
    )
    assert "V7:witness_missing" in codes(a, kn)[0]


def test_v7_with_known_witness_passes_and_unknown_fails(kn):
    ok = answer(
        Claim(
            type=ClaimType.SYSTEM_CONCLUSION,
            flagged_as_system=True,
            text="ייתכן שיש כאן טעות סופר",
            citations=[cit("Gem 1:1", witness_id=5)],
        )
    )
    bad = answer(
        Claim(
            type=ClaimType.SYSTEM_CONCLUSION,
            flagged_as_system=True,
            text="ייתכן שיש כאן טעות סופר",
            citations=[cit("Gem 1:1", witness_id=99)],
        )
    )
    assert not {x for x in codes(ok, kn)[0] if x.startswith("V7")}
    assert "V7:witness_unknown" in codes(bad, kn)[0]


def test_v7_reported_by_a_source_or_denied_passes(kn):
    reported = answer(
        Claim(
            type=ClaimType.INTERPRETATION,
            text="המפרש כותב שהוא טעות סופר",
            citations=[cit("Gem 1:1", attribution="המפרש")],
        )
    )
    denied = answer(
        Claim(
            type=ClaimType.SYSTEM_CONCLUSION,
            flagged_as_system=True,
            text="לא נמצא עד, ואין לומר שיש כאן טעות סופר",
        )
    )
    for a in (reported, denied):
        assert not {x for x in codes(a, kn)[0] if x.startswith("V7")}


# V8
def test_v8_missing_notice_fails_and_notice_passes(kn):
    q = Claim(type=ClaimType.QUOTE, text="לשון", citations=[cit("Gem 1:1")])
    assert "V8:coverage_notice_missing" in codes(answer(q, books=()), kn)[0]
    assert not {x for x in codes(answer(q, books=("פוסק",)), kn)[0] if x.startswith("V8")}


def test_v8_listing_a_checked_core_work_is_wrong(kn):
    q = Claim(type=ClaimType.QUOTE, text="לשון", citations=[cit("Gem 1:1")])
    assert "V8:coverage_notice_wrong" in codes(answer(q, books=("Codes", "Gem")), kn)[0]


def test_v8_skipped_without_tables_or_scope():
    q = Claim(type=ClaimType.QUOTE, text="לשון", citations=[cit("Gem 1:1")])
    r = verify_answer(answer(q, books=()), list(TEXTS), STORE, knowledge=None, scope_keys=SCOPE)
    assert r.passed and any(s.startswith("V8:") for s in r.skipped)
    assert any(s.startswith("V5:") for s in r.skipped)


# V9
def test_v9_kibalnu_without_citation_fails_with_citation_passes(kn):
    bad = answer(
        Claim(type=ClaimType.SYSTEM_CONCLUSION, flagged_as_system=True, text="כך קיבלנו להלכה")
    )
    ok = answer(Claim(type=ClaimType.INTERPRETATION, text="כך נהגו", citations=[cit("Codes 1:1")]))
    assert "V9:kibalnu_undocumented" in codes(bad, kn)[0]
    assert not {x for x in codes(ok, kn)[0] if x.startswith("V9")}


# V10
def test_v10_sa_decisive_fails_quote_and_neutral_pass(kn):
    bad = answer(
        Claim(
            type=ClaimType.SYSTEM_CONCLUSION,
            flagged_as_system=True,
            text='השו"ע פסק כך, וכך הלכה ואין לזוז',
        )
    )
    neutral = answer(
        Claim(type=ClaimType.INTERPRETATION, text='השו"ע פסק כך', citations=[cit("Codes 1:1")])
    )
    assert "V10:sa_decisive_phrasing" in codes(bad, kn)[0]
    assert not {x for x in codes(neutral, kn)[0] if x.startswith("V10")}


def test_v0_answer_still_validates():
    a = Answer.model_validate(
        {
            "claims": [
                {
                    "type": "quote",
                    "text": "x",
                    "citations": [{"ref": "Gem 1:1", "quoted_text": "טקסט Gem שורה 1"}],
                }
            ],
            "disclaimer": "d",
        }
    )
    assert a.scope is None and a.claims[0].citations[0].source_role == "primary"
