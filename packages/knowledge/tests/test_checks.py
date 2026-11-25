"""K-checks as pure functions (no DB)."""

from __future__ import annotations

from rav_knowledge.checks import (
    CheckContext,
    EvRec,
    SegInfo,
    run_checks,
)
from rav_knowledge.evidence import Located
from rav_knowledge.schemas import (
    Chakira,
    Ev,
    Holder,
    Issue,
    Machloket,
    Position,
    SourceClaim,
    SourceClaimTarget,
    Step,
    SugyaBoundaries,
    SugyaSummary,
    validate_body,
)


def seg(label="S1", role="primary", work="Shabbat", slug="gemara", order=0) -> SegInfo:
    return SegInfo(
        label,
        order + 1,
        f"Shabbat 21a:{order + 1}",
        1,
        work,
        slug,
        "canonical",
        role,
        "amoraim",
        order,
    )


def rec(path, ev_role="states", status="located", info=None, text="claim", seg_label="S1") -> EvRec:
    return EvRec(
        path,
        text,
        Ev(seg=seg_label, quote="q", role=ev_role),
        Located(status),
        info if info is not None else seg(seg_label),
    )


def run(model, evidence=(), **kw):
    return run_checks(CheckContext(kind="x", model=model, evidence=list(evidence), **kw))


def codes(r, level=None):
    return sorted({f.code for f in r.findings if level in (None, f.level)})


def pos(**kw) -> Position:
    base = dict(
        issue="issue:1",
        holder=Holder(label="רב", work="Shabbat", attribution="direct"),
        view_he="חייב",
    )
    return Position(**{**base, **kw})


# K1
def test_k1_schema_rejects_long_text_and_unknown_kind():
    model, errs = validate_body(
        "sugya_summary",
        {
            "title_he": "x",
            "steps": [{"id": "s1", "move": "statement", "text_he": " ".join(["מילה"] * 26)}],
        },
    )
    assert model is None and errs
    assert validate_body("nope", {})[0] is None
    assert validate_body("issue", {"canonical_q_he": "מה"})[0] is not None


# K2/K3/K4
def test_k2_no_evidence():
    assert "K2" in codes(run(pos()))


def test_k3_critical_role_not_found_and_fraction():
    r = run(pos(), [rec("evidence[0]", "states", "not_found")])
    assert "K3" in codes(r, "error")
    # non critical role: only the fraction rule applies (1/1 > 10%)
    r2 = run(pos(), [rec("evidence[0]", "supports", "not_found")])
    assert any(f.code == "K3" for f in r2.errors)
    ok = [rec(f"evidence[{i}]", "supports") for i in range(10)] + [
        rec("evidence[10]", "supports", "not_found")
    ]
    assert "K3" not in codes(run(pos(), ok), "error") or True  # 1/11 < 10%
    assert not [f for f in run(pos(), ok).errors if f.code == "K3"]


def test_k4_out_of_scope():
    r = run(pos(), [rec("evidence[0]", status="out_of_scope", info=None)])
    assert "K4" in codes(r, "error")


# K5
def test_k5_lens_only_evidence_is_downgraded_to_reading():
    m = pos()
    r = run(m, [rec("evidence[0]", "states", info=seg(role="lens", work="Acharon", slug="a"))])
    assert r.ok and m.reading_only and "position" in r.downgrades
    assert "K5" in codes(r, "warn")


def test_k5_direct_needs_holders_own_text():
    m = pos(holder=Holder(label="רש״י", entity_slug="rashi", work="Rashi on Shabbat"))
    r = run(m, [rec("evidence[0]", info=seg(role="primary"))])  # evidence is the Gemara, not Rashi
    assert "K5" in codes(r, "error")
    m2 = pos(holder=Holder(label="רב", attribution="reported", reported_by="Shabbat"))
    assert run(m2, [rec("evidence[0]", "attributes")]).ok
    m3 = pos(holder=Holder(label="רב", attribution="reported"))
    assert "K5" in codes(run(m3, [rec("evidence[0]", "attributes")]), "error")


# K6
def test_k6_stub_leaves_unchecked_and_a_checker_can_reject():
    class No:
        def check(self, claim, quote):
            return "not_entailed"

    r = run(pos(), [rec("evidence[0]")])
    assert r.entail["evidence[0]"] == "unchecked" and r.ok
    assert "K6" in codes(run(pos(), [rec("evidence[0]")], entailment=No()), "error")


# K7
def test_k7_coverage_is_only_a_warning():
    m = Issue(canonical_q_he="מה הדין")
    r = run_checks(
        CheckContext(
            kind="issue", model=m, evidence=[rec("evidence[0]")], coverage={7: "not_ingested"}
        )
    )
    assert "K7" in codes(r, "warn") and r.ok


# K8
def test_k8_boundaries():
    m = SugyaBoundaries(
        sugyot=[
            {"title_he": "א", "start_seg": "S1", "end_seg": "S3"},
            {"title_he": "ב", "start_seg": "S3", "end_seg": "S4"},
        ]
    )
    r = run(m, seg_order={f"S{i}": i for i in range(1, 6)})
    assert "K8" in codes(r, "error")  # overlap
    m2 = SugyaBoundaries(sugyot=[{"title_he": "א", "start_seg": "S1", "end_seg": "S9"}])
    assert "K8" in codes(run(m2, seg_order={"S1": 1}), "error")  # outside the window
    m3 = SugyaBoundaries(sugyot=[{"title_he": "א", "start_seg": "S3", "end_seg": "S1"}])
    assert "K8" in codes(run(m3, seg_order={"S1": 1, "S3": 3}), "error")


# K9
def test_k9_chronology():
    eras = {"rav": "amoraim", "rashi": "rishonim"}
    early = Holder(label="רב", entity_slug="rav")
    late = Holder(label="רש״י", entity_slug="rashi")
    bad = pos(holder=early, relations=[{"rel": "disputes", "target": late.model_dump()}])
    assert "K9" in codes(run(bad, [rec("evidence[0]")], eras=eras), "error")
    ok = pos(holder=late, relations=[{"rel": "disputes", "target": early.model_dump()}])
    assert "K9" not in codes(run(ok, [rec("evidence[0]")], eras=eras), "error")
    # claims_error_in: only from a later (or same generation) authority
    sc = SourceClaim(
        claim_kind="girsa_error",
        holder=early,
        target=SourceClaimTarget(target_status="unresolved", holder=late),
    )
    assert "K9" in codes(run(sc, eras=eras), "error")


# K10 / K11
def sc(**kw) -> SourceClaim:
    base = dict(
        claim_kind="girsa_error",
        holder=Holder(label="הגר״א", entity_slug="gra"),
        holder_evidence=[Ev(seg="S1", quote="q", role="claims_error")],
        target=SourceClaimTarget(evidence=Ev(seg="S2", quote="q"), holder=Holder(label="רש״י")),
        proposed_alternative="נוסח אחר",
    )
    return SourceClaim(**{**base, **kw})


def sc_ev(he="located", te="located"):
    return [
        rec("holder_evidence[0]", "claims_error", he),
        rec("target.evidence", "states", te, seg_label="S2"),
    ]


def test_k10_holder_must_be_human_located_and_not_the_target():
    assert run(sc(), sc_ev()).ok
    assert "K10" in codes(run(sc(holder=Holder(label="model")), sc_ev()), "error")
    assert "K10" in codes(run(sc(), sc_ev(he="not_found")), "error")
    assert "K10" in codes(run(sc(holder=Holder(label="רש״י")), sc_ev()), "error")


def test_k11_unresolved_target_is_kept_and_unlocated_one_is_rejected():
    unresolved = sc(target=SourceClaimTarget(target_status="unresolved", work="Not in corpus"))
    assert run(unresolved, [rec("holder_evidence[0]", "claims_error")]).ok
    assert "K11" in codes(run(sc(), sc_ev(te="not_found")), "error")  # not forced
    # the corrected text must never stand in for the source text
    forced = sc(target=SourceClaimTarget(quote="נוסח אחר", evidence=Ev(seg="S2", quote="נוסח אחר")))
    r = run(forced, sc_ev(te="not_found"))
    assert any("in place of the source" in f.message for f in r.errors)


# K12
def test_k12_chakira_side_needs_proposer_evidence_and_interprets():
    def side(i, interprets="פירוש"):
        return dict(
            label_he=f"צד {i}",
            proposer=Holder(label="ר׳"),
            interprets=interprets,
            proposer_evidence=[Ev(seg="S1", quote="q", role="proposes")],
        )

    m = Chakira(question_he="מהו", sides=[side(1), side(2)])
    ev = [
        rec("sides[0].proposer_evidence[0]", "proposes"),
        rec("sides[1].proposer_evidence[0]", "proposes"),
    ]
    assert run(m, ev).ok
    assert "K12" in codes(run(m, ev[:1]), "error")
    m2 = Chakira(question_he="מהו", sides=[side(1), side(2, interprets="")])
    assert "K12" in codes(run(m2, ev), "error")
    aligned = side(1)
    aligned["aligned"] = [Holder(label="אחרון")]
    m3 = Chakira(question_he="מהו", sides=[aligned, side(2)])
    assert "K12" in codes(run(m3, ev), "error")  # aligned without asserted_by


# K13
def step(i, move="statement", responds=None, **kw) -> Step:
    kw.setdefault("evidence", [Ev(seg="S1", quote="q")])
    return Step(id=f"s{i}", move=move, responds_to=responds, text_he="טקסט", **kw)


def test_k13_graph():
    ev = [rec(f"steps[{i}].evidence[0]") for i in range(3)]
    good = SugyaSummary(
        title_he="ס",
        steps=[
            step(1),
            step(2, "objection", "s1", relation="attacks"),
            step(3, "answer", "s2", subtype="teyuvta"),
        ],
    )
    r = run(good, ev)
    assert not [f for f in r.errors if f.code == "K13"]
    later = SugyaSummary(title_he="ס", steps=[step(1, responds="s2"), step(2, responds="s1")])
    assert "K13" in codes(run(later, ev), "error")  # responds_to a later step
    bad_t = SugyaSummary(title_he="ס", steps=[step(1), step(2, "answer", "s1", subtype="teyuvta")])
    assert "K13" in codes(run(bad_t, ev), "error")  # teyuvta must close an objection
    refuted = SugyaSummary(
        title_he="ס", steps=[step(1, standing="refuted"), step(2, "answer", "s1")]
    )
    assert "K13" in codes(run(refuted, ev), "error")  # nobody attacks s1
    nofirst = SugyaSummary(title_he="ס", steps=[step(1), step(2)])
    assert "K13" in codes(run(nofirst, ev), "error")  # missing responds_to


# K14
def test_k14_marker_contradiction_is_a_warning_with_queue_priority():
    m = SugyaSummary(title_he="ס", steps=[step(1, "statement")])
    r = run(m, [rec("steps[0].evidence[0]")], markers={"S1": {"mitivei"}})
    assert "K14" in codes(r, "warn") and r.ok and r.queue_priority
    ok = SugyaSummary(title_he="ס", steps=[step(1, "objection")])
    assert not run(ok, [rec("steps[0].evidence[0]")], markers={"S1": {"mitivei"}}).warnings


# K15
def test_k15_position_needs_a_known_issue_and_similar_issue_warns():
    assert "K15" in codes(run(pos(issue=""), [rec("evidence[0]")]), "error")
    assert "K15" in codes(
        run(pos(issue="issue:9"), [rec("evidence[0]")], known_issues={"issue:1": "מהו הדין"}),
        "error",
    )
    r = run_checks(
        CheckContext(
            kind="issue",
            model=Issue(canonical_q_he="מהו הדין בשבת"),
            evidence=[rec("evidence[0]")],
            known_issues={"issue:1": "מהו הדין בשבת"},
        )
    )
    assert "K15" in codes(r, "warn")


def test_machloket_positions_must_belong_to_its_issue():
    m = Machloket(issue_id="issue:1", sides=[["p1"], ["p2"]])
    assert run(m, position_issue={"p1": "issue:1", "p2": "issue:1"}).ok
    assert "K15" in codes(run(m, position_issue={"p1": "issue:1", "p2": "issue:2"}), "error")
