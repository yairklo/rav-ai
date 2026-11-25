"""Seed of the knowledge layer (kl_*) on top of the harness corpus seed. Hebrew text is toy."""

from __future__ import annotations

from types import SimpleNamespace

from rav_ai.db.models import (
    Annotation,
    Author,
    Edge,
    KlCoverage,
    KlEvidence,
    KlItem,
    KlItemSegment,
    KlJob,
    KlLink,
    KlReception,
    KlRelation,
    KlSource,
    KlWitness,
    Segment,
    SegmentText,
    Snapshot,
    Version,
    Work,
)
from sqlalchemy.orm import Session
from textnorm import normalize

SHABBAT_TEXT = "מצוה להניח נר חנוכה על פתח ביתו מבחוץ"
RASHI_TEXT = "פתח ביתו - בטפח הסמוך לפתח"
TUR_TEXT = "ומצותה להניח על פתח ביתו"
SA_TEXT = "ומצותה להניח על פתח ביתו מבחוץ"
BY_TEXT = 'ובית יוסף מביא דברי הרשב"א'
LENS_TEXT = 'לפי הבנתי ברש"י כאן'
GAMAL = "גמל טעון פשתן"
PESACH_TEXT = "מצה שמורה לשם מצה"
SCOPE = "Shabbat 21a-24a"


def _pos(label, view, side=None, level=None, **extra):
    return {
        "issue": "issue:candle-place",
        "holder": {"label": label, "known_via": "direct"},
        "view_he": view,
        "side_label": side,
        "normative_level": level,
        **extra,
    }


def seed_kl(s: Session, *, min_coverage_gap: bool = False) -> SimpleNamespace:
    snap = s.query(Snapshot).first()
    author = s.query(Author).first()
    w = SimpleNamespace()
    n = {"i": 0}

    def work(
        title,
        period="rishonim",
        tier="rishonim",
        role="primary",
        license_tier="PUBLIC_COMMERCIAL",
        weight="core",
        pub=True,
    ):
        wk = Work(
            title=title,
            author_id=author.id,
            category="Test",
            category_path=["Test"],
            period=period,
            section_names=["Daf", "Line"],
            schema={},
        )
        s.add(wk)
        s.flush()
        v = Version(
            work_id=wk.id,
            source="sefaria",
            language="he",
            version_title="V",
            license="Public Domain" if pub else "Copyrighted",
            can_index=True,
            can_display_public=pub,
            can_train=False,
            is_primary=True,
            first_snapshot_id=snap.id,
            last_snapshot_id=snap.id,
        )
        s.add(v)
        s.add(
            KlSource(
                work_id=wk.id,
                tier=tier,
                role=role,
                license_tier=license_tier,
                weight_tier=weight,
            )
        )
        s.flush()
        return wk, v

    def seg(wk, v, ref, text, parent):
        sg = Segment(
            ref=ref,
            work_id=wk.id,
            node_path=[],
            section_path=[],
            indices=[],
            section_names=[],
            sort_key=ref,
            parent_ref=parent,
        )
        s.add(sg)
        s.flush()
        s.add(
            SegmentText(
                segment_id=sg.id,
                version_id=v.id,
                text_raw=text,
                raw_hash="h" + ref,
                text_a=normalize(text, "A").text,
                text_b=normalize(text, "B").text,
                offsets_a=b"",
                offsets_b=b"",
            )
        )
        s.flush()
        return sg

    shab, shab_v = work("Shabbat", "amoraim", "canonical")
    rashi, rashi_v = work("Rashi on Shabbat")
    tur, tur_v = work("Tur, Orach Chayim", tier="codes")
    sa, sa_v = work("Shulchan Arukh, Orach Chayim", tier="codes")
    by, by_v = work("Beit Yosef, Orach Chayim", tier="nosei_kelim")
    mag, mag_v = work("Magen Avraham", tier="acharonim", weight="major")
    lens, lens_v = work("Lens on Rashi", tier="modern_secondary", role="lens", weight="extended")
    weiss, weiss_v = work(
        "Weiss Principles",
        tier="modern_secondary",
        role="lens",
        license_tier="PERSONAL_ONLY_RESTRICTED",
        weight="extended",
        pub=False,
    )
    cm, cm_v = work("Shulchan Arukh, Choshen Mishpat", tier="codes")
    pes, pes_v = work("Pesachim", "amoraim", "canonical")
    tos, _ = work("Tosafot on Shabbat")
    w.shab22 = seg(shab, shab_v, "Shabbat 22a:1", SHABBAT_TEXT, "Shabbat 22a")
    w.shab21 = seg(shab, shab_v, "Shabbat 21b:1", SHABBAT_TEXT, "Shabbat 21b")
    w.rashi = seg(rashi, rashi_v, "Rashi on Shabbat 22a:1:1", RASHI_TEXT, "Rashi on Shabbat 22a:1")
    w.tur = seg(tur, tur_v, "Tur, Orach Chayim 670:1", TUR_TEXT, "Tur, Orach Chayim 670")
    w.sa = seg(
        sa, sa_v, "Shulchan Arukh, Orach Chayim 670:1", SA_TEXT, "Shulchan Arukh, Orach Chayim 670"
    )
    w.by = seg(by, by_v, "Beit Yosef, Orach Chayim 670:1", BY_TEXT, "Beit Yosef, Orach Chayim 670")
    w.mag = seg(mag, mag_v, "Magen Avraham 670:1", TUR_TEXT, "Magen Avraham 670")
    w.lens = seg(lens, lens_v, "Lens on Rashi 1:1", LENS_TEXT, "Lens on Rashi 1")
    w.cm = seg(
        cm,
        cm_v,
        "Shulchan Arukh, Choshen Mishpat 418:1",
        GAMAL,
        "Shulchan Arukh, Choshen Mishpat 418",
    )
    w.pes = seg(pes, pes_v, "Pesachim 112a:1", PESACH_TEXT, "Pesachim 112a")
    w.works = SimpleNamespace(
        shab=shab, rashi=rashi, tur=tur, sa=sa, by=by, mag=mag, lens=lens, weiss=weiss, tos=tos
    )

    job = KlJob(stage="s2", scope_key=SCOPE, status="promoted")
    job2 = KlJob(stage="s2", scope_key=SCOPE, status="done")
    s.add_all([job, job2])
    s.flush()

    def item(
        kind,
        key,
        body,
        *,
        work=None,
        segs=(),
        status="verified",
        conf=0.9,
        stale=False,
        pub=True,
        current=True,
        text="",
        scope=SCOPE,
        job_id=None,
        prompt_version="v1",
    ):
        n["i"] += 1
        it = KlItem(
            kind=kind,
            item_key=key,
            version=1,
            scope_key=scope,
            work_id=work.id if work else None,
            body=body,
            schema_version=2,
            status=status,
            is_current=current,
            stale=stale,
            confidence=conf,
            input_hash=f"h{n['i']}",
            can_display_public=pub,
            can_train=False,
            search_text_he=normalize(text, "B").text if text else None,
            job_id=job_id,
            prompt_version=prompt_version,
        )
        s.add(it)
        s.flush()
        for i, sg in enumerate(segs):
            s.add(KlItemSegment(item_id=it.id, segment_id=sg.id, role="member", ord=i))
        s.flush()
        return it

    def ev(it, sg, claim=None, role="states", quote="x"):
        e = KlEvidence(
            item_id=it.id,
            claim_id=claim,
            segment_id=sg.id,
            quote=quote,
            role=role,
            locate_status="located",
        )
        s.add(e)
        s.flush()
        return e

    w.sugya = item(
        "sugya",
        "sugya:shab-22a",
        {"title_he": "מקום הדלקה"},
        work=shab,
        segs=[w.shab22],
        text="נר חנוכה פתח",
        job_id=job.id,
    )
    w.summary = item(
        "sugya_summary",
        "summary:shab-22a",
        {
            "title_he": "מקום",
            "steps": [
                {
                    "id": "S1",
                    "move": "question",
                    "text_he": "היכן מניחים " + "מילה " * 40,
                    "standing": "accepted",
                    "readings": [
                        {"text_he": "הסבר ראשון", "holder": {"label": 'רשב"א'}},
                        {"text_he": "הסבר שני", "holder": {"label": 'ריטב"א'}},
                    ],
                }
            ],
        },
        work=shab,
        segs=[w.shab22],
    )
    s.add(KlLink(from_item_id=w.summary.id, to_item_id=w.sugya.id, rel="summarizes"))
    ev(w.summary, w.shab22, "steps[0].evidence[0]")

    w.issue = item(
        "issue",
        "issue:candle-place",
        {"canonical_q_he": "היכן מניחים נר חנוכה", "anchors": []},
        work=shab,
        segs=[w.shab22],
        text="היכן מניחים נר חנוכה פתח ביתו",
        job_id=job.id,
    )
    w.issue2 = item(
        "issue",
        "issue:gamal-pishtan",
        {"canonical_q_he": "גמל טעון פשתן"},
        work=cm,
        segs=[w.cm],
        text="גמל טעון פשתן נר",
        scope="Shulchan Arukh, Choshen Mishpat 418-418",
    )
    s.add(KlLink(from_item_id=w.issue2.id, to_item_id=w.issue.id, rel="depends_on"))
    ev(w.issue2, w.cm)
    ev(w.issue, w.shab22)

    def position(key, wk, body, sg, **kw):
        p = item("position", key, body, work=wk, segs=[sg], **kw)
        s.add(KlLink(from_item_id=p.id, to_item_id=w.issue.id, rel="position_on"))
        s.flush()
        ev(p, sg, "evidence[0]")
        return p

    w.p_rashi = position(
        "pos:rashi", rashi, _pos('רש"י', "בטפח הסמוך לפתח", "א"), w.rashi, job_id=job.id
    )
    w.p_tur = position(
        "pos:tur", tur, _pos("טור", "על פתח ביתו", "א", unusual_interpretation=True), w.tur
    )
    w.p_sa = position("pos:sa", sa, _pos("המחבר", "מבחוץ", "א", "documented_practice"), w.sa)
    w.p_by = position("pos:by", by, _pos("בית יוסף", 'מביא הרשב"א', "ב"), w.by)
    w.p_mag = position("pos:mag", mag, _pos("מגן אברהם", "כטור", "א"), w.mag)
    w.p_lens = position("pos:lens", lens, _pos("עדשה", "לפי הבנתי", "א"), w.lens)
    w.p_rej = position("pos:rejected", tur, _pos("נדחה", "x"), w.tur, status="rejected")
    w.p_stale = position("pos:stale", tur, _pos("ישן", "x"), w.tur, stale=True)
    w.p_low = position("pos:low", tur, _pos("חלש", "x"), w.tur, conf=0.1, status="generated")
    w.p_nopub = position("pos:nopub", tur, _pos("סגור", "x"), w.tur, pub=False)
    w.p_old = position("pos:old", tur, _pos("ישן", "x"), w.tur, current=False)

    # reception: the Rema-like acharon is unusual; rejected only for lemaase
    s.add_all(
        [
            KlReception(
                target_item_id=w.p_rashi.id,
                target_claim="",
                context="iyun",
                weight_label="primary",
                basis={"why": "x"},
                policy_version="v3",
            ),
            KlReception(
                target_item_id=w.p_rashi.id,
                target_claim="",
                context="lemaase",
                weight_label="rejected",
                basis={},
                policy_version="v3",
            ),
            KlReception(
                target_item_id=w.p_tur.id,
                target_claim="",
                context="lemaase",
                weight_label="primary",
                basis={"unusual_interpretation": True},
                policy_version="v3",
            ),
        ]
    )
    # relations: gemara-ish chain of four levels around the siman (psak chain)
    gem = item("position", "pos:gemara", _pos("הגמרא", "הדלקה בפתח", "א"), work=shab, segs=[w.tur])
    ev(gem, w.shab22)
    e1 = ev(w.p_rashi, w.rashi, "evidence[1]")
    s.add_all(
        [
            KlRelation(
                from_item_id=w.p_rashi.id,
                to_item_id=gem.id,
                rel="derived_from",
                attribution="direct",
                stance="asserts",
                evidence_ids=[e1.id],
            ),
            KlRelation(
                from_item_id=w.p_tur.id,
                to_item_id=w.p_rashi.id,
                rel="rules_like",
                attribution="direct",
                stance="asserts",
                evidence_ids=[e1.id],
            ),
            KlRelation(
                from_item_id=w.p_mag.id,
                to_item_id=w.p_tur.id,
                rel="rules_like",
                attribution="inferred_by_system",
                stance="suggests",
                evidence_ids=[e1.id],
            ),
            KlRelation(
                from_item_id=w.p_by.id, to_item_id=w.p_lens.id, rel="disputes", evidence_ids=[]
            ),
        ]
    )
    w.gem = gem
    w.rel_tur = s.query(KlRelation).filter_by(from_item_id=w.p_tur.id).one()

    w.sc = item(
        "source_claim",
        "claim:err",
        {
            "claim_kind": "girsa_error",
            "target": {
                "work": "Rashi",
                "ref": "Rashi on Shabbat 22a:1:1",
                "quote": "בטפח",
                "target_status": "located",
            },
            "holder": {"label": 'הב"ח'},
            "proposed_alternative": "בטפחיים",
        },
        work=rashi,
        segs=[w.rashi],
    )
    s.add(KlLink(from_item_id=w.sc.id, to_item_id=w.p_rashi.id, rel="discusses"))
    s.flush()
    ev(w.sc, w.rashi)

    chak_body = {
        "question_he": "האם מצוה בפתח",
        "sides": [
            {"label_he": "בפתח", "proposer": {"label": 'רש"י'}, "aligned": [{"label": "טור"}]},
            {"label_he": "בחוץ", "proposer": {"label": "בית יוסף"}},
        ],
    }
    w.chak_ok = item(
        "chakira",
        "chak:ok",
        {**chak_body, "cluster_status": "verified"},
        work=rashi,
        segs=[w.shab22],
        text="מצוה בפתח",
    )
    w.chak_no = item(
        "chakira",
        "chak:no",
        {**chak_body, "cluster_status": "candidate"},
        work=rashi,
        segs=[w.shab22],
        text="מצוה בפתח",
    )
    ev(w.chak_ok, w.shab22, "sides[0].proposer_evidence[0]")
    w.mk = item(
        "machloket",
        "mk:1",
        {"issue_id": "issue:candle-place", "sides": [["pos:rashi"], ["pos:by"]], "crux_he": "מקום"},
        work=shab,
        segs=[w.shab22],
    )
    w.oq = item(
        "open_question",
        "oq:1",
        {"issue": "issue:candle-place", "kind": "teiku", "text_he": "לא נפשט"},
        work=shab,
        segs=[w.shab22],
    )
    w.mp = item(
        "method_principle",
        "mp:1",
        {"principle_he": "לשקול כוונה", "source": {"label": "הרב וייס"}},
        work=weiss,
        segs=[w.shab22],
        text="כוונה",
    )
    wit = KlWitness(siglum="כי-X", kind="manuscript", can_index=True, can_display_public=True)
    s.add(wit)
    s.flush()
    w.witness = wit
    w.girsa = item(
        "girsa_variant",
        "girsa:1",
        {"witness_siglum": "כי-X", "variant_text": "ליתא"},
        work=shab,
        segs=[w.shab22],
    )
    w.witness_lonely = KlWitness(
        siglum="כי-Y", kind="manuscript", can_index=False, can_display_public=False
    )
    s.add(w.witness_lonely)

    s.add(
        Edge(
            segment_a_id=w.shab22.id,
            segment_b_id=w.tur.id,
            edge_type="ein mishpat / ner mitsvah",
            origin="sefaria",
            snapshot_id=snap.id,
            citation_a="שבת כב",
            citation_b="טור תר",
        )
    )
    s.add(
        Annotation(
            segment_id=w.shab22.id,
            version_id=shab_v.id,
            span_start=0,
            span_end=4,
            kind="abbrev",
            candidates=[{"text": "מצוה להניח"}],
            chosen="מצוה להניח",
            origin="s0",
            status="verified",
        )
    )
    if min_coverage_gap:
        s.add(KlCoverage(scope_key=SCOPE, work_id=tos.id, status="not_ingested"))
    s.commit()
    return w
