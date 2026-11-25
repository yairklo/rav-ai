from __future__ import annotations

import json

from rav_ai.db.models import Author, KlItem, KlLink, KlSource, Snapshot, Version
from sqlalchemy import select
from sqlalchemy.orm import Session

from conftest import add_work
from fakes import ScriptedLLM
from helpers import cfg, pipeline, run, s1_responder, s2_responder
from rav_knowledge import cli, queries
from rav_knowledge.licensing import admit, item_permission
from rav_knowledge.machloket import build_machloket_candidate
from rav_knowledge.sources import LENS_RECORDS, seed_lens_records
from rav_knowledge.store import load_input, write_item

# ---- licensing -------------------------------------------------------------------------------


def _extra_works(engine):
    """lens works: not indexable / personal-only / unregistered / reference-only."""
    texts = {"1:1": "קטע של מקור"}
    out = {}
    with Session(engine) as s:
        snap = s.scalars(select(Snapshot)).first()
        au = s.scalars(select(Author)).first()
        specs = {
            "NoIndex": dict(can_index=False, source=("acharonim", "lens")),
            "Personal": dict(
                source=("lamdanut", "lens"),
                lic="PERSONAL_ONLY_RESTRICTED",
                can_display=False,
                can_train=False,
            ),
            "Unreg": dict(),
            "RefOnly": dict(source=("modern_secondary", "reference_only")),
        }
        for name, kw in specs.items():
            _, _, segs = add_work(s, snap, au, name, "Talmud", ["Talmud"], texts, **kw)
            out[name] = segs["1:1"]
        s.commit()
    return out


def test_lens_without_can_index_and_unregistered_sources_never_enter(world):
    seg = _extra_works(world.engine)
    c = cfg(mode="public")
    with Session(world.engine) as s:
        cases = (
            ("NoIndex", "cannot_index"),
            ("Unreg", "unregistered_source"),
            ("RefOnly", "reference_only"),
            ("Personal", "personal_only_restricted"),
        )
        for name, why in cases:
            inp = load_input(s, [seg[name]], c)
            assert inp.segs == [] and inp.excluded == {why: 1}, name


def test_personal_only_needs_personal_mode_and_the_explicit_flag(world):
    seg = _extra_works(world.engine)
    with Session(world.engine) as s:
        for mode, allow, admitted in (("public", True, False), ("personal", False, False),
                                      ("personal", True, True)):  # fmt: skip
            c = cfg(mode=mode, allow_personal_only=allow)
            inp = load_input(s, [seg["Personal"]], c)
            assert bool(inp.segs) is admitted, (mode, allow)
        used = [(x.version, x.source) for x in inp.segs]
        perm = item_permission(used)
        assert (perm.can_display_public, perm.can_train) == (False, False)  # inherited


def test_permission_is_an_and_and_fails_closed():
    v = lambda d, t: Version(can_display_public=d, can_train=t)  # noqa: E731
    assert item_permission([]).can_display_public is False
    p = item_permission([(v(True, True), None), (v(True, False), None)])
    assert (p.can_display_public, p.can_train) == (True, False)
    nc = KlSource(license_tier="NON_COMMERCIAL_ONLY")
    assert item_permission([(v(True, True), nc)]).can_train is False
    assert item_permission([(v(True, True), None)], steinsaltz_aid=True).can_display_public is False
    assert admit(None, v(True, True), cfg())[0] is False


def test_lens_records_are_registered_personal_only_and_idempotent(world):
    with Session(world.engine) as s:
        assert seed_lens_records(s) == len(LENS_RECORDS)
        assert seed_lens_records(s) == 0
        s.commit()
        rows = {r.work_id: r for r in s.scalars(select(KlSource))}
        from rav_ai.db.models import Work

        by_title = {w.title: rows[w.id] for w in s.scalars(select(Work)) if w.id in rows}
    for t in ("Asif", "Minchat Asher", "Rav Asher Weiss - Shiurim (written)",
              "Rav Asher Weiss - Shiurim (transcripts)"):  # fmt: skip
        assert by_title[t].license_tier == "PERSONAL_ONLY_RESTRICTED" and by_title[t].role == "lens"
        assert "can_train=false" in by_title[t].note
    assert by_title["Virtual Beit Midrash (VBM)"].role == "reference_only"
    assert by_title["HebrewBooks"].role == "reference_only"


# ---- machloket + the three axes ---------------------------------------------------------------


def _item(s, kind, key, body, **kw):
    return write_item(s, kind=kind, item_key=key, scope_key="s", body=body, input_hash=key,
                      job_id=None, model=None, prompt_version="t", prompt_hash=None,
                      schema_version=1, **kw)  # fmt: skip


def _setup_issue(engine, labels):
    with Session(engine) as s:
        sg = _item(s, "sugya", "sugya:S1", {"title_he": "סוגיה"}, title_he="סוגיה")
        iss = _item(s, "issue", "issue:1", {"canonical_q_he": "מהו"}, title_he="מהו")
        s.add(KlLink(from_item_id=iss.id, to_item_id=sg.id, rel="raised_in"))
        for i, side in enumerate(labels):
            body = {"issue": "issue:1", "side_label": side, "holder": {"tier": "canonical"}}
            p = _item(s, "position", f"position:{i}", body)
            s.add(KlLink(from_item_id=p.id, to_item_id=iss.id, rel="position_on"))
        s.commit()
        return iss.id


def test_machloket_is_a_full_item_created_for_two_or_more_sides(world):
    _setup_issue(world.engine, ["רב", "שמואל"])
    with Session(world.engine) as s:
        iss = s.scalars(select(KlItem).where(KlItem.item_key == "issue:1")).one()
        m = build_machloket_candidate(s, iss)
        s.commit()
        assert m.kind == "machloket" and m.status == "generated" and m.is_current
        assert m.body["issue_id"] == "issue:1" and len(m.body["sides"]) == 2
        assert m.body["sugya_ids"] == ["sugya:S1"] and m.body["machloket_kind"] == "gemara"
        assert build_machloket_candidate(s, iss).id == m.id  # idempotent
    # the three axes, directly
    with Session(world.engine) as s:
        assert [i.item_key for i in queries.issues_by_sugya(s, "sugya:S1")] == ["issue:1"]
        assert [i.item_key for i in queries.machlokot_by_issue(s, "issue:1")] == [
            "machloket:issue:1"
        ]
        assert [i.item_key for i in queries.sugyot_by_machloket(s, "machloket:issue:1")] == [
            "sugya:S1"
        ]
        assert queries.issues_by_sugya(s, "nope") == []


def test_no_machloket_for_one_side_or_one_position(world):
    _setup_issue(world.engine, ["אותו צד", "אותו צד"])
    with Session(world.engine) as s:
        iss = s.scalars(select(KlItem).where(KlItem.item_key == "issue:1")).one()
        assert build_machloket_candidate(s, iss) is None


# ---- CLI + estimate -----------------------------------------------------------------------------


def test_cli_estimate_status_and_promote_refusal(world, capsys, monkeypatch, tmp_path):
    url = world.engine.test_url
    assert cli.main(["--db", url, "estimate", "Shabbat 21a-22b", "--stages", "s0b,s1,s2"]) == 0
    out = capsys.readouterr().out
    assert "TOTAL: $" in out and "21a" in out and "s1" in out and "token count:" in out
    assert cli.main(["--db", url, "run", "s1", "Shabbat 21a-22b", "--dry-run"]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["tasks"] == 3 and dry["est_cost_usd"] > 0 and dry["job_id"] is None  # no writes
    run(world.engine, "s0", None, "Shabbat 21a-22b")
    assert cli.main(["--db", url, "status"]) == 0
    assert '"stage": "s0"' in capsys.readouterr().out
    assert cli.main(["--db", url, "promote", "--job", "999"]) == 2
    monkeypatch.setattr("rav_knowledge.runner.REPORT_DIR", tmp_path)
    assert cli.main(["--db", url, "report", "--job", "1"]) == 0
    assert (tmp_path / "job_1.json").exists()
    assert cli.main(["--db", url, "seed-lenses"]) == 0
    assert cli.main(["--db", url, "invalidate"]) == 0
    assert cli.main(["--db", url, "conflicts"]) == 0


def test_report_gate_is_used_by_promote(world, monkeypatch, tmp_path):
    import pytest

    from rav_knowledge import runner

    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path)
    j1 = pipeline(world.engine, ScriptedLLM(s1_responder()), ScriptedLLM(s2_responder)).job_id
    rep = runner.report(world.engine, j1)
    assert rep["gate"]["passed"] and rep["located_fraction"] == 1.0
    assert rep["evidence"]["located"] == 6
    c = cfg()
    c.models = {**c.models, "s2": "anthropic/other"}
    j2 = run(world.engine, "s2", ScriptedLLM(s2_responder), config=c).job_id
    # a failing kl-report gate blocks promote even though the K checks pass
    (tmp_path / f"job_{j2}.json").write_text(json.dumps({"gate": {"passed": False}}))
    with pytest.raises(runner.PromoteRefused):
        runner.promote(world.engine, j2)
    (tmp_path / f"job_{j2}.json").write_text(json.dumps({"gate": {"passed": True}}))
    assert runner.promote(world.engine, j2)["gate"]["source"] == "kl-report"


def test_unimplemented_stage_is_reserved_in_the_dag(world):
    import pytest

    from rav_knowledge.stages import DAG

    assert {"s3", "s3l", "s4", "s4g", "s5"} <= set(DAG)
    with pytest.raises(ValueError, match="reserved"):
        run(world.engine, "s3", None)
