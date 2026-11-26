"""config/kl_sources_seed.toml: the reproducible kl_sources registry (fail-closed licensing)."""

from __future__ import annotations

import pytest
from rav_ai.db.models import KlSource, Version, Work
from sqlalchemy import select
from sqlalchemy.orm import Session

from rav_knowledge.sources import SEED_PATH, load_seed, seed_sources


def test_shipped_seed_is_valid_and_documents_every_skip():
    seed = load_seed()
    titles = {e["title"] for e in seed["source"]}
    assert "Shabbat" in titles and "Rashi on Shabbat" in titles and "Tosafot on Shabbat" in titles
    shabbat = next(e for e in seed["source"] if e["title"] == "Shabbat")
    assert (shabbat["tier"], shabbat["role"], shabbat["license_tier"]) == (
        "canonical",
        "primary",
        "NON_COMMERCIAL_ONLY",
    )
    # unknown / missing licenses and [לאמת] works are never registered, only reported
    skipped = {e["title"] for e in seed["not_registered"]}
    for t in (
        "Meiri on Shabbat",
        "Haggahot Ya'avetz on Shabbat",
        "Tziyyun LeNefesh Chayyah on Shabbat",
        "Rabbeinu Chananel on Shabbat",
        "Rashba on Shabbat",
    ):
        assert t in skipped and t not in titles
    assert all(e.get("draft") for e in seed["source"])  # every row cites the draft


def _seed(tmp_path, body: str):
    p = tmp_path / "seed.toml"
    p.write_text(body, encoding="utf-8")
    return p


def test_malformed_seed_is_refused(tmp_path):
    with pytest.raises(ValueError, match="license_tier is required"):
        load_seed(_seed(tmp_path, '[[source]]\ntitle="X"\ntier="rishonim"\nrole="primary"\n'))
    with pytest.raises(ValueError, match="tier"):
        load_seed(
            _seed(
                tmp_path,
                '[[source]]\ntitle="X"\ntier="gaonim"\nrole="primary"\n'
                'license_tier="PUBLIC_COMMERCIAL"\n',
            )
        )
    with pytest.raises(ValueError, match="reason"):
        load_seed(_seed(tmp_path, '[[not_registered]]\ntitle="X"\n'))


def test_seed_registers_rechecks_license_and_is_idempotent(world, tmp_path):
    with Session(world.engine) as s:  # Steinsaltz on Shabbat: make its only version non-commercial
        w = s.scalars(select(Work).where(Work.title == "Steinsaltz on Shabbat")).one()
        for v in s.scalars(select(Version).where(Version.work_id == w.id)):
            v.can_display_public = False
        s.commit()
    body = """
[[source]]
title = "Rashi on Shabbat"
tier = "rishonim"
role = "primary"
weight_tier = "core"
license_tier = "PUBLIC_COMMERCIAL"
draft = "§1"

[[source]]
title = "Steinsaltz on Shabbat"
tier = "modern_secondary"
role = "primary"
license_tier = "PUBLIC_COMMERCIAL"   # looser than its CC-BY-NC version: refused

[[source]]
title = "No Such Work on Shabbat"
tier = "acharonim"
role = "primary"
license_tier = "PUBLIC_COMMERCIAL"

[[not_registered]]
title = "Meiri on Shabbat"
reason = "license unknown"
"""
    p = _seed(tmp_path, body)
    with Session(world.engine) as s:
        dry = seed_sources(s, p, dry_run=True)
        assert [r["title"] for r in dry["registered"]] == ["Rashi on Shabbat"]
        s.rollback()
        rashi_id = s.scalars(select(Work.id).where(Work.title == "Rashi on Shabbat")).one()
        assert s.get(KlSource, rashi_id).role == "lens"  # dry run wrote nothing
        res = seed_sources(s, p)
        s.commit()
    refused = {r["title"]: r["reason"] for r in res["refused"]}
    assert "looser than NON_COMMERCIAL_ONLY" in refused["Steinsaltz on Shabbat"]
    assert refused["No Such Work on Shabbat"] == "work not in the corpus"
    assert res["not_registered"] == [{"title": "Meiri on Shabbat", "reason": "license unknown"}]
    with Session(world.engine) as s:
        src = s.get(KlSource, rashi_id)
        assert (src.tier, src.role, src.license_tier, src.weight_tier) == (
            "rishonim",
            "primary",
            "PUBLIC_COMMERCIAL",
            "core",
        )
        assert src.license_basis.startswith("Sefaria: ") and "core-books-draft" in src.note
        stein = s.scalars(select(Work).where(Work.title == "Steinsaltz on Shabbat")).one()
        assert s.get(KlSource, stein.id) is None
        n = len(list(s.scalars(select(KlSource))))
        seed_sources(s, p)  # idempotent
        s.commit()
        assert len(list(s.scalars(select(KlSource)))) == n


def test_seed_path_is_in_config():
    assert SEED_PATH.name == "kl_sources_seed.toml" and SEED_PATH.parent.name == "config"
