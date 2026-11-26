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


def _personal_policy(tmp_path):
    from rav_ai.config import POLICY_PATH

    p = tmp_path / "policy.toml"
    p.write_text(
        POLICY_PATH.read_text(encoding="utf-8")
        + '\n[licenses."personal-use-only"]\ncan_index = true\ncan_display_public = false\n'
        "can_train = false\npersonal_only = true\n",
        encoding="utf-8",
    )
    return p


def _set_only_version(engine, title, license_str, display):
    with Session(engine) as s:
        w = s.scalars(select(Work).where(Work.title == title)).one()
        for v in s.scalars(select(Version).where(Version.work_id == w.id)):
            v.license, v.can_index, v.can_display_public, v.can_train = (
                license_str,
                True,
                display,
                False,
            )
        s.commit()
        return w.id


def test_personal_only_version_requires_personal_tier(world, tmp_path):
    from rav_knowledge.sources import required_license_tier

    policy = _personal_policy(tmp_path)
    wid = _set_only_version(world.engine, "Steinsaltz on Shabbat", "Personal-Use-Only", False)
    with Session(world.engine) as s:
        need, basis = required_license_tier(s, s.get(Work, wid), policy)
        assert need == "PERSONAL_ONLY_RESTRICTED" and "Personal-Use-Only" in basis[0]
    entry = (
        '[[source]]\ntitle = "Steinsaltz on Shabbat"\ntier = "modern_secondary"\nrole = "lens"\n'
    )
    # negative: a looser declaration (non-commercial, or public) is refused and nothing is written
    for declared in ("NON_COMMERCIAL_ONLY", "PUBLIC_COMMERCIAL"):
        p = _seed(tmp_path, entry + f'license_tier = "{declared}"\n')
        with Session(world.engine) as s:
            res = seed_sources(s, p, policy_path=policy)
            s.commit()
            assert res["registered"] == []
            assert "looser than PERSONAL_ONLY_RESTRICTED" in res["refused"][0]["reason"]
            assert s.get(KlSource, wid) is None
    p = _seed(tmp_path, entry + 'license_tier = "PERSONAL_ONLY_RESTRICTED"\n')
    with Session(world.engine) as s:
        res = seed_sources(s, p, policy_path=policy)
        s.commit()
        assert [r["title"] for r in res["registered"]] == ["Steinsaltz on Shabbat"]
        assert s.get(KlSource, wid).license_tier == "PERSONAL_ONLY_RESTRICTED"


def test_cc_by_nc_stays_non_commercial_not_personal(world):
    from rav_knowledge.sources import required_license_tier

    wid = _set_only_version(world.engine, "Steinsaltz on Shabbat", "CC-BY-NC", False)
    with Session(world.engine) as s:  # the shipped policy marks no class personal_only
        assert required_license_tier(s, s.get(Work, wid))[0] == "NON_COMMERCIAL_ONLY"


def test_shipped_seed_covers_the_codes_and_lists_the_requested_skips():
    seed = load_seed()
    by = {e["title"]: e for e in seed["source"]}
    for t in (
        "Shulchan Arukh, Orach Chayim",
        "Tur",
        "Magen Avraham",
        "Mishnah Berurah",
        "Turei Zahav on Shulchan Arukh, Orach Chayim",
        "Beur HaGra on Shulchan Arukh, Orach Chayim",
    ):
        assert t in by and by[t]["draft"].startswith(("§3", "§4"))
    assert by["Shulchan Arukh, Orach Chayim"]["tier"] == "codes"
    assert by["Magen Avraham"]["tier"] == "nosei_kelim"
    skipped = {e["title"] for e in seed["not_registered"]}
    for t in (
        "Sefat Emet",
        "Ohr Zarua",
        "Bnei Yissaschar",
        "Mishneh Torah, Scroll of Esther and Hanukkah",
        "Maggid Mishneh on Mishneh Torah, Scroll of Esther and Hanukkah",
        "Sha'ar HaTziyun",
    ):
        assert t in skipped and t not in by
