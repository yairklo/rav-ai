"""Fixtures: a synthetic mini-export (same layout as the real bucket) + a throw-away Postgres DB.

Requires the docker-compose Postgres (`docker compose up -d`); tests skip if it is unreachable.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from rav_ai.config import LicensePolicy
from rav_ai.db.session import make_engine, migrate

ADMIN_URL = "postgresql+psycopg://rav:rav@localhost:54329/postgres"
SA = "Shulchan Arukh, Orach Chayim"
MB = "Mishnah Berurah"
BER = "Berakhot"
RASHI = "Rashi on Berakhot"
HEB1 = "יִתְגַּבֵּר כַּאֲרִי <b>לַעֲמֹד</b>  בַּבֹּקֶר\n"  # niqqud, markup, odd whitespace


def write_json(root: Path, rel: str, doc) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


def schema_doc(title, cats, section_names, address_types, era="AH", dependence=None, base=None):
    d = {
        "title": title,
        "heTitle": title,
        "categories": cats,
        "era": era,
        "authors": [{"en": f"Author of {title}", "he": "מחבר", "slug": f"author-{title}"}],
        "sectionNames": section_names,
        "compDate": [1500],
        "schema": {
            "nodeType": "JaggedArrayNode",
            "depth": len(section_names),
            "addressTypes": address_types,
            "sectionNames": section_names,
            "title": title,
            "key": title,
        },
    }
    if dependence:
        d["dependence"] = dependence
        d["base_text_titles"] = [{"en": base, "he": base}]
    return d


def version_doc(title, lang, vtitle, lic, text, primary=False):
    d = {
        "title": title,
        "language": lang,
        "versionTitle": vtitle,
        "text": text,
        "isPrimary": primary,
        "priority": 1.0,
        "status": "locked",
        "versionSource": "http://example.org",
    }
    if lic is not None:
        d["license"] = lic
    return d


def build_export(r: Path, sa_second: str = "סעיף ב") -> None:
    (r).mkdir(parents=True, exist_ok=True)
    (r / "last_export.txt").write_text("2026-01-01T00:00:00", encoding="utf-8")
    write_json(
        r,
        "table_of_contents.json",
        [
            {
                "contents": [
                    {"title": SA},
                    {"title": MB},
                    {"title": BER},
                    {"title": RASHI},
                    {"title": "English Only Book"},
                ]
            }
        ],
    )
    write_json(
        r,
        f"schemas/{SA.replace(' ', '_')}.json",
        schema_doc(SA, ["Halakhah", "Shulchan Arukh"], ["Siman", "Seif"], ["Siman", "Seif"]),
    )
    base = f"json/Halakhah/Shulchan Arukh/{SA}"
    write_json(
        r,
        f"{base}/Hebrew/Torat Emet.json",
        version_doc(
            SA,
            "he",
            "Torat Emet",
            "Public Domain",
            [[HEB1, sa_second], ["", "סעיף ג בסימן ב"]],
            primary=True,
        ),
    )
    write_json(
        r,
        f"{base}/Hebrew/NC Edition.json",
        version_doc(SA, "he", "NC Edition", "CC-BY-NC", [["נ״ק א", "נ״ק ב"]]),
    )
    write_json(
        r,
        f"{base}/Hebrew/Restricted.json",
        version_doc(SA, "he", "Restricted", "Copyrighted", [["מוגן"]]),
    )
    write_json(
        r,
        f"{base}/Hebrew/No License.json",
        version_doc(SA, "he", "No License", None, [["בלי רישיון"]]),
    )
    write_json(r, f"{base}/English/English.json", version_doc(SA, "en", "Eng", "CC0", [["x"]]))
    write_json(r, f"{base}/Hebrew/merged.json", {"title": SA, "text": [["ignored"]]})

    write_json(
        r,
        f"schemas/{MB.replace(' ', '_')}.json",
        schema_doc(
            MB,
            ["Halakhah", "Shulchan Arukh", "Commentary"],
            ["Siman", "Seif"],
            ["Siman", "Seif"],
            dependence="Commentary",
            base=SA,
        ),
    )
    write_json(
        r,
        f"json/Halakhah/Shulchan Arukh/Commentary/{MB}/Hebrew/MB.json",
        version_doc(MB, "he", "MB", "CC-BY-SA", [['משנ"ב א', 'משנ"ב ב']], primary=True),
    )

    write_json(
        r,
        f"schemas/{BER}.json",
        schema_doc(BER, ["Talmud", "Bavli"], ["Daf", "Line"], ["Talmud", "Integer"], era="A"),
    )
    write_json(
        r,
        f"json/Talmud/Bavli/Seder Zeraim/{BER}/Hebrew/Vilna.json",
        version_doc(
            BER, "he", "Vilna", "Public Domain", [[], [], ["a1", "a2"], ["b1"]], primary=True
        ),
    )
    write_json(
        r,
        f"schemas/{RASHI.replace(' ', '_')}.json",
        schema_doc(
            RASHI,
            ["Talmud", "Bavli", "Rishonim on Talmud", "Rashi"],
            ["Daf", "Line", "Comment"],
            ["Talmud", "Integer", "Integer"],
            era="RI",
            dependence="Commentary",
            base=BER,
        ),
    )
    write_json(
        r,
        f"json/Talmud/Bavli/Rishonim on Talmud/Rashi/{RASHI}/Hebrew/Rashi.json",
        version_doc(
            RASHI,
            "he",
            "Rashi",
            "Public Domain",
            [[], [], [["r a1 c1", "r a1 c2"], ["r a2 c1"]], []],
            primary=True,
        ),
    )

    (r / "links").mkdir(exist_ok=True)
    hdr = "Citation 1,Citation 2,Conection Type,Text 1,Text 2,Category 1,Category 2\n"
    rows = [
        (f"{SA} 1:1", f"{MB} 1:1", "commentary", SA, MB),
        (f"{SA} 1:1-2", f"{BER} 2a:1-2", "", SA, BER),  # range <-> range
        (f"{BER} 2b:1", f"{SA} 9:9", "quotation", BER, SA),  # SA 9:9 does not exist
        (f"{RASHI} 2a:1:1", f"{BER} 2a:1", "commentary", RASHI, BER),
        (f"{BER} 2a", f"{MB} 1", "", BER, MB),  # section-level endpoints
        (f"{SA} 1:1", "English Only Book 1", "", SA, "English Only Book"),
        (f"{SA} 1:1", "Mystery Book 1", "", SA, "Mystery Book"),
    ]
    body = "".join(
        f'"{a}","{b}",{t},"{x}","{y}",c,d\n' for a, b, t, x, y in rows for c, d in [("h", "h")]
    )
    (r / "links" / "links0.csv").write_text(hdr + body, encoding="utf-8")
    (r / "links" / "links_by_book.csv").write_text("Text 1,Text 2,Link Count\na,b,3\n")


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    r = tmp_path / "raw"
    build_export(r)
    return r


@pytest.fixture
def policy() -> LicensePolicy:
    return LicensePolicy.load()


@pytest.fixture
def engine():
    name = f"rav_test_{uuid.uuid4().hex[:8]}"
    try:
        admin = create_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")
        with admin.connect() as c:
            c.execute(text(f'CREATE DATABASE "{name}"'))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Postgres not available (docker compose up -d): {e}")
    url = f"postgresql+psycopg://rav:rav@localhost:54329/{name}"
    migrate(url)
    eng = make_engine(url)
    yield eng
    eng.dispose()
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
    admin.dispose()
