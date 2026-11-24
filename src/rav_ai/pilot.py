"""Pilot: Hilkhot Chanukah, assembled layer by layer from the already-ingested corpus.

Nothing is ingested here: the whole Hebrew Sefaria export is already in the database. This
module selects the pilot slice by layer, measures it, checks that the chain
Gemara -> Rishonim -> Rambam -> Tur/Beit Yosef -> Shulchan Arukh (+Rema, nosei kelim)
-> Mishnah Berurah is connected by edges, and lists gaps (missing texts / missing links).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.orm import Session

SEED_WORK = "Shabbat"
SEED_DAFIM = ["21a", "21b", "22a", "22b", "23a", "23b", "24a"]  # Shabbat 21a-24a
SIMAN_RANGE = (670, 693)  # Orach Chayim, Hilkhot Chanukah (Tur/BY/SA/MB numbering)
RAMBAM_CHAPTERS = ("3", "4")
RISHONIM = {  # label -> title regexes (Postgres ~*) of works commenting on the Bavli
    "Rashi": r"^Rashi on Shabbat$",
    "Tosafot": r"^Tosafot on Shabbat$",
    "Ramban": r"^(Chiddushei HaRamban|Ramban) on Shabbat$",
    "Rashba": r"^(Chiddushei HaRashba|Rashba) on Shabbat$",
    "Ritva": r"^(Chiddushei HaRitva|Ritva) on Shabbat$",
    "Ran": r"^(Chiddushei HaRan|Ran) on Shabbat$",
    "Rosh": r"^Rosh on Shabbat$",
}
LAYERS = [
    "gemara_seed",
    "mishnah",
    "gemara_other",
    "rishonim",
    "rambam",
    "tur",
    "beit_yosef",
    "shulchan_arukh",
    "nosei_keilim",
    "mishnah_berurah",
]
CHAIN = [
    ("gemara", ["gemara_seed", "gemara_other", "mishnah"]),
    ("rishonim", ["rishonim"]),
    ("rambam", ["rambam"]),
    ("tur", ["tur"]),
    ("beit_yosef", ["beit_yosef"]),
    ("shulchan_arukh", ["shulchan_arukh"]),
    ("nosei_keilim", ["nosei_keilim"]),
    ("mishnah_berurah", ["mishnah_berurah"]),
]


@dataclass
class PilotResult:
    layers: dict = field(default_factory=dict)  # layer -> {segments, words, works:[...]}
    edge_matrix: dict = field(default_factory=dict)  # (layerA, layerB) -> count
    edge_types: dict = field(default_factory=dict)
    gaps: list = field(default_factory=list)
    chain: dict = field(default_factory=dict)
    works_found: dict = field(default_factory=dict)


def _fill(s: Session, layer: str, sql: str, params: dict | None = None) -> None:
    s.execute(
        text(f"""INSERT INTO pilot_seg(segment_id, layer)
        SELECT q.id, '{layer}' FROM ({sql}) q ON CONFLICT DO NOTHING"""),
        params or {},
    )


def build(s: Session) -> PilotResult:
    r = PilotResult()
    s.execute(text("DROP TABLE IF EXISTS pilot_seg"))
    s.execute(
        text("""CREATE TEMP TABLE pilot_seg (segment_id bigint, layer text,
                      PRIMARY KEY(segment_id, layer))""")
    )

    # 1. Bavli Shabbat 21a-24a (seed) and mishnah
    _fill(
        s,
        "gemara_seed",
        """SELECT s.id FROM segments s JOIN works w ON w.id=s.work_id
        WHERE w.title=:t AND s.section_path->>0 = ANY(:d)""",
        {"t": SEED_WORK, "d": SEED_DAFIM},
    )
    _fill(
        s,
        "mishnah",
        """SELECT DISTINCT o.id FROM pilot_seg p
        JOIN edges e ON p.segment_id IN (e.segment_a_id, e.segment_b_id)
        JOIN segments o ON o.id = CASE WHEN e.segment_a_id=p.segment_id THEN e.segment_b_id
                                       ELSE e.segment_a_id END
        JOIN works w ON w.id=o.work_id
        WHERE p.layer='gemara_seed' AND w.category='Mishnah' AND w.dependence IS NULL""",
    )
    _fill(
        s,
        "mishnah",
        """SELECT s.id FROM segments s JOIN works w ON w.id=s.work_id
        WHERE w.title='Mishnah Shabbat' AND s.section_path->>0='2'""",
    )
    # other places in Shas the seed links to (any Bavli base text)
    _fill(
        s,
        "gemara_other",
        """SELECT DISTINCT o.id FROM pilot_seg p
        JOIN edges e ON p.segment_id IN (e.segment_a_id, e.segment_b_id)
        JOIN segments o ON o.id = CASE WHEN e.segment_a_id=p.segment_id THEN e.segment_b_id
                                       ELSE e.segment_a_id END
        JOIN works w ON w.id=o.work_id
        WHERE p.layer='gemara_seed' AND w.category='Talmud' AND w.dependence IS NULL
          AND w.category_path ? 'Bavli'
          AND o.id NOT IN (SELECT segment_id FROM pilot_seg WHERE layer='gemara_seed')""",
    )

    # 2. rishonim on the sugya: commentary segments linked to any Talmud-layer segment
    for label, rx in RISHONIM.items():
        row = s.execute(text("SELECT id FROM works WHERE title ~* :rx"), {"rx": rx}).first()
        r.works_found[label] = bool(row)
    rx_all = "|".join(f"({v})" for v in RISHONIM.values())
    _fill(
        s,
        "rishonim",
        """SELECT DISTINCT o.id FROM pilot_seg p
        JOIN edges e ON p.segment_id IN (e.segment_a_id, e.segment_b_id)
        JOIN segments o ON o.id = CASE WHEN e.segment_a_id=p.segment_id THEN e.segment_b_id
                                       ELSE e.segment_a_id END
        JOIN works w ON w.id=o.work_id
        WHERE p.layer IN ('gemara_seed','gemara_other','mishnah') AND w.title ~* :rx""",
        {"rx": rx_all},
    )

    # 3. Rambam, Tur, Beit Yosef, SA, nosei keilim, MB
    _fill(
        s,
        "rambam",
        """SELECT s.id FROM segments s JOIN works w ON w.id=s.work_id
        WHERE w.title ~* '^Mishneh Torah, .*(Megillah|Megilah).*(Chanukah|Hanukkah|Chanuka)'
          AND s.section_path->>0 = ANY(:c)""",
        {"c": list(RAMBAM_CHAPTERS)},
    )
    lo, hi = SIMAN_RANGE
    for layer, where in (
        ("tur", "w.title = 'Tur, Orach Chayim'"),
        ("beit_yosef", "w.title = 'Beit Yosef, Orach Chayim'"),
        ("shulchan_arukh", "w.title = 'Shulchan Arukh, Orach Chayim'"),
        ("mishnah_berurah", "w.title = 'Mishnah Berurah'"),
        (
            "nosei_keilim",
            """w.dependence='Commentary' AND w.title <> 'Mishnah Berurah'
            AND w.base_work_id =
                (SELECT id FROM works WHERE title='Shulchan Arukh, Orach Chayim')""",
        ),
    ):
        _fill(
            s,
            layer,
            f"""SELECT s.id FROM segments s JOIN works w ON w.id=s.work_id
            WHERE {where} AND s.section_path->>0 ~ '^[0-9]+$'
              AND (s.section_path->>0)::int BETWEEN {lo} AND {hi}""",
        )

    # measure layers
    for layer, n, words in s.execute(
        text("""
        SELECT p.layer, count(DISTINCT p.segment_id),
               COALESCE(sum(t.word_count) FILTER (WHERE v.is_primary), 0)
        FROM pilot_seg p LEFT JOIN segment_text t ON t.segment_id=p.segment_id
        LEFT JOIN versions v ON v.id=t.version_id GROUP BY p.layer""")
    ):
        r.layers[layer] = {"segments": n, "words": int(words)}
    for layer, works in s.execute(
        text("""
        SELECT p.layer, array_agg(DISTINCT w.title ORDER BY w.title) FROM pilot_seg p
        JOIN segments sg ON sg.id=p.segment_id JOIN works w ON w.id=sg.work_id GROUP BY p.layer""")
    ):
        r.layers.setdefault(layer, {})["works"] = works

    # 4. edges inside the pilot, by layer pair and type
    s.execute(text("DROP TABLE IF EXISTS pilot_edges"))
    s.execute(
        text("""CREATE TEMP TABLE pilot_edges AS
        SELECT DISTINCT e.id, e.segment_a_id a, e.segment_b_id b, e.edge_type, pa.layer la,
               pb.layer lb FROM edges e JOIN pilot_seg pa ON pa.segment_id=e.segment_a_id
        JOIN pilot_seg pb ON pb.segment_id=e.segment_b_id WHERE pa.layer <> pb.layer
          OR e.segment_a_id <> e.segment_b_id""")
    )
    for la, lb, n in s.execute(
        text("""SELECT LEAST(la,lb), GREATEST(la,lb), count(*)
                                       FROM pilot_edges GROUP BY 1,2 ORDER BY 3 DESC""")
    ):
        r.edge_matrix[(la, lb)] = n
    for la, lb, et, n in s.execute(
        text("""SELECT LEAST(la,lb), GREATEST(la,lb), edge_type, count(*)
                                           FROM pilot_edges GROUP BY 1,2,3 ORDER BY 4 DESC""")
    ):
        r.edge_types.setdefault((la, lb), {})[et or "(untyped)"] = n

    _chain_and_gaps(s, r)
    return r


def _link_cover(s: Session, layer: str, targets: list[str]) -> tuple[int, int, list[str]]:
    """(#segments in layer, #with >=1 edge to any target layer, sample of uncovered refs)."""
    rows = s.execute(
        text("""
        SELECT sg.ref, EXISTS (
            SELECT 1 FROM pilot_edges e JOIN pilot_seg q ON
                 (q.segment_id = CASE WHEN e.a=p.segment_id THEN e.b ELSE e.a END)
            WHERE p.segment_id IN (e.a, e.b) AND q.layer = ANY(:t)) ok
        FROM pilot_seg p JOIN segments sg ON sg.id=p.segment_id
        WHERE p.layer=:l ORDER BY sg.node_index, sg.sort_key"""),
        {"l": layer, "t": targets},
    ).all()
    miss = [ref for ref, ok in rows if not ok]
    return len(rows), len(rows) - len(miss), miss


def _chain_and_gaps(s: Session, r: PilotResult) -> None:
    pairs = [
        ("gemara_seed", ["rishonim"], "Gemara segment with no Rishonim commentary link"),
        (
            "rambam",
            ["gemara_seed", "gemara_other", "mishnah"],
            "Rambam segment with no link back to Gemara/Mishnah",
        ),
        (
            "tur",
            ["rambam", "gemara_seed", "gemara_other", "beit_yosef"],
            "Tur segment with no link to Gemara/Rambam/Beit Yosef",
        ),
        ("beit_yosef", ["tur"], "Beit Yosef segment with no link to Tur"),
        (
            "shulchan_arukh",
            ["tur", "beit_yosef", "rambam"],
            "Shulchan Arukh seif with no link to Tur/Beit Yosef/Rambam",
        ),
        (
            "shulchan_arukh",
            ["gemara_seed", "gemara_other", "mishnah"],
            "Shulchan Arukh seif with no direct link to Gemara",
        ),
        ("mishnah_berurah", ["shulchan_arukh"], "Mishnah Berurah segment with no link to SA"),
        ("nosei_keilim", ["shulchan_arukh"], "Nosei keilim segment with no link to SA"),
    ]
    for layer, targets, desc in pairs:
        total, ok, miss = _link_cover(s, layer, targets)
        r.chain[f"{layer} -> {'/'.join(targets)}"] = {"total": total, "linked": ok}
        if total and miss:
            r.gaps.append(
                {
                    "kind": "missing_link",
                    "desc": desc,
                    "layer": layer,
                    "missing": len(miss),
                    "of": total,
                    "sample": miss[:8],
                }
            )
    # empty layers / expected works that Sefaria lacks
    for layer in LAYERS:
        if not r.layers.get(layer, {}).get("segments"):
            r.gaps.append(
                {"kind": "missing_text", "desc": f"layer '{layer}' has no segments", "layer": layer}
            )
    for label, found in r.works_found.items():
        if not found:
            r.gaps.append(
                {
                    "kind": "missing_text",
                    "desc": f"no work for Rishon '{label}' on Shabbat in Sefaria (Hebrew)",
                }
            )
    # transitive reachability: every SA seif -> ... -> gemara_seed through pilot edges
    adj: dict[int, set[int]] = defaultdict(set)
    for a, b in s.execute(text("SELECT a, b FROM pilot_edges")):
        adj[a].add(b)
        adj[b].add(a)
    layer_of = defaultdict(set)
    for sid, layer in s.execute(text("SELECT segment_id, layer FROM pilot_seg")):
        layer_of[sid].add(layer)
    seeds = {sid for sid, ls in layer_of.items() if "gemara_seed" in ls}
    reach, dq = set(seeds), deque(seeds)
    while dq:
        x = dq.popleft()
        for y in adj[x]:
            if y not in reach:
                reach.add(y)
                dq.append(y)
    for layer in ("shulchan_arukh", "mishnah_berurah"):
        ids = [sid for sid, ls in layer_of.items() if layer in ls]
        ok = sum(1 for i in ids if i in reach)
        r.chain[f"{layer} reachable from Gemara seed (any path in pilot)"] = {
            "total": len(ids),
            "linked": ok,
        }
        if ids and ok < len(ids):
            refs = [
                x
                for (x,) in s.execute(
                    text(
                        "SELECT ref FROM segments WHERE id = ANY(:i) ORDER BY node_index, sort_key"
                    ),
                    {"i": [i for i in ids if i not in reach]},
                )
            ]
            r.gaps.append(
                {
                    "kind": "disconnected",
                    "desc": f"{layer} segments not reachable from the Gemara seed",
                    "layer": layer,
                    "missing": len(refs),
                    "of": len(ids),
                    "sample": refs[:8],
                }
            )


def render(r: PilotResult) -> str:
    L = [
        "# Pilot: Hilkhot Chanukah",
        "",
        f"Seed: Bavli {SEED_WORK} {SEED_DAFIM[0]}-{SEED_DAFIM[-1]}; Orach Chayim "
        f"{SIMAN_RANGE[0]}-{SIMAN_RANGE[1]}; Rambam Megillah & Chanukah ch. "
        f"{'-'.join(RAMBAM_CHAPTERS)}.",
        "",
        "## Layers (primary versions)",
        "",
        "| layer | segments | words | works |",
        "|---|---:|---:|---|",
    ]
    for layer in LAYERS:
        d = r.layers.get(layer, {})
        works = ", ".join(d.get("works", [])[:12]) + (
            " ..." if len(d.get("works", [])) > 12 else ""
        )
        L.append(f"| {layer} | {d.get('segments', 0):,} | {d.get('words', 0):,} | {works} |")
    L += [
        "",
        "## Edges between pilot layers",
        "",
        "| layer A | layer B | edges | types |",
        "|---|---|---:|---|",
    ]
    for (a, b), n in r.edge_matrix.items():
        types = ", ".join(f"{t}:{c}" for t, c in list(r.edge_types.get((a, b), {}).items())[:6])
        L.append(f"| {a} | {b} | {n:,} | {types} |")
    L += ["", "## Chain connectivity", "", "| check | linked / total |", "|---|---:|"]
    for k, v in r.chain.items():
        L.append(f"| {k} | {v['linked']:,} / {v['total']:,} |")
    L += ["", "## Gaps", ""]
    if not r.gaps:
        L.append("None found.")
    for g in r.gaps:
        extra = (
            f" ({g['missing']:,}/{g['of']:,}; e.g. {', '.join(g['sample'])})" if "of" in g else ""
        )
        L.append(f"- **{g['kind']}**: {g['desc']}{extra}")
    return "\n".join(L) + "\n"
