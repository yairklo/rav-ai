# rav-ai — corpus foundation

Infrastructure stage of a source-grounded halakhic Q&A system: ingest all of Sefaria's Hebrew and
Aramaic texts, links and licenses into a canonical PostgreSQL database. **No retrieval,
embeddings, agents or LLM calls live here.**

## Setup

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and Docker.

```powershell
docker compose up -d --wait      # Postgres 16 + pgvector on localhost:54329 (user/pw/db: rav/rav/rav_ai)
uv sync
uv run pytest                    # needs the Postgres container; creates/drops throw-away databases
uv run ruff check .
```

Database URL: `RAV_DATABASE_URL` or `--db` (default `postgresql+psycopg://rav:rav@localhost:54329/rav_ai`).
Schema is managed with Alembic (`uv run alembic upgrade head`; the ingest CLI also migrates).
The `vector` extension is created by the first migration; nothing uses it yet.

## Pipeline

```powershell
# 1. Download (resumable via .part + HTTP Range; idempotent; data/raw is git-ignored)
uv run rav-download --all --language Hebrew --links      # all Hebrew/Aramaic version files + schemas + links
uv run rav-download --book "Berakhot" --language Hebrew  # or one book / --category Talmud
uv run rav-download --all --dry-run                      # show what would be fetched

# 2. Ingest = a dated snapshot (structure, text + 2 normalized forms, license flags, edges)
uv run rav-ingest sefaria                  # resumable; unchanged files are skipped by hash
uv run rav-ingest diff                     # what changed in the latest snapshot
uv run rav-ingest recompute-flags          # re-apply config/license_policy.toml, no re-import

# 3. Reports
uv run rav-license-report                  # reports/licenses.csv (per version) + license_summary.md
uv run rav-pilot                           # reports/pilot_hanukkah.md (layers, chain, gaps)
uv run rav-token-stats                     # reports/token_stats.md (tiktoken tokens per word)

# 4. Inspect / export
uv run rav-show "Shabbat 21b:5"            # text + every layer above and below, by type and period
RAV_DEPLOYMENT_MODE=public uv run rav-show "Shabbat 21b:5"   # display gate: can_display_public
uv run rav-export --purpose train --out train.jsonl           # export gate: can_train
```

## Data model (`src/rav_ai/db/models.py`)

| table | purpose |
|---|---|
| `snapshots`, `snapshot_changes` | one row per import (export label + date); per-segment-text hash diff (added/changed/removed) against the previous state |
| `authors` | name, period, birth/death year (nullable — Sefaria's export has no life dates) |
| `works` | book, author, category, `dependence` + `base_work_id` (commentary on …), era/period, schema |
| `versions` | an edition: source, license (as exported), **`can_index`, `can_display_public`, `can_train`** (computed from `config/license_policy.toml`, stored here, enforced at display/export — **never filtered at ingest**), file hash |
| `segments` | smallest unit; canonical Sefaria `ref` (unique), hierarchy (`section_path`, `indices`, `parent_ref`) |
| `segment_text` | per (segment, version): `text_raw` (verbatim) + `text_a`, `text_b` + `offsets_a/b` maps + `raw_hash` |
| `edges` | segment ↔ segment, Sefaria link type verbatim, `origin` (sefaria/heuristic/llm), `confidence`, optional char ranges |
| `unresolved_links` | Sefaria links that could not be resolved, with reason (gap report) |
| `quote_spans`, `annotations`, `cards`, `card_dependencies` | empty, for later stages (quotations, abbreviation candidates, AI analysis cards with `stale`) |

### Normalized forms (`src/rav_ai/normalize.py`)

- **A**: markup/entities/footnotes removed, niqqud and te'amim removed, geresh/gershayim/quote/dash
  variants unified, presentation forms decomposed, whitespace collapsed. Spelling (ktiv male/haser)
  is *not* altered; abbreviations are *not* expanded.
- **B**: A without any punctuation (maqaf/dash → space); tokenizable on whitespace (search form).
- **Offset maps** (`offsets_a/b`): zlib'd varint chunk lists `(raw gap, n_len, r_len)`;
  `OffsetMap.from_bytes(b).to_raw(ns, ne)` returns the exact raw `[start, end)` for a normalized range.

### Edges

Links are stored once as exported; query either side. Citations that are sections or ranges expand
to segment pairs (capped at 3,000 pairs per link; larger range-to-range links become one
`range_collapsed` edge with `a_span`/`b_span` recording the original sizes).
Sefaria links carry no confidence or character ranges, so those columns are NULL for `origin='sefaria'`.

## Layout

```
config/license_policy.toml   license -> can_index / can_display_public / can_train
src/rav_ai/sefaria/          bucket access, layout, schema walker, citation resolver
src/rav_ai/normalize.py      normalization + offset maps
src/rav_ai/ingest/           sefaria.py (snapshots, diff), links.py (edges)
src/rav_ai/lookup.py         license-gated reads + layered neighbourhood
src/rav_ai/pilot.py          Hilkhot Chanukah pilot slice and gap analysis
alembic/ tests/ reports/
```
