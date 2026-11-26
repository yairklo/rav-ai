# rav-evals

Answer schema, deterministic citation verifier (V1-V4), golden-set format and eval runner.

```powershell
cd packages/evals
uv sync
uv run pytest
uv run rav-evals golden/hanukkah.draft.yaml --answer-fn mymod:answer_fn --sources sources.json
```

`answer_fn(GoldenQuestion) -> RunOutput` (answer + retrieved refs + model + tool calls).
`sources.json` is `{"ref": "text"}` for the in-memory store (tests). Against the real corpus use
`--database-url postgresql+psycopg://...` instead: `PostgresSourceStore` answers V1 (ref exists,
indexable version) and V2 (stored `segment_text.text_b`, same normalization policy as
`rav_evals.normalization`), and every result is logged to the `runs` table (migration 0002).

Integration tests need Postgres (`docker compose up -d` in the repo root); they skip without it,
and fail when `REQUIRE_DB=1` (set in CI).
Reports are written to `--out-dir` as `report.json` and `report.md`.

Golden file `golden/hanukkah.draft.yaml` is a DRAFT: every ref is a `TODO` placeholder and
needs human review. Checks that depend on placeholders are reported as `skipped`.

## KL-3: knowledge-layer evaluation

Gates and thresholds: `config/evals.toml` (all initial values are estimates, to calibrate).

### Verifier V5-V10 (a measurement; the harness's `validate_answer` is the run-time guard)

| check | fails when |
|---|---|
| V5 `secondary_as_primary` | a citation marked `source_role="primary"` is from a lens work (`kl_sources.role='lens'`) |
| V6 `candidate_as_proof` / `relation_unknown` | a candidate relation whose two endpoints were not both retrieved AND cited in the claim (each candidate on its own); a relation id that does not exist. An `edges` row is a candidate unless its stored origin is `human` (fail closed); `kl_relations` / `kl_links` are not links - their `evidence_origin` is a separate metric |
| V7 `witness_missing` / `witness_unknown` | a variant / scribal-error claim with no known `witness_id`, unless attributed to another holder (an attributed citation from another work) |
| V8 `coverage_notice_missing` / `_wrong` / `scope_mismatch` / `out_of_scope_notice_missing` | scope comes from the question / run. In scope: an unchecked core work (`kl_coverage`) must be listed, and the answer may not call itself out_of_scope. Out of scope: a full answer (sources + conclusion) opening with the notice "שאלה זו מחוץ להיקף שכבת הידע שנבדקה; התשובה לא עברה אימות שכבת ידע."; no conclusion = soft failure (reported only) |
| V9 `kibalnu_undocumented` | acceptance / custom language ("קיבלנו", "נהגו") without a citation |
| V10 `sa_decisive_phrasing` | decisive phrasing about the Shulchan Arukh in the system's voice |

A check whose `kl_*` table (or the whole lookup) is missing is skipped and listed in `skipped`;
in `regress` a skipped hard gate (V5/V6/V7) on content it should have checked is a failure. V7/V9/V10 use the marker
lists in `config/evals.toml` (deterministic heuristics; V7 ignores claims that deny an error).

### Commands

```
rav-evals golden-status                     # questions per category / split, every pending TODO
rav-evals regress [--update]                # recorded fixtures vs baselines/metrics.json (CI, no key)
rav-evals run --knowledge off|base|base+lens --split dev [--record-fixtures fixtures/runs]
rav-evals compare reports/evals/run-off-dev reports/evals/run-base-dev [reports/evals/run-base_lens-dev]
rav-evals pairs-sample --n 200 --seed 20261125   # -> golden/kl/pairs.draft.jsonl
rav-evals pairs-kappa                       # kappa (2 raters) / self agreement (1 rater)
rav-evals pairs-recheck --rater yair        # blind recheck set: 10% of pairs rated >= 14 days ago
rav-evals attribution [--sample N]          # misattribution rate, Wilson CI, gate
rav-evals review --rater yair               # the rating tool, http://127.0.0.1:8765
```

**Start rating (one command):** `uv run rav-evals review --rater yair` - opens the browser on
127.0.0.1. Three queues: answers (claim by claim next to the sources read; `a` ✓, `x` ✗, `e` fix,
`1`-`9`/`0` error type, `s` attribution subtype, `q`/`w`/`r` ok/minor/major), pairs (`1`-`6`
label, `d` direction, `c` confidence, `a` save; `/pairs?recheck=1` is the blind recheck, without
the earlier rating), knowledge-layer items (queue order: K14/K15 warnings, model disagreement,
weighty relations, confidence < 0.75, items used by reviewed runs, 10% random; `a` approve ->
verified, `x` reject -> rejected, `e` hide the selected claim - the item stays `generated`;
`1`-`9`/`0`/`t` error type, required for reject / fix; `v` diff vs the previous version).
Hiding takes effect only once the harness reads `hidden_claims` (KL-2). Synthetic seed runs and
heldout runs are not in the answers queue; heldout errors go to `errors-heldout-<pv>.md`. Ratings are appended to `golden/kl/ratings/<rater>.<queue>.jsonl` - commit them.
Rejected claims are exported to `reports/kl/errors-<prompt_version>.md` and become regression
fixtures (`fixtures/runs/rejected-*.json`).

### Gold principles

Nothing in `golden/kl` is filled in from a model's memory. Every value not marked by a human is a
`TODO` and the question is reported `pending` (never `passed`). `split: heldout` questions are
never used to tune prompts and run only with `--split heldout`. With a single rater there is no
kappa: the report says so and gives self agreement on a blind recheck after 14 days (target 85%,
undecided below `self_agreement_min_n` = 20 rechecks). `load_golden` loads `dev` by default.

`regress` fails on: a hard gate above 0 or skipped, a baseline fixture that is missing, a smaller
category, a metric that had a baseline value and is now missing, zero fixtures, an unknown
`hard_zero` key, and a rejected claim that is no longer caught by the codes it had at export.

### Live runs (manual, never in CI)

1. Set the key the harness model needs (e.g. `$env:ANTHROPIC_API_KEY = "..."`) and have the corpus
   database up (`docker compose up -d`).
2. Baseline first, without the layer: `rav-evals run --knowledge off --split dev`.
3. Then `--knowledge base` and `--knowledge base+lens` (needs the KL-2 harness), and
   `rav-evals compare` the three run directories.
4. To keep a live run as a regression case add `--record-fixtures packages/evals/fixtures/runs`:
   each question is saved with its tool transcript, answer, the text of every source read and a
   snapshot of the `kl_*` rows the checks read. Then `rav-evals regress --update` and commit.

`fixtures/runs/seed-*.json` are SYNTHETIC (neutral placeholder text, `synthetic: true`), made by
`scripts/make_seed_fixtures.py`; replace them with recorded live runs as they come.
