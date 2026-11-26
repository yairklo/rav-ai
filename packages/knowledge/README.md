# rav-knowledge (`rav_knowledge`, CLI `rav-kl`)

The knowledge layer over the corpus: sugyot and their move graph, issues, positions, machlokot,
source claims, with every claim anchored to a span of the **raw** text. The text is the authority;
the layer is a map. A secondary source is a lens (what it says is attributed to it); sources can be
wrong (an error is somebody's claim, with evidence); licensing is fail-closed.

Stages (DAG): `s0` deterministic annotations (acronyms ranked by context, citations, discourse /
quote / error markers) -> `s0b` constrained abbreviation choice -> `s1` sugya boundaries (sliding
window, merged per scope) -> `s2` basic Gemara-only summary. `s3 s3l s4 s4g s5` are reserved.

```
rav-kl estimate "Shabbat 21a-24a" --stages s0b,s1,s2     # no LLM calls
rav-kl run s0 "Shabbat 21a-24a"                          # deterministic
rav-kl run s1 "Shabbat 21a-24a" --dry-run | --budget-usd 5
rav-kl report --job N ; rav-kl promote --job N [--force "why"]   # rollback = promote an old job
rav-kl invalidate [--segments 1,2]                       # re-anchor, stale only if the quote is gone
rav-kl register-work "Shabbat" --tier canonical --role primary --license-tier PUBLIC_COMMERCIAL   # fail-closed registry: the license tier is required
rav-kl seed-lenses ; rav-kl conflicts ; rav-kl status
```

Config: `config/knowledge.toml` (models per stage, second model, prices - placeholders, thresholds,
`mode`, `allow_personal_only`, `use_steinsaltz_aid`, `anchor_format`, `backend`).

### Backends (`kl.backend`, env `RAV_KL_BACKEND` overrides)

- `api` (default): LiteLLM with a provider key (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`).
- `claude_cli`: the local `claude` CLI on a Claude subscription, no API key (the key is even
  removed from the child's environment). `rav-kl run` checks `claude auth status` first and stops
  with "run `claude auth login`" otherwise. Binary: `kl.claude_cli_bin` / `RAV_KL_CLAUDE_BIN`
  (on Windows the npm `claude.cmd` shim is resolved to `claude.exe`). One structured output per
  call (`--json-schema` from the `submit_<kind>` tool, system prompt in a file, input on stdin, an
  empty temp cwd); validation and the one retry are the same `call_tool` / `generate` path as the
  API. Limits: no `temperature` / `max_tokens` flags (recorded in the job stats as
  `unsupported_params`); `second_model` (another family) is skipped and counted as
  `second_model_skipped`; `--budget-usd` is ignored. `cost_usd` is the CLI's `total_cost_usd`
  (`cost_kind=subscription`), the reported model and CLI version are in `kl_jobs.stats` and per
  task in `kl_tasks.result._llm`. `backend` is part of the params / input hash, so CLI and API
  jobs never mix on resume.
- A usage / rate limit stops the run cleanly: the task's writes are rolled back, it goes back to
  `pending`, the job is `aborted`, and the resume command is printed (exit 3). `--limit N` runs at
  most N new tasks and stops the same way.

```
$env:RAV_KL_BACKEND = "claude_cli"
rav-kl run s0 "Shabbat 21a"; rav-kl run s0b "Shabbat 21a"; rav-kl run s1 "Shabbat 21a"; rav-kl run s2 "Shabbat 21a" --limit 5
```

## Behaviour worth knowing

- **S1 long sugyot.** A sugya cut by a window edge is carried to the next window (`open_end` /
  `open_start`, flags from the model and derived from the edge when windows overlap). After the
  merge every loaded segment must belong to exactly one sugya; a hole or overlap is a K8 error in
  the sugya's `check_report` (`queue_priority`), counted in the job stats.
- **Evidence location.** A quote matches only on word boundaries; a quote shorter than
  `quote_min_words` / `quote_min_chars` (config) or with more than one ordered occurrence is
  `ambiguous`, never `located`.
- **Roles.** Evidence of a step, of `position.evidence` and of `holder_evidence` defaults to
  `states` / `attributes`, so K3 treats an unfound quote as an error.
- **Overrides.** `edit` and `set_boundary` never change an item: they create a new verified
  version (`prompt_version="human"`, `model` null) and make it current. A conflict is counted only
  when the new output differs from the decision.
- **Links between versions.** The axis queries (`issues_by_sugya`, `machlokot_by_issue`,
  `sugyot_by_machloket`, `positions_by_issue`) resolve by `item_key` (any version of the target)
  and return only current items.
- **`reading_only` positions** (evidence only from a lens, holder not the lens author) record
  `read_by` and `attribution_grade="inferred_by_source"` and are excluded from machlokot and from
  every axis query. **In KL-4 they become `kl_relations` of type `reads_as` or `aligns_with_side`,
  and no longer remain a position.**
- **Promote gate.** `gate_max_rejected_fraction` (default 0.1) and `gate_min_located_fraction`
  in `[thresholds]`.
- **For KL-2 (search).** FTS queries must use exactly the expression
  `to_tsvector('simple', coalesce(search_text_he, ''))`, otherwise the GIN index
  `ix_kl_items_search` is not used.
