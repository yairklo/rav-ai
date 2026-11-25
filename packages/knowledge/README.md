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
rav-kl register-work "Shabbat" --tier canonical --role primary   # kl_sources is a fail-closed registry
rav-kl seed-lenses ; rav-kl conflicts ; rav-kl status
```

Config: `config/knowledge.toml` (models per stage, second model, prices - placeholders, thresholds,
`mode`, `allow_personal_only`, `use_steinsaltz_aid`, `anchor_format`).
