# rav-evals

Answer schema, deterministic citation verifier (V1-V4), golden-set format and eval runner.

```powershell
cd packages/evals
uv sync
uv run pytest
uv run rav-evals golden/hanukkah.draft.yaml --answer-fn mymod:answer_fn --sources sources.json
```

`answer_fn(GoldenQuestion) -> RunOutput` (answer + retrieved refs + model + tool calls).
`sources.json` is `{"ref": "text"}` for the fake store; swap in a DB-backed `SourceStore` later.
Reports are written to `--out-dir` as `report.json` and `report.md`.

Golden file `golden/hanukkah.draft.yaml` is a DRAFT: every ref is a `TODO` placeholder and
needs human review. Checks that depend on placeholders are reported as `skipped`.
