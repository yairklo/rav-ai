"""A fake ``claude`` CLI for tests: no network, no real claude.

Env:
  FAKE_CLAUDE_SCENARIO  comma list consumed one per ``-p`` call (the last repeats):
                        ok | badjson | noso | schema | exit1 | limit | noauth
  FAKE_CLAUDE_STATE     counter file (which scenario is next)
  FAKE_CLAUDE_LOG       JSON lines: argv, stdin, system prompt, cwd listing, key visibility
  FAKE_CLAUDE_AUTH      ok (default) | loggedout | apikey   (for ``auth status``)
  FAKE_CLAUDE_AUTH_LOG  JSON lines for ``auth status`` calls: which provider keys were visible
Scenarios: ok | badjson | noso | schema | exit1 | limit | noauth | iserror_valid
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _arg(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


def _scenario() -> str:
    seq = [x for x in os.environ.get("FAKE_CLAUDE_SCENARIO", "ok").split(",") if x]
    state = os.environ.get("FAKE_CLAUDE_STATE")
    n = 0
    if state:
        p = Path(state)
        n = int(p.read_text()) if p.exists() else 0
        p.write_text(str(n + 1))
    return seq[min(n, len(seq) - 1)]


def _answer(schema: dict, system: str, user: str, model: str) -> dict:
    props = set(schema.get("properties", {}))
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if "choices" in props:
        return {"choices": [{"id": int(i), "choice": 0} for i in re.findall(r"id=(\d+)", user)]}
    from helpers import s1_responder, s2_responder

    if "sugyot" in props:
        return s1_responder()(msgs, "sugya", model)
    return s2_responder(msgs, "sugya_summary", model)


def main() -> int:
    argv = sys.argv[1:]
    if argv == ["--version"]:
        print("9.9.9 (Claude Code fake)")
        return 0
    if argv[:2] == ["auth", "status"]:
        if os.environ.get("FAKE_CLAUDE_AUTH_LOG"):
            with open(os.environ["FAKE_CLAUDE_AUTH_LOG"], "a", encoding="utf-8") as f:
                seen = [k for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN") if k in os.environ]
                f.write(json.dumps({"argv": argv, "keys_visible": seen}) + "\n")
        mode = os.environ.get("FAKE_CLAUDE_AUTH", "ok")
        if mode == "loggedout":
            print(json.dumps({"loggedIn": False}))
            return 1
        method = "claude.ai" if mode == "ok" else "api_key"
        print(json.dumps({"loggedIn": True, "authMethod": method, "email": "secret@example"}))
        return 0
    stdin = sys.stdin.buffer.read().decode("utf-8")  # like the real CLI
    schema = json.loads(_arg(argv, "--json-schema") or "{}")
    system = Path(_arg(argv, "--system-prompt-file")).read_text(encoding="utf-8")
    model = _arg(argv, "--model") or "?"
    if os.environ.get("FAKE_CLAUDE_LOG"):
        with open(os.environ["FAKE_CLAUDE_LOG"], "a", encoding="utf-8") as f:
            rec = {
                "argv": argv,
                "stdin": stdin,
                "system": system,
                "cwd_files": os.listdir("."),
                "api_key_visible": "ANTHROPIC_API_KEY" in os.environ,
                "auth_token_visible": "ANTHROPIC_AUTH_TOKEN" in os.environ,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    sc = _scenario()
    base = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "num_turns": 2,
        "duration_ms": 10,
        "total_cost_usd": 0.0125,
        "usage": {
            "input_tokens": 10,
            "cache_creation_input_tokens": 900,
            "cache_read_input_tokens": 90,
            "output_tokens": 50,
        },
        "modelUsage": {model: {"inputTokens": 10, "outputTokens": 50}},
    }
    if sc == "ok":
        so = _answer(schema, system, stdin, model)
        print(json.dumps({**base, "result": json.dumps(so), "structured_output": so}))
        return 0
    if sc == "iserror_valid":  # is_error=true even though structured_output is valid
        so = _answer(schema, system, stdin, model)
        print(json.dumps({**base, "is_error": True, "result": "partial", "structured_output": so}))
        return 0
    if sc == "badjson":
        print("this is {not json")
        return 0
    if sc == "noso":
        print(json.dumps({**base, "result": "I could not comply"}))
        return 0
    if sc == "schema":
        print(json.dumps({**base, "structured_output": {"unexpected": 1}}))
        return 0
    if sc == "exit1":
        print("boom: internal failure", file=sys.stderr)
        return 1
    if sc == "limit":
        print(
            json.dumps(
                {
                    **base,
                    "subtype": "error_during_execution",
                    "is_error": True,
                    "result": "You've hit your limit · resets 5pm",
                }
            )
        )
        return 1
    if sc == "noauth":
        print(json.dumps({**base, "is_error": True, "result": "Not logged in · Please run /login"}))
        return 1
    raise SystemExit(f"unknown scenario {sc}")


if __name__ == "__main__":
    raise SystemExit(main())
