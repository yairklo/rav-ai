"""LLM backends for rav-kl.

``api``        - ``rav_harness.llm.LiteLLMClient`` (unchanged; needs a provider API key).
``claude_cli`` - :class:`ClaudeCLIClient`: the local ``claude`` CLI on the user's Claude
                 subscription (``claude auth login``). No API key is read or needed.

Both implement ``rav_harness.llm.LLM`` (``complete -> LLMResponse``), so ``llm_io.call_tool``
and ``core.generate`` stay the only path to Pydantic validation (K1) and its one retry.

A Gemini CLI backend would be a third class with the same ``complete`` contract; not built.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from rav_harness.llm import LLMResponse, ToolCall

from .config import KLConfig

BACKENDS = ("api", "claude_cli")

# The child never sees a provider key: the CLI must bill the subscription, not an API account.
_STRIPPED_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
_WIN_CMDLINE_MAX = 8000  # cmd.exe limit is 8191; only matters if we end up running a .cmd shim

_LIMIT_MARKERS = ("rate_limit", "billing_error", "usage limit", "hit your limit")
_AUTH_MARKERS = (
    "not logged in",
    "please run /login",
    "claude auth login",
    "invalid api key",
    "authentication_error",
    "oauth token has expired",
)
LOGIN_HINT = "the claude CLI is not logged in to a Claude subscription: run `claude auth login`"


class StopRun(Exception):
    """The run cannot continue (not a task failure): the task goes back to pending."""


class UsageLimitError(StopRun):
    """Subscription usage limit / rate limit reached."""


class CLIAuthError(StopRun):
    """The CLI is not logged in (or not with a claude.ai subscription)."""


class CLINotFound(StopRun):
    pass


def resolve_bin(name: str) -> str:
    """``claude`` -> an executable path. On Windows prefer the real ``claude.exe`` behind the npm
    ``claude.cmd`` shim: a .cmd runs through cmd.exe (8191-char line, mangles JSON quotes)."""
    p = shutil.which(name)
    if p is None:
        raise CLINotFound(f"claude CLI not found ({name!r}); install it or set RAV_KL_CLAUDE_BIN")
    if os.name == "nt" and p.lower().endswith((".cmd", ".bat")):
        exe = Path(p).parent / "node_modules" / "@anthropic-ai" / "claude-code" / "bin"
        if (exe / "claude.exe").exists():
            return str(exe / "claude.exe")
    return p


def cli_model(model_id: str) -> str:
    """LiteLLM id -> ``--model`` value. ``anthropic/claude-opus-5-5`` -> ``claude-opus-5-5``;
    an alias (opus / sonnet / haiku / fable) passes through. Other providers are refused."""
    provider, _, name = model_id.rpartition("/")
    if provider and provider != "anthropic":
        raise ValueError(f"claude_cli runs Anthropic models only, not {model_id!r}")
    return name


def child_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV}


def _has(text: str, markers: tuple[str, ...]) -> bool:
    t = text.lower()
    return any(m in t for m in markers)


class ClaudeCLIClient:
    """``claude -p --output-format json --json-schema ...``: one structured output per call."""

    backend = "claude_cli"

    def __init__(
        self,
        bin: str = "claude",
        *,
        cmd: list[str] | None = None,
        timeout: float = 1800.0,
    ):
        self._bin = bin
        self._cmd = cmd  # tests: [python, fake_claude.py]
        self.timeout = timeout
        self._version: str | None = None
        self._ids = itertools.count(1)
        self.calls: list[dict[str, Any]] = []  # per-call meta, drained per task by the runner
        self.models_reported: set[str] = set()
        self.cli_cost_usd = 0.0

    @property
    def cmd(self) -> list[str]:
        if self._cmd is None:
            self._cmd = [resolve_bin(self._bin)]
        return self._cmd

    def _run(self, args: list[str], *, stdin: str | None = None, cwd: str | None = None):
        argv = [*self.cmd, *args]
        if os.name == "nt" and argv[0].lower().endswith((".cmd", ".bat")):
            if len(subprocess.list2cmdline(argv)) > _WIN_CMDLINE_MAX:
                raise RuntimeError("claude CLI command line too long for a .cmd shim")
        return subprocess.run(  # noqa: S603 - argv list, no shell
            argv,
            input=stdin,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
            env=child_env(),
            timeout=self.timeout,
        )

    def version(self) -> str:
        if self._version is None:
            p = self._run(["--version"])
            self._version = (p.stdout or p.stderr).strip() if p.returncode == 0 else "unknown"
        return self._version

    def ensure_logged_in(self) -> dict[str, Any]:
        """``claude auth status`` must exit 0 with authMethod claude.ai. Never prints secrets."""
        try:
            p = self._run(["auth", "status", "--json"])
        except FileNotFoundError as e:
            raise CLINotFound(str(e)) from e
        try:
            st = json.loads(p.stdout or "{}")
        except ValueError:
            st = {}
        if p.returncode != 0 or not st.get("loggedIn") or st.get("authMethod") != "claude.ai":
            raise CLIAuthError(
                f"{LOGIN_HINT} (auth status exit={p.returncode}, "
                f"authMethod={st.get('authMethod')!r})"
            )
        return {k: st.get(k) for k in ("authMethod", "apiProvider", "subscriptionType")}

    def job_meta(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "cli_version": self.version(),
            "models_reported": sorted(self.models_reported),
            "cost_kind": "subscription",
            "cli_cost_usd": round(self.cli_cost_usd, 6),
            "unsupported_params": ["temperature", "max_tokens"],  # the CLI has no such flags
        }

    def drain_calls(self) -> list[dict[str, Any]]:
        out, self.calls = self.calls, []
        return out

    def complete(self, messages, tools, *, model, temperature, max_tokens) -> LLMResponse:
        # temperature / max_tokens: no CLI flags; recorded as unsupported in job_meta
        fn = tools[0]["function"]
        name = fn["name"]
        schema = json.dumps(fn["parameters"], ensure_ascii=True, separators=(",", ":"))
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
        call_id = f"cli-{next(self._ids)}"
        meta: dict[str, Any] = {"call": call_id, "model_requested": model}
        with tempfile.TemporaryDirectory(prefix="rav-kl-cli-") as d:
            sp = Path(d) / "system.txt"
            sp.write_text(system, encoding="utf-8")
            work = Path(d) / "cwd"
            work.mkdir()
            args = [
                "-p",
                "--output-format",
                "json",
                "--json-schema",
                schema,
                "--model",
                cli_model(model),
                "--tools",
                "",
                "--no-session-persistence",
                "--permission-mode",
                "dontAsk",
                "--system-prompt-file",
                str(sp),
            ]
            try:
                p = self._run(args, stdin=user, cwd=str(work))
            except subprocess.TimeoutExpired:
                return self._fail(name, call_id, meta, model, f"timeout after {self.timeout}s")
        out, err = p.stdout or "", p.stderr or ""
        try:
            data = json.loads(out)
            if not isinstance(data, dict):
                raise ValueError("not an object")
        except ValueError:
            data = None
        blob = f"{out}\n{err}"
        failed = p.returncode != 0 or data is None or bool(data.get("is_error"))
        if failed:
            d = data or {}
            if (
                _has(blob, _LIMIT_MARKERS)
                or d.get("api_error_status") == 429
                or _has(str(d.get("subtype", "")), _LIMIT_MARKERS)
            ):
                raise UsageLimitError(f"claude CLI usage/rate limit: {_snippet(d, blob)}")
            if _has(blob, _AUTH_MARKERS):
                raise CLIAuthError(f"{LOGIN_HINT}: {_snippet(d, blob)}")
        if data is None:
            return self._fail(
                name, call_id, meta, model, f"exit={p.returncode}; stdout is not JSON: {blob}"
            )
        usage = data.get("usage") or {}
        in_tok = sum(
            int(usage.get(k) or 0)
            for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        )
        out_tok = int(usage.get("output_tokens") or 0)
        cost = float(data.get("total_cost_usd") or 0.0)
        reported = ",".join(sorted((data.get("modelUsage") or {}).keys())) or model
        self.models_reported.update(x for x in reported.split(",") if x)
        self.cli_cost_usd += cost
        meta.update(
            model_reported=reported,
            subtype=data.get("subtype"),
            num_turns=data.get("num_turns"),
            duration_ms=data.get("duration_ms"),
            total_cost_usd=cost,
        )
        resp = dict(
            input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost, model=reported
        )  # cost: CLI-reported (notional under a subscription), 0 when absent
        if failed:
            err_msg = f"exit={p.returncode} is_error={data.get('is_error')}: {_snippet(data, blob)}"
            return self._fail(name, call_id, meta, model, err_msg, **resp)
        so = data.get("structured_output")
        if not isinstance(so, dict):
            return self._fail(
                name,
                call_id,
                meta,
                model,
                f"no structured_output (subtype={data.get('subtype')})",
                **resp,
            )
        meta["ok"] = True
        self.calls.append(meta)
        raw = json.dumps(so, ensure_ascii=False)
        return LLMResponse(content=None, tool_calls=[ToolCall(call_id, name, so, raw)], **resp)

    def _fail(self, name, call_id, meta, requested, error, **resp) -> LLMResponse:
        """A failed call is a submit_<kind> call whose arguments are an error: call_tool reports it
        and core.generate retries once, exactly like invalid tool arguments on the API path."""
        msg = f"claude_cli call failed: {error}"[:1500]
        meta.update(ok=False, error=msg)
        self.calls.append(meta)
        resp.setdefault("model", requested)
        resp.setdefault("cost_usd", 0.0)
        return LLMResponse(content=None, tool_calls=[ToolCall(call_id, name, {}, "", msg)], **resp)


def _snippet(data: dict, blob: str) -> str:
    if data.get("result"):
        return str(data["result"])[:500]
    return blob.strip()[:500]


def make_llm(cfg: KLConfig):
    if cfg.backend == "claude_cli":
        return ClaudeCLIClient(cfg.claude_cli_bin)
    from rav_harness.llm import LiteLLMClient

    return LiteLLMClient()
