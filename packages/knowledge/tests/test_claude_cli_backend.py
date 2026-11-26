"""backend=claude_cli: the ``claude`` CLI on a subscription, driven through a fake binary.

No network and no real ``claude``: ``fake_claude.py`` plays the CLI (see its docstring)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from rav_ai.db.models import KlItem, KlJob, KlTask
from sqlalchemy import select
from sqlalchemy.orm import Session

from fakes import ScriptedLLM
from helpers import SCOPE, cfg, run, s1_responder, s2_responder
from rav_knowledge import backends
from rav_knowledge.backends import (
    ClaudeCLIClient,
    CLIAuthError,
    UsageLimitError,
    cli_model,
    make_llm,
)
from rav_knowledge.cli import main as rav_kl
from rav_knowledge.config import CONFIG_PATH, load_config
from rav_knowledge.llm_io import build_messages, call_tool, tool_def
from rav_knowledge.prompts import Prompt
from rav_knowledge.schemas import TOOL_MODELS

FAKE = Path(__file__).parent / "fake_claude.py"


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """Returns set(scenario) -> a fresh ClaudeCLIClient on the fake; log() -> calls so far."""
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-reach-the-cli")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "must-not-reach-the-cli-either")
    monkeypatch.setenv("FAKE_CLAUDE_AUTH_LOG", str(tmp_path / "auth.jsonl"))

    class F:
        def set(self, scenario: str = "ok", auth: str = "ok") -> ClaudeCLIClient:
            state = tmp_path / f"state-{scenario}-{auth}-{id(object())}"
            monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", scenario)
            monkeypatch.setenv("FAKE_CLAUDE_STATE", str(state))
            monkeypatch.setenv("FAKE_CLAUDE_AUTH", auth)
            return ClaudeCLIClient(cmd=[sys.executable, str(FAKE)], timeout=120)

        def auth_log(self) -> list[dict]:
            p = tmp_path / "auth.jsonl"
            if not p.exists():
                return []
            return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]

        def log(self) -> list[dict]:
            if not log.exists():
                return []
            return [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]

    return F()


def cli_cfg(**kw):
    c = cfg(backend="claude_cli")
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def items(engine, kind):
    with Session(engine) as s:
        return list(s.scalars(select(KlItem).where(KlItem.kind == kind).order_by(KlItem.id)))


def tasks(engine, job_id):
    with Session(engine) as s:
        return {
            t.task_key: t
            for t in s.scalars(select(KlTask).where(KlTask.job_id == job_id).order_by(KlTask.id))
        }


# ---- selection / mapping (no DB) -----------------------------------------------------------


def test_backend_selection_config_and_env(tmp_path, monkeypatch):
    monkeypatch.delenv("RAV_KL_BACKEND", raising=False)
    monkeypatch.delenv("RAV_KL_CLAUDE_BIN", raising=False)
    base = load_config()
    assert base.backend == "api" and base.claude_cli_bin == "claude"
    assert "backend" not in base.params()  # API hashes / job params unchanged
    toml = tmp_path / "k.toml"
    toml.write_text(
        CONFIG_PATH.read_text(encoding="utf-8")
        .replace('backend = "api"', 'backend = "claude_cli"')
        .replace('claude_cli_bin = "claude"', 'claude_cli_bin = "my-claude"'),
        encoding="utf-8",
    )
    c = load_config(toml)
    assert c.backend == "claude_cli" and c.claude_cli_bin == "my-claude"
    assert c.params()["backend"] == "claude_cli" and c.params() != base.params()
    assert isinstance(make_llm(c), ClaudeCLIClient)
    monkeypatch.setenv("RAV_KL_BACKEND", "api")  # env beats the file
    monkeypatch.setenv("RAV_KL_CLAUDE_BIN", "other-claude")
    c2 = load_config(toml)
    assert c2.backend == "api" and c2.claude_cli_bin == "other-claude"
    assert type(make_llm(c2)).__name__ == "LiteLLMClient"
    monkeypatch.setenv("RAV_KL_BACKEND", "claude_cli")
    assert load_config().backend == "claude_cli"
    monkeypatch.setenv("RAV_KL_BACKEND", "gemini")
    with pytest.raises(ValueError, match="backend"):
        load_config()


def test_model_mapping():
    assert cli_model("anthropic/claude-opus-5-5") == "claude-opus-5-5"
    assert cli_model("anthropic/claude-haiku-5-5") == "claude-haiku-5-5"
    assert cli_model("sonnet") == "sonnet"
    with pytest.raises(ValueError, match="Anthropic"):
        cli_model("openai/gpt-5.1")


def test_complete_command_line_and_parsing(fake):
    llm = fake.set("ok")
    td = tool_def("sugya")
    msgs = [
        {"role": "system", "content": "מערכת: הנחיות"},
        {"role": "user", "content": "[S1|canonical|Shabbat] Shabbat 21a:1 ⟩ אמר רב"},
    ]
    r = llm.complete(msgs, [td], model="anthropic/claude-opus-5-5", temperature=0.0, max_tokens=9)
    (rec,) = fake.log()
    argv = rec["argv"]
    assert argv[0] == "-p" and _flag(argv, "--output-format") == "json"
    assert json.loads(_flag(argv, "--json-schema")) == td["function"]["parameters"]
    assert _flag(argv, "--model") == "claude-opus-5-5"
    assert _flag(argv, "--tools") == "" and _flag(argv, "--permission-mode") == "dontAsk"
    assert "--no-session-persistence" in argv
    assert "--bare" not in argv and "--max-turns" not in argv
    assert not any("temperature" in a or "max-tokens" in a for a in argv)  # no invented flags
    assert rec["system"] == "מערכת: הנחיות" and rec["stdin"] == msgs[1]["content"]
    assert rec["cwd_files"] == [] and rec["api_key_visible"] is False
    (tc,) = r.tool_calls
    assert tc.name == "submit_sugya" and tc.arguments_error is None
    TOOL_MODELS["sugya"].model_validate(tc.arguments)
    assert (r.input_tokens, r.output_tokens) == (1000, 50)
    assert r.cost_usd == 0.0125 and r.model == "claude-opus-5-5"
    meta = llm.job_meta()
    assert meta["backend"] == "claude_cli" and meta["cli_version"].startswith("9.9.9")
    assert meta["cost_kind"] == "subscription" and meta["models_reported"] == ["claude-opus-5-5"]


@pytest.mark.parametrize(
    "scenario,needle",
    [
        ("badjson", "not JSON"),
        ("noso", "no structured_output"),
        ("exit1", "exit=1"),
    ],
)
def test_failed_call_is_a_documented_tool_error(fake, scenario, needle):
    llm = fake.set(scenario)
    prompt = Prompt("s1", "kl1.0", "system text", "h")
    res = call_tool(
        llm, cfg(), "anthropic/claude-opus-5-5", build_messages(prompt, [], cfg()), "sugya"
    )
    assert res.args is None and "claude_cli call failed" in res.error and needle in res.error


def test_usage_limit_and_auth_errors_raise(fake):
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    with pytest.raises(UsageLimitError, match="hit your limit"):
        fake.set("limit").complete(
            msgs, [tool_def("sugya")], model="opus", temperature=0, max_tokens=1
        )
    with pytest.raises(CLIAuthError, match="claude auth login"):
        fake.set("noauth").complete(
            msgs, [tool_def("sugya")], model="opus", temperature=0, max_tokens=1
        )


def test_not_logged_in_fails_clearly(fake):
    assert fake.set(auth="ok").ensure_logged_in()["authMethod"] == "claude.ai"
    for auth in ("loggedout", "apikey"):  # an API-key login is not the subscription
        with pytest.raises(CLIAuthError, match="claude auth login") as e:
            fake.set(auth=auth).ensure_logged_in()
        assert "secret@example" not in str(e.value)


# ---- through the runner (Postgres) -------------------------------------------------------------


def test_pipeline_on_cli_validates_and_records(world, fake):
    c = cli_cfg(second_model_fraction=1.0)  # every task would get a shadow run on the API
    c.second_model = "openai/gpt-5.1"
    run(world.engine, "s0", None, config=c)
    r1 = run(world.engine, "s1", fake.set("ok"), config=c)
    r2 = run(world.engine, "s2", fake.set("ok"), config=c)
    assert (r1.failed, r2.failed) == (0, 0) and r2.done == 3
    # the same Pydantic-validated outputs as the API path with ScriptedLLM
    assert [i.body["title_he"] for i in items(world.engine, "sugya")] == [
        "סוגיה א",
        "סוגיה ב",
        "סוגיה ג",
    ]
    summ = items(world.engine, "sugya_summary")
    assert len(summ) == 3 and all(i.model == c.model_for("s2") for i in summ)  # no shadow items
    assert all(TOOL_MODELS["sugya_summary"].model_validate(i.body) for i in summ)
    assert not any("gpt" in x["argv"][x["argv"].index("--model") + 1] for x in fake.log())
    with Session(world.engine) as s:
        job = s.get(KlJob, r2.job_id)
        assert job.params["backend"] == "claude_cli"
        st = job.stats
        assert st["backend"] == "claude_cli" and st["cost_kind"] == "subscription"
        assert st["cli_version"].startswith("9.9.9") and st["models_reported"] == [
            "claude-opus-5-5"
        ]
        assert st["second_model_skipped"] == 3 and "temperature" in st["unsupported_params"]
        assert abs(job.cost_usd - 3 * 0.0125) < 1e-9 and job.input_tokens == 3000
    for t in tasks(world.engine, r2.job_id).values():
        assert t.result["_llm"]["backend"] == "claude_cli"
        assert t.result["_llm"]["models_reported"] == ["claude-opus-5-5"]
        assert t.cost_usd == 0.0125


def test_cli_and_api_jobs_do_not_mix_on_resume(world, fake):
    run(world.engine, "s0", None)
    api = run(world.engine, "s1", ScriptedLLM(s1_responder()))
    cli = run(world.engine, "s1", fake.set("ok"), config=cli_cfg())
    assert cli.job_id != api.job_id and cli.skipped == 0 and cli.done == 3


@pytest.mark.parametrize("scenario,needle", [("badjson", "not JSON"), ("schema", "K1: schema")])
def test_invalid_output_retries_once_then_fails_documented(world, fake, scenario, needle):
    run(world.engine, "s0", None)
    run(world.engine, "s1", ScriptedLLM(s1_responder()))
    r = run(world.engine, "s2", fake.set(scenario), config=cli_cfg())
    assert r.status == "done" and r.done == 3
    assert len(fake.log()) == 6  # attempt + exactly one retry per task
    for it in items(world.engine, "sugya_summary"):
        assert it.status == "rejected" and not it.is_current  # never accepted silently
    with Session(world.engine) as s:
        rejected = s.scalars(select(KlItem).where(KlItem.kind == "sugya_summary")).first()
        assert rejected is not None
    errs = [
        x for t in tasks(world.engine, r.job_id).values() for x in t.result["_llm"]["failed_calls"]
    ]
    if scenario == "badjson":
        assert len(errs) == 6 and all(needle in e for e in errs)


def test_nonzero_exit_fails_the_task_in_s1(world, fake):
    run(world.engine, "s0", None)
    r = run(world.engine, "s1", fake.set("exit1"), config=cli_cfg())
    assert r.failed >= 1 and r.status == "failed"
    t = next(t for t in tasks(world.engine, r.job_id).values() if t.state == "failed")
    assert "claude_cli call failed" in t.last_error and "exit=1" in t.last_error
    assert items(world.engine, "sugya") == []


def test_usage_limit_stops_cleanly_and_resumes(world, fake):
    run(world.engine, "s0", None)
    run(world.engine, "s1", ScriptedLLM(s1_responder()))
    c = cli_cfg()
    r = run(world.engine, "s2", fake.set("ok,limit"), config=c)
    assert r.status == "aborted" and r.failed == 0 and r.done == 1
    assert "UsageLimitError" in r.stopped and 'rav-kl run s2 "' in r.resume
    assert "RAV_KL_BACKEND" in r.resume and "claude_cli" in r.resume  # backend explicit
    ts = tasks(world.engine, r.job_id)
    states = sorted(t.state for t in ts.values())
    assert states == ["done", "pending", "pending"]  # the hit task is pending, not failed
    hit = next(t for t in ts.values() if t.last_error)
    assert hit.state == "pending" and "hit your limit" in hit.last_error
    assert len(items(world.engine, "sugya_summary")) == 1  # nothing written for the hit task
    with Session(world.engine) as s:
        assert s.get(KlJob, r.job_id).status == "aborted"
    done_before = {k: t.updated_at for k, t in ts.items() if t.state == "done"}
    r2 = run(world.engine, "s2", fake.set("ok"), config=c)
    assert r2.job_id == r.job_id and r2.status == "done" and r2.done == 3
    ts2 = tasks(world.engine, r.job_id)
    assert all(ts2[k].updated_at == v for k, v in done_before.items())  # done tasks untouched
    assert len(items(world.engine, "sugya_summary")) == 3


def test_auth_lost_mid_run_is_a_clean_stop(world, fake):
    run(world.engine, "s0", None)
    r = run(world.engine, "s1", fake.set("noauth"), config=cli_cfg())
    assert r.status == "aborted" and r.failed == 0 and "claude auth login" in r.stopped
    assert {t.state for t in tasks(world.engine, r.job_id).values()} == {"pending"}


def test_limit_runs_at_most_n_new_tasks(world):
    run(world.engine, "s0", None)
    run(world.engine, "s1", ScriptedLLM(s1_responder()))
    llm = ScriptedLLM(s2_responder)
    r = run(world.engine, "s2", llm, limit=1)
    assert r.done == 1 and llm.n == 1 and r.status == "aborted"
    assert r.stopped == "--limit 1 reached" and r.resume.endswith("--limit 1")
    r2 = run(world.engine, "s2", ScriptedLLM(s2_responder), limit=5)
    assert r2.job_id == r.job_id and r2.done == 3 and r2.status == "done" and r2.resume is None


# ---- rav-kl run ------------------------------------------------------------------------------


def test_rav_kl_run_refuses_when_not_logged_in(world, fake, monkeypatch, capsys):
    monkeypatch.setenv("RAV_KL_BACKEND", "claude_cli")
    client = fake.set(auth="loggedout")
    monkeypatch.setattr(backends, "make_llm", lambda c: client)
    rav_kl(["--db", world.engine.test_url, "run", "s0", SCOPE])  # s0 needs no LLM / login
    capsys.readouterr()
    rc = rav_kl(["--db", world.engine.test_url, "run", "s1", SCOPE])
    err = capsys.readouterr().err
    assert rc == 2 and "claude auth login" in err and fake.log() == []
    with Session(world.engine) as s:  # nothing started; no fallback to the API path
        assert s.scalars(select(KlJob).where(KlJob.stage == "s1")).first() is None


def test_rav_kl_run_limit_prints_resume(world, fake, monkeypatch, capsys):
    monkeypatch.setenv("RAV_KL_BACKEND", "claude_cli")
    client = fake.set("ok")
    monkeypatch.setattr(backends, "make_llm", lambda c: client)
    rav_kl(["--db", world.engine.test_url, "run", "s0", SCOPE])
    capsys.readouterr()
    rc = rav_kl(["--db", world.engine.test_url, "run", "s1", SCOPE, "--limit", "1"])
    err = capsys.readouterr().err
    assert rc == 3 and "resume with: " in err and "rav-kl run s1" in err
    assert "RAV_KL_BACKEND" in err and len(fake.log()) == 1


def _flag(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


# ---- follow-up fixes ---------------------------------------------------------------------------


def test_is_error_with_valid_structured_output_is_a_failure(fake):
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "[S1|canonical|Shabbat] Shabbat 21a:1 ⟩ אמר רב"},
    ]
    r = fake.set("iserror_valid").complete(
        msgs, [tool_def("sugya")], model="opus", temperature=0, max_tokens=1
    )
    (tc,) = r.tool_calls
    assert tc.arguments == {} and "is_error=True" in tc.arguments_error  # never accepted
    res = call_tool(fake.set("iserror_valid"), cfg(), "anthropic/claude-opus-5-5", msgs, "sugya")
    assert res.args is None and "claude_cli call failed" in res.error


def test_provider_keys_never_reach_auth_status_or_calls(fake):
    fake.set(auth="ok").ensure_logged_in()
    (rec,) = fake.auth_log()
    assert rec["argv"][:2] == ["auth", "status"] and rec["keys_visible"] == []
    fake.set("ok").complete(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "Shabbat 21a:1"}],
        [tool_def("sugya")],
        model="opus",
        temperature=0,
        max_tokens=1,
    )
    (call,) = fake.log()
    assert call["api_key_visible"] is False and call["auth_token_visible"] is False


def test_argv_too_long_fails_clearly(monkeypatch):
    huge = "x" * 200_000
    with pytest.raises(backends.CLICommandTooLong, match="too large"):
        backends.check_argv(["claude.exe", "--json-schema", huge])
    backends.check_argv(["claude.exe", "--json-schema", "x" * 3000])  # a real schema fits
    monkeypatch.setattr(backends.os, "name", "nt")
    with pytest.raises(backends.CLICommandTooLong, match="CreateProcess"):
        backends.check_argv(["C:/x/claude.exe", "x" * 33_000])
    with pytest.raises(backends.CLICommandTooLong, match="cmd.exe"):
        backends.check_argv(["C:/x/claude.cmd", "x" * 9_000])
    backends.check_argv(["C:/x/claude.exe", "x" * 9_000])  # fine for the real exe


def test_too_long_argv_stops_the_run_cleanly(world, fake, monkeypatch):
    monkeypatch.setattr(backends, "_POSIX_ARG_MAX", 400)  # above --version, below the schema
    monkeypatch.setattr(backends, "_WIN_EXE_MAX", 400)
    run(world.engine, "s0", None)
    r = run(world.engine, "s1", fake.set("ok"), config=cli_cfg())
    assert r.status == "aborted" and r.failed == 0 and "CLICommandTooLong" in r.stopped
    assert fake.log() == []  # nothing was spawned


def test_cmd_shim_without_exe_is_refused(tmp_path, monkeypatch):
    shim = tmp_path / "claude.cmd"
    shim.write_text("@echo off\n")
    monkeypatch.setattr(backends.shutil, "which", lambda name: str(shim))
    with pytest.raises(backends.CLINotFound, match="no claude.exe next to it"):
        backends.resolve_bin("claude")
    exe = tmp_path / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    assert backends.resolve_bin("claude") == str(exe)


def test_limit_counts_only_tasks_that_call_the_llm(world):
    run(world.engine, "s0", None)
    llm = ScriptedLLM(s1_responder())
    r = run(world.engine, "s1", llm, limit=2)  # two windows (LLM) + the merge (deterministic)
    assert llm.n == 2 and r.done == 3 and r.status == "done" and r.stopped is None


def test_resume_command_names_the_backend(monkeypatch):
    from rav_knowledge.runner import resume_command

    monkeypatch.setattr("rav_knowledge.runner.os.name", "posix")
    assert resume_command("s2", "Shabbat 21a", "claude_cli", 3) == (
        'RAV_KL_BACKEND=claude_cli rav-kl run s2 "Shabbat 21a" --limit 3'
    )
    monkeypatch.setattr("rav_knowledge.runner.os.name", "nt")
    assert resume_command("s1", "Shabbat 21a", "api") == (
        '$env:RAV_KL_BACKEND = "api"; rav-kl run s1 "Shabbat 21a"'
    )
