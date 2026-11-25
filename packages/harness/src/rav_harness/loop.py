"""The agent loop: a few dozen lines on purpose. Model call -> tool calls -> repeat, until the
model calls ``submit_answer`` with a valid Answer (or limits are hit)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError
from rav_evals.schema import Answer, ClaimType

from .config import HarnessConfig, load_config
from .llm import LLM, LLMResponse, ToolCall
from .prompt import DEFAULT_DISCLAIMER, PROMPT_VERSION, prompt_for, prompt_hash
from .tools import ToolBox, timed_call
from .validators import KnowledgeContext, validate_knowledge

SUBMIT = "submit_answer"
_SYSTEM_TYPES = (ClaimType.SYSTEM_CONCLUSION, ClaimType.CHIDUSH)


def submit_spec() -> dict[str, Any]:
    schema = Answer.model_json_schema()
    schema["properties"].pop("is_psak", None)  # always False; the model does not decide it
    return {
        "type": "function",
        "function": {
            "name": SUBMIT,
            "description": "Submit the final answer. Call exactly once, when done.",
            "parameters": schema,
        },
    }


@dataclass
class HarnessResult:
    question: str
    answer: Answer | None
    retrieved_refs: list[str]
    tool_calls: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    model: str
    steps: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_ms: int
    error: str | None = None
    run_id: str | None = None
    rejected_answers: list[str] = field(default_factory=list)
    prompt_version: str = PROMPT_VERSION
    prompt_hash: str = field(default_factory=prompt_hash)
    knowledge: str = "off"
    scope_gate: dict[str, Any] | None = None
    knowledge_items: list[dict[str, Any]] = field(default_factory=list)  # runs.knowledge_items


def validate_answer(
    arguments: dict[str, Any], retrieved: set[str], kl: KnowledgeContext | None = None
) -> tuple[Answer | None, str]:
    """Deterministic guards on top of the schema. Returns (answer, error-for-the-model).
    ``kl`` (knowledge on) adds the knowledge-layer checks of :mod:`rav_harness.validators`."""
    data = {k: v for k, v in arguments.items() if k != "is_psak"}
    try:
        answer = Answer.model_validate(data)
    except ValidationError as e:
        return None, f"answer does not match the schema: {e}"
    problems = []
    if kl is not None:
        problems += validate_knowledge(answer, retrieved, kl)
    for i, c in enumerate(answer.claims):
        if c.type in _SYSTEM_TYPES and not c.flagged_as_system:
            problems.append(f"claim {i} ({c.type}) must set flagged_as_system=true")
        for cit in c.citations:
            if cit.ref not in retrieved and not (kl and cit.ref in kl.layer_refs):
                problems.append(
                    f"claim {i} cites {cit.ref!r}, which was not retrieved; "
                    "fetch it with get_segment first"
                )
    if problems:
        return None, "; ".join(problems)
    answer.is_psak = False
    if not answer.disclaimer.strip():
        answer.disclaimer = DEFAULT_DISCLAIMER
    return answer, ""


class Harness:
    def __init__(
        self,
        llm: LLM,
        tools: ToolBox,
        config: HarnessConfig | None = None,
        *,
        recorder=None,  # callable(HarnessResult, question_id, golden_version) -> run_id
    ):
        self.llm, self.tools = llm, tools
        self.config = config or load_config()
        self.recorder = recorder

    def run(
        self, question: str, *, question_id: str | None = None, golden_version: str | None = None
    ) -> HarnessResult:
        cfg = self.config
        t0 = time.perf_counter()
        calls_log: list[dict[str, Any]] = []
        kl_tools = getattr(self.tools, "kl", None) if cfg.knowledge != "off" else None
        version, system_prompt, p_hash = prompt_for(cfg.knowledge if kl_tools else "off")
        user_content = question
        gate_dict: dict[str, Any] | None = None
        if kl_tools is not None:
            g0 = time.perf_counter()
            gate = kl_tools.run_scope_gate(question)
            gate_dict = gate.as_dict()
            calls_log.append(
                {
                    "name": "scope_gate",
                    "arguments": {"question": question},
                    "ms": int((time.perf_counter() - g0) * 1000),
                    "error": None,
                }
            )
            user_content = (
                question
                + "\n\n[שער תחום, שלב דטרמיניסטי; השתמש בו, אל תצטט אותו]\n"
                + json.dumps(gate_dict, ensure_ascii=False)
            )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        specs = [*self.tools.specs(), submit_spec()]
        rejected: list[str] = []
        tokens_in = tokens_out = steps = 0
        cost: float | None = None
        model = cfg.model
        answer: Answer | None = None
        error: str | None = None

        while answer is None:
            if steps >= cfg.max_steps:
                error = f"max_steps ({cfg.max_steps}) reached without a valid answer"
                break
            steps += 1
            try:
                resp: LLMResponse = self.llm.complete(
                    messages,
                    specs,
                    model=cfg.model,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                )
            except Exception as e:  # noqa: BLE001 - recorded, not raised
                error = f"llm error: {type(e).__name__}: {e}"
                break
            tokens_in += resp.input_tokens
            tokens_out += resp.output_tokens
            if resp.cost_usd is not None:
                cost = (cost or 0.0) + resp.cost_usd
            model = resp.model or model
            messages.append(resp.assistant_message())

            if not resp.tool_calls:
                messages.append({"role": "user", "content": f"Call {SUBMIT} to give the answer."})
                continue
            for call in resp.tool_calls:
                answer, result = self._handle(call, calls_log, rejected)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
                if answer is not None:
                    break
            if answer is None and len(rejected) > cfg.max_answer_retries:
                error = f"answer rejected {len(rejected)} times: {rejected[-1]}"
                break

        res = HarnessResult(
            question=question,
            answer=answer,
            retrieved_refs=list(self.tools.retrieved_refs),
            tool_calls=calls_log,
            messages=messages,
            model=model,
            steps=steps,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            cost_usd=cost,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            error=error,
            rejected_answers=rejected,
            prompt_version=version,
            prompt_hash=p_hash,
            knowledge=cfg.knowledge if kl_tools else "off",
            scope_gate=gate_dict,
            knowledge_items=(
                [
                    {
                        "meta": {"knowledge": cfg.knowledge, "scope_gate": gate_dict},
                        "tool": "_meta",
                    },
                    *kl_tools.ctx.records,
                ]
                if kl_tools is not None
                else []
            ),
        )
        if self.recorder:
            res.run_id = self.recorder(res, question_id, golden_version)
        return res

    def _handle(self, call: ToolCall, log: list, rejected: list[str]):
        if call.arguments_error:
            log.append(
                {"name": call.name, "arguments": call.raw_arguments, "error": call.arguments_error}
            )
            return None, {"error": call.arguments_error}
        if call.name == SUBMIT:
            ans, err = validate_answer(
                call.arguments,
                set(self.tools.retrieved_refs),
                getattr(getattr(self.tools, "kl", None), "ctx", None)
                if self.config.knowledge != "off"
                else None,
            )
            log.append({"name": SUBMIT, "arguments": call.arguments, "error": err or None})
            if ans is None:
                rejected.append(err)
                return None, {"accepted": False, "error": err}
            return ans, {"accepted": True}
        result, ms = timed_call(self.tools, call.name, call.arguments)
        log.append(
            {"name": call.name, "arguments": call.arguments, "ms": ms, "error": result.get("error")}
        )
        return None, result
