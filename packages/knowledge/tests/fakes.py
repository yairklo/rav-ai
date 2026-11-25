"""Scripted LLM: no network. The responder gets (messages, tool_name, model) and returns the
tool arguments (dict) or None for 'no tool call'."""

from __future__ import annotations

import json
from typing import Any

from rav_harness.llm import LLMResponse, ToolCall


class ScriptedLLM:
    def __init__(self, responder):
        self.responder = responder
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages, tools, *, model, temperature, max_tokens) -> LLMResponse:
        name = tools[0]["function"]["name"]
        self.calls.append({"messages": messages, "tool": name, "model": model})
        args = self.responder(messages, name.removeprefix("submit_"), model)
        if args is None:
            return LLMResponse(content="no", input_tokens=100, output_tokens=5, model=model)
        raw = json.dumps(args, ensure_ascii=False)
        return LLMResponse(
            content=None,
            tool_calls=[ToolCall("c1", name, args, raw)],
            input_tokens=1000,
            output_tokens=200,
            cost_usd=None,
            model=model,
        )

    @property
    def n(self) -> int:
        return len(self.calls)
