"""Model layer: a tiny protocol over LiteLLM so the loop (and tests) never touch the network."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""  # as sent by the model (kept for the audit log)
    arguments_error: str | None = None  # set when the arguments were not a JSON object


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    model: str = ""

    def assistant_message(self) -> dict[str, Any]:
        """The assistant turn in OpenAI chat format, to append to the transcript."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.name,
                        "arguments": c.raw_arguments or json.dumps(c.arguments, ensure_ascii=False),
                    },
                }
                for c in self.tool_calls
            ]
        return msg


class LLM(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse: ...


def parse_tool_call(call_id: str, name: str, raw: str | None) -> ToolCall:
    raw = raw or ""
    try:
        args = json.loads(raw) if raw.strip() else {}
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
    except ValueError as e:
        return ToolCall(call_id, name, {}, raw, f"invalid arguments: {e}")
    return ToolCall(call_id, name, args, raw)


class LiteLLMClient:
    """Real client. The model id is any LiteLLM id, chosen in config/harness.toml."""

    def complete(self, messages, tools, *, model, temperature, max_tokens) -> LLMResponse:
        import litellm  # imported lazily: heavy, and tests never need it

        resp = litellm.completion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        msg = resp.choices[0].message
        calls = [
            parse_tool_call(tc.id, tc.function.name, tc.function.arguments)
            for tc in (getattr(msg, "tool_calls", None) or [])
        ]
        usage = getattr(resp, "usage", None)
        try:
            cost = float(litellm.completion_cost(completion_response=resp))
        except Exception:  # noqa: BLE001 - unknown model pricing must not fail the run
            cost = None
        return LLMResponse(
            content=msg.content,
            tool_calls=calls,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cost_usd=cost,
            model=getattr(resp, "model", None) or model,
        )
