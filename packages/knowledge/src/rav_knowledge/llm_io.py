"""Building LLM input and calling the model through ``rav_harness.llm``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from rav_harness.llm import LLM

from .config import KLConfig
from .prompts import Prompt
from .schemas import TOOL_MODELS
from .store import InputSeg


def numbered_text(text: str) -> str:
    """anchor_format=numbered: every word carries its index, e.g. ``אמר⟨0⟩ רב⟨1⟩``."""
    return " ".join(f"{w}⟨{i}⟩" for i, w in enumerate(text.split()))


def format_line(x: InputSeg, anchor_format: str = "quote") -> str:
    text = x.text.text_a
    if anchor_format == "numbered" and not x.is_aid:
        text = numbered_text(text)
    tier = x.info.tier or "-"
    return f"[{x.info.label}|{tier}|{x.info.work_title}] {x.info.ref} ⟩ {text}"


def build_messages(
    prompt: Prompt,
    segs: list[InputSeg],
    cfg: KLConfig,
    *,
    aids: list[InputSeg] | None = None,
    prior: list[tuple[str, int, str]] | None = None,
    extra: str = "",
    feedback: str | None = None,
) -> list[dict[str, Any]]:
    """Stable prefix (system) + the input. Prior items appear only under the 'aid notes' header."""
    system = prompt.text
    if cfg.anchor_format == "numbered":
        system += (
            "\n\nכל מילה בקלט מסומנת באינדקס ⟨n⟩. בכל ראיה החזר גם w_start: אינדקס המילה "
            "הראשונה של הציטוט. את הציטוט עצמו כתוב ללא סימוני האינדקס."
        )
    parts = [format_line(x, cfg.anchor_format) for x in segs]
    if aids:
        parts += ["", "עזרי קריאה (אינם ראיה, אסור לצטט כ-seg):"]
        parts += [format_line(a) for a in aids]
    if prior:
        parts += ["", "הערות עזר, עלולות לטעות (אינן מקור):"]
        parts += [f"- {key} v{ver}: {summary}" for key, ver, summary in prior]
    if extra:
        parts += ["", extra]
    if feedback:
        parts += ["", "הגשה קודמת נפסלה בבדיקות הדטרמיניסטיות. תקן:", feedback]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(parts)},
    ]


def tool_def(name: str) -> dict[str, Any]:
    model = TOOL_MODELS[name]
    schema = model.model_json_schema()
    schema.setdefault("properties", {}).setdefault("notes", {"type": ["string", "null"]})
    return {
        "type": "function",
        "function": {
            "name": f"submit_{name}",
            "description": f"Submit the {name} result. Evidence is quotes, never offsets.",
            "parameters": schema,
        },
    }


@dataclass
class CallResult:
    args: dict[str, Any] | None
    error: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    model: str


def call_tool(
    llm: LLM, cfg: KLConfig, model: str, messages: list[dict[str, Any]], tool_name: str
) -> CallResult:
    resp = llm.complete(
        messages,
        [tool_def(tool_name)],
        model=model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    cost = resp.cost_usd
    if cost is None:
        cost = cfg.cost(model, resp.input_tokens, resp.output_tokens)
    want = f"submit_{tool_name}"
    for c in resp.tool_calls:
        if c.name == want:
            if c.arguments_error:
                return CallResult(
                    None, c.arguments_error, resp.input_tokens, resp.output_tokens, cost, model
                )
            return CallResult(c.arguments, None, resp.input_tokens, resp.output_tokens, cost, model)
    return CallResult(
        None, f"no {want} call in the response", resp.input_tokens, resp.output_tokens, cost, model
    )


def count_tokens(text: str, model: str | None = None) -> tuple[int, str]:
    """(tokens, method). Provider count_tokens when an API key is present, else tiktoken."""
    import os

    if model and model.startswith("anthropic/") and os.environ.get("ANTHROPIC_API_KEY"):
        try:  # pragma: no cover - needs network
            import litellm

            n = litellm.token_counter(model=model, messages=[{"role": "user", "content": text}])
            return int(n), "provider"
        except Exception:  # noqa: BLE001
            pass
    import tiktoken

    return len(tiktoken.get_encoding("cl100k_base").encode(text)), "tiktoken~"


def summarize_item(kind: str, body: dict) -> str:
    return json.dumps(body, ensure_ascii=False)[:300]
