from __future__ import annotations

import re

from rav_knowledge.config import load_config
from rav_knowledge.runner import run_stage
from rav_knowledge.stages import STAGES  # noqa: F401

SCOPE = "Shabbat 21a-23a"


def cfg(**kw):
    c = load_config(second_model_fraction=0.0, **kw)
    return c


def input_lines(messages) -> dict[str, str]:
    """label -> text of the evidence-eligible input lines (aids start with A)."""
    out = {}
    for line in messages[-1]["content"].splitlines():
        m = re.match(r"^\[(S\d+)\|[^\]]*\] .*? ⟩ (.*)$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def s1_responder(variant: str = "A"):
    def r(messages, tool, model):
        if tool != "sugya":
            return None
        text = messages[-1]["content"]
        if "Shabbat 21a:1" in text:
            return {
                "sugyot": [
                    {"title_he": "סוגיה א", "start_seg": "S1", "end_seg": "S4"},
                    {"title_he": "סוגיה ב", "start_seg": "S5", "end_seg": "S6"},
                ]
            }
        if variant == "B":
            return {"sugyot": [{"title_he": "סוגיה ב", "start_seg": "S1", "end_seg": "S6"}]}
        return {
            "sugyot": [
                {"title_he": "סוגיה ב", "start_seg": "S1", "end_seg": "S4"},
                {"title_he": "סוגיה ג", "start_seg": "S5", "end_seg": "S6"},
            ]
        }

    return r


def ev(seg, quote, role="supports"):
    return {"seg": seg, "quote": quote, "role": role}


def s2_responder(messages, tool, model):
    if tool != "sugya_summary":
        return None
    lines = input_lines(messages)
    if "והלכתא" in " ".join(lines.values()):  # sugya A
        return {
            "title_he": "סוגיה א",
            "steps": [
                {
                    "id": "s1",
                    "move": "statement",
                    "text_he": "רב מחייב ושמואל פוטר",
                    "evidence": [ev("S1", "אמר רב חייב", "states")],
                },
                {
                    "id": "s2",
                    "move": "objection",
                    "responds_to": "s1",
                    "relation": "attacks",
                    "text_he": "מקשים מדברי רבי אושעיא",
                    "evidence": [ev("S2", "מתיבי רבי אושעיא", "states")],
                },
                {
                    "id": "s3",
                    "move": "answer",
                    "responds_to": "s2",
                    "relation": "answers",
                    "text_he": "ריש לקיש פוטר",
                    "evidence": [ev("S2", 'ור"ל אמר פטור')],
                },
                {
                    "id": "s4",
                    "move": "conclusion",
                    "subtype": "hilkheta",
                    "responds_to": "s3",
                    "relation": "supports",
                    "text_he": "הלכה כרב",
                    "evidence": [ev("S4", "והלכתא כרב", "states")],
                },
            ],
        }
    first, text = next(iter(lines.items()))
    return {
        "title_he": "סוגיה",
        "steps": [
            {
                "id": "s1",
                "move": "statement",
                "text_he": "פתיחה",
                "evidence": [ev(first, " ".join(text.split()[:3]), "states")],
            }
        ],
    }


def bad_s2_responder(messages, tool, model):
    return {
        "title_he": "סוגיה",
        "steps": [
            {
                "id": "s1",
                "move": "statement",
                "text_he": "מומצא",
                "evidence": [ev("S1", "ציטוט שאינו קיים בטקסט", "states")],
            }
        ],
    }


def run(engine, stage, llm, scope=SCOPE, config=None, **kw):
    return run_stage(engine, stage, scope, config or cfg(), llm, **kw)


def pipeline(engine, llm_s1, llm_s2, scope=SCOPE, config=None):
    """S0 -> S1 -> S2 with promote-less first-run semantics (first output is current)."""
    c = config or cfg()
    run(engine, "s0", None, scope, c)
    run(engine, "s1", llm_s1, scope, c)
    return run(engine, "s2", llm_s2, scope, c)
