"""``rav-evals compare A B [C]``: side-by-side metrics of two or three eval bundles (e.g.
knowledge off / base / base+lens). Writes ``reports/evals/compare-<labels>.md`` and ``.json``."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROWS: list[tuple[str, str]] = [
    ("source_recall", "Source recall (required refs cited)"),
    ("required_sides_rate", "Required sides mentioned"),
    ("rubric_pass_rate", "Rubric pass rate"),
    ("verifier_pass_rate", "Verifier pass"),
    ("secondary_as_primary", "secondary_as_primary (must be 0)"),
    ("candidate_as_proof", "candidate_as_proof (must be 0)"),
    ("coverage_notice_fidelity", "Coverage notice fidelity (V8)"),
    ("sycophancy_pass_rate", "Sycophancy pass rate"),
    ("accepted_false_error", "accepted_false_error (must be 0)"),
    ("flip_without_source_rate", "flip_without_source rate"),
    ("novel_pass_rate", "Novel pass rate"),
    ("cost_usd", "Cost (USD)"),
]


def load_bundle(path: Path) -> dict[str, Any]:
    p = Path(path)
    if p.is_dir():
        p = p / "bundle.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    data.setdefault("label", p.parent.name)
    return data


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def compare(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [b["label"] for b in bundles]
    rows = {k: [b["metrics"].get(k) for b in bundles] for k, _ in ROWS}
    return {
        "labels": labels,
        "rows": rows,
        "kl_items_in_failed": {
            b["label"]: b["metrics"].get("kl_items_in_failed", {}) for b in bundles
        },
    }


def to_markdown(c: dict[str, Any]) -> str:
    lines = ["# Eval comparison", "", "| metric | " + " | ".join(c["labels"]) + " |"]
    lines.append("|---" * (len(c["labels"]) + 1) + "|")
    for k, title in ROWS:
        lines.append(f"| {title} | " + " | ".join(_fmt(v) for v in c["rows"][k]) + " |")
    lines += ["", "## Knowledge-layer items used in failed questions", ""]
    for label, items in c["kl_items_in_failed"].items():
        lines.append(f"- **{label}**: " + (", ".join(f"{k} x{v}" for k, v in items.items()) or "-"))
    return "\n".join(lines) + "\n"


def write_compare(c: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9_+=-]+", "_", "_vs_".join(c["labels"]))[:120]
    jp, mp = out_dir / f"compare-{slug}.json", out_dir / f"compare-{slug}.md"
    jp.write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")
    mp.write_text(to_markdown(c), encoding="utf-8")
    return jp, mp
