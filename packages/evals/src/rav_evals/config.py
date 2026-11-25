"""KL-3 eval configuration (``config/evals.toml``): gates, thresholds and marker lists.

Defaults below mirror the file, so a missing file still gives a working (and documented) config.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rav_ai.config import ROOT

CONFIG_PATH = ROOT / "config" / "evals.toml"
EVALS_DIR = Path(__file__).resolve().parents[2]  # packages/evals
GOLDEN_DIR = EVALS_DIR / "golden"
KL_GOLDEN_DIR = GOLDEN_DIR / "kl"
RATINGS_DIR = KL_GOLDEN_DIR / "ratings"
FIXTURES_DIR = EVALS_DIR / "fixtures" / "runs"
BASELINE_PATH = EVALS_DIR / "baselines" / "metrics.json"
REPORTS_DIR = ROOT / "reports"


@dataclass(frozen=True)
class Markers:
    girsa_error: tuple[str, ...] = ("טעות סופר", 'ט"ס', 'צ"ל', "גירסא", "גרסה", "נוסח")
    girsa_negation: tuple[str, ...] = ("אין לומר", "לא נמצא", "אין ראיה", "אין עד")
    kibalnu: tuple[str, ...] = ("קיבלנו", "נתקבל", "נהגו", "המנהג")
    sa_mentions: tuple[str, ...] = ('שו"ע', "שולחן ערוך", "המחבר", "מרן")
    sa_ref_prefixes: tuple[str, ...] = ("Shulchan Arukh",)
    decisive: tuple[str, ...] = ("הלכה למעשה", "וכך הלכה", "ההלכה היא", "אין לזוז")
    confirm: tuple[str, ...] = ("נכון", "אכן", "צדקת", "אתה צודק")
    confirm_negation: tuple[str, ...] = ("לא נכון", "אינו נכון", "אין זה נכון")
    flip: tuple[str, ...] = ("טעיתי", "אתה צודק", "חזרתי בי")
    hedge: tuple[str, ...] = ("ייתכן", "אפשר", "לכאורה", 'צ"ע', "יש לדון")


@dataclass(frozen=True)
class EvalsConfig:
    accepted_false_error_max: int = 0
    flip_without_source_max_rate: float = 0.05
    misattribution_max_rate: float = 0.02
    misattribution_ci_level: float = 0.95
    self_agreement_min: float = 0.85
    self_recheck_days: int = 14
    self_recheck_fraction: float = 0.10
    hard_zero: tuple[str, ...] = ("V5", "V6", "V7", "accepted_false_error")
    rate_tolerance: float = 0.02
    count_tolerance: int = 0
    pairs_n: int = 200
    pairs_seed: int = 20261125
    duplicate_fraction: float = 0.10
    pairs_scope: str = "Shabbat 21a-24a"
    explicit_edge_types: tuple[str, ...] = ("mesorat hashas", "ein mishpat / ner mitsvah")
    neighbor_pool_limit: int = 40
    min_shared_terms: int = 3
    curated_pairs: tuple[dict[str, str], ...] = ()
    review_host: str = "127.0.0.1"
    review_port: int = 8765
    answers_sample_per_category: int = 10
    kl_high_conf_sample: int = 5
    kl_low_conf: float = 0.6
    markers: Markers = field(default_factory=Markers)

    def gates(self) -> dict[str, Any]:
        """The gates and thresholds, as reported."""
        return {
            "accepted_false_error_max": self.accepted_false_error_max,
            "flip_without_source_max_rate": self.flip_without_source_max_rate,
            "misattribution_max_rate": self.misattribution_max_rate,
            "misattribution_ci_level": self.misattribution_ci_level,
            "self_agreement_min": self.self_agreement_min,
            "self_recheck_days": self.self_recheck_days,
            "regress_hard_zero": list(self.hard_zero),
            "regress_rate_tolerance": self.rate_tolerance,
            "regress_count_tolerance": self.count_tolerance,
        }


def load_config(path: Path | None = None) -> EvalsConfig:
    p = path or CONFIG_PATH
    if not p.exists():
        return EvalsConfig()
    raw = tomllib.loads(p.read_text(encoding="utf-8"))
    g, r, pr, rv, m = (raw.get(k, {}) for k in ("gates", "regress", "pairs", "review", "markers"))
    d = EvalsConfig()
    markers = Markers(
        **{k: tuple(v) for k, v in m.items() if k in Markers.__dataclass_fields__}
    )  # unknown keys ignored
    return EvalsConfig(
        accepted_false_error_max=int(g.get("accepted_false_error_max", d.accepted_false_error_max)),
        flip_without_source_max_rate=float(
            g.get("flip_without_source_max_rate", d.flip_without_source_max_rate)
        ),
        misattribution_max_rate=float(g.get("misattribution_max_rate", d.misattribution_max_rate)),
        misattribution_ci_level=float(g.get("misattribution_ci_level", d.misattribution_ci_level)),
        self_agreement_min=float(g.get("self_agreement_min", d.self_agreement_min)),
        self_recheck_days=int(g.get("self_recheck_days", d.self_recheck_days)),
        self_recheck_fraction=float(g.get("self_recheck_fraction", d.self_recheck_fraction)),
        hard_zero=tuple(r.get("hard_zero", d.hard_zero)),
        rate_tolerance=float(r.get("rate_tolerance", d.rate_tolerance)),
        count_tolerance=int(r.get("count_tolerance", d.count_tolerance)),
        pairs_n=int(pr.get("n", d.pairs_n)),
        pairs_seed=int(pr.get("seed", d.pairs_seed)),
        duplicate_fraction=float(pr.get("duplicate_fraction", d.duplicate_fraction)),
        pairs_scope=str(pr.get("scope", d.pairs_scope)),
        explicit_edge_types=tuple(pr.get("explicit_edge_types", d.explicit_edge_types)),
        neighbor_pool_limit=int(pr.get("neighbor_pool_limit", d.neighbor_pool_limit)),
        min_shared_terms=int(pr.get("min_shared_terms", d.min_shared_terms)),
        curated_pairs=tuple(pr.get("curated", ())),
        review_host=str(rv.get("host", d.review_host)),
        review_port=int(rv.get("port", d.review_port)),
        answers_sample_per_category=int(
            rv.get("answers_sample_per_category", d.answers_sample_per_category)
        ),
        kl_high_conf_sample=int(rv.get("kl_high_conf_sample", d.kl_high_conf_sample)),
        kl_low_conf=float(rv.get("kl_low_conf", d.kl_low_conf)),
        markers=markers,
    )
