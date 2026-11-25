"""config/knowledge.toml -> typed settings."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from rav_ai.config import ROOT

CONFIG_PATH = ROOT / "config" / "knowledge.toml"
STAGES = ("s0", "s0b", "s1", "s2", "s3", "s3l", "s4", "s4g", "s5")
IMPLEMENTED_STAGES = ("s0", "s0b", "s1", "s2")


@dataclass(frozen=True)
class Price:
    input: float  # USD per million tokens
    output: float


@dataclass
class KLConfig:
    mode: str = "personal"
    allow_personal_only: bool = False
    use_steinsaltz_aid_flag: bool | None = None
    steinsaltz_prefix: str = "Steinsaltz on "
    anchor_format: str = "quote"
    second_model_fraction: float = 0.10
    batch: bool = False
    batch_discount: float = 0.5
    temperature: float = 0.0
    max_tokens: int = 8000
    window_amudim: int = 3
    window_overlap: int = 1
    budget_usd: float = 0.0
    models: dict[str, str] = field(default_factory=dict)
    second_model: str | None = None
    prices: dict[str, Price] = field(default_factory=dict)
    out_ratio: dict[str, float] = field(default_factory=dict)
    k3_not_found_max_fraction: float = 0.10
    step_max_words: int = 25
    issue_similarity_warn: float = 0.6

    @property
    def use_steinsaltz_aid(self) -> bool:
        """Default: on only in personal mode. Never evidence, never citation (see prompts)."""
        if self.use_steinsaltz_aid_flag is None:
            return self.mode == "personal"
        return self.use_steinsaltz_aid_flag

    def model_for(self, stage: str) -> str:
        return self.models.get(stage) or next(iter(self.models.values()), "unknown")

    def price(self, model: str) -> Price | None:
        return self.prices.get(model)

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        p = self.price(model)
        if p is None:
            return None
        c = (input_tokens * p.input + output_tokens * p.output) / 1_000_000
        return c * self.batch_discount if self.batch else c

    def params(self) -> dict:
        """What affects output; goes into every task input_hash."""
        return {
            "mode": self.mode,
            "anchor_format": self.anchor_format,
            "steinsaltz": self.use_steinsaltz_aid,
            "temperature": self.temperature,
            "allow_personal_only": self.allow_personal_only,
        }


def load_config(path: Path | None = None, **overrides) -> KLConfig:
    raw = tomllib.loads((path or CONFIG_PATH).read_text(encoding="utf-8"))
    kl = raw.get("kl", {})
    models = dict(raw.get("models", {}))
    second = models.pop("second_model", None)
    prices_raw = dict(raw.get("prices", {}))
    discount = float(prices_raw.pop("batch_discount", kl.get("batch_discount", 0.5)))
    prices = {
        m: Price(float(v["input"]), float(v["output"]))
        for m, v in prices_raw.items()
        if isinstance(v, dict)
    }
    flag = str(kl.get("use_steinsaltz_aid", "")).strip().lower()
    th = raw.get("thresholds", {})
    cfg = KLConfig(
        mode=kl.get("mode", "personal"),
        allow_personal_only=bool(kl.get("allow_personal_only", False)),
        use_steinsaltz_aid_flag={"true": True, "false": False}.get(flag),
        steinsaltz_prefix=kl.get("steinsaltz_prefix", "Steinsaltz on "),
        anchor_format=kl.get("anchor_format", "quote"),
        second_model_fraction=float(kl.get("second_model_fraction", 0.10)),
        batch=bool(kl.get("batch", False)),
        batch_discount=discount,
        temperature=float(kl.get("temperature", 0.0)),
        max_tokens=int(kl.get("max_tokens", 8000)),
        window_amudim=int(kl.get("window_amudim", 3)),
        window_overlap=int(kl.get("window_overlap", 1)),
        budget_usd=float(kl.get("budget_usd", 0.0)),
        models=models,
        second_model=second,
        prices=prices,
        out_ratio={k: float(v) for k, v in raw.get("out_ratio", {}).items()},
        k3_not_found_max_fraction=float(th.get("k3_not_found_max_fraction", 0.10)),
        step_max_words=int(th.get("step_max_words", 25)),
        issue_similarity_warn=float(th.get("issue_similarity_warn", 0.6)),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    if cfg.anchor_format not in ("quote", "numbered"):
        raise ValueError("anchor_format must be quote|numbered")
    if cfg.mode not in ("personal", "public"):
        raise ValueError("mode must be personal|public")
    return cfg
