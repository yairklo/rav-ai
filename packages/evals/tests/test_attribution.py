"""Wilson CI and the misattribution gate (PASS / FAIL / UNDECIDED)."""

from __future__ import annotations

import pytest

from rav_evals.attribution import (
    attribution_gate,
    attribution_units,
    misattribution_report,
    needed_n,
    stratified_sample,
    wilson,
)
from rav_evals.schema import Answer, Citation, Claim, ClaimType


def test_wilson_known_values():
    lo, hi = wilson(5, 100)
    assert lo == pytest.approx(0.02154, abs=1e-4) and hi == pytest.approx(0.11175, abs=1e-4)
    lo, hi = wilson(0, 100)
    assert lo == 0.0 and hi == pytest.approx(0.03699, abs=1e-4)
    assert wilson(0, 0) == (0.0, 1.0)


def test_gate_pass_fail_undecided():
    assert attribution_gate(0, 400).verdict == "PASS"  # upper ~0.95% < 2%
    assert attribution_gate(3, 100).verdict == "FAIL"  # 3% >= 2%
    u = attribution_gate(1, 100)  # 1% but upper ~5.4%
    assert u.verdict == "UNDECIDED" and u.needed_n and u.needed_n > 100
    assert u.as_dict()["label"] == "לא מוכרע, הגדל מדגם"
    assert attribution_gate(0, 0).verdict == "UNDECIDED"


def test_needed_n_zero_rate():
    n = needed_n(0.0, 0.02)
    assert n == 189  # z^2 * (1/0.02 - 1) = 188.2
    assert wilson(0, n)[1] < 0.02 <= wilson(0, n - 1)[1]
    assert needed_n(0.03, 0.02) is None


def test_units_strata_and_report():
    a = Answer(
        disclaimer="d",
        claims=[
            Claim(
                type=ClaimType.INTERPRETATION,
                text="x",
                citations=[Citation(ref="R", quoted_text="q")],
            ),
            Claim(
                type=ClaimType.INTERPRETATION,
                text="y",
                citations=[Citation(ref="L", quoted_text="q", source_role="secondary_lens")],
            ),
            Claim(type=ClaimType.QUOTE, text="z", citations=[Citation(ref="R", quoted_text="q")]),
        ],
    )
    units = attribution_units("run1", "dispute", a) + attribution_units("run2", "novel", a)
    assert {u.source_role for u in units} == {"primary", "secondary_lens"} and len(units) == 4
    s1 = stratified_sample(units, 3, seed=1)
    assert (
        s1 == stratified_sample(units, 3, seed=1)
        and len({(u.category, u.source_role) for u in s1}) == 3
    )
    ratings = [{"unit_id": f"u{i}", "verdict": "ok"} for i in range(98)]
    ratings += [
        {
            "unit_id": "w1",
            "verdict": "wrong",
            "error_type": "wrong_attribution",
            "subtype": "hava_amina_as_conclusion",
        },
        {
            "unit_id": "w2",
            "verdict": "wrong",
            "error_type": "wrong_attribution",
            "subtype": "wrong_holder",
        },
    ]
    rep = misattribution_report(ratings)
    assert rep["n"] == 100 and rep["wrong"] == 2 and rep["verdict"] == "FAIL"
    assert rep["by_subtype"] == {"hava_amina_as_conclusion": 1, "wrong_holder": 1}
