"""pairs-sample (deterministic, balanced, swaps, 10% duplicates) and pairs-kappa."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from rav_evals.pairs import (
    BUCKETS,
    LABELS,
    PairCandidate,
    cohen_kappa,
    kappa_markdown,
    pairs_kappa,
    per_class,
    sample_pairs,
)


class FakeSource:
    def __init__(self, n=150, short=None):
        self.n, self.short = n, short or {}

    def _mk(self, tag):
        k = self.short.get(tag, self.n)
        return [PairCandidate(f"A {tag} {i}", f"B {tag} {i}") for i in range(k)]

    def explicit(self):
        return self._mk("explicit")

    def neighbors(self):
        return self._mk("neighbor")

    def hard_negatives(self):
        return self._mk("hard_negative")


def test_sample_is_deterministic_and_balanced():
    a = sample_pairs(FakeSource(), 200, seed=7)
    b = sample_pairs(FakeSource(), 200, seed=7)
    c = sample_pairs(FakeSource(), 200, seed=8)
    assert a.records == b.records
    assert a.records != c.records
    s = a.summary()
    assert s["pairs"] == 200 and s["by_bucket"] == {
        "explicit": 67,
        "neighbor": 67,
        "hard_negative": 66,
    }
    assert s["duplicates"] == 20 and s["records"] == 220
    assert 70 <= s["swapped"] <= 130  # ~50%
    base = {r["pair_id"]: r for r in a.records if r["dup_of"] is None}
    for d in (r for r in a.records if r["dup_of"]):
        o = base[d["dup_of"]]
        assert d["swapped"] is (not o["swapped"])  # shown in the reversed order
        assert (d["a_ref"], d["b_ref"]) == (o["a_ref"], o["b_ref"])
    assert all(r["ratings"] == [] for r in a.records)  # never pre-labelled
    assert {r["unit_kind"] for r in a.records} == {"segment_range"}


def test_short_bucket_is_filled_and_reported():
    res = sample_pairs(FakeSource(short={"hard_negative": 10}), 90, seed=1, duplicate_fraction=0)
    assert res.shortfall == {"hard_negative": 20}
    assert res.summary()["pairs"] == 90 and res.bucket_counts()["hard_negative"] == 10


def test_curated_pairs_go_first_and_todo_is_skipped():
    cur = [{"a_ref": "X 1", "b_ref": "Y 1", "note": "hard"}, {"a_ref": "TODO: x", "b_ref": "Z"}]
    res = sample_pairs(FakeSource(), 30, seed=3, curated=cur, duplicate_fraction=0)
    refs = {(r["a_ref"], r["b_ref"]) for r in res.records if r["source_bucket"] == "hard_negative"}
    assert ("X 1", "Y 1") in refs and not any("TODO" in a for a, _ in refs)


def test_kappa_known_values():
    a = ["same_as", "unrelated", "depends_on", "similar_to"] * 5
    assert cohen_kappa(a, a) == pytest.approx(1.0)
    # independent with equal marginals: po = pe = 0.5 -> kappa = 0
    x = ["same_as", "same_as", "unrelated", "unrelated"] * 10
    y = ["same_as", "unrelated", "same_as", "unrelated"] * 10
    assert cohen_kappa(x, y) == pytest.approx(0.0)
    assert cohen_kappa(["same_as"] * 4, ["same_as"] * 4) is None  # undefined, not faked
    pc = per_class(x, y)
    assert pc["sub_issue_of"] == {"support": 0, "precision": None, "recall": None}  # empty class
    assert pc["same_as"]["precision"] == pytest.approx(0.5)


def _pairs(n):
    return [
        {
            "pair_id": f"p{i}",
            "a_ref": f"a{i}",
            "b_ref": f"b{i}",
            "dup_of": None,
            "swapped": False,
            "ratings": [],
        }
        for i in range(n)
    ]


T0 = datetime(2026, 11, 1, tzinfo=UTC)


def _r(pid, rater, label, days=0, **kw):
    return {
        "pair_id": pid,
        "rater_id": rater,
        "label": label,
        "confidence": 2,
        "ts": (T0 + timedelta(days=days)).isoformat(),
        **kw,
    }


def test_pairs_kappa_two_raters_ci_and_confusion():
    pairs = _pairs(40)
    labels = [LABELS[i % 4] for i in range(40)]
    ratings = [_r(f"p{i}", "yair", labels[i]) for i in range(40)]
    ratings += [_r(f"p{i}", "second", labels[i]) for i in range(40)]
    rep = pairs_kappa(pairs, ratings, reps=200)
    assert rep["kappa"]["kappa"] == pytest.approx(1.0)
    lo, hi = rep["kappa"]["ci95"]
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)
    assert rep["confusion"]["same_as"]["same_as"] == 10
    assert rep["per_class"]["unrelated"]["recall"] is None  # class never used
    assert "kappa" in kappa_markdown(rep).lower()


def test_single_rater_reports_no_kappa_and_self_agreement():
    pairs = _pairs(20)
    ratings = [_r(f"p{i}", "yair", "same_as" if i % 2 else "unrelated") for i in range(20)]
    # blind recheck after 15 days on 2 pairs (10%): one agrees, one does not
    ratings += [
        _r("p1", "yair", "same_as", days=15, recheck=True),
        _r("p2", "yair", "same_as", days=15, recheck=True),
        _r("p3", "yair", "unrelated", days=3, recheck=True),
    ]  # too early: ignored
    rep = pairs_kappa(pairs, ratings)
    assert rep["kappa"] is None and "no kappa" in rep["kappa_note"]
    sa = rep["self_agreement"]["yair"]
    assert sa["n"] == 2 and sa["agreement"] == pytest.approx(0.5) and not sa["meets_target"]
    assert "no kappa" in kappa_markdown(rep)


def test_order_stability_uses_reversed_duplicates_and_direction():
    pairs = _pairs(2) + [
        {
            "pair_id": "p0-dup",
            "dup_of": "p0",
            "a_ref": "a0",
            "b_ref": "b0",
            "swapped": True,
            "ratings": [],
        }
    ]
    ratings = [
        _r("p0", "yair", "depends_on", direction="ab"),
        _r("p0-dup", "yair", "depends_on", direction="ab"),
    ]
    assert pairs_kappa(pairs, ratings)["order_stability"]["yair"] == {"n": 1, "agreement": 1.0}
    ratings[1] = _r("p0-dup", "yair", "depends_on", direction="ba")
    assert pairs_kappa(pairs, ratings)["order_stability"]["yair"]["agreement"] == 0.0


def test_buckets_constant():
    assert BUCKETS == ("explicit", "neighbor", "hard_negative")
