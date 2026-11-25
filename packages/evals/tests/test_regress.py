"""rav-evals regress over recorded fixtures (no API key): passes on the baseline, fails when a
candidate_as_proof is injected, and keeps rejected claims that were caught, caught."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from rav_evals.cli import main as cli_main
from rav_evals.config import BASELINE_PATH, FIXTURES_DIR
from rav_evals.kl_metrics import regress


def test_regress_passes_on_recorded_fixtures():
    res = regress(FIXTURES_DIR, BASELINE_PATH)
    assert res["passed"], res["failures"]
    assert res["fixtures"] >= 8
    assert res["metrics"]["V6"] == 0 and res["metrics"]["accepted_false_error"] == 0


def _copy(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    shutil.copytree(FIXTURES_DIR, d)
    return d


def test_regress_fails_on_injected_candidate_as_proof(tmp_path):
    d = _copy(tmp_path)
    p = d / "seed-relation.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    claim = data["output"]["answer"]["claims"][0]
    claim["citations"][0]["ref"] = "Seed Rishon 1a:1"  # endpoints no longer cited
    claim["citations"][0]["quoted_text"] = "פירוש לדוגמה"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    res = regress(d, BASELINE_PATH)
    assert not res["passed"]
    assert any("hard gate V6" in f for f in res["failures"])
    assert cli_main(["regress", "--fixtures", str(d), "--baseline", str(BASELINE_PATH)]) == 1


def test_regress_fails_on_rate_drop_beyond_tolerance(tmp_path):
    d = _copy(tmp_path)
    p = d / "seed-snif.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    data["output"]["answer"]["claims"][0]["flags"] = []  # the snif got filtered
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    res = regress(d, BASELINE_PATH)
    assert not res["passed"] and any("rubric_pass_rate dropped" in f for f in res["failures"])


def test_rejected_claim_that_was_caught_must_stay_caught(tmp_path):
    d = _copy(tmp_path)
    rc = {
        "fixture_id": "rejected-seed-direct-c9",
        "kind": "rejected_claim",
        "run_fixture": "seed-direct",
        "claim_index": 9,
        "error_type": "candidate_as_proof",
        "auto_caught": True,
        "claim": {
            "type": "system_conclusion",
            "flagged_as_system": True,
            "text": "יש כאן טעות סופר",
        },
    }
    (d / "rejected-seed-direct-c9.json").write_text(json.dumps(rc, ensure_ascii=False), "utf-8")
    assert regress(d, BASELINE_PATH)["passed"]  # V7 catches it
    rc["claim"]["text"] = "טענה רגילה"  # no longer caught by anything
    (d / "rejected-seed-direct-c9.json").write_text(json.dumps(rc, ensure_ascii=False), "utf-8")
    res = regress(d, BASELINE_PATH)
    assert not res["passed"] and any("no longer caught" in f for f in res["failures"])


def test_compare_writes_md_and_json(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    assert cli_main(["regress", "--out", str(a)]) == 0
    data = json.loads(a.read_text(encoding="utf-8"))
    data["label"] = "other"
    b.write_text(json.dumps(data), encoding="utf-8")
    assert cli_main(["compare", str(a), str(b), "--out-dir", str(tmp_path / "rep")]) == 0
    outs = sorted(p.suffix for p in (tmp_path / "rep").iterdir())
    assert outs == [".json", ".md"]
