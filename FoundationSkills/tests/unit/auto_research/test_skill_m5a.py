"""M5a (multi-objective campaigns) integration tests: claim gate, close mapping, rows pinning, back-compat."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from foundationskills.core.status import Status
from foundationskills.skills.auto_research import accept
from foundationskills.skills.auto_research.campaign import campaign_hash
from foundationskills.skills.auto_research.ledger import Ledger, ledger_files
from foundationskills.skills.auto_research.proposers import rows_digest
from foundationskills.skills.auto_research.skill import (
    AutoResearchSkill,
    _env_payload,
    _finding_message,
    _job_payload,
    _result,
    _spec,
    _val,
    _val_throughput,
)

try:  # the same relative-import approach test_skill_m3.py uses
    from .test_auto_research_skill import _ctx, _materialize, _request, _stage
except ImportError:  # pragma: no cover - a flat (non-package) test layout
    from tests.unit.auto_research.test_auto_research_skill import _ctx, _materialize, _request, _stage

APPROVER = "arbiter"
SEEDS = (101, 102, 103)
OBJECTIVES = [{"metric": "val_accuracy", "direction": "max"}, {"metric": "throughput", "direction": "max"}]
BASELINE = {101: _val_throughput(0.5, 100.0), 102: _val_throughput(0.501, 101.0), 103: _val_throughput(0.502, 102.0)}
TRADEOFF = _val_throughput(0.9, 50.0)  # wins val_accuracy (+0.4) and loses throughput (-50; tau ~2.8)
DOMINATING = _val_throughput(0.9, 150.0)  # wins both objectives
REGRESSING = _val_throughput(0.1, 50.0)  # loses both objectives
REFUSED = getattr(Status, "REFUSED", Status.RED)


def _mo_spec(**over: Any) -> dict[str, Any]:
    """Multi-objective spec: objectives[0] == objective, distinct metrics, none a guardrail (AR-IN-009)."""
    return _spec(objectives=[dict(o) for o in OBJECTIVES], **over)


def _rows(candidates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Baseline rows plus a 3-seed candidate set per trial (the baseline varies per seed: noise floor calibrates)."""
    rows = [_result("baseline", "baseline", seed, dict(m)) for seed, m in BASELINE.items()]
    for trial, metrics in candidates.items():
        rows += [_result(trial, "candidate", seed, dict(metrics)) for seed in SEEDS]
    return rows


def _messages(result: Any) -> list[tuple[str, str]]:
    return [(str(f.rule_id), _finding_message(f)) for f in result.findings]


def _claim_staged(tmp_path: Any, spec: dict[str, Any], trial: str, metrics: dict[str, Any],
                  base: dict[int, dict[str, Any]] = BASELINE) -> None:
    """Baseline plus one complete 3-seed confirm set for ``trial`` (job_submitted + trial_result per seed)."""
    campaign = str(spec.get("id") or "unnamed-campaign")
    events: list[tuple[str, str, str, dict[str, Any]]] = [
        ("campaign_approved", campaign, "-", {"spec_hash": campaign_hash(spec), "approver": APPROVER}),
        ("launch_envelope", campaign, "-", _env_payload(spec, {"max_runs": 6, "gpu_hours_total": 24.0})),
        *[("trial_result", campaign, "baseline", _result("baseline", "baseline", seed, dict(m)))
          for seed, m in base.items()],
    ]
    for seed in SEEDS:
        job = _job_payload(str(800 + seed), trial, kind="eval_only")
        job.update({"seed": seed, "phase": "confirm", "gpu_hours_est": 1.0})
        events.append(("job_submitted", campaign, trial, job))
        events.append(("trial_result", campaign, trial, _result(trial, "confirm", seed, dict(metrics))))
    for rel, text in ledger_files(events).items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _claim(tmp_path: Any, spec: dict[str, Any], trial: str) -> Any:
    return AutoResearchSkill().execute(
        _request(tmp_path, action="claim", campaign_spec=spec, campaign_confirm=campaign_hash(spec),
                 approver=APPROVER, trial=trial),
        _ctx(tmp_path),
    )


def _close(tmp_path: Any, spec: dict[str, Any], rows: list[dict[str, Any]]) -> Any:
    _stage(tmp_path, spec, rows)
    return AutoResearchSkill().execute(
        _request(tmp_path, action="close", campaign_spec=spec, campaign_confirm=campaign_hash(spec),
                 approver="tester", stop_reason="budget exhausted"),
        _ctx(tmp_path),
    )


def _docs(root: Any) -> list[dict[str, Any]]:
    """Every JSON document written under the workdir (the close report artefact among them)."""
    found = []
    for path in sorted(p for p in Path(root).rglob("*") if p.is_file()):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(doc, dict):
            found.append(doc)
    return found


def _lookup(doc: Any, key: str) -> Any:
    if not isinstance(doc, dict):
        return None
    if doc.get(key) is not None:
        return doc[key]
    for value in doc.values():
        hit = _lookup(value, key)
        if hit is not None:
            return hit
    return None


def _field(result: Any, tmp_path: Any, key: str) -> Any:
    """One close key read from the payload or from the report artefact underneath the workdir."""
    for doc in [result.payload, *_docs(tmp_path)]:
        hit = _lookup(doc, key)
        if hit is not None:
            return hit
    raise AssertionError("the close result carries no " + repr(key))


def _report(tmp_path: Any, *keys: str) -> dict[str, Any]:
    for doc in _docs(tmp_path):
        hits = {key: _lookup(doc, key) for key in keys}
        if all(value is not None for value in hits.values()):
            return hits
    raise AssertionError("no close report artefact carrying " + repr(keys))


def _digest(spec: dict[str, Any]) -> str:
    """proposers.rows_digest over an empty row set."""
    return rows_digest(spec, [], [], {}, [])


class TestClaimGate:
    """M5a claim gate (A6/AR-RS-008): weak-Pareto dominance is the only claimable verdict."""

    def test_a_tradeoff_confirm_set_is_refused_and_appends_nothing(self, tmp_path):
        spec = _mo_spec()
        _claim_staged(tmp_path, spec, "t1", TRADEOFF)
        before = len(Ledger(tmp_path / "ledger").entries())

        result = _claim(tmp_path, spec, "t1")

        assert result.status is REFUSED
        assert ("AR-RS-008", "claim_refused_not_dominating:t1") in _messages(result)
        assert len(Ledger(tmp_path / "ledger").entries()) == before

    def test_a_dominating_confirm_set_passes_and_appends_one_claim(self, tmp_path):
        spec = _mo_spec()
        _claim_staged(tmp_path, spec, "t1", DOMINATING)
        before = len(Ledger(tmp_path / "ledger").entries())

        result = _claim(tmp_path, spec, "t1")

        assert result.status is Status.PASS
        appended = Ledger(tmp_path / "ledger").entries()[before:]
        assert len(appended) == 1
        assert "claim" in str(appended[0]["op"]) or "chain" in str(appended[0]["op"])

    def test_a_single_objective_no_gain_set_keeps_firing_ar_rs_007(self, tmp_path):
        spec = _spec()
        base = {101: _val(0.5), 102: _val(0.502), 103: _val(0.501)}
        _claim_staged(tmp_path, spec, "t1", _val(0.501), base=base)

        result = _claim(tmp_path, spec, "t1")

        assert result.status is REFUSED
        assert ("AR-RS-007", "claim_no_gain:t1") in _messages(result)
        assert not [rule for rule, _ in _messages(result) if rule == "AR-RS-008"]


class TestCloseMapping:
    """M5a close mapping (A6): a trade-off is no_gain at RED with the AR-HO-008 disclosure, never a claim."""

    def test_a_tradeoff_only_campaign_is_no_gain_red_with_ar_ho_008(self, tmp_path):
        result = _close(tmp_path, _mo_spec(), _rows({"t1": TRADEOFF}))

        assert result.status is Status.RED
        assert _field(result, tmp_path, "outcome") == "no_gain"
        rules = [rule for rule, _ in _messages(result)]
        assert "AR-HO-001" in rules and "AR-HO-008" in rules
        assert result.payload["tradeoffs"] == ["t1"]
        assert result.payload["objectives"] == OBJECTIVES
        assert set(result.payload["frontier"]) >= {"entries", "excluded", "order"}
        assert result.payload["frontier"]["order"] == "presentation_only"
        report = _report(tmp_path, "tradeoffs", "frontier", "objectives")
        assert report["tradeoffs"] == ["t1"] and report["objectives"] == OBJECTIVES
        assert report["frontier"]["order"] == "presentation_only"

    def test_a_tradeoff_plus_regression_is_no_gain_not_regressed(self, tmp_path):
        result = _close(tmp_path, _mo_spec(), _rows({"t1": TRADEOFF, "t2": REGRESSING}))

        assert result.status is Status.RED
        assert _field(result, tmp_path, "outcome") == "no_gain"
        assert result.payload["tradeoffs"] == ["t1"]

    def test_a_dominating_candidate_improves_without_ar_ho_008(self, tmp_path):
        result = _close(tmp_path, _mo_spec(), _rows({"t1": DOMINATING}))

        assert result.status is not Status.RED
        assert _field(result, tmp_path, "outcome") == "improved"
        assert "AR-HO-008" not in [rule for rule, _ in _messages(result)]
        assert result.payload["tradeoffs"] == []


class TestObjectiveOnlyBackCompat:
    """M4 back-compat (A2/C1): an objective-only close stays byte-identical - no M5a keys, M4 decisions."""

    def test_an_objective_only_close_carries_no_m5a_keys_and_m4_decisions(self, tmp_path):
        spec = _spec()
        baseline = [_result("baseline", "baseline", seed, _val(v)) for seed, v in ((101, 0.5), (102, 0.502), (103, 0.501))]
        rows = [_result("t1", "candidate", seed, _val(0.7)) for seed in SEEDS]

        result = _close(tmp_path, spec, baseline + rows)

        assert not [key for key in ("objectives", "tradeoffs", "frontier") if key in result.payload]
        metric = spec["objective"]["metric"]
        assert _field(result, tmp_path, "decisions") == {"t1": accept.decide(spec, baseline, rows, metric)}


class TestRowsDigest:
    """Test-only pin: rows_digest hashes the whole spec, so objectives are frozen for free."""

    def test_the_rows_digest_pins_the_objectives(self):
        one, other = _mo_spec(), _mo_spec()
        assert _digest(one) == _digest(other)
        flipped = _spec(objectives=[{"metric": o["metric"], "direction": "min"} for o in OBJECTIVES])
        assert _digest(one) != _digest(flipped)


class TestMustFire:
    """The M5a must-fire fixtures are present and each fires its own rule id."""

    @pytest.mark.parametrize("rule", ["AR-IN-009", "AR-RS-008", "AR-HO-008"])
    def test_each_m5a_fixture_fires_its_own_rule(self, rule, tmp_path):
        fixtures = AutoResearchSkill().must_fire_fixtures()
        assert rule in fixtures
        workdir = tmp_path / rule
        workdir.mkdir(parents=True, exist_ok=True)
        request = _materialize(fixtures[rule], workdir)

        result = AutoResearchSkill().execute(request, _ctx(workdir))

        assert rule in [rid for rid, _ in _messages(result)]
