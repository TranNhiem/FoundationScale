"""M4 propose: the C5 replay record on every ``proposal`` and AR-PR-002 (C3) for ``optuna-cma``.

The proposal payload carries the self-contained replay record ``proposers.reverify`` replays at close:
``replay_inputs`` (the exact inputs ``select`` saw), ``package_version`` (C4) and ``replay_status``. The
record is NEVER faked: a ``current`` state that cannot be canonicalised (e.g. NaN) records None and
"unmeasured". ``optuna-cma`` (CMA-ES) is numeric-only: a categorical axis REFUSES (AR-PR-002) - it is
never quietly dropped. CPU-only: no optuna, no network, no fork.
"""
from __future__ import annotations

from typing import Any

from foundationskills.core.status import Status
from foundationskills.skills.auto_research.campaign import campaign_hash, check_proposer_spec
from foundationskills.skills.auto_research.ledger import Ledger
from foundationskills.skills.auto_research.skill import AutoResearchSkill, _finding_message, _spec

try:  # the same relative-import approach test_skill_m2_claim.py uses
    from .test_auto_research_skill import _baseline_rows, _ctx, _materialize, _request, _stage
except ImportError:  # pragma: no cover - a flat (non-package) test layout
    from tests.unit.auto_research.test_auto_research_skill import (  # type: ignore[no-redef]
        _baseline_rows,
        _ctx,
        _materialize,
        _request,
        _stage,
    )


# ---- small copies of test_skill_m3.py's helpers ----------------------------

def _messages(result: Any) -> list[tuple[str, str]]:
    return [(str(f.rule_id), _finding_message(f)) for f in result.findings]


def _proposal_payloads(tmp_path: Any) -> list[dict[str, Any]]:
    ledger = Ledger(tmp_path / "ledger")
    return [dict(ledger.payload(entry)) for entry in ledger.entries() if entry.get("op") == "proposal"]


def _propose(tmp_path: Any, current: dict[str, Any], symptoms: list[str]) -> Any:
    """One catalog ``propose`` over the staged fixture campaign (exactly one ``proposal`` op, B1)."""
    spec = _spec()
    _stage(tmp_path, spec, _baseline_rows())
    return AutoResearchSkill().execute(
        _request(
            tmp_path, action="propose", campaign_spec=spec, campaign_confirm=campaign_hash(spec),
            current=current, symptoms=symptoms,
        ),
        _ctx(tmp_path),
    )


class TestReplayRecord:
    """C5: the ``proposal`` payload is the ledgered symptom log - self-contained and never faked."""

    def test_the_proposal_record_carries_replay_inputs_package_version_and_replay_status(self, tmp_path):
        symptoms = ["the loss kinked at step 20"]

        result = _propose(tmp_path, {"optim.lr": 1e-3}, list(symptoms))

        assert result.status is Status.PASS
        ledger = Ledger(tmp_path / "ledger")
        record = _proposal_payloads(tmp_path)[0]
        assert record["replay_inputs"] == {
            "current": {"optim.lr": 1e-3},
            "symptoms": symptoms,
            "results_count": len(ledger.results("ar-fixture")),
            "launches_count": len(ledger.launches("ar-fixture")),
        }
        assert record["package_version"] == "builtin"       # C4: the catalog's own build
        assert record["replay_status"] == "byte_identical"  # canonical inputs AND a recorded version

    def test_the_propose_response_payload_carries_the_replay_status(self, tmp_path):
        result = _propose(tmp_path, {"optim.lr": 1e-3}, [])

        assert result.status is Status.PASS
        assert result.payload["replay_status"] == "byte_identical"

    def test_the_record_is_never_faked_when_the_current_state_cannot_be_canonicalised(self, tmp_path):
        result = _propose(tmp_path, {"optim.lr": float("nan")}, [])

        assert result.status is Status.PASS
        record = _proposal_payloads(tmp_path)[0]
        assert record["replay_inputs"] is None          # canonical() refused it: report the hole (C5)
        assert record["replay_status"] == "unmeasured"  # never 0 or False
        assert record["package_version"] == "builtin"
        assert result.payload["replay_status"] == "unmeasured"


class TestOptunaCmaAxes:
    """C3 / AR-PR-002: ``optuna-cma`` is numeric-only - a categorical axis REFUSES, never drops."""

    def test_optuna_cma_with_a_categorical_axis_refuses_on_a_check_request(self, tmp_path):
        spec = _spec(proposer={"name": "optuna-cma"})

        result = AutoResearchSkill().execute(
            _request(tmp_path, action="check", campaign_spec=spec, campaign_confirm=campaign_hash(spec)),
            _ctx(tmp_path),
        )

        assert result.status is Status.REFUSED
        assert ("AR-PR-002", "proposer_axis_unsupported:optuna-cma:train.method") in _messages(result)

    def test_one_finding_per_categorical_axis_in_axis_order_and_only_for_optuna_cma(self):
        axes = [
            {"key": "train.method", "type": "categorical", "values": ["full", "lora"]},
            {"key": "optim.lr", "type": "log_float", "min": 1e-6, "max": 1e-3},
            {"key": "optim.wd", "type": "categorical", "values": ["none", "0.1"]},
        ]

        assert check_proposer_spec(_spec(proposer={"name": "optuna-cma"}, axes=axes)) == [
            ("AR-PR-002", "proposer_axis_unsupported:optuna-cma:train.method"),
            ("AR-PR-002", "proposer_axis_unsupported:optuna-cma:optim.wd"),
        ]
        assert check_proposer_spec(_spec(proposer={"name": "optuna"}, axes=axes)) == []
        assert check_proposer_spec(_spec(proposer={"name": "catalog"}, axes=axes)) == []

    def test_a_numeric_only_axis_set_fires_no_ar_pr_002(self, tmp_path):
        spec = _spec(proposer={"name": "optuna-cma"}, axes=[
            {"key": "optim.lr", "type": "log_float", "min": 1e-6, "max": 1e-3},
            {"key": "lora.rank", "type": "int", "min": 4, "max": 64},
        ])
        assert check_proposer_spec(spec) == []  # the rule is proposer + axis scoped

        result = AutoResearchSkill().execute(
            _request(tmp_path, action="check", campaign_spec=spec, campaign_confirm=campaign_hash(spec)),
            _ctx(tmp_path),
        )

        assert result.status is not Status.REFUSED
        assert "AR-PR-002" not in {rule_id for rule_id, _ in _messages(result)}


class TestArPr002Fixture:
    """The AR-PR-002 MUST_FIRE fixture is a check request (pure data, C9)."""

    def test_the_ar_pr_002_in_code_fixture_refuses_on_a_check_request(self, tmp_path):
        fixtures = AutoResearchSkill().must_fire_fixtures()
        assert set(fixtures["AR-PR-002"]) <= {"files", "request"}

        request = _materialize(fixtures["AR-PR-002"], tmp_path)

        result = AutoResearchSkill().execute(request, _ctx(tmp_path))

        assert result.status is Status.REFUSED
        assert ("AR-PR-002", "proposer_axis_unsupported:optuna-cma:train.method") in _messages(result)
