"""M4 close: the replay re-check over every recorded proposal (C7) and the AR-HO-007 rule.

``close`` re-checks each ``ledger.proposals(campaign)`` entry before the seal. A byte-identical
replay is a clean handoff; a drifted replay fires exactly one AR-HO-007 finding at RED while the
outcome stays whatever it was; an unmeasured re-check is a declined claim counted
``proposal_replay_unmeasured:<reason>:<i>``. Recorded ``proposal`` ops are staged directly
(``ledger.append("proposal", ...)``) so these tests never depend on the propose branch. Stubs go in
through ``proposer_registry`` (B9); NEVER the MUST_FIRE fixtures. CPU-only: no optuna, no network.
"""
from __future__ import annotations

from typing import Any

from foundationskills.core.status import Status
from foundationskills.skills.auto_research.campaign import campaign_hash
from foundationskills.skills.auto_research.ledger import Ledger
from foundationskills.skills.auto_research.propose import propose as catalog_propose
from foundationskills.skills.auto_research.proposers import DEFAULT_K, CatalogProposer
from foundationskills.skills.auto_research.skill import AutoResearchSkill, _finding_message, _spec

try:  # the same relative-import approach test_skill_m3.py uses
    from .test_auto_research_skill import _baseline_rows, _ctx, _materialize, _request, _stage
except ImportError:  # pragma: no cover - a flat (non-package) test layout
    from tests.unit.auto_research.test_auto_research_skill import (  # type: ignore[no-redef]
        _baseline_rows,
        _ctx,
        _materialize,
        _request,
        _stage,
    )

CAMPAIGN = "ar-fixture"
TAMPERED = [{"idea": "tampered", "delta": {}, "score": None, "reason": "tampered"}]


class _BumpedVersionProposer:
    """A model stub whose package moved on: its recorded build can never replay (unmeasured)."""

    name = "optuna"
    version = "stub-1"
    package_version = "2.0"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def propose(self, spec: Any, results: Any, launches: Any, current: Any, symptoms: Any, *, k: int):
        return [{"idea": "stub-0", "delta": {"optim.lr": 2e-4}}], []


# ---- staging: one recorded `proposal` op plus one close (C7) ----------------

def _messages(result: Any) -> list[tuple[str, str]]:
    return [(str(f.rule_id), _finding_message(f)) for f in result.findings]


def _close_request(tmp_path: Any, spec: dict[str, Any]) -> dict[str, Any]:
    return _request(
        tmp_path, action="close", campaign_spec=spec, campaign_confirm=campaign_hash(spec),
        approver="tester", stop_reason="budget exhausted",
    )


def _record(
    cards: list[dict[str, Any]], results: Any, launches: Any, current: Any, symptoms: Any,
    *, proposer: dict[str, Any] | None = None, package_version: str | None = "builtin", replay: bool = True,
) -> dict[str, Any]:
    """One recorded ``proposal`` as the propose branch writes it; ``replay`` off is an M3 (C5-free) one."""
    record: dict[str, Any] = {
        "proposer": dict(proposer or {"name": "catalog", "version": "1"}),
        "requested": 3,
        "seed": 0,
        "rows_digest": None,
        "k": DEFAULT_K,
        "fallback": None,
        "cards": list(cards),
        "drops": [],
        "stats": {"out": len(cards), "dropped": 0},
    }
    if replay:
        record |= {
            "package_version": package_version,
            "replay_status": "byte_identical",
            "replay_inputs": {
                "current": current,
                "symptoms": symptoms,
                "results_count": len(results),
                "launches_count": len(launches),
            },
        }
    return record


def _close_staged(tmp_path: Any, spec: dict[str, Any], build: Any, *, skill: Any = None) -> Any:
    """Stage the baseline rows plus one recorded ``proposal`` (``build(ledger)``) and run one close."""
    _stage(tmp_path, spec, _baseline_rows())
    ledger = Ledger(tmp_path / "ledger")
    ledger.append("proposal", CAMPAIGN, "-", build(ledger))
    return (skill or AutoResearchSkill()).execute(_close_request(tmp_path, spec), _ctx(tmp_path))


class TestProposalReplayAtClose:
    """C7: drift is one AR-HO-007 at RED; a clean or unmeasured re-check never moves the status."""

    def test_a_clean_catalog_proposal_replays_byte_identically(self, tmp_path):
        spec, current, symptoms = _spec(), {"optim.lr": 1e-3}, ["loss diverging"]

        def build(ledger: Ledger) -> dict[str, Any]:
            results, launches = ledger.results(CAMPAIGN), ledger.launches(CAMPAIGN)
            cards = catalog_propose(spec, results, current, symptoms, k=DEFAULT_K, launches=launches)
            return _record(cards, results, launches, current, symptoms)

        result = _close_staged(tmp_path, spec, build)

        assert result.payload["proposals"] == {"checked": 1, "byte_identical": 1, "drifted": [], "unmeasured": 0}
        assert "AR-HO-007" not in [rule for rule, _ in _messages(result)]
        assert not [drop for drop in result.payload["drops"] if str(drop).startswith("proposal_replay_unmeasured")]

    def test_tampered_cards_drift_into_one_ar_ho_007_at_red_without_moving_the_outcome(self, tmp_path):
        spec, current, symptoms = _spec(), {"optim.lr": 1e-3}, ["loss diverging"]

        def clean(ledger: Ledger) -> dict[str, Any]:
            results, launches = ledger.results(CAMPAIGN), ledger.launches(CAMPAIGN)
            cards = catalog_propose(spec, results, current, symptoms, k=DEFAULT_K, launches=launches)
            return _record(cards, results, launches, current, symptoms)

        def forged(ledger: Ledger) -> dict[str, Any]:
            results, launches = ledger.results(CAMPAIGN), ledger.launches(CAMPAIGN)
            return _record(TAMPERED, results, launches, current, symptoms)

        baseline = _close_staged(tmp_path / "clean", spec, clean)
        result = _close_staged(tmp_path / "drifted", spec, forged)

        assert result.payload["proposals"] == {"checked": 1, "byte_identical": 0, "drifted": [0], "unmeasured": 0}
        assert result.payload["outcome"] == baseline.payload["outcome"]  # drift never moves the outcome (C7)
        assert result.status is Status.RED
        assert result.payload["recommendation"].startswith("investigate proposal replay drift before adopting: ")
        assert (
            "AR-HO-007",
            "proposal replay drifted at close: proposal(s) 0 (search provenance is not reproducible)",
        ) in _messages(result)

    def test_an_m3_record_without_replay_keys_is_a_declined_unmeasured_claim(self, tmp_path):
        spec = _spec()

        def build(ledger: Ledger) -> dict[str, Any]:
            results, launches = ledger.results(CAMPAIGN), ledger.launches(CAMPAIGN)
            return _record([{"idea": "m3"}], results, launches, {}, [], replay=False)

        result = _close_staged(tmp_path, spec, build)

        assert result.payload["proposals"] == {"checked": 1, "byte_identical": 0, "drifted": [], "unmeasured": 1}
        assert "proposal_replay_unmeasured:legacy_proposal:0" in result.payload["drops"]
        assert "proposal_replay_unmeasured:legacy_proposal:0" in result.payload["budgets"]["drops"]
        assert "AR-HO-007" not in [rule for rule, _ in _messages(result)]

    def test_a_moved_package_is_a_declined_version_changed_without_the_model(self, tmp_path):
        spec = _spec()

        def build(ledger: Ledger) -> dict[str, Any]:
            results, launches = ledger.results(CAMPAIGN), ledger.launches(CAMPAIGN)
            return _record(
                [{"idea": "stub-0", "delta": {"optim.lr": 2e-4}}], results, launches, {}, [],
                proposer={"name": "optuna", "version": "stub-1"}, package_version="1.0",
            )

        skill = AutoResearchSkill(proposer_registry={"catalog": CatalogProposer, "optuna": _BumpedVersionProposer})
        result = _close_staged(tmp_path, spec, build, skill=skill)

        assert result.payload["proposals"] == {"checked": 1, "byte_identical": 0, "drifted": [], "unmeasured": 1}
        assert "proposal_replay_unmeasured:version_changed:optuna:0" in result.payload["drops"]
        assert "AR-HO-007" not in [rule for rule, _ in _messages(result)]


class TestNewRuleFixtures:
    def test_the_ar_ho_007_fixture_fires_the_drift_finding(self, tmp_path):
        request = _materialize(AutoResearchSkill().must_fire_fixtures()["AR-HO-007"], tmp_path)

        result = AutoResearchSkill().execute(request, _ctx(tmp_path))

        assert result.status is Status.RED
        assert "AR-HO-007" in [rule for rule, _ in _messages(result)]
        assert result.payload["proposals"] == {"checked": 1, "byte_identical": 0, "drifted": [0], "unmeasured": 0}

    def test_the_ar_ho_007_fixture_carries_only_files_and_request(self):
        fixtures = AutoResearchSkill().must_fire_fixtures()
        assert "AR-HO-007" in fixtures
        assert set(fixtures["AR-HO-007"]) <= {"files", "request"}


class TestProposeThenClose:
    """End to end: the record the propose branch writes (C5) is exactly what close re-checks (C6/C7)."""

    def test_a_real_propose_then_close_replays_byte_identically(self, tmp_path):
        spec = _spec()
        _stage(tmp_path, spec, _baseline_rows())
        skill = AutoResearchSkill()
        propose_request = _request(
            tmp_path, action="propose", campaign_spec=spec, campaign_confirm=campaign_hash(spec),
            current={"optim.lr": 1e-3}, symptoms=["loss diverging"],
        )
        proposed = skill.execute(propose_request, _ctx(tmp_path))
        assert proposed.status is Status.PASS and proposed.payload["replay_status"] == "byte_identical"

        closed = skill.execute(_close_request(tmp_path, spec), _ctx(tmp_path))

        assert closed.payload["proposals"] == {"checked": 1, "byte_identical": 1, "drifted": [], "unmeasured": 0}
        assert "AR-HO-007" not in [rule for rule, _ in _messages(closed)]

    def test_a_nan_hint_propose_is_an_unmeasured_replay_at_close(self, tmp_path):
        spec = _spec()
        _stage(tmp_path, spec, _baseline_rows())
        skill = AutoResearchSkill()
        propose_request = _request(
            tmp_path, action="propose", campaign_spec=spec, campaign_confirm=campaign_hash(spec),
            current={"optim.lr": float("nan")}, symptoms=[],
        )
        assert skill.execute(propose_request, _ctx(tmp_path)).payload["replay_status"] == "unmeasured"

        closed = skill.execute(_close_request(tmp_path, spec), _ctx(tmp_path))

        assert closed.payload["proposals"]["unmeasured"] == 1
        assert "proposal_replay_unmeasured:legacy_proposal:0" in closed.payload["drops"]
