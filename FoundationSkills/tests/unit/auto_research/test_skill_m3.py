"""M3 wiring: one ``proposal`` op per ``propose`` and the model-gate rules (AR-IN-008 / AR-PR-001).

``propose`` is a MUTATING action (B1): it appends exactly one ``proposal`` op per call, and AR-LG-002
refuses it after ``campaign_closed``. The ideas catalog stays the byte-identity baseline; model
proposers are lazy optional extras whose failures are named, counted ``proposer_fallback_catalog:<reason>``
drops - except under ``proposer.require_model``, where AR-PR-001 refuses the request (or the model's
replay is not byte-identical). Unit tests inject stub proposers via the constructor kwarg
``proposer_registry`` (B9); NEVER the MUST_FIRE fixtures. CPU-only: no optuna, no network, no fork.
"""
from __future__ import annotations

from typing import Any

import pytest

from foundationskills.core.status import Status
from foundationskills.skills.auto_research.campaign import campaign_hash
from foundationskills.skills.auto_research.ledger import Ledger, canonical
from foundationskills.skills.auto_research.locks import MUTATING_ACTIONS
from foundationskills.skills.auto_research.propose import propose as catalog_propose
from foundationskills.skills.auto_research.proposers import DEFAULT_K, CatalogProposer
from foundationskills.skills.auto_research.skill import (
    AutoResearchSkill,
    _finding_message,
    _result,
    _spec,
    _val,
)

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

# The documented proposal payload key set (no budget keys, no run counts; M4/C5 adds the replay record).
PROPOSAL_KEYS = {"proposer", "requested", "seed", "rows_digest", "k", "fallback", "cards", "drops", "stats",
                 "replay_inputs", "package_version", "replay_status"}


# ---- model stubs (tests inject them through proposer_registry; fixtures never do) ----

def _stub_cards(k: int = DEFAULT_K) -> list[dict[str, Any]]:
    """Deterministic in-axes cards: every delta fits the fixture spec's optim.lr axis."""
    return [{"idea": f"stub-{index}", "delta": {"optim.lr": 2e-4}, "kind": "model"} for index in range(k)]


class _StubProposer:
    """A deterministic model stub: every fresh instance proposes byte-identical cards (verified)."""

    name = "optuna"
    version = "stub-1"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def propose(self, spec: Any, results: Any, launches: Any, current: Any, symptoms: Any, *, k: int):
        return _stub_cards(k), []


class _FlipProposer:
    """A model whose cards move under it: its replay can never be byte-identical (unverified)."""

    name = "optuna"
    version = "flip-1"
    calls = 0

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def propose(self, spec: Any, results: Any, launches: Any, current: Any, symptoms: Any, *, k: int):
        cls = type(self)
        cls.calls += 1
        return [{"idea": f"flip-{cls.calls}", "delta": {"optim.lr": 2e-4}, "kind": "model"}], []


class _LateFlipProposer:
    """Reproducible for the check-time select (calls 1-2), never again at run time (the model moved)."""

    name = "optuna"
    version = "late-1"
    calls = 0

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def propose(self, spec: Any, results: Any, launches: Any, current: Any, symptoms: Any, *, k: int):
        cls = type(self)
        cls.calls += 1
        idea = "late-0" if cls.calls <= 2 else f"late-{cls.calls}"
        return [{"idea": idea, "delta": {"optim.lr": 2e-4}, "kind": "model"}], []


def _registry(factory: Any) -> dict[str, Any]:
    """The test-only proposer registry (B9): the catalog baseline plus one model factory."""
    return {"catalog": CatalogProposer, "optuna": factory}


# ---- staging helpers -------------------------------------------------------

def _model_rows(trials: int = 2) -> list[dict[str, Any]]:
    """Measured candidate rows (one model row each): ok, non-limited, with recovered axis params (B2/B6)."""
    return [
        _result(f"t{index}", "candidate", 101, _val(0.5 + index * 0.01), delta={"optim.lr": 5e-5 * (index + 1)})
        for index in range(1, trials + 1)
    ]


def _optuna_spec(require_model: bool, min_rows: int = 10) -> dict[str, Any]:
    return _spec(proposer={"name": "optuna", "require_model": require_model, "min_rows": min_rows})


def _propose_request(tmp_path: Any, spec: dict[str, Any]) -> dict[str, Any]:
    return _request(
        tmp_path, action="propose", campaign_spec=spec, campaign_confirm=campaign_hash(spec),
        current={"optim.lr": 1e-3}, symptoms=[],
    )


def _messages(result: Any) -> list[tuple[str, str]]:
    return [(str(f.rule_id), _finding_message(f)) for f in result.findings]


def _joined(result: Any) -> str:
    parts = [str(getattr(result, "refusal", "")), str(getattr(result, "payload", ""))]
    parts.extend(_finding_message(f) for f in result.findings)
    return " ".join(parts)


def _ops(tmp_path: Any) -> list[str]:
    return [str(entry.get("op")) for entry in Ledger(tmp_path / "ledger").entries()]


def _proposal_payloads(tmp_path: Any) -> list[dict[str, Any]]:
    ledger = Ledger(tmp_path / "ledger")
    return [dict(ledger.payload(entry)) for entry in ledger.entries() if entry.get("op") == "proposal"]


class TestProposeLedger:
    """B1: one ``propose`` call appends exactly one ``proposal`` op carrying the catalog's cards."""

    def test_default_propose_matches_the_catalog_and_records_one_proposal(self, tmp_path):
        spec = _spec()
        _stage(tmp_path, spec, _baseline_rows())
        current, symptoms = {"optim.lr": 1e-3}, ["loss diverging"]

        result = AutoResearchSkill().execute(
            _request(tmp_path, action="propose", campaign_spec=spec, current=dict(current), symptoms=list(symptoms)),
            _ctx(tmp_path),
        )

        assert result.status is Status.PASS
        ledger = Ledger(tmp_path / "ledger")
        expected = catalog_propose(
            spec, ledger.results("ar-fixture"), current, symptoms, k=3, launches=ledger.launches("ar-fixture"),
        )
        assert canonical(result.payload["cards"]) == canonical(expected)
        payloads = _proposal_payloads(tmp_path)
        assert len(payloads) == 1
        assert payloads[0]["proposer"] == {"name": "catalog", "version": "1"}
        assert payloads[0]["fallback"] is None
        assert str(payloads[0]["rows_digest"]).startswith("sha256:")
        assert str(result.payload["rows_digest"]).startswith("sha256:")

    def test_every_propose_call_appends_exactly_one_proposal(self, tmp_path):
        spec = _spec()
        _stage(tmp_path, spec, _baseline_rows())
        skill = AutoResearchSkill()
        for step in (1, 2):
            result = skill.execute(_propose_request(tmp_path, spec), _ctx(tmp_path))
            assert result.status is Status.PASS
            assert _ops(tmp_path).count("proposal") == step  # never more than one per call

    def test_the_proposal_payload_carries_no_budget_or_run_count_keys(self, tmp_path):
        spec = _spec()
        _stage(tmp_path, spec, _baseline_rows())

        result = AutoResearchSkill().execute(_propose_request(tmp_path, spec), _ctx(tmp_path))

        assert result.status is Status.PASS
        payload = _proposal_payloads(tmp_path)[0]
        assert set(payload) == PROPOSAL_KEYS
        assert not set(payload) & {"budget", "gpu_hours", "gpu_hours_total", "max_runs", "runs", "runs_left"}
        assert payload["stats"] == {"out": len(payload["cards"]), "dropped": len(payload["drops"])}


class TestModelGate:
    """B3: non-strict falls back to the counted catalog; ``require_model`` refuses (AR-PR-001, B7)."""

    def test_non_strict_optuna_falls_back_to_the_catalog_with_a_counted_drop(self, tmp_path):
        spec = _optuna_spec(require_model=False)
        _stage(tmp_path, spec, [*_baseline_rows(), *_model_rows(2)])

        result = AutoResearchSkill().execute(_propose_request(tmp_path, spec), _ctx(tmp_path))

        assert result.status is Status.PASS
        payload = result.payload
        assert payload["proposer"] == {"name": "catalog", "version": "1"}
        assert payload["fallback"] == "below_min_rows:2/10"  # the row gate fires first (B7)
        assert "proposer_fallback_catalog:below_min_rows:2/10" in payload["drops"]
        assert payload["dropped"]["by_reason"]["proposer_fallback_catalog"] == 1
        assert set(payload["dropped"]["by_reason"]) <= {"proposer_fallback_catalog"} | {
            reason for reason in payload["dropped"]["by_reason"] if reason.startswith("catalog_")}
        assert payload["dropped"]["total"] == len(payload["drops"])
        assert _ops(tmp_path).count("proposal") == 1  # the counted fallback is never a refusal

    def test_strict_below_min_rows_refuses_and_writes_no_proposal(self, tmp_path):
        spec = _optuna_spec(require_model=True)
        _stage(tmp_path, spec, [*_baseline_rows(), *_model_rows(2)])

        result = AutoResearchSkill().execute(_propose_request(tmp_path, spec), _ctx(tmp_path))

        assert result.status is Status.REFUSED
        assert ("AR-PR-001", "proposer_unavailable:below_min_rows:2/10") in _messages(result)
        assert "proposer_unavailable:below_min_rows:2/10" in _joined(result)
        assert "proposal" not in _ops(tmp_path)

    def test_strict_passes_on_a_deterministic_in_axes_stub(self, tmp_path):
        spec = _optuna_spec(require_model=True)
        _stage(tmp_path, spec, [*_baseline_rows(), *_model_rows(10)])

        skill = AutoResearchSkill(proposer_registry=_registry(_StubProposer))
        result = skill.execute(_propose_request(tmp_path, spec), _ctx(tmp_path))

        assert result.status is Status.PASS
        assert result.payload["proposer"] == {"name": "optuna", "version": "stub-1"}
        assert canonical(result.payload["cards"]) == canonical(_stub_cards(DEFAULT_K))
        assert result.payload["fallback"] is None
        payloads = _proposal_payloads(tmp_path)
        assert len(payloads) == 1 and payloads[0]["proposer"] == {"name": "optuna", "version": "stub-1"}

    def test_strict_refuses_a_non_reproducible_model_as_unverified(self, tmp_path):
        spec = _optuna_spec(require_model=True)
        _stage(tmp_path, spec, [*_baseline_rows(), *_model_rows(10)])

        skill = AutoResearchSkill(proposer_registry=_registry(_FlipProposer))
        result = skill.execute(_propose_request(tmp_path, spec), _ctx(tmp_path))

        assert result.status is Status.REFUSED
        assert ("AR-PR-001", "proposer_unverified:optuna") in _messages(result)
        assert "proposer_unverified:optuna" in _joined(result)
        assert "proposal" not in _ops(tmp_path)


    def test_strict_gate_is_re_asserted_at_run_time(self, tmp_path):
        spec = _optuna_spec(require_model=True)
        _stage(tmp_path, spec, [*_baseline_rows(), *_model_rows(10)])
        _LateFlipProposer.calls = 0

        skill = AutoResearchSkill(proposer_registry=_registry(_LateFlipProposer))
        result = skill.execute(_propose_request(tmp_path, spec), _ctx(tmp_path))

        assert result.status is Status.REFUSED
        assert "proposer_unverified:optuna" in _joined(result)
        assert "proposal" not in _ops(tmp_path)


class TestProposeInputRules:
    def test_vizier_proposer_name_fires_ar_in_008(self, tmp_path):
        spec = _spec(proposer={"name": "vizier", "min_rows": 0})

        result = AutoResearchSkill().execute(
            _request(tmp_path, action="check", campaign_spec=spec, campaign_confirm=campaign_hash(spec)),
            _ctx(tmp_path),
        )

        assert result.status is Status.REFUSED
        assert "AR-IN-008" in [rule_id for rule_id, _ in _messages(result)]
        assert not (tmp_path / "ledger").exists()

    def test_propose_after_campaign_closed_is_refused_without_a_proposal(self, tmp_path):
        spec = _spec()
        _stage(tmp_path, spec, _baseline_rows())
        Ledger(tmp_path / "ledger").append(
            "campaign_closed", "ar-fixture", "-", {"stop_reason": "budget exhausted"},
        )

        result = AutoResearchSkill().execute(_propose_request(tmp_path, spec), _ctx(tmp_path))

        assert result.status is Status.REFUSED
        assert ("AR-LG-002", "campaign_closed") in _messages(result)
        assert "proposal" not in _ops(tmp_path)

    def test_propose_is_documented_as_a_mutating_action(self):
        assert "propose" in MUTATING_ACTIONS and len(MUTATING_ACTIONS) == 7


class TestNewRuleFixtures:
    @pytest.mark.parametrize("rule_id", ["AR-IN-008", "AR-PR-001"])
    def test_every_new_rule_id_has_an_in_code_fixture(self, rule_id):
        fixtures = AutoResearchSkill().must_fire_fixtures()
        assert rule_id in fixtures
        assert set(fixtures[rule_id]) <= {"files", "request"}

    def test_the_ar_pr_001_fixture_refuses_with_the_named_reason(self, tmp_path):
        request = _materialize(AutoResearchSkill().must_fire_fixtures()["AR-PR-001"], tmp_path)

        result = AutoResearchSkill().execute(request, _ctx(tmp_path))

        assert result.status is Status.REFUSED
        assert ("AR-PR-001", "proposer_unavailable:below_min_rows:2/10") in _messages(result)
        assert "proposal" not in _ops(tmp_path)

    def test_the_ar_in_008_fixture_refuses(self, tmp_path):
        request = _materialize(AutoResearchSkill().must_fire_fixtures()["AR-IN-008"], tmp_path)

        result = AutoResearchSkill().execute(request, _ctx(tmp_path))

        assert result.status is Status.REFUSED
        assert "AR-IN-008" in [rule_id for rule_id, _ in _messages(result)]
