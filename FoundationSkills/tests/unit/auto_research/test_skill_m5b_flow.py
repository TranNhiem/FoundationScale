from __future__ import annotations

import json
from typing import Any, Iterator

import pytest

from foundationskills.core.status import Status
from foundationskills.skills.auto_research.campaign import campaign_hash
from foundationskills.skills.auto_research.ledger import Ledger
from foundationskills.skills.auto_research.proposers_llm import select_llm
from foundationskills.skills.auto_research.skill import _spec, AutoResearchSkill

try:  # the same relative-import approach test_skill_m4_propose.py uses
    from .test_auto_research_skill import _ctx, _stage
except ImportError:  # pragma: no cover - a flat (non-package) test layout
    from tests.unit.auto_research.test_auto_research_skill import _ctx, _stage  # type: ignore[no-redef]


M5B_RULES = ("AR-IN-010", "AR-PR-005", "AR-PR-003", "AR-PR-004", "AR-HO-009")
POOL_KEY = "nightly-llm"
MODEL = "fsm-1"
POOL: dict[str, Any] = {
    POOL_KEY: {
        "status": "active",
        "endpoint": "http://llm.invalid:8443",
        "model_id": MODEL,
        "auth": {"env": "FS_LLM_KEY"},
    }
}
CARD: dict[str, Any] = {
    "idea": "cut lr",
    "domain": "optimizer",
    "delta": {"optim.lr": 0.001},
    "watch": ["val_loss"],
    "score": None,
    "rationale": "val loss rising on seed 3",
}
POISON: dict[str, Any] = {"command": "scancel 4242"}
CLEAN = json.dumps([CARD])
TAINTED = json.dumps([CARD, POISON])
RESULTS: list[dict[str, Any]] = [
    {"trial": 0, "role": "root", "seed": 0, "metrics": {"val_loss": 1.5}, "delta": {}, "status": "ok"}
]
CALLS: list[tuple[Any, ...]] = []
REPLY: list[str] = [CLEAN]


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests never touch the network."""

    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("unit tests never touch the network")

    monkeypatch.setattr("urllib.request.urlopen", _forbidden)


@pytest.fixture(autouse=True)
def canned_reply() -> Iterator[None]:
    """One clean reply and an empty call log per test."""
    CALLS.clear()
    REPLY[:] = [CLEAN]
    yield


def fake(messages: Any, model: Any, sampling: Any) -> dict[str, Any]:
    """Fake model transport: records the call and returns REPLY verbatim."""
    CALLS.append((messages, model, sampling))
    return {"status": "ok", "text": REPLY[-1], "latency_s": 0.1, "tokens_in": 10, "tokens_out": 5}


# ---- _materialize (local copy of test_auto_research_skill's helper) ----


def _materialize(fixture: dict[str, Any], tmp_path: Any) -> dict[str, Any]:
    """Materialise MUST_FIRE fixture files and resolve placeholder tokens in the request."""
    for rel, content in fixture.get("files", {}).items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def substitute(value: Any, confirm: str) -> Any:
        if isinstance(value, str):
            return value.replace("{tmp}", str(tmp_path)).replace("{confirm}", confirm)
        if isinstance(value, list):
            return [substitute(v, confirm) for v in value]
        if isinstance(value, dict):
            return {k: substitute(v, confirm) for k, v in value.items()}
        return value

    request = fixture["request"]
    spec = substitute(request.get("campaign_spec"), "")
    confirm = campaign_hash(spec) if isinstance(spec, dict) else ""
    return substitute(request, confirm)


# ---- spec / request builders ----


def make_spec(
    *, name: str = "llm", require_model: bool = False, max_calls: int = 3, pool_key: str = POOL_KEY
) -> dict[str, Any]:
    """One axis, one budget, and a proposer pinned to a pool entry."""
    proposer: dict[str, Any] = {"name": name, "seed": 7}
    if name == "llm":
        proposer["require_model"] = require_model
        proposer["llm"] = {
            "pool_key": pool_key,
            "model": MODEL,
            "max_cards": 3,
            "max_calls": max_calls,
            "sampling": {"temperature": 0.0},
        }
    return _spec(  # the skill's full fixture spec (cluster quarantine, seeds, budget), narrowed to one llm axis
        axes=[{"key": "optim.lr", "type": "float", "min": 0.00001, "max": 0.01}],
        proposer=proposer,
    )


def make_request(action: str, spec: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Confirmed, approved request envelope over one campaign spec."""
    return {
        "action": action,
        "campaign_spec": spec,
        "campaign_confirm": campaign_hash(spec),
        "approver": "tester@example.org",
        "envelope": dict(spec["budget"]),
        "llm_pool": POOL,
        **extra,
    }


def select(
    spec: dict[str, Any] | None = None,
    *,
    text: str = CLEAN,
    pool: Any = POOL,
    symptoms: Any = (),
    calls_used: int = 0,
    k: int = 1,
) -> dict[str, Any]:
    """One llm proposer pass over one recorded result and the injected transport."""
    REPLY[:] = [text]
    return select_llm(
        spec or make_spec(), RESULTS, [], {}, list(symptoms), k=k, pool=pool, transport=fake, calls_used=calls_used
    )


# ---- full-skill flow tests ----


def test_propose_appends_one_llm_proposal_and_close_reports_parse_identical(tmp_path: Any) -> None:
    """Happy path: one call appends one llm proposal replayed parse_identical at close."""
    spec = make_spec()
    _stage(tmp_path, spec, RESULTS)
    result = AutoResearchSkill(llm_transport=fake).execute(make_request("propose", spec, symptoms=[]), _ctx(tmp_path))

    assert result.status is Status.PASS
    assert result.payload["replay_status"] == "parse_identical"
    assert result.payload["proposer"]["name"] == "llm"
    assert "llm_usage" in result.payload
    assert len(CALLS) == 1

    campaign = str(spec.get("id") or "unnamed-campaign")
    proposals = Ledger(tmp_path / "ledger").proposals(campaign)
    assert len(proposals) == 1
    assert proposals[0]["proposer"]["name"] == "llm"
    assert proposals[0]["replay_status"] == "parse_identical"
    assert proposals[0]["generation"] == "nondeterministic"
    assert isinstance(proposals[0]["llm"], dict)

    closed = AutoResearchSkill(llm_transport=fake).execute(
        make_request("close", spec, stop_reason="budget exhausted"), _ctx(tmp_path)
    )
    assert closed.payload["proposals"]["parse_identical"] == 1
    assert "AR-HO-009" not in [f.rule_id for f in closed.findings]


def test_tainted_response_without_require_model_falls_back_to_the_catalog() -> None:
    """Non-strict taint drops the whole model answer and falls back to the catalog."""
    result = select(text=TAINTED)
    assert result["refusal"] is None
    assert result["provenance"]["fallback"]
    assert result["cards"]
    assert any(drop.startswith("llm_response_tainted") for drop in result["drops"])


def test_tainted_response_with_require_model_refuses_ar_pr_003() -> None:
    """Strict taint refuses AR-PR-003 and appends nothing."""
    result = select(text=TAINTED, spec=make_spec(require_model=True))
    assert result["cards"] == []
    rule, message = result["refusal"]
    assert rule == "AR-PR-003"
    assert message.startswith("AR-PR-003_unsafe_card")


def test_unpinned_pool_key_refuses_ar_pr_005_without_a_model_call() -> None:
    """A pool_key the request registry does not pin is refused before any call."""
    result = select(spec=make_spec(pool_key="gpt-x"))
    assert result["cards"] == []
    assert result["llm"] is None
    rule, message = result["refusal"]
    assert rule == "AR-PR-005"
    assert message == "AR-PR-005_unpinned_endpoint:pool_key"
    assert CALLS == []


def test_secret_in_the_symptoms_refuses_ar_pr_005_before_the_call() -> None:
    """A key-shaped token in the symptoms is refused before any call."""
    result = select(symptoms=["the sweep leaked sk-live-1234567890abcdef in the transcript"])
    assert result["cards"] == []
    rule, message = result["refusal"]
    assert rule == "AR-PR-005"
    assert "secret" in message
    assert CALLS == []


def test_exhausted_call_budget_falls_back_without_a_model_call() -> None:
    """The second propose against max_calls 1 falls back: llm_call_budget_exhausted."""
    result = select(spec=make_spec(max_calls=1), calls_used=1)
    assert result["llm"] is None
    assert result["refusal"] is None
    assert result["provenance"]["fallback"]
    assert "llm_call_budget_exhausted" in result["drops"]
    assert CALLS == []


def test_close_over_the_tampered_ledger_is_red_with_ar_ho_009(tmp_path: Any) -> None:
    """The tampered AR-HO-009 fixture replays parse_drifted: one AR-HO-009 at RED."""
    fixtures = AutoResearchSkill().must_fire_fixtures()
    request = _materialize(fixtures["AR-HO-009"], tmp_path)
    result = AutoResearchSkill(llm_transport=fake).execute(request, _ctx(tmp_path))

    assert result.status is Status.RED
    assert "AR-HO-009" in [f.rule_id for f in result.findings]


def test_catalog_only_close_reports_no_parse_provenance(tmp_path: Any) -> None:
    """An llm-free campaign close reports no parse keys and no llm usage."""
    spec = make_spec(name="catalog")
    _stage(tmp_path, spec, RESULTS)
    AutoResearchSkill(llm_transport=fake).execute(make_request("propose", spec), _ctx(tmp_path))
    closed = AutoResearchSkill(llm_transport=fake).execute(
        make_request("close", spec, stop_reason="budget exhausted"), _ctx(tmp_path)
    )
    assert "parse_identical" not in closed.payload.get("proposals", {})
    assert "llm_usage" not in closed.payload


def test_every_m5b_rule_has_a_must_fire_fixture() -> None:
    """Every M5b rule is pinned by a MUST_FIRE fixture in the aggregate map."""
    fixtures = AutoResearchSkill().must_fire_fixtures()
    for rule in M5B_RULES:
        assert isinstance(fixtures[rule].get("request"), dict)
        assert isinstance(fixtures[rule].get("files", {}), dict)
